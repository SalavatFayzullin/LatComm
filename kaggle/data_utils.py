"""Pinned LatComm data preparation and a disk-backed Example sequence.

No datasets.load_dataset materialization: original JSONL records are retained.
An offset index allows random access without retaining two million texts in RAM.
"""
from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np

from latcomm.data import Example, render_mcq_prompt

DATASET = "huaweilin/LatComm"
FOLDERS = {"stage1": "subsets/english_reconstruction_v1/original", "stage2": "original"}
SPLITS = {"train": "train", "dev": "validation", "fast_dev": "fast_validation", "test": "test"}
EXPECTED = {"stage1": {"train": 2_000_000, "dev": 15_043, "test": 14_983},
            "stage2": {"train": 282_441, "dev": 2_036, "test": 55_890}}
INDEX_VERSION = 1


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inspect_jsonl(path):
    digest, count, last = sha256(), 0, b""
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            count += block.count(b"\n")
            last = block[-1:]
    if last and last != b"\n":
        count += 1
    return {"sha256": digest.hexdigest(), "records": count, "bytes": Path(path).stat().st_size}


def prepare_data(root: Path, *, revision: str, profile: str = "full", token=None,
                 local_source: Path | None = None) -> dict:
    """Download original gz files at a pinned commit, or use a local source copy.

    `smoke` explicitly keeps the first 384 train records and 16 per eval split.
    Full data is never truncated, re-split, or re-formatted.
    """
    root = Path(root)
    if profile not in ("full", "smoke"):
        raise ValueError("profile must be full or smoke")
    receipt = root / "preparation.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["revision"] != revision or saved["profile"] != profile:
            raise ValueError("Prepared data revision/profile changed; use a separate data directory")
        for relative, info in saved["files"].items():
            path = root / relative
            if not path.is_file() or digest_file(path) != info["sha256"]:
                raise ValueError(f"Prepared data changed or is missing: {path}")
        return saved

    files = [f"{stage}/{split}.jsonl" for stage in FOLDERS for split in SPLITS]
    preexisting = all((root / relative).is_file() for relative in files)
    if not preexisting:
        root.mkdir(parents=True, exist_ok=True)
    result = {"dataset": DATASET, "revision": revision, "profile": profile,
              "source": "local" if local_source or preexisting else "huggingface", "files": {}}
    for stage, folder in FOLDERS.items():
        for split, remote_split in SPLITS.items():
            relative = f"{stage}/{split}.jsonl"
            target = root / relative
            if not preexisting:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = None
                if local_source:
                    local_source = Path(local_source)
                    candidates = (local_source / stage / f"{split}.jsonl",
                                  local_source / folder / f"{remote_split}.jsonl.gz")
                    source = next((p for p in candidates if p.is_file()), None)
                    if source is None:
                        raise FileNotFoundError(f"No local data file for {stage}/{split}; expected {candidates}")
                else:
                    from huggingface_hub import hf_hub_download
                    source = Path(hf_hub_download(DATASET, f"{folder}/{remote_split}.jsonl.gz",
                                                  repo_type="dataset", revision=revision, token=token))
                temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
                opener = gzip.open if source.suffix == ".gz" else open
                try:
                    with opener(source, "rb") as src, temporary.open("wb") as out:
                        if profile == "smoke":
                            limit = 384 if split == "train" else 16
                            for index, line in enumerate(src):
                                if index >= limit:
                                    break
                                out.write(line)
                        else:
                            shutil.copyfileobj(src, out, length=4 << 20)
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            info = inspect_jsonl(target)
            expected = EXPECTED[stage].get(split) if profile == "full" else (384 if split == "train" else 16)
            if not info["records"] or expected is not None and info["records"] != expected:
                raise ValueError(f"Unexpected record count in {target}: {info['records']} (expected {expected})")
            result["files"][relative] = info
            print(f"Data {relative}: {info['records']:,} records", flush=True)
    if os.access(root, os.W_OK):
        atomic_json(receipt, result)
    return result


def parse_example(record, data, where="record"):
    try:
        context = "\n\n".join(part["text"] for part in record["evidence"])
        if not isinstance(context, str) or not context:
            raise ValueError("empty context")
        response = record["response"]
        if data.response_format == "mcq":
            choices = record["choices"]
            if not 2 <= len(choices) <= 26 or not all(isinstance(c, str) for c in choices):
                raise ValueError("invalid multiple-choice options")
            if len(response) != 1 or response not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]:
                raise ValueError("invalid gold option letter")
            prompt = render_mcq_prompt(record["question"], choices, data.mcq_template)
            num_choices = len(choices)
        else:
            if response != context:
                raise ValueError("Stage 1 response must equal context")
            if record.get("choices"):
                raise ValueError("Stage 1 cannot have answer choices")
            prompt, num_choices = record["question"], 0
        return Example(record["sample_id"], record["provenance"]["dataset"], context,
                       prompt, response, num_choices)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid LatComm {where}: {error}") from error


class IndexedExamples(Sequence):
    """Memory-map [byte offset, context character count] pairs for JSONL.

    Indexes live in a writable cache, never next to read-only Kaggle Input data.
    Concurrent ranks publish their own temporary index by atomic rename; every
    index describes the same immutable content hash, so publication is race-safe.
    """

    def __init__(self, path, data, *, index_dir=None):
        self.path = Path(path).resolve()
        self.data = data
        self.sha256 = digest_file(self.path)
        root = Path(index_dir or os.environ.get("LATCOMM_INDEX_DIR", "/kaggle/working/latcomm-index"))
        if os.name == "nt" and index_dir is None and "LATCOMM_INDEX_DIR" not in os.environ:
            root = self.path.parent / ".latcomm-index"
        root.mkdir(parents=True, exist_ok=True)
        identity = sha256((self.sha256 + str(INDEX_VERSION)).encode()).hexdigest()
        target = root / f"{identity}.idx"
        meta = root / f"{identity}.json"
        if not meta.exists() or not target.exists():
            temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{time.time_ns()}")
            count = 0
            try:
                with self.path.open("rb") as src, temporary.open("wb") as out:
                    while True:
                        offset = src.tell()
                        line = src.readline()
                        if not line:
                            break
                        record = json.loads(line)
                        context = "\n\n".join(part["text"] for part in record["evidence"])
                        out.write(struct.pack("<QQ", offset, len(context)))
                        count += 1
                    out.flush()
                    os.fsync(out.fileno())
                # Replacing an identical mmap is unnecessary on Windows.
                if not target.exists():
                    os.replace(temporary, target)
                else:
                    temporary.unlink()
                atomic_json(meta, {"version": INDEX_VERSION, "sha256": self.sha256, "records": count})
            finally:
                temporary.unlink(missing_ok=True)
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info["sha256"] != self.sha256 or target.stat().st_size != info["records"] * 16:
            raise ValueError(f"Invalid offset index for {self.path}")
        self._count = info["records"]
        self._index = np.memmap(target, dtype="<u8", mode="r", shape=(self._count, 2)) if self._count else None
        self.context_lengths = self._index[:, 1] if self._count else np.empty(0, dtype=np.uint64)
        self._handle = None
        self._pid = None

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._pid = os.getpid()
        self._handle.seek(int(self._index[index, 0]))
        return parse_example(json.loads(self._handle.readline()), self.data, f"{self.path}:{index + 1}")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = state["_pid"] = None
        return state

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()
