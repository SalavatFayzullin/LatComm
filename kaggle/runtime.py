"""Kaggle execution adapter for the unmodified LatComm implementation.

The bridge keeps upstream's FP32 master parameters and FP32 arithmetic. Only
frozen backbones use the selected 16-bit format. FP16 is a hardware adaptation,
not a bitwise reproduction of the paper's BF16 experiment.

Checkpoints are written only between complete optimizer updates. Exit status 75
means that the current session stopped at its requested limit and can resume.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
from importlib.metadata import version
import json
from itertools import islice
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

if os.environ.get("LATCOMM_REPO"):
    sys.path.insert(0, os.environ["LATCOMM_REPO"])
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import IndexedExamples
from progress import CounterProgress, ProgressLoader, TrainingProgress, configure_logging, format_metrics
from latcomm.checkpoint import load_bridge, save_bridge
from latcomm.config import LatCommConfig, load_config
from latcomm.data import Collator, LengthSortedBatchSampler, ShuffleSampler, pad, tokenize
from latcomm.distributed import DistributedContext, initialize_distributed
from latcomm.evaluate import evaluate as upstream_evaluate, score_responses
from latcomm.model import build_model, load_tokenizer, response_logits
from latcomm.packets import PacketCache, context_key, to_stored
from latcomm.train import build_scheduler, seed_everything

LOGGER = logging.getLogger("latcomm.kaggle")
FORMAT_VERSION = 1
LOSS_NORMALIZER = 32768.0
SESSION_LIMIT_EXIT = 75
SHARD_ROWS = 64
BEST_POLICY = {
    "metric": "loss", "mode": "min", "comparison": "strictly_lower",
    "ties": "keep_existing", "stage1_step_zero": False, "stage2_step_zero": True,
}


def memory_snapshot() -> dict[str, float]:
    result = {}
    if torch.cuda.is_available():
        result["gpu_allocated_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        result["gpu_reserved_gib"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
        result["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    try:
        import psutil
        result["process_ram_gib"] = round(psutil.Process().memory_info().rss / 1024**3, 3)
        result["system_ram_available_gib"] = round(psutil.virtual_memory().available / 1024**3, 3)
    except ImportError:
        pass
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit = int(Path(limit_path).read_text().strip())
            used = int(Path(used_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 2**60:
            result["container_ram_limit_gib"] = round(limit / 1024**3, 3)
            result["container_ram_used_gib"] = round(used / 1024**3, 3)
            break
    return result


class MicrobatchProgress:
    """Report completed work without CUDA synchronization or RNG changes."""

    def __init__(self, rank: int, *, interval: float = 30.0, summary=None):
        self.rank = rank
        self.interval = interval
        self.summary = summary
        self.completed = 0
        self.retry = 0

    def start(self, next_step: int, target: int, *, phase: str = "train") -> None:
        self.next_step = next_step
        self.target = target
        self.mode = phase
        self.completed = 0
        self.retry = 0
        self.started = self.last_report = self.batch_started = time.monotonic()
        self.emit("update_start", microbatches=target)

    def emit(self, kind: str, **values) -> None:
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("%s", json.dumps({"type": kind, "rank": self.rank,
                                          "next_step": self.next_step, "mode": self.mode,
                                          "retry": self.retry, **memory_snapshot(), **values}))

    def phase(self, name: str) -> None:
        if self.completed == 0:
            self.emit("microbatch_phase", microbatch=1, phase=name)

    def begin_microbatch(self) -> None:
        self.batch_started = time.monotonic()

    def finish_microbatch(self) -> None:
        self.completed += 1
        now = time.monotonic()
        if self.completed == 1 or self.completed == self.target or now - self.last_report >= self.interval:
            elapsed = now - self.started
            self.emit("microbatch_progress", microbatch=self.completed, accumulation=self.target,
                      microbatch_seconds=round(now - self.batch_started, 3),
                      elapsed_seconds=round(elapsed, 3))
            self.last_report = now
        if self.summary:
            self.summary.microbatch(self.next_step, self.completed, self.target, now - self.started)

    def repeat(self, attempt: int, old_scale: float, new_scale: float) -> None:
        self.retry = attempt
        if self.rank == 0:
            LOGGER.warning("Step %d: FP16 overflow; retry %d of the same %d microbatches, loss scale %g -> %g",
                           self.next_step, attempt, self.target, old_scale, new_scale)
        self.emit("fp16_retry", attempt=attempt, microbatches_to_repeat=self.target,
                  old_loss_scale=old_scale, new_loss_scale=new_scale)
        self.completed = 0


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def precision_config(config: LatCommConfig, precision: str) -> LatCommConfig:
    """Bypass the upstream dtype literal only for frozen FP16 model loading."""
    dtype = "float16" if precision == "fp16" else "bfloat16"
    return config.model_copy(update={
        "encoder": config.encoder.model_copy(update={"dtype": dtype}),
        "receiver": config.receiver.model_copy(update={"dtype": dtype}),
    })


def enable_fp16_stability(precision: str) -> None:
    """Keep Qwen3.5's fallback Q/K normalization backward out of FP16.

    Transformers 5.15.1 normalizes Q/K before the delta-rule fallback upcasts
    them. At zero or small vectors, rsqrt backward overflows its FP16
    intermediates even when the output and upstream gradient are finite.
    Reducing the loss scale cannot fix that intermediate overflow. Keep this
    normalization in FP32 and cast its result back to the backbone dtype.
    The frozen weights, architecture, epsilon and training objective stay put.
    """
    if precision != "fp16":
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5

    original = modeling_qwen3_5.l2norm
    if getattr(original, "_latcomm_fp32", False):
        return

    def stable_l2norm(x, dim=-1, eps=1e-6):
        if x.dtype != torch.float16:
            return original(x, dim=dim, eps=eps)
        work = x.float()
        return (work * torch.rsqrt((work * work).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)

    stable_l2norm._latcomm_fp32 = True
    modeling_qwen3_5.l2norm = stable_l2norm
    LOGGER.debug("FP16 stability: Qwen3.5 fallback Q/K L2 normalization computes in FP32")


def build_runtime_model(config, precision: str, device, **kwargs):
    enable_fp16_stability(precision)
    return build_model(precision_config(config, precision), device, **kwargs)


def code_identity(source_commit: str) -> dict[str, Any]:
    import latcomm
    package = Path(latcomm.__file__).resolve().parent
    files = {path.name: file_hash(path) for path in sorted(package.glob("*.py"))}
    files["kaggle/runtime.py"] = file_hash(__file__)
    helper = Path(__file__).with_name("data_utils.py")
    if helper.exists():
        files["kaggle/data_utils.py"] = file_hash(helper)
    files["kaggle/progress.py"] = file_hash(Path(__file__).with_name("progress.py"))
    libraries = {name: version(name) for name in ("torch", "transformers", "numpy", "pydantic", "safetensors")}
    libraries["python"] = sys.version.split()[0]
    return {"source_commit": source_commit or os.environ.get("LATCOMM_SOURCE_COMMIT", ""),
            "files": files, "libraries": libraries}


def compressor_hash(bridge) -> str:
    digest = sha256()
    for name, value in sorted(bridge.compressors.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_compressor_hash(checkpoint: str | Path) -> str:
    from safetensors import safe_open
    digest = sha256()
    with safe_open(str(Path(checkpoint) / "bridge.safetensors"), framework="pt", device="cpu") as reader:
        for name in sorted(key for key in reader.keys() if key.startswith("compressors.")):
            tensor = reader.get_tensor(name).contiguous()
            digest.update(name.removeprefix("compressors.").encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_stage1_checkpoint(checkpoint: str | Path, config) -> dict[str, Any]:
    root = Path(checkpoint)
    info = read_json(root / "checkpoint.json")
    saved = info["config"]
    for side in ("encoder", "receiver", "bridge"):
        current = getattr(config, side).model_dump(mode="json")
        if saved[side] != current:
            raise ValueError(f"Stage 1 checkpoint {side} settings differ from this stage")
    if saved["training"]["trainable"] != "all":
        raise ValueError("Stage 2 initialization and packet encoding require a Stage 1 bridge")
    metadata = root / "runtime.json"
    if metadata.exists() and not read_json(metadata).get("complete", False):
        raise ValueError("Stage 1 is paused; finish it before encoding or starting Stage 2")
    return info


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), torch.cuda.current_device())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def packet_fingerprint(directory: str | Path) -> dict[str, Any]:
    """Verify payloads as well as their index; moving an intact cache is allowed."""
    root = Path(directory)
    meta = read_json(root / "meta.json")
    if not meta.get("complete", False) or "provenance" not in meta:
        raise ValueError("Packet cache is incomplete or lacks runtime provenance; run runtime.py encode first")
    indices = {path.name: file_hash(path) for path in sorted(root.glob("index-rank*.json"))}
    if not indices:
        raise ValueError("Packet cache has no indices")
    referenced: set[str] = set()
    count = 0
    for name in indices:
        entries = read_json(root / name)
        count += len(entries)
        referenced.update(item[0] for item in entries.values())
    for name in sorted(referenced):
        expected = meta.get("shards", {}).get(name)
        if expected is None or file_hash(root / name) != expected:
            raise ValueError(f"Packet shard changed or has no checksum: {name}")
    if count != meta["contexts"]:
        raise ValueError("Packet cache index count does not match its manifest")
    return {"meta": meta, "indices": indices}


def settings_fingerprint(
    config: LatCommConfig, precision: str, distributed: DistributedContext,
    source_commit: str, train_path: Path, eval_path: Path, packet_identity: Any,
) -> dict[str, Any]:
    settings = config.model_dump(mode="json")
    # Paths can change when Kaggle outputs are attached to a subsequent session.
    # Contents, architecture, optimizer settings and the complete schedule cannot.
    settings["training"].pop("output_dir")
    checkpoint = settings["training"].pop("init_checkpoint")
    settings["data"].pop("dataset_root")
    settings["data"].pop("packet_cache")
    init_hash = file_hash(Path(checkpoint) / "bridge.safetensors") if checkpoint else None
    return {
        "format": FORMAT_VERSION, "settings": settings, "precision": precision,
        "world_size": distributed.world_size, "code": code_identity(source_commit),
        "data": {"train": file_hash(train_path), "eval": file_hash(eval_path)},
        "initial_bridge": init_hash, "packets": packet_identity,
        "loss_normalizer": LOSS_NORMALIZER,
    }


class OffsetBatchSampler:
    """Skip already-used indices before tokenization and packet loading."""
    def __init__(self, base):
        self.base = base
        self.start = 0
        self.full_length = len(base)

    def __iter__(self):
        return islice(iter(self.base), self.start, None)

    def __len__(self):
        return max(0, self.full_length - self.start)


def make_loader(examples, collator, config, distributed, *, training: bool) -> tuple[DataLoader, Any]:
    """Use upstream samplers without materializing two million Example objects."""
    options: dict[str, Any] = {
        "collate_fn": collator, "num_workers": config.data.num_workers,
        "persistent_workers": config.data.num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
        # DataLoader seeding must never consume the model's restored RNG stream.
        "generator": torch.Generator().manual_seed(config.training.seed + distributed.rank),
    }
    if not training:
        from torch.utils.data import Subset
        subset = Subset(examples, range(distributed.rank, len(examples), distributed.world_size))
        return DataLoader(subset, batch_size=config.training.per_device_eval_batch_size, **options), None
    sampler = ShuffleSampler(len(examples), seed=config.training.seed,
                             rank=distributed.rank, world_size=distributed.world_size)
    batch_size = config.training.per_device_batch_size
    if config.data.sort_within_update:
        lengths = examples.context_lengths
        batch_sampler = LengthSortedBatchSampler(
            sampler, lengths, batch_size=batch_size,
            accumulation=config.training.gradient_accumulation_steps,
        )
    else:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    return DataLoader(examples, batch_sampler=OffsetBatchSampler(batch_sampler), **options), sampler


def save_training_checkpoint(output, name, model, config, optimizer, scheduler, scaler,
                             distributed, fingerprint, progress, *, complete=False, statistics=None) -> Path:
    all_rng = distributed.all_gather(rng_state())
    all_statistics = distributed.all_gather(statistics or {})
    destination = output / "checkpoints" / name
    if distributed.is_main:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{name}.incomplete-{os.getpid()}")
        if temporary.exists():
            # A previous interruption may leave only this known, local staging directory.
            resolved = temporary.resolve()
            if resolved.parent != destination.parent.resolve():
                raise ValueError("Checkpoint staging directory escaped output/checkpoints")
            shutil.rmtree(resolved)
        save_bridge(temporary, model.bridge, config, progress["step"])
        state = {"format": FORMAT_VERSION, "fingerprint": fingerprint,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": all_rng,
                 "statistics": all_statistics, "progress": progress, "complete": complete}
        torch.save(state, temporary / "training_state.pt")
        atomic_json(temporary / "runtime.json", {
            "progress": progress, "complete": complete,
            "fingerprint_sha256": canonical_hash(fingerprint),
            "precision": fingerprint["precision"],
            "world_size": distributed.world_size,
        })
        if destination.exists():
            # Never overwrite a published checkpoint. Same-step session saves can
            # simply reuse it only when their exact state/progress is already saved.
            existing = read_json(destination / "runtime.json")
            if (existing["progress"] != progress or existing["complete"] != complete
                    or existing["fingerprint_sha256"] != canonical_hash(fingerprint)):
                raise ValueError(f"Checkpoint already exists with different progress: {destination}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        atomic_json(output / "latest.json", {
            "path": str(destination.relative_to(output)), "step": progress["step"], "complete": complete,
        })
        # Retention follows publication, so interruption cannot remove the last
        # usable checkpoint. Restrict deletion to our exact names and directory.
        checkpoint_root = (output / "checkpoints").resolve()
        checkpoints = sorted(
            (path for path in checkpoint_root.iterdir()
             if re.fullmatch(r"step-\d{8}", path.name) and path.is_dir()
             and (path / "training_state.pt").is_file()),
            key=lambda path: int(path.name[5:]), reverse=True,
        )
        for old in checkpoints[2:]:
            if old.resolve().parent != checkpoint_root:
                raise ValueError("Checkpoint retention path escaped output/checkpoints")
            shutil.rmtree(old)
    distributed.barrier()
    return destination


def read_best_checkpoint(output: Path, fingerprint: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate a persisted selection without rebinding it to another experiment."""
    pointer = output / "best.json"
    if not pointer.exists():
        return None
    best = read_json(pointer)
    if (best.get("format") != FORMAT_VERSION or best.get("policy") != BEST_POLICY
            or best.get("fingerprint_sha256") != canonical_hash(fingerprint)
            or best.get("provenance") != fingerprint):
        raise ValueError("Best checkpoint rejected: source, data, settings or selection policy changed")
    step = best.get("step")
    value = best.get("value")
    if (type(step) is not int or step < 0 or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError("Best checkpoint has invalid validation loss or step")
    relative = f"checkpoints/best-step-{step:08d}"
    root = (output / "checkpoints").resolve()
    checkpoint = (output / relative).resolve()
    if best.get("path") != relative or checkpoint.parent != root:
        raise ValueError("Best checkpoint path escaped output/checkpoints")
    metadata = read_json(checkpoint / "runtime.json")
    info = read_json(checkpoint / "checkpoint.json")
    if (metadata.get("kind") != "validation_best_bridge" or metadata.get("resumable") is not False
            or metadata.get("complete") is not False or metadata.get("selection") != best
            or info.get("step") != step or info.get("config") != best.get("config")
            or (checkpoint / "training_state.pt").exists()
            or file_hash(checkpoint / "bridge.safetensors") != best.get("bridge_sha256")):
        raise ValueError("Best checkpoint metadata or bridge checksum changed")
    return best


def save_best_checkpoint(output, model, config, fingerprint, progress, metrics):
    """Publish bridge-only validation minima; never change exact-resume state.

    Step zero participates only in Stage 2. Non-finite losses are not candidates;
    equal losses retain the already selected checkpoint. Publication precedes
    retention so a crash always leaves the previous or the new selection usable.
    Called on the main rank inside the evaluation RNG-preservation block.
    """
    best = read_best_checkpoint(output, fingerprint)
    step, loss = progress["step"], float(metrics["loss"])
    if (not math.isfinite(loss) or (step == 0 and config.training.trainable == "all")
            or (best is not None and loss >= best["value"])):
        return best
    name = f"best-step-{step:08d}"
    root = (output / "checkpoints").resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    temporary = root / f".{name}.incomplete-{os.getpid()}"
    for path in (destination, temporary):
        if path.resolve().parent != root:
            raise ValueError("Best checkpoint staging path escaped output/checkpoints")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_bridge(temporary, model.bridge, config, step)
    selected = {
        "format": FORMAT_VERSION, "policy": dict(BEST_POLICY),
        "metric": "loss", "value": loss, "step": step,
        "path": f"checkpoints/{name}", "fingerprint_sha256": canonical_hash(fingerprint),
        "bridge_sha256": file_hash(temporary / "bridge.safetensors"),
        "config": config.model_dump(mode="json"), "provenance": fingerprint,
    }
    metadata = {"kind": "validation_best_bridge", "resumable": False, "complete": False,
                "progress": dict(progress), "fingerprint_sha256": canonical_hash(fingerprint),
                "selection": selected}
    atomic_json(temporary / "runtime.json", metadata)
    if destination.exists():
        # An interruption may have published the bridge before best.json. Reuse
        # only the identical candidate, never overwrite an immutable directory.
        existing = read_json(destination / "runtime.json")
        if (existing != metadata
                or file_hash(destination / "bridge.safetensors") != selected["bridge_sha256"]):
            raise ValueError(f"Best checkpoint already exists with different state: {destination}")
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, destination)
    atomic_json(output / "best.json", selected)
    for old in root.iterdir():
        if old.name == name or not re.fullmatch(r"best-step-\d{8}", old.name) or not old.is_dir():
            continue
        if old.resolve().parent != root:
            raise ValueError("Best checkpoint retention path escaped output/checkpoints")
        old_metadata = old / "runtime.json"
        if not old_metadata.is_file() or (old / "training_state.pt").exists():
            continue
        info = read_json(old_metadata)
        if (info.get("kind") == "validation_best_bridge" and info.get("resumable") is False
                and info.get("fingerprint_sha256") == canonical_hash(fingerprint)):
            shutil.rmtree(old)
    LOGGER.info("Validation best | step %d | loss=%.4f | bridge: %s", step, loss, destination)
    return selected


def resolve_resume(output: Path, requested: str) -> Path | None:
    if requested == "none":
        if (output / "latest.json").exists() or (output / "best.json").exists():
            raise ValueError("Output already contains a run; use --resume auto or choose another output_dir")
        return None
    if requested != "auto":
        path = Path(requested)
        if (path / "latest.json").exists():
            return resolve_resume(path, "auto")
        if not (path / "training_state.pt").is_file():
            raise ValueError("Resume requires an exact training checkpoint with training_state.pt; "
                             "a validation-best bridge is for evaluation only")
        return path
    latest = output / "latest.json"
    if not latest.exists():
        return None
    path = (output / read_json(latest)["path"]).resolve()
    if output.resolve() not in path.parents:
        raise ValueError("latest.json checkpoint escaped output directory")
    return path


def forward_backward(model, batch, scaler) -> tuple[Any, torch.Tensor]:
    reporter = getattr(model, "_microbatch_progress", None)
    if reporter:
        reporter.phase("packet_read" if "packet" in batch else "encoder_and_compressors")
    packet = batch["packet"] if "packet" in batch else model.encode_context(batch["context"])
    if not bool(torch.isfinite(packet).all()):
        raise FloatingPointError("Non-finite context packet; selected backbone precision overflowed")
    with model.reading(packet, batch["prompt"]) as reader:
        if reporter:
            reporter.phase("receiver_forward")
        score = score_responses(response_logits(model.receiver, batch["prompt"]), batch,
                                checkpoint_loss=True)
        loss = score.nll.sum() / LOSS_NORMALIZER
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss; selected backbone precision overflowed")
        # Keep hooks active through checkpoint recomputation. FP32 bridge arithmetic
        # is deliberate: no autocast surrounds the bridge, pooling, or gate softmax.
        if reporter:
            reporter.phase("backward")
        scaler.scale(loss).backward()
    gate = torch.stack(list(reader.gates.values())).mean()
    return score, gate


def benchmark_microbatches(model, loader, optimizer, scaler, args, distributed, output, accumulation):
    """Measure real forward/backward calls without making an optimizer update."""
    count = min(args.benchmark_microbatches, len(loader))
    reporter = MicrobatchProgress(distributed.rank, interval=0.0)
    model._microbatch_progress = reporter
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reporter.start(1, count, phase="benchmark")
    started = time.monotonic()
    durations = []
    examples = 0
    # Length sorting puts the shortest examples first. Sample across the first
    # update instead of benchmarking only its shortest few microbatches.
    width = min(len(loader), max(accumulation, count))
    batches = list(islice(iter(loader.batch_sampler), width))
    positions = [i * (width - 1) // max(1, count - 1) for i in range(count)]
    for position in positions:
        batch = loader.collate_fn([loader.dataset[index] for index in batches[position]])
        reporter.begin_microbatch()
        before = time.monotonic()
        forward_backward(model, batch, scaler)
        # Benchmark timings include completed GPU work, rather than only the
        # asynchronous launches. Normal training synchronizes scalar statistics.
        examples += len(batch["ids"])
        if distributed.device.type == "cuda":
            torch.cuda.synchronize(distributed.device)
        durations.append(time.monotonic() - before)
        reporter.finish_microbatch()
    scaler.unscale_(optimizer)
    finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                 for group in optimizer.param_groups for parameter in group["params"])
    own = {"rank": distributed.rank, "microbatches": count, "examples": examples,
           "sampled_microbatch_positions": positions,
           "elapsed_seconds": time.monotonic() - started,
           "seconds_per_microbatch": sum(durations) / len(durations),
           "microbatch_seconds": durations, "finite_gradients": finite}
    ranks = distributed.all_gather(own)
    result = {"type": "benchmark", "optimizer_updates": 0, "ranks": ranks,
              "gradient_accumulation_steps": accumulation,
              "rough_update_seconds": max(rank["seconds_per_microbatch"] for rank in ranks) * accumulation,
              "note": "Small cold-start sample; excludes optimizer/communication time and full-update overflow retries."}
    optimizer.zero_grad(set_to_none=True)
    if distributed.is_main:
        atomic_json(output / "benchmark.json", result)
        LOGGER.info("%s", json.dumps(result))
    return 0


def train(args) -> int:
    config = load_config(args.config)
    distributed = initialize_distributed()
    try:
        return _train(config, args, distributed)
    finally:
        distributed.close()


def _train(config, args, distributed) -> int:
    output = Path(config.training.output_dir)
    stage = "Stage 1" if config.training.trainable == "all" else "Stage 2"
    if distributed.is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "session-paused.json").unlink(missing_ok=True)
    distributed.barrier()
    resume = resolve_resume(output, args.resume)
    if getattr(args, "benchmark_microbatches", 0) and resume is not None:
        raise ValueError("Benchmark requires a fresh workdir; existing training checkpoints are preserved")
    seed_everything(config.training.seed)
    np.random.seed(config.training.seed)
    cache_identity = packet_fingerprint(config.data.packet_cache) if config.data.packet_cache else None
    if config.training.trainable == "readers_and_gates" and config.data.packet_cache is None:
        raise ValueError("Stage 2 requires a stored packet cache")
    if config.training.trainable == "readers_and_gates" and not config.training.init_checkpoint:
        raise ValueError("Stage 2 requires the complete final Stage 1 checkpoint")
    if config.training.trainable == "readers_and_gates":
        validate_stage1_checkpoint(config.training.init_checkpoint, config)
    if config.training.trainable == "all" and config.data.packet_cache:
        raise ValueError("Stage 1 must train compressors from encoder states, not cached packets")

    train_path = Path(config.data.dataset_root) / f"{config.data.train_split}.jsonl"
    eval_path = Path(config.data.dataset_root) / f"{config.data.eval_split}.jsonl"
    fingerprint = settings_fingerprint(config, args.precision, distributed, args.source_commit,
                                       train_path, eval_path, cache_identity)
    read_best_checkpoint(output, fingerprint)
    state = None
    if resume is not None:
        if not (resume / "training_state.pt").is_file():
            raise ValueError("Resume requires training_state.pt; a validation-best bridge is for evaluation only")
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["format"] != FORMAT_VERSION or state["fingerprint"] != fingerprint:
            raise ValueError("Resume rejected: data, model, source, precision, world size, or training settings changed")
        if state.get("complete"):
            if distributed.is_main:
                LOGGER.info("%s already completed: %s", stage, resume)
            return 0

    if distributed.is_main:
        LOGGER.info("%s | loading frozen models...", stage)
    model = build_runtime_model(config, args.precision, distributed.device,
                                load_encoder=config.data.packet_cache is None)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("rank %d: models loaded; memory=%s", distributed.rank, json.dumps(memory_snapshot()))
    if config.training.init_checkpoint:
        load_bridge(config.training.init_checkpoint, model.bridge)
    if cache_identity is not None:
        if cache_identity["meta"]["provenance"]["compressor"] != compressor_hash(model.bridge):
            raise ValueError("Cached packets were made by a different Stage 1 compressor")
        if cache_identity["meta"]["packet_shape"] != list(model.bridge.packet_shape):
            raise ValueError("Cached packet shape differs from the bridge")
        if cache_identity["meta"]["provenance"]["encoder"] != config.encoder.model_dump(mode="json"):
            raise ValueError("Cached packets were made by a different encoder configuration")
        if cache_identity["meta"]["provenance"]["precision"] != args.precision:
            raise ValueError("Cached packets were encoded with a different backbone precision")
        if cache_identity["meta"]["provenance"]["code"] != code_identity(args.source_commit):
            raise ValueError("Cached packets were encoded by different source code")
    if config.training.trainable == "readers_and_gates":
        model.bridge.compressors.requires_grad_(False)
    if config.training.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    parameters = [parameter for parameter in model.bridge.parameters() if parameter.requires_grad]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Runtime requires FP32 bridge master parameters")
    packets = PacketCache(config.data.packet_cache) if config.data.packet_cache else None
    collator = Collator(load_tokenizer(config.encoder), load_tokenizer(config.receiver), config.data,
                        with_context=packets is None, packets=packets)
    if distributed.is_main:
        LOGGER.info("%s | preparing data indices...", stage)
    train_examples = IndexedExamples(train_path, config.data)
    eval_examples = IndexedExamples(eval_path, config.data)
    if not len(train_examples) or len(eval_examples) < distributed.world_size:
        raise ValueError("Training must be nonempty and eval needs at least one example per rank")
    loader, sampler = make_loader(train_examples, collator, config, distributed, training=True)
    eval_loader, _ = make_loader(eval_examples, collator, config, distributed, training=False)
    accumulation = config.training.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = config.training.max_steps or config.training.num_epochs * steps_per_epoch
    if total_steps <= 0:
        raise ValueError("Training schedule is empty")
    if distributed.is_main:
        LOGGER.info("%s | %s examples | %d epochs | %d updates | global batch %d",
                    stage, f"{len(train_examples):,}", config.training.num_epochs, total_steps,
                    config.training.per_device_batch_size * accumulation * distributed.world_size)
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    scheduler = build_scheduler(optimizer, config, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and distributed.device.type == "cuda",
                                  init_scale=1024.0)
    if getattr(args, "benchmark_microbatches", 0):
        return benchmark_microbatches(model, loader, optimizer, scaler, args, distributed, output, accumulation)
    progress = {"step": 0, "epoch": 0, "next_batch": 0, "overflow_retries": 0}
    if state is not None:
        load_bridge(resume, model.bridge)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        progress = state["progress"]
        if len(state["rng"]) != distributed.world_size:
            raise ValueError("Resume checkpoint is missing a rank's RNG state")
        restore_rng(state["rng"][distributed.rank])
        if distributed.is_main:
            LOGGER.info("%s | resumed after step %d/%d", stage, progress["step"], total_steps)
    elif distributed.is_main:
        atomic_json(output / "config.json", config.model_dump(mode="json"))
        atomic_json(output / "environment.json", {
            "torch": torch.__version__, "precision": args.precision,
            "bridge_dtype": "float32", "packet_storage": "bfloat16",
            "source": fingerprint["code"], "world_size": distributed.world_size,
            "cuda": torch.version.cuda,
        })
    stats = dict.fromkeys(("nll", "tokens", "correct_tokens", "examples", "correct", "gate"), 0.0)
    if state is not None:
        stats = state["statistics"][distributed.rank]
    summary = TrainingProgress(stage, total_steps, config.training.num_epochs, steps_per_epoch,
                               enabled=distributed.is_main, memory=memory_snapshot)

    def log(kind, metrics):
        if distributed.is_main:
            record = {"type": kind, "step": progress["step"], **metrics}
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            LOGGER.debug("%s", json.dumps(record, sort_keys=True))
            if kind == "eval":
                LOGGER.info("%s | validation at step %d/%d | %s", stage, progress["step"],
                            total_steps, format_metrics(metrics))

    def evaluate_now():
        # Evaluation is deterministic and must not perturb continuation RNG.
        previous_rng = rng_state()
        if distributed.is_main:
            LOGGER.info("%s | validating at step %d/%d (%d examples)...", stage, progress["step"],
                        total_steps, len(eval_examples))
        try:
            metrics, _ = upstream_evaluate(model, ProgressLoader(
                eval_loader, f"{stage} | validation batches on worker 0", enabled=distributed.is_main), distributed)
            log("eval", metrics)
            if distributed.is_main:
                save_best_checkpoint(output, model, config, fingerprint, dict(progress), metrics)
        finally:
            restore_rng(previous_rng)

    def save(complete=False):
        name = "final" if complete else f"step-{progress['step']:08d}"
        LOGGER.debug("rank %d: saving checkpoint %s", distributed.rank, name)
        if complete and distributed.is_main:
            LOGGER.info("%s | saving final checkpoint...", stage)
        result = save_training_checkpoint(output, name, model, config, optimizer, scheduler, scaler,
                                         distributed, fingerprint, dict(progress), complete=complete, statistics=dict(stats))
        LOGGER.debug("rank %d: checkpoint %s saved", distributed.rank, name)
        if complete and distributed.is_main:
            LOGGER.info("%s complete | final checkpoint: %s", stage, result)
        return result

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if state is None:
        evaluate_now()
        save()  # A valid resume point exists even before a very long first update.
    summary.last_report = summary.last_update_report = time.monotonic()
    started = time.monotonic()
    deadline = float(os.environ.get("LATCOMM_DEADLINE_MONOTONIC", "inf"))
    session_steps = 0
    consecutive_overflows = 0
    update_tokens = 0
    micro_steps = 0
    buffered_batches = []
    update_rng = update_stats = None
    reporter = MicrobatchProgress(distributed.rank, summary=summary)
    model._microbatch_progress = reporter

    def accumulate(batch):
        nonlocal update_tokens
        reporter.begin_microbatch()
        score, gate = forward_backward(model, batch, scaler)
        tokens = int(score.token_counts.sum())
        examples = len(batch["ids"])
        update_tokens += tokens
        stats["nll"] += float(score.nll.detach().sum())
        stats["tokens"] += tokens
        stats["correct_tokens"] += int(score.correct_tokens.sum())
        stats["examples"] += examples
        stats["gate"] += float(gate) * examples
        if score.predictions is not None:
            stats["correct"] += int((score.predictions.cpu() == batch["gold_options"]).sum())
        reporter.finish_microbatch()

    while progress["step"] < total_steps:
        epoch = progress["epoch"]
        start_batch = progress["next_batch"]
        if distributed.is_main:
            LOGGER.info("%s | epoch %d/%d starting | completed %d/%d updates",
                        stage, epoch + 1, config.training.num_epochs, progress["step"], total_steps)
        sampler.set_epoch(epoch)
        loader.batch_sampler.start = start_batch
        loader.generator.manual_seed(config.training.seed + epoch * distributed.world_size + distributed.rank)
        for batch_index, batch in enumerate(loader, start=start_batch):
            if micro_steps == 0:
                update_rng, update_stats = rng_state(), dict(stats)
                buffered_batches = []
                reporter.start(progress["step"] + 1,
                               min(accumulation, loader.batch_sampler.full_length - batch_index))
            buffered_batches.append(batch)
            accumulate(batch)
            micro_steps += 1
            end_epoch = batch_index + 1 == loader.batch_sampler.full_length
            if micro_steps < accumulation and not end_epoch:
                continue
            next_progress = {**progress, "epoch": epoch + int(end_epoch),
                             "next_batch": 0 if end_epoch else batch_index + 1}
            while True:
                scaler.unscale_(optimizer)
                # Reduce the same FP32 gradient tensors on every rank.
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
                if distributed.sum(int(not finite)) == 0:
                    break
                if not scaler.is_enabled():
                    raise FloatingPointError("Non-finite FP32 bridge gradients")
                old_scale = scaler.get_scale()
                scaler.update(new_scale=old_scale / 2)
                optimizer.zero_grad(set_to_none=True)
                consecutive_overflows += 1
                next_progress["overflow_retries"] += 1
                reporter.repeat(consecutive_overflows, old_scale, scaler.get_scale())
                if consecutive_overflows >= 16:
                    raise FloatingPointError("FP16 gradients overflowed in 16 consecutive updates")
                # Replay exactly the same update and stochastic operations. No
                # data is skipped and the paper's epoch count remains unchanged.
                restore_rng(update_rng)
                stats = dict(update_stats)
                update_tokens = 0
                for retry_batch in buffered_batches:
                    accumulate(retry_batch)
            consecutive_overflows = 0
            reporter.emit("optimizer_update", phase="gradient_reduction_and_step")
            total_tokens = distributed.sum(update_tokens)
            distributed.sum_gradients(parameters)  # Float32 sum, not a world-size mean.
            for parameter in parameters:
                parameter.grad.mul_(LOSS_NORMALIZER / total_tokens)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.max_grad_norm,
                                                  error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress = {**next_progress, "step": progress["step"] + 1}
            session_steps += 1
            micro_steps = update_tokens = 0
            buffered_batches = []
            latest_metrics = None
            if progress["step"] % config.training.log_every == 0:
                shards = distributed.all_gather(stats)
                merged = {key: sum(shard[key] for shard in shards) for key in stats}
                metrics = {"loss": merged["nll"] / merged["tokens"],
                           "token_accuracy": merged["correct_tokens"] / merged["tokens"],
                           "gate_mean": merged["gate"] / merged["examples"],
                           "grad_norm": float(norm), "lr": scheduler.get_last_lr()[0],
                           "loss_scale": scaler.get_scale(),
                           "step_seconds": round(time.monotonic() - reporter.started, 3)}
                if config.data.response_format == "mcq":
                    metrics["accuracy"] = merged["correct"] / merged["examples"]
                log("train", metrics)
                latest_metrics = metrics
                stats = dict.fromkeys(stats, 0.0)
            complete = progress["step"] == total_steps
            summary.update(progress["step"], time.monotonic() - reporter.started, latest_metrics,
                           force=end_epoch or complete)
            if end_epoch and distributed.is_main:
                LOGGER.info("%s | epoch %d/%d complete", stage, epoch + 1, config.training.num_epochs)
            if progress["step"] % config.training.eval_every == 0 or end_epoch or complete:
                evaluate_now()
            if complete:
                save(complete=True)
                return 0
            capped = (args.session_steps > 0 and session_steps >= args.session_steps) or (
                args.session_hours > 0 and time.monotonic() - started >= args.session_hours * 3600) or (
                time.monotonic() >= deadline)
            capped = distributed.sum(int(capped)) > 0
            if capped or progress["step"] % args.checkpoint_every == 0:
                save()
            if capped:
                if distributed.is_main:
                    atomic_json(output / "session-paused.json", {
                        "step": progress["step"], "epoch": progress["epoch"],
                        "next_batch": progress["next_batch"], "exit_code": SESSION_LIMIT_EXIT,
                        "checkpoint": read_json(output / "latest.json")["path"],
                        "reason": "session_limit",
                    })
                distributed.barrier()
                if distributed.is_main:
                    LOGGER.info("%s paused at step %d/%d; checkpoint saved, resume with --resume auto (exit 75)",
                                stage, progress["step"], total_steps)
                return SESSION_LIMIT_EXIT
        # A finished data pass must advance the epoch cursor.
        if progress["epoch"] == epoch:
            raise RuntimeError("Training made no progress through its data epoch")
    raise RuntimeError("Unexpected end of training")


def _encoding_sources(config, splits=None) -> list[Path]:
    root = Path(config.data.dataset_root)
    if splits:
        return [root / f"{name}.jsonl" for name in splits]
    paths = [root / f"{name}.jsonl" for name in ("train", "fast_dev", "test")]
    return [path for path in paths if path.exists()]


def encode(args) -> int:
    config = load_config(args.config)
    distributed = initialize_distributed()
    try:
        return _encode(config, args, distributed)
    finally:
        distributed.close()


def _encode(config, args, distributed) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = _encoding_sources(config, getattr(args, "splits", None))
    if not sources:
        raise ValueError("No JSONL splits found to encode")
    checkpoint_info = validate_stage1_checkpoint(args.checkpoint, config)
    packet_shape = [len(checkpoint_info["layer_pairs"]), config.bridge.num_slots, config.bridge.context_dim]
    provenance = {
        "format": FORMAT_VERSION, "compressor": checkpoint_compressor_hash(args.checkpoint),
        "encoder": config.encoder.model_dump(mode="json"),
        "layer_pairs": checkpoint_info["layer_pairs"],
        "packet_shape": packet_shape, "precision": args.precision,
        "storage": "bfloat16", "code": code_identity(getattr(args, "source_commit", "")),
        "sources": {path.name: file_hash(path) for path in sources},
        "max_context_length": config.data.max_context_length,
        "world_size": distributed.world_size,
    }
    meta_path = output / "meta.json"
    if meta_path.exists() and read_json(meta_path).get("provenance") != provenance:
        raise ValueError("Encoding resume rejected: encoder, compressor, data, source, precision or world size changed")
    if meta_path.exists() and read_json(meta_path).get("complete", False):
        # Attached Kaggle datasets are read-only. A completed compatible cache
        # needs checksum verification only, with no model load or file writes.
        packet_fingerprint(output)
        if distributed.is_main:
            LOGGER.info("Packets | reusing complete, verified cache: %s", output)
        return 0
    manifest_path = output / f"encoding-rank{distributed.rank}.json"
    index_path = output / f"index-rank{distributed.rank}.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"provenance": provenance, "shards": {}}
    if manifest["provenance"] != provenance:
        raise ValueError("Per-rank encoding provenance changed")
    index = read_json(index_path) if index_path.exists() else {}
    for name in {item[0] for item in index.values()}:
        if file_hash(output / name) != manifest["shards"].get(name):
            raise ValueError(f"Existing encoded shard failed checksum: {name}")

    # Stream records before loading a GPU model. Keep only context hashes and
    # source row coordinates; identical contexts are encoded exactly once.
    datasets = [IndexedExamples(path, config.data) for path in sources]
    unique: dict[str, tuple[int, int]] = {}
    for dataset_index, dataset in enumerate(datasets):
        for row, example in enumerate(dataset):
            unique.setdefault(context_key(example.context), (dataset_index, row))
    if not set(index).issubset(unique):
        raise ValueError("Existing packet index contains contexts absent from the current data")
    existing = sum(distributed.all_gather(len(index)))
    remaining = len(unique) - existing
    if distributed.is_main:
        LOGGER.info("Packets | %s unique contexts, %s already saved | loading encoder...",
                    f"{len(unique):,}", f"{existing:,}")
    payload_bytes = remaining * math.prod(packet_shape) * 2
    headroom = max(256 * 1024**2, math.ceil(payload_bytes * 0.05))
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < payload_bytes + headroom:
        raise OSError(
            f"Insufficient disk for packet cache: {remaining:,} contexts need approximately "
            f"{payload_bytes / 1024**3:.2f} GiB plus {headroom / 1024**3:.2f} GiB headroom; "
            f"only {free_bytes / 1024**3:.2f} GiB is free. Attach a compatible completed cache "
            "or choose a larger writable packet-cache directory."
        )
    seed_everything(config.training.seed)
    model = build_runtime_model(config, args.precision, distributed.device, load_receiver=False)
    load_bridge(args.checkpoint, model.bridge)
    if list(model.bridge.packet_shape) != packet_shape or compressor_hash(model.bridge) != provenance["compressor"]:
        raise ValueError("Checkpoint metadata differs from its actual compressor tensors")
    model.eval()
    if distributed.is_main:
        atomic_json(meta_path, {"dtype": "bfloat16", "packet_shape": list(model.bridge.packet_shape),
                                "provenance": provenance, "contexts": 0, "complete": False})
    distributed.barrier()
    tokenizer = load_tokenizer(config.encoder)
    encoding_progress = CounterProgress("Packets saved", len(unique), initial=existing, enabled=distributed.is_main)
    pending_keys: list[str] = []
    pending_packets: list[np.ndarray] = []
    next_shard = 0
    pattern = f"packets-rank{distributed.rank}-*.npy"
    for path in output.glob(pattern):
        next_shard = max(next_shard, int(path.stem.rsplit("-", 1)[1]) + 1)

    def flush():
        nonlocal next_shard, pending_keys, pending_packets
        if not pending_keys:
            return
        name = f"packets-rank{distributed.rank}-{next_shard:05d}.npy"
        path = output / name
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            np.save(handle, np.stack(pending_packets), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        manifest["shards"][name] = file_hash(path)
        for row, key in enumerate(pending_keys):
            index[key] = [name, row]
        # Publish checksum before publishing any index that references the shard.
        atomic_json(manifest_path, manifest)
        atomic_json(index_path, index)
        next_shard += 1
        pending_keys, pending_packets = [], []

    keys: list[str] = []
    contexts: list[str] = []

    def run_batch():
        nonlocal keys, contexts
        token_rows = [tokenize(tokenizer, context, config.data.max_context_length, "context", key)
                      for key, context in zip(keys, contexts, strict=True)]
        ids, mask = pad(token_rows, tokenizer.pad_token_id)
        with torch.no_grad():
            packet = model.encode_context({"input_ids": ids, "attention_mask": mask})
        if not bool(torch.isfinite(packet).all()):
            raise FloatingPointError("Non-finite encoded packet; backbone precision overflowed")
        pending_keys.extend(keys)
        pending_packets.extend(to_stored(packet, "bfloat16"))
        keys, contexts = [], []
        if len(pending_keys) >= SHARD_ROWS:
            flush()
        if encoding_progress.due():
            # Read only published indices. No extra distributed collective is
            # needed, and another worker may have unfinished packets in memory.
            saved = len(index)
            for rank in range(distributed.world_size):
                other = output / f"index-rank{rank}.json"
                if rank != distributed.rank and other.exists():
                    saved += len(read_json(other))
            encoding_progress.update(saved)

    try:
        for key, (dataset_index, row) in unique.items():
            if int(key, 16) % distributed.world_size != distributed.rank or key in index:
                continue
            example = datasets[dataset_index][row]
            keys.append(key)
            contexts.append(example.context)
            if len(keys) == args.batch_size:
                run_batch()
        if keys:
            run_batch()
    finally:
        flush()
    if not index_path.exists():
        atomic_json(index_path, index)
        atomic_json(manifest_path, manifest)
    if distributed.is_main and distributed.world_size > 1:
        LOGGER.info("Packets | local encoding finished; synchronizing workers...")
    counts = distributed.all_gather({"count": len(index), "shards": manifest["shards"]})
    if distributed.is_main:
        total = sum(item["count"] for item in counts)
        if total != len(unique):
            raise ValueError("Encoded index does not cover exactly all unique contexts")
        shards = {name: digest for item in counts for name, digest in item["shards"].items()}
        atomic_json(meta_path, {"dtype": "bfloat16", "packet_shape": list(model.bridge.packet_shape),
                                "provenance": provenance, "contexts": total, "shards": shards, "complete": True})
        encoding_progress.update(total, force=True)
        LOGGER.info("Packets complete | %s unique contexts | cache: %s", f"{total:,}", output)
    distributed.barrier()
    return 0


def evaluate_command(args) -> int:
    config = load_config(args.config)
    distributed = initialize_distributed()
    try:
        if distributed.is_main:
            LOGGER.info("Test | loading final model and packet cache...")
        packets = None
        identity = None
        if config.data.packet_cache:
            identity = packet_fingerprint(config.data.packet_cache)
            packets = PacketCache(config.data.packet_cache)
        model = build_runtime_model(config, args.precision, distributed.device,
                                    load_encoder=packets is None)
        load_bridge(args.checkpoint, model.bridge)
        if identity and identity["meta"]["provenance"]["compressor"] != compressor_hash(model.bridge):
            raise ValueError("Evaluation cache comes from different compressors")
        if identity:
            provenance = identity["meta"]["provenance"]
            if (provenance["encoder"] != config.encoder.model_dump(mode="json")
                    or provenance["precision"] != args.precision
                    or provenance["code"] != code_identity(getattr(args, "source_commit", ""))
                    or identity["meta"]["packet_shape"] != list(model.bridge.packet_shape)):
                raise ValueError("Evaluation cache encoder, precision, source or shape differs")
        model.eval()
        examples = IndexedExamples(args.data, config.data)
        if len(examples) < distributed.world_size:
            raise ValueError("Evaluation needs at least one example per rank")
        collator = Collator(load_tokenizer(config.encoder), load_tokenizer(config.receiver), config.data,
                            with_context=packets is None, packets=packets)
        eval_config = config.model_copy(update={"training": config.training.model_copy(update={
            "per_device_eval_batch_size": 1,
        })})
        loader, _ = make_loader(examples, collator, eval_config, distributed, training=False)
        if distributed.is_main:
            LOGGER.info("Test | evaluating %s examples...", f"{len(examples):,}")
        metrics, records = upstream_evaluate(model, ProgressLoader(
            loader, "Test batches on worker 0", enabled=distributed.is_main), distributed, packet_dtype="bfloat16")
        if distributed.is_main:
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "metrics.json", metrics)
            atomic_json(output / "provenance.json", {
                "data_sha256": file_hash(args.data), "bridge_sha256": file_hash(Path(args.checkpoint) / "bridge.safetensors"),
                "precision": args.precision, "packet_storage": "bfloat16",
                "source": code_identity(getattr(args, "source_commit", "")),
            })
            temporary = output / "predictions.jsonl.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            os.replace(temporary, output / "predictions.jsonl")
            LOGGER.info("Test complete | %s | results: %s", format_metrics(metrics), output)
        distributed.barrier()
        return 0
    finally:
        distributed.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("train", "encode", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--precision", required=True, choices=("fp16", "bf16"))
        command.add_argument("--source-commit", default=os.environ.get("LATCOMM_SOURCE_COMMIT", ""))
        command.add_argument("--verbose", action="store_true", help="Show per-worker debug events and detailed metrics")
        if name != "train":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--resume", default="auto")
            command.add_argument("--session-steps", type=int, default=0)
            command.add_argument("--session-hours", type=float, default=0)
            command.add_argument("--checkpoint-every", type=int, default=25)
            command.add_argument("--benchmark-microbatches", type=int, default=0,
                                 help="Time N microbatches without optimizer updates; requires a fresh workdir")
        if name == "encode":
            command.add_argument("--batch-size", type=int, default=1)
            command.add_argument("--splits", nargs="+")
        if name == "evaluate":
            command.add_argument("--data", required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "train":
        if args.session_steps < 0 or args.session_hours < 0 or args.checkpoint_every < 1 or args.benchmark_microbatches < 0:
            raise ValueError("Session limits must be nonnegative and checkpoint interval positive")
        return train(args)
    if args.command == "encode":
        if args.batch_size < 1:
            raise ValueError("Encoding batch size must be positive")
        return encode(args)
    return evaluate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
