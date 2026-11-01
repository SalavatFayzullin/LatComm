"""Data correctness checks; synthetic inputs, no Hub downloads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as data
from latcomm.config import DataConfig


def record(index, mcq=False):
    text = f"Контекст {index}: text with a newline\nand a second line."
    value = {"sample_id": str(index), "question": "Which?" if mcq else "Reproduce the passage.",
             "evidence": [{"source_record_id": str(index), "text": text}],
             "response": "B" if mcq else text, "provenance": {"dataset": "synthetic"}}
    if mcq:
        value["choices"] = ["first", "second", "third", "fourth"]
    return value


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def test_index_matches_upstream_and_handles_unicode(tmp_path):
    from latcomm.data import load_examples
    path = tmp_path / "data/train.jsonl"
    write(path, [record(i, mcq=True) for i in range(4)])
    config = DataConfig(dataset_root=str(path.parent), response_format="mcq")
    indexed = data.IndexedExamples(path, config, index_dir=tmp_path / "indexes")
    assert list(indexed) == load_examples(path, config)
    assert indexed[-1].sample_id == "3"
    assert indexed[1:3] == load_examples(path, config)[1:3]
    assert indexed.context_lengths.tolist() == [len(indexed[i].context) for i in range(4)]
    with pytest.raises(IndexError):
        indexed[4]
    indexed.close()
    again = data.IndexedExamples(path, config, index_dir=tmp_path / "indexes")
    assert again.sha256 == indexed.sha256
    assert again[0] == indexed[0]


def test_stage1_and_gold_validation(tmp_path):
    text = DataConfig(dataset_root=str(tmp_path), response_format="text")
    mcq = text.model_copy(update={"response_format": "mcq"})
    assert data.parse_example(record(0), text).response == record(0)["response"]
    bad = record(0)
    bad["response"] = "a summary instead of the original context"
    with pytest.raises(ValueError, match="must equal context"):
        data.parse_example(bad, text)
    bad = record(0, mcq=True)
    bad["response"] = "Z"
    with pytest.raises(ValueError, match="gold option"):
        data.parse_example(bad, mcq)


def test_full_preserves_bytes_and_receipt_rejects_tampering(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "prepared"
    monkeypatch.setattr(data, "EXPECTED", {stage: {split: 3 for split in data.SPLITS} for stage in data.FOLDERS})
    for stage in data.FOLDERS:
        for split in data.SPLITS:
            write(source / stage / f"{split}.jsonl", [record(i, stage == "stage2") for i in range(3)])
    receipt = data.prepare_data(target, revision="fixed-commit", local_source=source)
    assert receipt["profile"] == "full"
    for relative in receipt["files"]:
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
    assert data.prepare_data(target, revision="fixed-commit", local_source=source) == receipt
    with pytest.raises(ValueError, match="revision/profile"):
        data.prepare_data(target, revision="other-commit", local_source=source)
    with (target / "stage1/train.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        data.prepare_data(target, revision="fixed-commit", local_source=source)


def test_smoke_is_explicit_and_keeps_original_records(tmp_path):
    source, target = tmp_path / "source", tmp_path / "prepared"
    for stage in data.FOLDERS:
        for split in data.SPLITS:
            write(source / stage / f"{split}.jsonl",
                  [record(i, stage == "stage2") for i in range(400 if split == "train" else 24)])
    receipt = data.prepare_data(target, revision="fixed-commit", profile="smoke", local_source=source)
    assert receipt["profile"] == "smoke"
    assert receipt["files"]["stage1/train.jsonl"]["records"] == 384
    assert receipt["files"]["stage2/test.jsonl"]["records"] == 16
    actual = (target / "stage2/train.jsonl").read_bytes().splitlines()
    expected = (source / "stage2/train.jsonl").read_bytes().splitlines()[:384]
    assert actual == expected


def test_inspect_unterminated_last_record(tmp_path):
    path = tmp_path / "two.jsonl"
    path.write_bytes(b"{}\n{}")
    assert data.inspect_jsonl(path)["records"] == 2
