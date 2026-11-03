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


def test_kaggle_sampling_is_balanced_reproducible_and_preserves_records(tmp_path, monkeypatch):
    policy = {"seed": 42,
              "stage1": {"train": 20, "dev": 4, "fast_dev": 4, "test": 4},
              "stage2_per_dataset": {"train": 3, "dev": 2, "fast_dev": 2, "test": 2},
              "stage2_datasets": 3}
    monkeypatch.setattr(data, "KAGGLE_POLICY", policy)
    source = tmp_path / "source"
    for split in data.SPLITS:
        write(source / "stage1" / f"{split}.jsonl", [record(i) for i in range(80)])
        rows = []
        for group in ("first", "second", "third"):
            for index in range(12):
                row = record(index, mcq=True)
                row["sample_id"] = f"{group}-{index}"
                row["provenance"]["dataset"] = group
                rows.append(row)
        write(source / "stage2" / f"{split}.jsonl", rows)
    first, second = tmp_path / "first", tmp_path / "second"
    receipt = data.prepare_data(first, revision="fixed", profile="kaggle", local_source=source)
    again = data.prepare_data(second, revision="fixed", profile="kaggle", local_source=source)
    assert receipt["sampling"] == policy
    assert receipt["files"] == again["files"]
    assert receipt["files"]["stage2/train.jsonl"]["datasets"] == {"first": 3, "second": 3, "third": 3}
    for relative, info in receipt["files"].items():
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        source_name = relative.replace("fast_dev", "dev") if relative.startswith("stage2") else relative
        originals = set((source / source_name).read_bytes().splitlines())
        assert all(line in originals for line in (first / relative).read_bytes().splitlines())
    assert (first / "stage2/dev.jsonl").read_bytes() == (first / "stage2/fast_dev.jsonl").read_bytes()
    # Reservoir sampling is not the first N lines of a source grouped by task.
    assert (first / "stage1/train.jsonl").read_bytes().splitlines() != (source / "stage1/train.jsonl").read_bytes().splitlines()[:20]


def test_kaggle_rejects_a_source_missing_tasks(tmp_path, monkeypatch):
    import io
    monkeypatch.setitem(data.KAGGLE_POLICY, "stage2_per_dataset", {"train": 1})
    with pytest.raises(ValueError, match="Cannot sample"):
        data.sample_kaggle(io.BytesIO(json.dumps(record(0, mcq=True)).encode() + b"\n"), "stage2", "train")


@pytest.mark.parametrize("profile", ["overnight", "nightly"])
def test_extended_profile_increases_only_reconstruction_training_sample(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(data, "KAGGLE_POLICY", {
        "seed": 42,
        "stage1": {"train": 2, "dev": 2, "fast_dev": 2, "test": 2},
        "stage2_per_dataset": {"train": 3, "dev": 2, "fast_dev": 2, "test": 2},
        "stage2_datasets": 1,
    })
    monkeypatch.setitem(data.EXTENDED_STAGE1_TRAIN, profile, 5)
    source = tmp_path / "source"
    for stage in data.FOLDERS:
        for split in data.SPLITS:
            write(source / stage / f"{split}.jsonl",
                  [record(f"{split}-{i}", mcq=stage == "stage2") for i in range(12)])

    pilot, extended = tmp_path / "pilot", tmp_path / profile
    pilot_receipt = data.prepare_data(pilot, revision="fixed", profile="kaggle", local_source=source)
    extended_receipt = data.prepare_data(extended, revision="fixed", profile=profile, local_source=source)
    assert pilot_receipt["files"]["stage1/train.jsonl"]["records"] == 2
    assert extended_receipt["files"]["stage1/train.jsonl"]["records"] == 5
    assert extended_receipt["profile"] == profile
    assert extended_receipt["sampling"]["stage1"]["train"] == 5
    for relative in pilot_receipt["files"]:
        if relative != "stage1/train.jsonl":
            assert (pilot / relative).read_bytes() == (extended / relative).read_bytes()
    assert data.prepare_data(extended, revision="fixed", profile=profile) == extended_receipt
    with pytest.raises(ValueError, match="revision/profile changed"):
        data.prepare_data(extended, revision="fixed", profile="kaggle")
