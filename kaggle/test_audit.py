"""CPU audit controls and provenance checks; no HF downloads or GPU required."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "tests")]
spec = importlib.util.spec_from_file_location("latcomm_audit", HERE / "audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

import runtime
import data_utils
from tiny import CharTokenizer, TinyLM
from latcomm.bridge import LatentBridge, select_layer_pairs
from latcomm.checkpoint import save_bridge
from latcomm.config import LatCommConfig
from latcomm.data import Example
from latcomm.distributed import DistributedContext
from latcomm import evaluate as scoring
from latcomm.model import LatCommModel, wrap_frozen_lm
from latcomm.packets import PacketWriter, context_key


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = LatCommConfig.model_validate({
        "encoder": {"model_name_or_path": "tiny-encoder", "revision": "fixed"},
        "receiver": {"model_name_or_path": "tiny-receiver", "revision": "fixed"},
        "bridge": {"context_dim": 8, "num_slots": 2, "num_attention_heads": 2, "gate_pooling_dim": 8},
        "data": {"dataset_root": "unused", "response_format": "mcq", "num_workers": 0},
        "training": {"output_dir": "unused", "trainable": "readers_and_gates"},
    })
    encoder = TinyLM(hidden_size=24, num_layers=3).requires_grad_(False).eval()
    receiver = TinyLM(hidden_size=16, num_layers=2).requires_grad_(False).eval()
    receiver.config.max_position_embeddings = 20000
    bridge = LatentBridge(select_layer_pairs(3, 2, 1.0), 24, 16, config.bridge)
    for reader in bridge.readers:
        torch.nn.init.normal_(reader.output_projection.weight, std=0.25)
    wrapped = wrap_frozen_lm(encoder, "cpu")
    wrapped.model = None  # Audit must work without encoder weights.
    model = LatCommModel(wrapped, wrap_frozen_lm(receiver, "cpu"), bridge)
    examples = [Example("z", "CLOTH", "context alpha", "Which?\nA. alpha\nB. beta\nAnswer: ", "A", 2),
                Example("a", "CLOTH", "context beta", "Which?\nA. alpha\nB. beta\nAnswer: ", "B", 2),
                Example("c", "CLOTH", "context alpha", "Another?\nA. alpha\nB. beta\nAnswer: ", "A", 2),
                Example("b", "task-two", "context gamma", "Pick? A. gamma B. delta C. none: ", "A", 3),
                Example("d", "task-two", "context delta", "Pick? A. gamma B. delta C. none: ", "B", 3)]
    generator = torch.Generator().manual_seed(123)
    values = {context_key(e.context): torch.randn(model.bridge.packet_shape, generator=generator)
              for e in examples}
    calls = []
    def packets(rows):
        calls.extend((row.sample_id, context_key(row.context)) for row in rows)
        return torch.stack([values[context_key(row.context)] for row in rows])
    return config, model, examples, packets, calls


def test_shuffle_uses_unique_context_derangement_within_task(setup):
    _, _, examples, _, _ = setup
    first = audit.shuffled_contexts(examples, context_key, 42)
    second = audit.shuffled_contexts(list(reversed(examples)), context_key, 42)
    assert first == second
    for example in examples:
        donor = first[(example.dataset, context_key(example.context))]
        assert donor.dataset == example.dataset
        assert donor.context != example.context
    # Repeated contexts share their donor; each unique context appears once in the cycle.
    for task in {e.dataset for e in examples}:
        own = {context_key(e.context) for e in examples if e.dataset == task}
        other = {context_key(e.context) for (group, _), e in first.items() if group == task}
        assert own == other
    with pytest.raises(ValueError, match="two distinct contexts"):
        audit.shuffled_contexts(examples[:1], context_key, 42)


def test_controls_disable_hooks_freeze_weights_keep_rows_and_correct_shift(setup, monkeypatch):
    config, model, examples, packets, calls = setup
    receiver = model.receiver.model
    snapshots = [p.detach().clone() for p in list(model.bridge.parameters()) + list(receiver.parameters())]
    observations = []
    original = receiver.forward
    def observe(*args, **kwargs):
        observations.append((torch.is_grad_enabled(), any(layer._forward_hooks for layer in model.receiver.layers)))
        return original(*args, **kwargs)
    monkeypatch.setattr(receiver, "forward", observe)
    monkeypatch.setattr(model, "encode_context", lambda *_: pytest.fail("Audit encoded a context"))
    results, lengths = audit.evaluate_controls(runtime, model, examples, packets, config,
        DistributedContext(), tokenizer=CharTokenizer(), scoring=scoring)
    assert set(results) == set(audit.MODES)
    assert all([r["sample_id"] for r in rows] == sorted(e.sample_id for e in examples)
               for rows in results.values())
    assert len(calls) == 2 * len(examples)  # Query-only and raw-text never read packets.
    assert observations == [(False, True), (False, True), (False, False), (False, False)] * len(examples)
    assert all(not layer._forward_hooks for layer in model.receiver.layers)
    parameters = list(model.bridge.parameters()) + list(receiver.parameters())
    assert all(not p.requires_grad and p.grad is None for p in parameters)
    assert all(torch.equal(before, after) for before, after in zip(snapshots, parameters))
    for rows in results.values():
        for row in rows:
            descriptor = lengths[row["sample_id"]]
            expected = descriptor["raw_prediction_position" if row["mode"] == "raw_text" else "prediction_position"]
            assert row["prediction_position"] == expected == row["input_tokens"] - 2
            assert row["tokens"] == 1
            assert set(row["option_scores"]) == set("ABC"[:len(row["option_scores"])])
            assert sum(s["conditional_probability"] for s in row["option_scores"].values()) == pytest.approx(1)
            if row["mode"] == "actual_packet":
                assert row["packet_context_key"] == row["own_context_key"]
            elif row["mode"] == "shuffled_packet":
                assert row["packet_context_key"] == row["shuffled_context_key"] != row["own_context_key"]
            else:
                assert row["packet_context_key"] is None
    actual, shuffled, query = (results[k] for k in audit.MODES[:3])
    # Nonzero readers and distinct packets make these real controls, not aliases.
    assert any(a["option_scores"] != b["option_scores"] for a, b in zip(actual, shuffled))
    assert any(a["option_scores"] != b["option_scores"] for a, b in zip(actual, query))


def test_raw_text_expands_saved_limit_without_truncating_and_checks_window(setup):
    config, _, examples, _, _ = setup
    long = replace(examples[0], context="FULL CONTEXT " * 400)
    descriptions, raw, limit = audit.prepare_inputs([long], CharTokenizer(), config.data, 20000, audit.MODES)
    text = raw[long.sample_id].prompt
    assert text == f"Context:\n{long.context}\n\n{long.prompt}"
    assert long.context in text and limit > 4096
    ids = CharTokenizer()(text, add_special_tokens=True)["input_ids"]
    assert descriptions[long.sample_id]["raw_prompt_tokens"] == len(ids) == limit
    with pytest.raises(ValueError, match="no truncation or sample skipping"):
        audit.prepare_inputs([long], CharTokenizer(), config.data, len(ids), audit.MODES)
    # A long raw input does not block a selected query-only audit.
    audit.prepare_inputs([long], CharTokenizer(), config.data, 100, ("query_only",))


def test_multitoken_options_are_rejected_instead_of_misscored(setup):
    config, _, examples, _, _ = setup
    class SplitOption(CharTokenizer):
        def __call__(self, text, add_special_tokens=True):
            if text == "A" and not add_special_tokens:
                return {"input_ids": [5, 6]}
            return super().__call__(text, add_special_tokens=add_special_tokens)
    with pytest.raises(ValueError, match="single token"):
        audit.prepare_inputs(examples, SplitOption(), config.data, 1000, audit.MODES)


def test_results_publish_all_source_rows_cloth_prompts_and_paired_metrics(setup, tmp_path):
    config, model, examples, packets, _ = setup
    results, lengths = audit.evaluate_controls(runtime, model, examples, packets, config,
        DistributedContext(), tokenizer=CharTokenizer(), scoring=scoring)
    sources = {e.sample_id: {"sample_id": e.sample_id, "original": e.context} for e in examples}
    summary = audit.make_summary(results, examples, scoring, {"training_source_commit": "old"})
    assert summary["chance_accuracy"]["overall"] == pytest.approx((3 / 2 + 2 / 3) / 5)
    assert summary["chance_accuracy"]["per_task"] == {"CLOTH": 0.5, "task-two": pytest.approx(1 / 3)}
    assert set(summary["comparisons_to_actual_packet"]) == set(audit.MODES[1:])
    output = tmp_path / "new-output"
    audit.publish_results(output, summary, results, examples, sources, lengths)
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 4 * len(examples)
    saved_sources = [json.loads(line) for line in (output / "test_source_records.jsonl").read_text().splitlines()]
    assert {row["sample_id"] for row in saved_sources} == set(sources)
    cloth = [json.loads(line) for line in (output / "cloth_examples.jsonl").read_text().splitlines()]
    assert len(cloth) == 3 and all(set(row["modes"]) == set(audit.MODES) for row in cloth)
    assert all(row["source_record"] == sources[row["sample_id"]] for row in cloth)
    before = audit.file_hash(output / "predictions.jsonl")
    with pytest.raises(ValueError, match="nonempty"):
        audit.publish_results(output, summary, results, examples, sources, lengths)
    assert audit.file_hash(output / "predictions.jsonl") == before
    with pytest.raises(ValueError, match="separate"):
        audit.check_output(output / "nested", (output,))


def test_paired_rows_cannot_silently_mismatch():
    with pytest.raises(ValueError, match="identical sample IDs"):
        audit.paired_comparison([{"sample_id": "a"}], [{"sample_id": "b"}])


def test_reproduction_check_reports_match_prediction_changes_and_missing_references(tmp_path):
    rows = [{"sample_id": "a", "dataset": "task", "nll": 0.5, "tokens": 1,
             "correct_tokens": 1, "prediction": "A", "gold": "A", "correct": 1},
            {"sample_id": "b", "dataset": "task", "nll": 1.5, "tokens": 1,
             "correct_tokens": 0, "prediction": "A", "gold": "B", "correct": 0}]
    results = {"actual_packet": rows}
    assert audit.reproduction_check(results, tmp_path, scoring)["status"] == "unavailable"
    assert audit.reproduction_check({}, tmp_path, scoring)["status"] == "unavailable"
    write_json(tmp_path / "results/metrics.json", scoring.summarize(rows))
    path = tmp_path / "results/predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    matched = audit.reproduction_check(results, tmp_path, scoring)
    assert matched["status"] == "match" and matched["prediction_difference_count"] == 0
    changed = deepcopy(rows)
    changed[1]["prediction"] = "B"
    path.write_text("".join(json.dumps(row) + "\n" for row in changed))
    report = audit.reproduction_check(results, tmp_path, scoring)
    assert report["status"] == "different"
    assert report["prediction_difference_count"] == 1 and report["prediction_difference_ids"] == ["b"]
    assert report["predictions"]["prediction_mismatch_ids"] == ["b"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(tmp_path / "results/metrics.json", {"accuracy": 0.5, "loss": 2.0})
    report = audit.reproduction_check(results, tmp_path, scoring)
    assert report["status"] == "different" and report["prediction_difference_count"] == 0
    assert report["metrics"]["values"]["loss"]["delta"] == -1.0


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifacts(setup, tmp_path, monkeypatch):
    config, model, examples, _, _ = setup
    run, data = tmp_path / "original-run", tmp_path / "original-data"
    cache = run / "packets"
    checkpoint = run / "outputs/stage2/checkpoints/final"
    save_bridge(checkpoint, model.bridge, config, 144)
    test = data / "stage2/test.jsonl"
    test.parent.mkdir(parents=True)
    test.write_text("".join(json.dumps({"sample_id": e.sample_id, "question": e.prompt,
        "choices": [f"choice {letter}" for letter in "ABC"[:e.num_choices]], "response": e.response,
        "evidence": [{"source_record_id": context_key(e.context), "text": e.context}],
        "provenance": {"dataset": e.dataset}}) + "\n" for e in examples), encoding="utf-8")
    test_hash = audit.file_hash(test)
    code = {"source_commit": "original-source", "files": {"runtime.py": "immutable"}, "libraries": {"torch": "exact"}}
    monkeypatch.setattr(runtime, "code_identity", lambda _: deepcopy(code))
    manifest = {"source_commit": "original-source", "dataset_revision": "original-data", "profile": "kaggle",
        "models": {side: {"name": getattr(config, side).model_name_or_path, "revision": "fixed"}
                   for side in ("encoder", "receiver")}}
    write_json(run / "manifest.json", manifest)
    write_json(data / "preparation.json", {"revision": "original-data", "profile": "kaggle", "files": {
        "stage2/test.jsonl": {"sha256": test_hash, "records": len(examples)}}})
    writer = PacketWriter(cache, 0, "bfloat16")
    unique = sorted({context_key(e.context) for e in examples})
    writer.add(unique, torch.randn(len(unique), *model.bridge.packet_shape))
    writer.close()
    shards = {p.name: audit.file_hash(p) for p in cache.glob("*.npy")}
    provenance = {"format": 1, "code": code, "encoder": config.encoder.model_dump(mode="json"),
        "precision": "fp16", "packet_shape": list(model.bridge.packet_shape),
        "layer_pairs": [list(pair) for pair in model.bridge.layer_pairs], "sources": {"test.jsonl": test_hash},
        "max_context_length": config.data.max_context_length, "compressor": runtime.compressor_hash(model.bridge)}
    write_json(cache / "meta.json", {"dtype": "bfloat16", "packet_shape": list(model.bridge.packet_shape),
        "contexts": len(unique), "complete": True, "provenance": provenance, "shards": shards})
    settings = config.model_dump(mode="json")
    settings["training"].pop("output_dir")
    settings["training"].pop("init_checkpoint")
    settings["data"].pop("dataset_root")
    settings["data"].pop("packet_cache")
    fingerprint = {"code": code, "precision": "fp16", "settings": settings,
                   "packets": runtime.packet_fingerprint(cache)}
    progress = {"step": 144, "epoch": 3, "next_batch": 0, "overflow_retries": 0}
    torch.save({"format": 1, "complete": True, "progress": progress, "fingerprint": fingerprint},
               checkpoint / "training_state.pt")
    write_json(checkpoint / "runtime.json", {"complete": True, "progress": progress,
        "fingerprint_sha256": runtime.canonical_hash(fingerprint), "precision": "fp16", "world_size": 2})
    write_json(run / "outputs/stage2/environment.json", {"precision": "fp16", "source": code,
        "bridge_dtype": "float32", "packet_storage": "bfloat16", "world_size": 2})
    write_json(run / "results/provenance.json", {"data_sha256": test_hash,
        "bridge_sha256": audit.file_hash(checkpoint / "bridge.safetensors"), "precision": "fp16", "source": code})
    args = argparse.Namespace(run_dir=run, data_dir=data, packet_cache=cache)
    return args, manifest, checkpoint


def test_original_run_validation_is_read_only_and_binds_test_and_cache(artifacts):
    args, manifest, checkpoint = artifacts
    files = list(args.run_dir.rglob("*")) + list(args.data_dir.rglob("*"))
    before = {p: audit.file_hash(p) for p in files if p.is_file()}
    config, actual, test, precision, identity, provenance = audit.validate_run(runtime, args, manifest)
    assert actual == checkpoint and precision == "fp16"
    assert provenance["checkpoint_selection"] == "original_final"
    assert provenance["test_sha256"] == audit.file_hash(test)
    assert identity["meta"]["complete"] and config.training.trainable == "readers_and_gates"
    assert before == {p: audit.file_hash(p) for p in before}


@pytest.mark.parametrize("change,match", [
    ("code", "code or library"), ("data", "Prepared test"), ("compressor", "compressors differ"),
    ("incomplete", "not complete"), ("revision", "pinned run model"),
    ("cache_source", "cache provenance"), ("checkpoint_hash", "final-test provenance"),
])
def test_changed_original_inputs_are_rejected(artifacts, monkeypatch, change, match):
    args, manifest, checkpoint = artifacts
    if change == "code":
        monkeypatch.setattr(runtime, "code_identity", lambda _: {"changed": True})
    elif change == "data":
        with (args.data_dir / "stage2/test.jsonl").open("a") as handle:
            handle.write("{}\n")
    elif change == "compressor":
        monkeypatch.setattr(runtime, "checkpoint_compressor_hash", lambda _: "wrong")
    elif change == "incomplete":
        value = audit.read_json(checkpoint / "runtime.json")
        value["complete"] = False
        write_json(checkpoint / "runtime.json", value)
    elif change == "revision":
        manifest["models"]["receiver"]["revision"] = "wrong"
    elif change == "cache_source":
        value = audit.read_json(args.packet_cache / "meta.json")
        value["provenance"]["sources"]["test.jsonl"] = "wrong"
        write_json(args.packet_cache / "meta.json", value)
    elif change == "checkpoint_hash":
        value = audit.read_json(args.run_dir / "results/provenance.json")
        value["bridge_sha256"] = "wrong"
        write_json(args.run_dir / "results/provenance.json", value)
    with pytest.raises(ValueError, match=match):
        audit.validate_run(runtime, args, manifest)


def test_main_evaluates_original_artifacts_without_training_or_changing_them(artifacts, setup, tmp_path, monkeypatch):
    args, _, _ = artifacts
    _, model, _, _, _ = setup
    monkeypatch.setattr(audit, "import_training_code", lambda *_: (runtime, data_utils, scoring))
    monkeypatch.setattr(audit, "git_head", lambda *_: "audit-source")
    monkeypatch.setattr(runtime, "initialize_distributed", lambda: DistributedContext())
    monkeypatch.setattr(runtime, "load_tokenizer", lambda *_: CharTokenizer())
    def build(*_args, **kwargs):
        assert kwargs == {"load_encoder": False}
        return model
    monkeypatch.setattr(runtime, "build_runtime_model", build)
    monkeypatch.setattr(runtime, "train", lambda *_: pytest.fail("Audit trained a model"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *_: pytest.fail("Audit made an optimizer update"))
    files = list(args.run_dir.rglob("*")) + list(args.data_dir.rglob("*"))
    before = {p: audit.file_hash(p) for p in files if p.is_file()}
    output = tmp_path / "audit-output"
    assert audit.main(["--run-dir", str(args.run_dir), "--data-dir", str(args.data_dir),
        "--packet-cache", str(args.packet_cache), "--training-repo", str(tmp_path / "training-source"),
        "--output", str(output)]) == 0
    summary = audit.read_json(output / "audit_summary.json")
    assert summary["examples"] == 5 and set(summary["modes"]) == set(audit.MODES)
    assert summary["provenance"]["training_source_commit"] == "original-source"
    assert summary["provenance"]["audit_source_commit"] == "audit-source"
    assert len(summary["provenance"]["audit_script_sha256"]) == 64
    assert before == {p: audit.file_hash(p) for p in before}


def test_existing_output_is_rejected_before_importing_training_code(tmp_path, monkeypatch):
    output = tmp_path / "results"
    output.mkdir()
    (output / "keep").write_text("original")
    monkeypatch.setattr(audit, "import_training_code", lambda *_: pytest.fail("Imported models before output check"))
    with pytest.raises(ValueError, match="nonempty"):
        audit.main(["--run-dir", str(tmp_path / "old"), "--data-dir", str(tmp_path / "data"),
            "--packet-cache", str(tmp_path / "packets"), "--training-repo", str(tmp_path / "source"),
            "--output", str(output)])
