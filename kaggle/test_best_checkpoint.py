"""Validation selection preserves exact training state; CPU only, no downloads."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_runtime_test as support
import runtime
from latcomm.config import LatCommConfig

cpu_only = support.cpu_only


def candidate(tmp_path, *, stage2=False):
    value = support.config_dict(tmp_path / "data", tmp_path / "run")
    if stage2:
        value["training"]["trainable"] = "readers_and_gates"
    config = LatCommConfig.model_validate(value)
    fingerprint = {"precision": "bf16", "code": {"source_commit": "test-source"},
                   "data": {"train": "train-digest", "eval": "validation-digest"}}
    return Path(config.training.output_dir), support.tiny_model(config), config, fingerprint


def save_candidate(parts, step, loss):
    output, model, config, fingerprint = parts
    return runtime.save_best_checkpoint(output, model, config, fingerprint,
                                        {"step": step, "epoch": 0, "next_batch": step}, {"loss": loss})


def test_selection_policy_and_bridge_only_resume_rejection(tmp_path):
    parts = candidate(tmp_path)
    output, _, _, fingerprint = parts
    assert save_candidate(parts, 0, 0.1) is None  # Stage 1 initialization is excluded.
    assert save_candidate(parts, 1, float("nan")) is None
    assert save_candidate(parts, 1, float("inf")) is None
    chosen = save_candidate(parts, 1, 1.0)
    pointer = (output / "best.json").read_bytes()
    assert save_candidate(parts, 2, 1.0) == chosen  # Ties retain the earlier selection.
    assert (output / "best.json").read_bytes() == pointer
    bridge = output / chosen["path"]
    assert not (bridge / "training_state.pt").exists()
    assert not (output / "latest.json").exists()
    with pytest.raises(ValueError, match="evaluation only"):
        runtime.resolve_resume(output, str(bridge))
    metadata = runtime.read_json(bridge / "runtime.json")
    assert metadata["complete"] is False and metadata["resumable"] is False
    assert runtime.read_best_checkpoint(output, fingerprint) == chosen
    stage2 = candidate(tmp_path / "stage2", stage2=True)
    assert save_candidate(stage2, 0, 2.0)["step"] == 0


def test_selection_rejects_other_run_and_tampered_payload_or_path(tmp_path):
    parts = candidate(tmp_path)
    output, _, _, fingerprint = parts
    best = save_candidate(parts, 1, 1.0)
    changed = deepcopy(fingerprint)
    changed["code"]["source_commit"] = "another-source"
    with pytest.raises(ValueError, match="source, data, settings"):
        runtime.read_best_checkpoint(output, changed)
    pointer = deepcopy(best)
    pointer["path"] = "../outside"
    runtime.atomic_json(output / "best.json", pointer)
    with pytest.raises(ValueError, match="escaped"):
        runtime.read_best_checkpoint(output, fingerprint)
    runtime.atomic_json(output / "best.json", best)
    with (output / best["path"] / "bridge.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        runtime.read_best_checkpoint(output, fingerprint)


def test_publication_failure_preserves_previous_selection_then_reuses_orphan(tmp_path, monkeypatch):
    parts = candidate(tmp_path)
    output, _, _, fingerprint = parts
    previous = save_candidate(parts, 1, 2.0)
    atomic = runtime.atomic_json

    def fail_pointer(path, value):
        if path == output / "best.json":
            raise OSError("simulated pointer publication interruption")
        return atomic(path, value)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(runtime, "atomic_json", fail_pointer)
        with pytest.raises(OSError, match="interruption"):
            save_candidate(parts, 2, 1.0)
    assert runtime.read_best_checkpoint(output, fingerprint) == previous
    assert (output / previous["path"]).is_dir()
    best = save_candidate(parts, 2, 1.0)
    assert best["step"] == 2
    assert not (output / previous["path"]).exists()
    assert runtime.read_best_checkpoint(output, fingerprint) == best


def controlled_validation(monkeypatch, losses):
    evaluate = runtime.upstream_evaluate
    values = iter(losses)

    def with_loss(*args, **kwargs):
        metrics, rows = evaluate(*args, **kwargs)
        return {**metrics, "loss": next(values)}, rows

    monkeypatch.setattr(runtime, "upstream_evaluate", with_loss)


def training_config(tmp_path, name):
    data = tmp_path / "data"
    if not data.exists():
        support.write_data(data)
    output, path = tmp_path / name, tmp_path / f"{name}.yaml"
    value = support.config_dict(data, output)
    value["training"]["num_epochs"] = 3  # Six updates, validations at 0, 2, 4, 6.
    support.save_config(path, value)
    return output, path


def test_best_survives_normal_retention_and_final_worse_without_changing_training(tmp_path, monkeypatch):
    selected, selected_config = training_config(tmp_path, "selected")
    baseline, baseline_config = training_config(tmp_path, "without-selection")
    with monkeypatch.context() as enabled:
        controlled_validation(enabled, [100.0, 1.0, 2.0, 3.0])
        assert runtime.train(support.train_args(selected_config)) == 0
    with monkeypatch.context() as disabled:
        controlled_validation(disabled, [100.0, 1.0, 2.0, 3.0])
        disabled.setattr(runtime, "save_best_checkpoint", lambda *_: None)
        assert runtime.train(support.train_args(baseline_config)) == 0
    best = runtime.read_json(selected / "best.json")
    assert best["step"] == 2 and best["value"] == 1.0
    assert (selected / best["path"]).is_dir()
    assert not (selected / "checkpoints/step-00000002").exists()
    assert len(list((selected / "checkpoints").glob("step-*"))) == 2
    assert Path(runtime.read_json(selected / "latest.json")["path"]) == Path("checkpoints/final")
    final_weights = load_file(str(selected / "checkpoints/final/bridge.safetensors"))
    best_weights = load_file(str(selected / best["path"] / "bridge.safetensors"))
    assert any(not torch.equal(final_weights[key], best_weights[key]) for key in final_weights)
    support.assert_nested_equal(final_weights, load_file(str(baseline / "checkpoints/final/bridge.safetensors")))
    states = [torch.load(root / "checkpoints/final/training_state.pt", weights_only=False, map_location="cpu")
              for root in (selected, baseline)]
    for field in ("optimizer", "scheduler", "scaler", "rng", "progress", "statistics"):
        support.assert_nested_equal(states[0][field], states[1][field])


def test_resumed_run_honors_previous_best_and_equal_loss(tmp_path, monkeypatch):
    output, config = training_config(tmp_path, "resumed")
    with monkeypatch.context() as first:
        controlled_validation(first, [100.0, 1.0])
        assert runtime.train(support.train_args(config, limit=2)) == 75
    previous = (output / "best.json").read_bytes()
    with monkeypatch.context() as continued:
        controlled_validation(continued, [1.0, 3.0])
        assert runtime.train(support.train_args(config)) == 0
    assert (output / "best.json").read_bytes() == previous
    best = runtime.read_json(output / "best.json")
    assert best["step"] == 2 and (output / best["path"]).is_dir()
    assert runtime.read_json(output / "latest.json")["complete"] is True
