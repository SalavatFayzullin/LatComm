"""Meaningful, download-free CPU checks for Kaggle continuation and packets."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
import yaml

KAGGLE = Path(__file__).resolve().parent
REPO = KAGGLE.parent
sys.path.insert(0, str(KAGGLE))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO))

import runtime
from tiny import CharTokenizer, TinyLM
from latcomm.bridge import LatentBridge, select_layer_pairs
from latcomm.config import BridgeConfig, LatCommConfig
from latcomm.model import LatCommModel, wrap_frozen_lm


def tiny_model(config, device="cpu", *, load_encoder=True, load_receiver=True):
    encoder = TinyLM(hidden_size=24, num_layers=3).requires_grad_(False).eval()
    receiver = TinyLM(hidden_size=16, num_layers=2).requires_grad_(False).eval()
    wrapped_encoder = wrap_frozen_lm(encoder, device)
    wrapped_receiver = wrap_frozen_lm(receiver, device)
    if not load_encoder:
        wrapped_encoder.model = None
    if not load_receiver:
        wrapped_receiver.model = None
    pairs = select_layer_pairs(3, 2, config.bridge.num_layer_pairs)
    return LatCommModel(wrapped_encoder, wrapped_receiver, LatentBridge(pairs, 24, 16, config.bridge))


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runtime, "build_model", tiny_model)
    monkeypatch.setattr(runtime, "load_tokenizer", lambda _: CharTokenizer())


def write_data(root, kind="text", count=8):
    root.mkdir(parents=True, exist_ok=True)
    for split, size in (("train", count), ("fast_dev", 2), ("test", 2)):
        rows = []
        for index in range(size):
            context = f"alpha beta {index % 4}"
            rows.append({
                "sample_id": f"{split}-{index}", "question": "Repeat.",
                "choices": ["alpha", "beta"] if kind == "mcq" else [],
                "response": "AB"[index % 2] if kind == "mcq" else context,
                "evidence": [{"source_record_id": f"c-{index % 4}", "text": context}],
                "provenance": {"dataset": "tiny"},
            })
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def config_dict(root, output, kind="text"):
    return {
        "encoder": {"model_name_or_path": "tiny-encoder"},
        "receiver": {"model_name_or_path": "tiny-receiver"},
        "bridge": {"context_dim": 8, "num_slots": 2, "num_attention_heads": 2, "gate_pooling_dim": 8},
        "data": {"dataset_root": str(root), "response_format": kind, "num_workers": 0,
                 "sort_within_update": True},
        "training": {
            "output_dir": str(output), "per_device_batch_size": 2, "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 2, "num_epochs": 2,
            "warmup_steps": 1, "scheduler": "cosine", "learning_rate": 1e-3,
            "log_every": 1, "eval_every": 2, "save_every": 500, "gradient_checkpointing": True,
        },
    }


def train_args(path, *, limit=0, resume="auto"):
    return argparse.Namespace(config=str(path), precision="bf16", source_commit="cpu-test-source",
                              session_steps=limit, session_hours=0, checkpoint_every=1, resume=resume)


def save_config(path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right, strict=True):
            assert_nested_equal(one, two)
    else:
        assert left == right


@pytest.mark.parametrize("relocate", [False, True])
def test_exact_resume_matches_uninterrupted_and_retains_two_checkpoints(tmp_path, relocate):
    data = tmp_path / "data"
    write_data(data)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    full_config, interrupted_config = tmp_path / "full.yaml", tmp_path / "interrupted.yaml"
    save_config(full_config, config_dict(data, full))
    save_config(interrupted_config, config_dict(data, interrupted))
    assert runtime.train(train_args(full_config)) == 0
    assert runtime.train(train_args(interrupted_config, limit=1)) == 75
    assert not (interrupted / "checkpoints" / "final").exists()
    assert (interrupted / "session-paused.json").exists()
    if relocate:
        new_data, new_output = tmp_path / "attached-data", tmp_path / "next-session"
        shutil.copytree(data, new_data)
        shutil.copytree(interrupted, new_output)
        interrupted = new_output
        save_config(interrupted_config, config_dict(new_data, new_output))
    assert runtime.train(train_args(interrupted_config)) == 0
    assert not (interrupted / "session-paused.json").exists()
    from safetensors.torch import load_file
    full_bridge = load_file(str(full / "checkpoints" / "final" / "bridge.safetensors"))
    resumed_bridge = load_file(str(interrupted / "checkpoints" / "final" / "bridge.safetensors"))
    assert_nested_equal(full_bridge, resumed_bridge)
    states = [torch.load(root / "checkpoints" / "final" / "training_state.pt",
                         weights_only=False, map_location="cpu") for root in (full, interrupted)]
    for field in ("optimizer", "scheduler", "scaler", "rng", "progress"):
        assert_nested_equal(states[0][field], states[1][field])
    assert len(list((interrupted / "checkpoints").glob("step-*"))) <= 2
    assert runtime.train(train_args(interrupted_config)) == 0


def test_resume_rejects_changed_data_and_settings(tmp_path):
    data, output = tmp_path / "data", tmp_path / "run"
    write_data(data)
    value = config_dict(data, output)
    path = tmp_path / "config.yaml"
    save_config(path, value)
    assert runtime.train(train_args(path, limit=1)) == 75
    changed = deepcopy(value)
    changed["training"]["gradient_accumulation_steps"] = 1
    save_config(path, changed)
    with pytest.raises(ValueError, match="Resume rejected"):
        runtime.train(train_args(path))
    save_config(path, value)
    with (data / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        runtime.train(train_args(path))


def test_resumable_bf16_packet_cache_and_stage2_freeze(tmp_path, monkeypatch):
    text, mcq = tmp_path / "text", tmp_path / "mcq"
    write_data(text)
    write_data(mcq, "mcq")
    config1, config2 = tmp_path / "stage1.yaml", tmp_path / "stage2.yaml"
    output1, output2 = tmp_path / "stage1", tmp_path / "stage2"
    save_config(config1, config_dict(text, output1))
    assert runtime.train(train_args(config1)) == 0
    final1 = output1 / "checkpoints" / "final"
    task = config_dict(mcq, output2, "mcq")
    task["data"]["packet_cache"] = str(tmp_path / "packets")
    task["training"].update(init_checkpoint=str(final1), trainable="readers_and_gates")
    save_config(config2, task)
    args = argparse.Namespace(config=str(config2), checkpoint=str(final1), output=str(tmp_path / "packets"),
                              precision="bf16", batch_size=1, splits=None, source_commit="cpu-test-source")
    assert runtime.encode(args) == 0
    meta = runtime.packet_fingerprint(tmp_path / "packets")["meta"]
    assert meta["dtype"] == "bfloat16" and meta["contexts"] == 4
    # A repeat must reuse all persisted contexts, not run the encoder again.
    with monkeypatch.context() as readonly:
        readonly.setattr(LatCommModel, "encode_context", lambda *_: pytest.fail("re-encoded cached context"))
        readonly.setattr(runtime, "build_model", lambda *_args, **_kwargs: pytest.fail("loaded model for complete cache"))
        readonly.setattr(runtime, "atomic_json", lambda *_: pytest.fail("wrote into complete read-only cache"))
        assert runtime.encode(args) == 0
    assert runtime.train(train_args(config2)) == 0
    from safetensors.torch import load_file
    before = load_file(str(final1 / "bridge.safetensors"))
    after = load_file(str(output2 / "checkpoints" / "final" / "bridge.safetensors"))
    assert all(torch.equal(before[name], value) for name, value in after.items() if name.startswith("compressors."))
    assert any(not torch.equal(before[name], value) for name, value in after.items() if name.startswith("readers."))
    evaluation = argparse.Namespace(config=str(config2), checkpoint=str(output2 / "checkpoints" / "final"),
                                    data=str(mcq / "test.jsonl"), output=str(tmp_path / "results"),
                                    precision="bf16", source_commit="cpu-test-source")
    assert runtime.evaluate_command(evaluation) == 0
    assert (tmp_path / "results" / "metrics.json").exists()
    assert len((tmp_path / "results" / "predictions.jsonl").read_text().splitlines()) == 2


def test_precision_adapter_changes_only_frozen_model_loading():
    config = LatCommConfig.model_validate(config_dict(Path("data"), Path("output")))
    adapted = runtime.precision_config(config, "fp16")
    assert config.encoder.dtype == "bfloat16"
    assert adapted.encoder.dtype == adapted.receiver.dtype == "float16"
    assert adapted.bridge == config.bridge


def test_overflow_retries_identical_update_without_skipping_examples(tmp_path, monkeypatch):
    data = tmp_path / "data"
    write_data(data)
    reference_path, retry_path = tmp_path / "reference.yaml", tmp_path / "retry.yaml"
    save_config(reference_path, config_dict(data, tmp_path / "reference"))
    save_config(retry_path, config_dict(data, tmp_path / "retry"))
    assert runtime.train(train_args(reference_path)) == 0

    class SimulatedScaler:
        def __init__(self, *args, **kwargs):
            self.value = 1024.0
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def is_enabled(self):
            return True
        def get_scale(self):
            return self.value
        def update(self, new_scale=None):
            if new_scale is not None:
                self.value = new_scale
        def step(self, optimizer):
            optimizer.step()
        def state_dict(self):
            return {"value": self.value}
        def load_state_dict(self, state):
            self.value = state["value"]

    original = runtime.forward_backward
    visits = []
    def poison_once(model, batch, scaler):
        visits.append(tuple(batch["ids"]))
        result = original(model, batch, scaler)
        if len(visits) == 1:
            next(model.bridge.parameters()).grad.fill_(float("inf"))
        return result
    monkeypatch.setattr(runtime.torch.amp, "GradScaler", SimulatedScaler)
    monkeypatch.setattr(runtime, "forward_backward", poison_once)
    assert runtime.train(train_args(retry_path)) == 0
    assert visits[0:2] == visits[2:4]
    final = tmp_path / "retry" / "checkpoints" / "final"
    state = torch.load(final / "training_state.pt", weights_only=False)
    assert state["progress"] == {"step": 4, "epoch": 2, "next_batch": 0, "overflow_retries": 1}
    from safetensors.torch import load_file
    assert_nested_equal(load_file(str(final / "bridge.safetensors")),
                        load_file(str(tmp_path / "reference" / "checkpoints" / "final" / "bridge.safetensors")))


def test_encoding_disk_preflight_runs_before_loading_models(tmp_path, monkeypatch):
    data = tmp_path / "data"
    write_data(data)
    config = tmp_path / "config.yaml"
    save_config(config, config_dict(data, tmp_path / "run"))
    assert runtime.train(train_args(config)) == 0
    from types import SimpleNamespace
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(runtime, "build_model", lambda *_args, **_kwargs: pytest.fail("model loaded before disk preflight"))
    args = argparse.Namespace(config=str(config), checkpoint=str(tmp_path / "run" / "checkpoints" / "final"),
                              output=str(tmp_path / "packets"), precision="bf16", batch_size=1,
                              splits=None, source_commit="cpu-test-source")
    with pytest.raises(OSError, match="Insufficient disk"):
        runtime.encode(args)


def test_progress_reports_microbatches_and_fp16_retries(caplog):
    import logging
    progress = runtime.MicrobatchProgress(1, interval=0.0)
    with caplog.at_level(logging.INFO, logger=runtime.LOGGER.name):
        progress.start(1, 192)
        progress.phase("backward")
        progress.begin_microbatch()
        progress.finish_microbatch()
        progress.repeat(1, 1024, 512)
        progress.phase("encoder_and_compressors")
    events = [json.loads(record.message) for record in caplog.records]
    completed = next(event for event in events if event["type"] == "microbatch_progress")
    assert completed["rank"] == 1 and completed["microbatch"] == 1 and completed["accumulation"] == 192
    retry = next(event for event in events if event["type"] == "fp16_retry")
    assert retry["microbatches_to_repeat"] == 192 and retry["new_loss_scale"] == 512


def test_benchmark_skips_evaluation_and_never_updates_weights(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "benchmark"
    write_data(data)
    config = tmp_path / "config.yaml"
    save_config(config, config_dict(data, output))
    args = train_args(config)
    args.benchmark_microbatches = 3
    monkeypatch.setattr(runtime, "upstream_evaluate", lambda *_: pytest.fail("benchmark ran full evaluation"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *_args, **_kwargs: pytest.fail("benchmark updated weights"))
    assert runtime.train(args) == 0
    result = json.loads((output / "benchmark.json").read_text())
    assert result["optimizer_updates"] == 0
    assert result["ranks"][0]["microbatches"] == 3
    assert result["ranks"][0]["sampled_microbatch_positions"] == [0, 1, 2]
    assert result["ranks"][0]["finite_gradients"]
    assert result["rough_update_seconds"] > 0
    assert not (output / "checkpoints").exists()
