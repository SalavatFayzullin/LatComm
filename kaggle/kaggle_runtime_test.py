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


@pytest.mark.parametrize("magnitude", [0.0, 0.001])
def test_qwen_fp16_normalization_keeps_small_vector_gradients_finite(monkeypatch, magnitude):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.l2norm
    monkeypatch.setattr(qwen, "l2norm", original)  # Restore the global after this test.
    values = torch.full((1, 4, 2, 16), magnitude, dtype=torch.float16, requires_grad=True)
    weights = torch.arange(values.numel(), dtype=torch.float16).reshape_as(values) / values.numel()
    output = original(values)
    assert torch.isfinite(output).all()
    (output * weights).sum().backward()
    assert not torch.isfinite(values.grad).all()  # Regression in the pinned fallback.

    runtime.enable_fp16_stability("fp16")
    installed = qwen.l2norm
    runtime.enable_fp16_stability("fp16")
    assert qwen.l2norm is installed
    values.grad = None
    stable = qwen.l2norm(values)
    (stable * weights).sum().backward()
    reference = values.detach().double().requires_grad_()
    normalized = reference / (reference.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    (normalized * weights.double()).sum().backward()
    assert stable.dtype == values.dtype
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(stable, normalized.half(), rtol=1e-3, atol=5e-4)
    torch.testing.assert_close(values.grad, reference.grad.half(), rtol=1e-3, atol=5e-4)


def test_runtime_installs_qwen_fp16_adapter_before_loading(monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.l2norm
    monkeypatch.setattr(qwen, "l2norm", original)
    runtime.enable_fp16_stability("bf16")
    assert qwen.l2norm is original
    config = LatCommConfig.model_validate(config_dict(Path("data"), Path("output")))
    def load(adapted, device, **kwargs):
        assert getattr(qwen.l2norm, "_latcomm_fp32", False)
        assert adapted.encoder.dtype == adapted.receiver.dtype == "float16"
        assert device == "cpu" and kwargs == {"load_receiver": False}
        return "loaded"
    monkeypatch.setattr(runtime, "build_model", load)
    assert runtime.build_runtime_model(config, "fp16", "cpu", load_receiver=False) == "loaded"
    for dtype in (torch.bfloat16, torch.float32):
        values = torch.randn(2, 3, 4, dtype=dtype)
        assert torch.equal(qwen.l2norm(values), original(values))


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_real_qwen_layers_and_latcomm_backward_are_finite_in_fp16(monkeypatch, checkpoint_layers):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    monkeypatch.setattr(qwen, "l2norm", qwen.l2norm)
    runtime.enable_fp16_stability("fp16")
    config = Qwen3_5TextConfig(
        vocab_size=80, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )
    config._attn_implementation = "sdpa"
    receiver = qwen.Qwen3_5ForCausalLM(config).half().requires_grad_(False).eval()
    # Zero Q/K is a valid input to normalization and reliably triggers the old
    # backward failure. Test the real hybrid layers, reader hooks and CE path.
    with torch.no_grad():
        for layer in receiver.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.in_proj_qkv.weight.zero_()
    encoder = TinyLM(hidden_size=24, num_layers=4).requires_grad_(False).eval()
    pairs = select_layer_pairs(4, 4, 1.0)
    bridge = LatentBridge(pairs, 24, 32, BridgeConfig(
        context_dim=8, num_slots=2, num_attention_heads=2, gate_pooling_dim=8))
    model = LatCommModel(wrap_frozen_lm(encoder, "cpu"), wrap_frozen_lm(receiver, "cpu"), bridge)
    if checkpoint_layers:
        model.enable_gradient_checkpointing()
    model.train()
    batch = {
        "context": {"input_ids": torch.tensor([[3, 4, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)},
        "prompt": {"input_ids": torch.tensor([[1, 7, 8, 9, 10, 11, 12, 13]]),
                   "attention_mask": torch.ones(1, 8, dtype=torch.long),
                   "prompt_end_positions": torch.tensor([2]),
                   "prediction_positions": torch.tensor([[2, 3, 4, 5, 6]])},
        "response_ids": torch.tensor([[9, 10, 11, 12, 13]]),
        "response_mask": torch.ones(1, 5, dtype=torch.bool),
    }
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = bridge.readers[0].output_projection.weight.detach().clone()
    for _ in range(2):
        score, _ = runtime.forward_backward(model, batch, scaler)
        assert torch.isfinite(score.nll).all()
        gradients = [parameter.grad for parameter in bridge.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(bridge.readers[0].output_projection.weight, before)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in receiver.parameters())


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
    with caplog.at_level(logging.DEBUG, logger=runtime.LOGGER.name):
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


def test_reduced_profile_prepares_data_and_completes_both_stages(tmp_path, monkeypatch):
    import data_utils
    spec = importlib.util.spec_from_file_location("pilot_run_config", KAGGLE / "run.py")
    orchestration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestration)
    policy = {"seed": 42,
              "stage1": {"train": 32, "dev": 2, "fast_dev": 2, "test": 2},
              "stage2_per_dataset": {"train": 4, "dev": 2, "fast_dev": 2, "test": 2},
              "stage2_datasets": 3}
    monkeypatch.setattr(data_utils, "KAGGLE_POLICY", policy)
    source = tmp_path / "source"
    write_data(source / "stage1", count=40)
    # prepare_data also preserves a dev split for a complete local source.
    shutil.copy2(source / "stage1/fast_dev.jsonl", source / "stage1/dev.jsonl")
    task_root = source / "stage2"
    task_root.mkdir(parents=True)
    for split in ("train", "dev", "fast_dev", "test"):
        rows = []
        for group in range(3):
            for index in range(5):
                context = f"context {group} {index}"
                rows.append({"sample_id": f"{split}-{group}-{index}", "question": "Which?",
                             "choices": ["alpha", "beta"], "response": "AB"[index % 2],
                             "evidence": [{"source_record_id": context, "text": context}],
                             "provenance": {"dataset": f"task-{group}"}})
        (task_root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = tmp_path / "data"
    receipt = data_utils.prepare_data(prepared, revision="fixed", profile="kaggle", local_source=source)
    args = argparse.Namespace(profile="kaggle", pair="qwen0.8b_to_qwen0.8b", data_dir=prepared,
                              workdir=tmp_path / "run", packet_cache=tmp_path / "packets",
                              microbatch=2, checkpoint_every=1)
    state = {"models": {side: {"revision": "fixed"} for side in ("encoder", "receiver")}}
    configs = orchestration.configure(args, REPO, state, receipt, 1)
    assert runtime.train(train_args(configs["stage1"])) == 0
    final1 = args.workdir / "outputs/stage1/checkpoints/final"
    encoded = argparse.Namespace(config=str(configs["stage2"]), checkpoint=str(final1), output=str(args.packet_cache),
                                 precision="bf16", batch_size=2, splits=None, source_commit="cpu-test-source")
    assert runtime.encode(encoded) == 0
    assert runtime.train(train_args(configs["stage2"])) == 0
    final2 = args.workdir / "outputs/stage2/checkpoints/final"
    evaluated = argparse.Namespace(config=str(configs["stage2"]), checkpoint=str(final2),
                                   data=str(prepared / "stage2/test.jsonl"), output=str(args.workdir / "results"),
                                   precision="bf16", source_commit="cpu-test-source")
    assert runtime.evaluate_command(evaluated) == 0
    metrics = json.loads((args.workdir / "results/metrics.json").read_text())
    assert len([name for name in metrics if name.startswith("accuracy/")]) == 3
