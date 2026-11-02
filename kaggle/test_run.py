"""CPU orchestration checks: no Hub, GPU, pip installation, or git network calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPEC = importlib.util.spec_from_file_location("latcomm_kaggle_run_tests", HERE / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def options(tmp_path, profile="full", pair="qwen4b_to_qwen0.8b"):
    work = tmp_path / "work"
    return SimpleNamespace(
        pair=pair, profile=profile, data_dir=tmp_path / "data",
        workdir=work, packet_cache=tmp_path / "packets",
        microbatch=1, checkpoint_every=5,
    )


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("pair", run.PAIRS)
def test_full_preserves_paper_method_and_global_batch(tmp_path, processes, pair):
    args = options(tmp_path, pair=pair)
    state = {"models": {"encoder": {"revision": "encoder-sha"},
                        "receiver": {"revision": "receiver-sha"}}}
    paths = run.configure(args, REPO, state, {}, processes)
    for stage, epochs, warmup, scheduler in (
        ("stage1", 1, 157, "constant"), ("stage2", 3, 45, "cosine")
    ):
        actual = yaml.safe_load(paths[stage].read_text(encoding="utf-8"))
        upstream = yaml.safe_load((REPO / "configs" / stage / f"{pair}.yaml").read_text())
        assert actual["bridge"] == upstream["bridge"]
        assert actual["bridge"]["num_slots"] == 32
        assert actual["bridge"]["context_dim"] == 256
        assert actual["bridge"]["num_layer_pairs"] == 1.0
        train = actual["training"]
        assert train["num_epochs"] == epochs
        assert train["warmup_steps"] == warmup
        assert train["scheduler"] == scheduler
        assert train["learning_rate"] == upstream["training"]["learning_rate"]
        assert train["trainable"] == upstream["training"]["trainable"]
        assert train["per_device_batch_size"] * train["gradient_accumulation_steps"] * processes == 384
        assert actual["data"]["dataset_root"] == str(args.data_dir / stage)
        assert train["output_dir"] == str(args.workdir / "outputs" / stage)
        for side in ("encoder", "receiver"):
            assert actual[side]["model_name_or_path"] == upstream[side]["model_name_or_path"]
            assert actual[side]["revision"] == state["models"][side]["revision"]
        if stage == "stage1":
            assert actual["data"]["response_format"] == "text"
            assert "packet_cache" not in actual["data"]
            assert "init_checkpoint" not in train
        else:
            assert actual["data"]["response_format"] == "mcq"
            assert actual["data"]["packet_cache"] == str(args.packet_cache)
            assert train["init_checkpoint"] == str(args.workdir / "outputs/stage1/checkpoints/final")


def test_smoke_is_explicit_and_does_not_mutate_upstream(tmp_path):
    state = {"models": {side: {"revision": "pinned"} for side in ("encoder", "receiver")}}
    upstream_path = REPO / "configs/stage1/qwen4b_to_qwen0.8b.yaml"
    before = upstream_path.read_bytes()
    full = run.configure(options(tmp_path / "full"), REPO, state, {}, 2)
    smoke = run.configure(options(tmp_path / "smoke", profile="smoke"), REPO, state, {}, 2)
    for stage in ("stage1", "stage2"):
        full_config = yaml.safe_load(full[stage].read_text())
        smoke_config = yaml.safe_load(smoke[stage].read_text())
        assert full_config["training"]["warmup_steps"] > 0
        assert smoke_config["training"]["warmup_steps"] == 0
        assert full_config["bridge"] == smoke_config["bridge"]
        assert full_config["training"]["num_epochs"] == smoke_config["training"]["num_epochs"]
        assert smoke_config["training"]["gradient_accumulation_steps"] == 192
    assert upstream_path.read_bytes() == before


def test_single_gpu_command_and_distributed_command():
    args = ["train", "--config", "stage1.yaml", "--precision", "fp16"]
    single = run.runtime_command(1, args)
    assert single == [run.sys.executable, str(run.HERE / "runtime.py"), *args]
    multi = run.runtime_command(2, args)
    assert multi == [run.sys.executable, "-m", "torch.distributed.run", "--standalone",
                     "--nproc_per_node", "2", str(run.HERE / "runtime.py"), *args]


def test_repository_comes_from_run_script_location():
    assert run.repository() == REPO


def test_copied_run_without_repository_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "HERE", tmp_path / "kaggle")
    with pytest.raises(RuntimeError, match="inside the cloned"):
        run.repository()


def test_source_identity_records_actual_fork_and_commit(monkeypatch):
    fork = "https://github.com/SalavatFayzullin/LatComm.git"
    commands = Mock(side_effect=["a" * 40 + "\n", fork + "\n"])
    monkeypatch.setattr(run.subprocess, "check_output", commands)
    assert run.source_identity(REPO) == {"source_url": fork, "source_commit": "a" * 40}
    assert commands.call_args_list[0].args[0] == ["git", "-C", str(REPO), "rev-parse", "HEAD"]
    assert commands.call_args_list[1].args[0] == ["git", "-C", str(REPO), "remote", "get-url", "origin"]


def test_setup_installs_current_checkout_without_cloning(tmp_path, monkeypatch):
    install = Mock()
    monkeypatch.setattr(run, "install", install)
    monkeypatch.setenv("LATCOMM_INDEX_DIR", str(tmp_path / "indexes"))
    assert run.main(["setup", "--workdir", str(tmp_path / "work")]) == 0
    install.assert_called_once_with(REPO)


def test_manifest_pins_fork_head_and_rejects_changed_source(tmp_path, monkeypatch):
    import huggingface_hub

    args = options(tmp_path)
    args.data_revision = None
    api = Mock()
    api.model_info.side_effect = [SimpleNamespace(sha="encoder-sha"), SimpleNamespace(sha="receiver-sha")]
    api.dataset_info.return_value = SimpleNamespace(sha="data-sha")
    factory = Mock(return_value=api)
    monkeypatch.setattr(huggingface_hub, "HfApi", factory)
    source = {"source_url": "https://github.com/SalavatFayzullin/LatComm.git", "source_commit": "a" * 40}
    state = run.manifest(args, REPO, source)
    assert state["source_url"] == source["source_url"]
    assert state["source_commit"] == "a" * 40
    assert state["dataset_revision"] == "data-sha"
    factory.reset_mock()
    assert run.manifest(args, REPO, source) == state
    factory.assert_not_called()
    with pytest.raises(RuntimeError, match="source_commit changed"):
        run.manifest(args, REPO, {**source, "source_commit": "b" * 40})


def test_runtime_environment_uses_same_checkout_and_commit():
    environment = run.environment(REPO, "a" * 40)
    assert environment["LATCOMM_SOURCE_COMMIT"] == "a" * 40
    assert environment["PYTHONPATH"].split(run.os.pathsep)[0] == str(REPO)


def test_setup_pins_pascal_compatible_wheels(tmp_path, monkeypatch):
    commands = Mock()
    monkeypatch.setattr(run, "call", commands)
    run.install(tmp_path)
    all_commands = [item.args[0] for item in commands.call_args_list]
    torch_install = all_commands[0]
    assert "torch==2.11.0" in torch_install
    assert "torchvision==0.26.0" in torch_install
    assert "torchaudio==2.11.0" in torch_install
    assert torch_install[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu126"]
    assert "transformers==5.15.1" in all_commands[1]
    assert all_commands[-1] == [run.sys.executable, "-m", "pip", "install", "--quiet",
                                "--disable-pip-version-check", "--no-deps", "-e", tmp_path]


def test_benchmark_runs_stage1_only_instead_of_continuing_pipeline(tmp_path, monkeypatch):
    import data_utils

    monkeypatch.setattr(run, "source_identity", lambda _: {"source_commit": "a" * 40})
    monkeypatch.setattr(run, "manifest", lambda *_: {"dataset_revision": "data-sha"})
    monkeypatch.setattr(data_utils, "prepare_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "hardware", lambda _: (2, "fp16", []))
    monkeypatch.setattr(run, "configure", lambda *_: {})
    launch = Mock(return_value=0)
    monkeypatch.setattr(run, "launch", launch)
    monkeypatch.setenv("LATCOMM_INDEX_DIR", str(tmp_path / "indexes"))
    assert run.main(["all", "--workdir", str(tmp_path / "benchmark"), "--benchmark-microbatches", "3"]) == 0
    assert launch.call_count == 1
    assert launch.call_args.args[5] == "stage1"
    assert launch.call_args.args[0].benchmark_microbatches == 3


def test_kaggle_profile_changes_budget_but_keeps_method(tmp_path):
    args = options(tmp_path, profile="kaggle", pair="qwen0.8b_to_qwen0.8b")
    state = {"models": {side: {"revision": "pinned"} for side in ("encoder", "receiver")}}
    paths = run.configure(args, REPO, state, {}, 2)
    expected = {"stage1": (2048, 1, 64, 2, "all", "constant"),
                "stage2": (1536, 3, 144, 3, "readers_and_gates", "cosine")}
    for stage, (examples, epochs, updates, warmup, trainable, scheduler) in expected.items():
        config = yaml.safe_load(paths[stage].read_text())
        original = yaml.safe_load((REPO / "configs" / stage / f"{args.pair}.yaml").read_text())
        training = config["training"]
        assert config["bridge"] == original["bridge"]
        assert training["per_device_batch_size"] * training["gradient_accumulation_steps"] * 2 == 32
        assert (examples + 31) // 32 * epochs == updates
        assert training["warmup_steps"] == warmup
        assert training["trainable"] == trainable
        assert training["scheduler"] == scheduler
        assert training["num_epochs"] == epochs
        assert config["receiver"]["model_name_or_path"] == config["encoder"]["model_name_or_path"] == "Qwen/Qwen3.5-0.8B"


@pytest.mark.parametrize("profile,pair", [("kaggle", "qwen0.8b_to_qwen0.8b"),
                                         ("full", "qwen4b_to_qwen0.8b"),
                                         ("smoke", "qwen4b_to_qwen0.8b")])
def test_profile_defaults_are_explicit_and_have_no_extra_stop_caps(tmp_path, monkeypatch, profile, pair):
    import data_utils
    captured = []
    monkeypatch.setattr(run, "source_identity", lambda _: {"source_commit": "a" * 40})
    def state(args, *_):
        captured.append(args)
        return {"dataset_revision": "data-sha"}
    monkeypatch.setattr(run, "manifest", state)
    monkeypatch.setattr(data_utils, "prepare_data", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("LATCOMM_INDEX_DIR", str(tmp_path / "indexes"))
    assert run.main(["prepare", "--profile", profile, "--workdir", str(tmp_path / "run")]) == 0
    assert captured[0].pair == pair
    assert captured[0].session_steps == captured[0].session_hours == 0
