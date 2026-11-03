"""Controls must never enter training or change its pinned checkout."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("controls_run_tests", HERE / "run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_audit_action_bypasses_training_manifest_data_preparation_and_configuration(tmp_path, monkeypatch):
    forbidden = lambda *_args, **_kwargs: pytest.fail("entered the training pipeline")
    for name in ("manifest", "configure", "launch", "install", "restore"):
        monkeypatch.setattr(run, name, forbidden)
    evaluate = Mock(return_value=0)
    monkeypatch.setattr(run, "audit_run", evaluate)
    old = tmp_path / "finished"
    out = tmp_path / "controls"
    assert run.main(["audit", "--workdir", str(out), "--audit-run", str(old)]) == 0
    assert evaluate.call_args.args[0].audit_run == old
    assert evaluate.call_args.args[0].workdir == out
    assert not (out / "manifest.json").exists()
    assert not old.exists()


def test_original_clean_source_is_reused_without_checkout_or_fetch(tmp_path, monkeypatch):
    import bootstrap
    original = tmp_path / "source"
    (original / ".git").mkdir(parents=True)
    state = {"source_url": bootstrap.REPO_URL, "source_commit": "a" * 40}
    monkeypatch.setattr(run, "source_identity", lambda _: state)
    monkeypatch.setattr(bootstrap, "git", Mock(side_effect=[state["source_commit"], state["source_url"], ""]))
    monkeypatch.setattr(bootstrap, "checkout", lambda *_: pytest.fail("changed the original source"))
    args = SimpleNamespace(training_repo=original, workdir=tmp_path / "controls")
    assert run.training_source(args, HERE.parent, state) == original


def test_changed_source_gets_a_separate_pinned_checkout(tmp_path, monkeypatch):
    import bootstrap
    original = tmp_path / "source"
    (original / ".git").mkdir(parents=True)
    state = {"source_url": bootstrap.REPO_URL, "source_commit": "a" * 40}
    monkeypatch.setattr(run, "source_identity", lambda _: state)
    monkeypatch.setattr(bootstrap, "git", lambda *_: "b" * 40)
    checkout = Mock(return_value=(tmp_path / "controls/training-source", state["source_commit"]))
    monkeypatch.setattr(bootstrap, "checkout", checkout)
    args = SimpleNamespace(training_repo=original, workdir=tmp_path / "controls")
    assert run.training_source(args, HERE.parent, state) == args.workdir / "training-source"
    options = checkout.call_args.args[0]
    assert options.commit == state["source_commit"]
    assert options.repo_dir != original
    assert options.repo_dir == args.workdir / "training-source"


def audit_fixture(tmp_path):
    original = tmp_path / "finished"
    stage = original / "outputs/stage2"
    (stage / "checkpoints/final").mkdir(parents=True)
    (original / "manifest.json").write_text(json.dumps({"profile": "kaggle"}))
    (stage / "environment.json").write_text(json.dumps({"precision": "fp16"}))
    (stage / "checkpoints/final/runtime.json").write_text(json.dumps({"complete": True}))
    data = tmp_path / "data"
    (data / "stage2").mkdir(parents=True)
    (data / "stage2/test.jsonl").write_text("{}\n")
    return SimpleNamespace(audit_run=original, workdir=tmp_path / "controls", precision="auto", profile="kaggle",
                           data_dir=data, packet_cache=None, audit_output=None, verbose=False,
                           gpus=0, microbatch=1, training_repo=tmp_path / "original-source")


@pytest.mark.parametrize("gpus", [1, 2])
def test_controls_launch_original_artifacts_with_new_tools_and_no_training(tmp_path, monkeypatch, gpus):
    args = audit_fixture(tmp_path)
    monkeypatch.setattr(run, "hardware", lambda _: (gpus, "fp16", []))
    monkeypatch.setattr(run, "training_source", lambda *_: args.training_repo)
    monkeypatch.setattr(run, "source_identity", lambda _: {"source_commit": "a" * 40})
    launch = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run.subprocess, "run", launch)
    before = {str(path): path.read_bytes() for path in args.audit_run.rglob("*") if path.is_file()}
    assert run.audit_run(args, HERE.parent) == 0
    command = launch.call_args.args[0]
    assert str(HERE / "audit.py") in command
    assert command[command.index("--run-dir") + 1] == str(args.audit_run)
    assert command[command.index("--training-repo") + 1] == str(args.training_repo)
    assert command[command.index("--packet-cache") + 1] == str(args.audit_run / "packets")
    assert command[command.index("--output") + 1] == str(args.workdir / "audit-results")
    assert ("torch.distributed.run" in command) == (gpus == 2)
    assert before == {str(path): path.read_bytes() for path in args.audit_run.rglob("*") if path.is_file()}
    assert not (args.workdir / "manifest.json").exists()


def test_controls_reject_precision_change_and_output_inside_trained_run(tmp_path, monkeypatch):
    args = audit_fixture(tmp_path)
    args.precision = "bf16"
    with pytest.raises(ValueError, match="precision"):
        run.audit_run(args, HERE.parent)
    args.precision = "auto"
    args.workdir = args.audit_run / "new-controls"
    with pytest.raises(ValueError, match="outside"):
        run.audit_run(args, HERE.parent)


def _execute_generated_cell(source, tmp_path, launch):
    # Keep the complete generated cell but replace only its OS/process services.
    tree = ast.parse(source)
    tree.body = [item for item in tree.body if not isinstance(item, (ast.Import, ast.ImportFrom))]
    def mapped_path(value):
        text = str(value)
        return tmp_path / text.lstrip("/") if text.startswith("/kaggle/") else Path(value)
    namespace = {"Path": mapped_path, "os": SimpleNamespace(environ={}),
                 "subprocess": SimpleNamespace(run=launch), "sys": SimpleNamespace(executable="python-test"),
                 "json": json}
    exec(compile(tree, "generated-controls-cell", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("compare", [False, True])
def test_standalone_cells_launch_only_audit_and_report_the_expected_summary(tmp_path, capsys, compare):
    modes = ("actual_packet", "shuffled_packet", "query_only", "raw_text")
    stage = {"examples": 16, "modes": {mode: {"accuracy": 0.5, "accuracy/cloth": 0.5, "loss": 2.0}
                                         for mode in modes},
             "chance_accuracy": {"overall": 0.25, "per_task": {"cloth": 0.25}},
             "reproduction_check": {"status": "match"}}
    if compare:
        summary = {"format": 1, "examples": 16, "stages": {"stage1": stage, "stage2": stage},
                   "comparisons": {}, "provenance": {}}
        name, relative = "Run_LatComm_Stage_Comparison.py", "stage-comparison-results/stage_comparison.json"
    else:
        summary, name, relative = stage, "Run_LatComm_Controls.py", "audit-results/audit_summary.json"
    target = tmp_path / "kaggle/working/latcomm-controls" / relative
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(summary), encoding="utf-8")
    original = tmp_path / "kaggle/working/latcomm-kaggle/checkpoint-marker"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"saved training artifacts")
    launch = Mock()
    namespace = _execute_generated_cell((HERE / name).read_text(encoding="utf-8"), tmp_path, launch)
    command = launch.call_args.args[0]
    assert launch.call_args.kwargs == {"check": True}
    assert command[command.index("--action") + 1] == "audit"
    assert command[command.index("--audit-run") + 1] == "/kaggle/working/latcomm-kaggle"
    assert command[command.index("--training-repo") + 1] == "/kaggle/working/LatComm"
    assert "--skip-setup" in command
    assert ("--compare-stage1" in command) == compare == namespace["COMPARE_STAGE1"]
    assert original.read_bytes() == b"saved training artifacts"
    shown = capsys.readouterr().out
    assert "16 одинаковых вопросах" in shown and "match" in shown and "Правильный пакет" in shown
    assert ("Stage 1 acc" in shown) == compare
    assert ("stage_comparison.json" in shown) == compare


def test_notebook_control_cells_are_independent_and_disabled_by_default(tmp_path, capsys):
    notebook = json.loads((HERE / "LatComm_Kaggle.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(item["source"]) for item in notebook["cells"] if item["cell_type"] == "code"]
    for flag in ("RUN_CONTROLS", "RUN_STAGE_COMPARISON"):
        selected = [source for source in sources if f"{flag} = False\n" in source]
        assert len(selected) == 1
        launch = Mock(side_effect=AssertionError("disabled cell launched a process"))
        namespace = _execute_generated_cell(selected[0], tmp_path, launch)
        assert not namespace[flag]
        launch.assert_not_called()
    assert not list(tmp_path.iterdir())
