"""Exercise real Git clone/resume locally; no network, pip, GPU or LLMs."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("latcomm_bootstrap_test", HERE / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "empty-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "remote"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.email", "latcomm-test@example.invalid")
    git(repo, "config", "user.name", "LatComm tests")
    git(repo, "config", "core.autocrlf", "false")
    for name in bootstrap.REQUIRED:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    (repo / "kaggle/run.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "with Path(os.environ['BOOTSTRAP_LOG']).open('a') as handle:\n"
        "    handle.write(json.dumps({'file': __file__, 'args': sys.argv[1:], "
        "'commit': os.environ['LATCOMM_SOURCE_COMMIT']}) + '\\n')\n"
        "raise SystemExit(int(os.environ.get('BOOTSTRAP_CODE', '0')) if sys.argv[1] != 'setup' else 0)\n",
        encoding="utf-8",
    )
    git(repo, "add", ".")
    git(repo, "commit", "--no-gpg-sign", "-m", "Initial fake fork")
    return repo


def arguments(tmp_path, remote, **extra):
    return SimpleNamespace(repo_dir=tmp_path / "checkout", repo_url=str(remote), ref="main",
                           commit=None, workdir=tmp_path / "work", restore=None, **extra)


def test_clone_installs_and_runs_file_from_fork(remote, tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BOOTSTRAP_LOG", str(log))
    monkeypatch.setenv("BOOTSTRAP_CODE", "75")
    repo = tmp_path / "clone"
    status = bootstrap.main(["--repo-url", str(remote), "--repo-dir", str(repo),
                             "--workdir", str(tmp_path / "work"), "--profile", "smoke", "--gpus", "2"])
    assert status == 75
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["args"][0] for r in records] == ["setup", "all"]
    assert all(Path(r["file"]).resolve() == repo / "kaggle/run.py" for r in records)
    assert all(r["commit"] == git(remote, "rev-parse", "HEAD") for r in records)
    assert "--profile" in records[1]["args"] and "smoke" in records[1]["args"]
    assert not (tmp_path / "work/upstream").exists()


def test_resume_uses_saved_commit_after_main_moves(remote, tmp_path):
    args = arguments(tmp_path, remote)
    repo, first = bootstrap.checkout(args)
    args.workdir.mkdir()
    (args.workdir / "manifest.json").write_text(json.dumps({"source_url": str(remote), "source_commit": first}))
    (remote / "new.txt").write_text("new version")
    git(remote, "add", ".")
    git(remote, "commit", "--no-gpg-sign", "-m", "Advance main")
    assert git(remote, "rev-parse", "HEAD") != first
    assert bootstrap.checkout(args) == (repo, first)
    assert git(repo, "rev-parse", "HEAD") == first


def test_restore_pins_commit_and_rejects_explicit_mismatch(remote, tmp_path):
    args = arguments(tmp_path, remote)
    saved = tmp_path / "saved"
    saved.mkdir()
    sha = git(remote, "rev-parse", "HEAD")
    (saved / "manifest.json").write_text(json.dumps({"source_url": str(remote), "source_commit": sha}))
    args.restore = saved
    assert bootstrap.pinned_commit(args) == sha
    args.commit = "b" * 40
    with pytest.raises(ValueError, match="differs from the saved run"):
        bootstrap.pinned_commit(args)


def test_dirty_clone_is_preserved(remote, tmp_path):
    args = arguments(tmp_path, remote)
    repo, _ = bootstrap.checkout(args)
    path = repo / "kaggle/run.py"
    original = path.read_text()
    path.write_text(original + "# local edit\n")
    with pytest.raises(RuntimeError, match="modified tracked files"):
        bootstrap.checkout(args)
    assert path.read_text() == original + "# local edit\n"


def test_missing_kaggle_files_explains_push_requirement(remote, tmp_path):
    git(remote, "rm", "kaggle/runtime.py")
    git(remote, "commit", "--no-gpg-sign", "-m", "Missing script")
    with pytest.raises(RuntimeError, match="Commit and push the kaggle/"):
        bootstrap.checkout(arguments(tmp_path, remote))


def test_unrelated_directory_is_preserved(remote, tmp_path):
    args = arguments(tmp_path, remote)
    args.repo_dir.mkdir()
    path = args.repo_dir / "user.txt"
    path.write_text("keep")
    with pytest.raises(RuntimeError, match="nonempty directory"):
        bootstrap.checkout(args)
    assert path.read_text() == "keep"


def test_existing_clone_of_another_repository_is_rejected(remote, tmp_path):
    args = arguments(tmp_path, remote)
    repo, _ = bootstrap.checkout(args)
    git(repo, "remote", "set-url", "origin", "https://example.invalid/other.git")
    with pytest.raises(RuntimeError, match="another repository"):
        bootstrap.checkout(args)


def test_skip_setup_only_runs_requested_action(remote, tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BOOTSTRAP_LOG", str(log))
    assert bootstrap.main(["--repo-url", str(remote), "--repo-dir", str(tmp_path / "clone"),
                           "--workdir", str(tmp_path / "work"), "--action", "prepare", "--skip-setup"]) == 0
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["args"][0] for r in records] == ["prepare"]
