"""Clone the user's LatComm fork, install the environment, run kaggle/run.py.

This entry point uses only Python's standard library and git. It can be copied
into a Kaggle cell or used by the accompanying lightweight notebook.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_URL = "https://github.com/SalavatFayzullin/LatComm.git"
REQUIRED = ("kaggle/run.py", "kaggle/runtime.py", "kaggle/data_utils.py", "latcomm/model.py")


def call(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def canonical_url(value: str) -> str:
    return value.rstrip("/").removesuffix(".git")


def pinned_commit(args) -> str | None:
    """A resumed experiment uses its saved SHA even when main has advanced."""
    pins = []
    for directory in (args.workdir, args.restore):
        if directory is None:
            continue
        path = Path(directory) / "manifest.json"
        if args.restore == directory and not path.is_file():
            raise FileNotFoundError(f"Restore directory has no manifest.json: {directory}")
        if not path.is_file():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        if canonical_url(state["source_url"]) != canonical_url(args.repo_url):
            raise ValueError("Saved run belongs to another repository; choose another workdir.")
        pins.append(state["source_commit"])
    if args.commit:
        pins.append(args.commit)
    if len(set(pins)) > 1:
        raise ValueError("Requested commit differs from the saved run; use its original commit or a new workdir.")
    commit = pins[0] if pins else None
    if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("A pinned commit must be a full 40-character Git SHA.")
    return commit


def checkout(args) -> tuple[Path, str]:
    repo = args.repo_dir.resolve()
    pin = pinned_commit(args)
    fresh = False
    if not args.ref or args.ref.startswith("-") or args.repo_url.startswith("-"):
        raise ValueError("Invalid repository URL or ref.")
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"Refusing to overwrite a nonempty directory: {repo}")
        repo.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", "--", args.repo_url, repo])
        fresh = True
    if canonical_url(git(repo, "remote", "get-url", "origin")) != canonical_url(args.repo_url):
        raise RuntimeError(f"{repo} belongs to another repository; choose another --repo-dir.")
    # Do not reset or overwrite local modifications in an existing Kaggle session.
    if not fresh and git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"Checkout has modified tracked files: {repo}. Keep it or choose another --repo-dir.")
    if pin:
        available = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{pin}^{{commit}}"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if available.returncode:
            call(["git", "-C", repo, "fetch", "--no-tags", "origin", pin])
        revision = pin
    else:
        call(["git", "-C", repo, "fetch", "--no-tags", "origin", args.ref])
        revision = git(repo, "rev-parse", "FETCH_HEAD^{commit}")
    call(["git", "-C", repo, "checkout", "--detach", revision])
    commit = git(repo, "rev-parse", "HEAD")
    missing = [name for name in REQUIRED if not (repo / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Fork revision {commit} is missing {', '.join(missing)}. "
            "Commit and push the kaggle/ scripts to your fork, then select that branch or commit."
        )
    print(f"LatComm fork: {args.repo_url}\nSource commit: {commit}", flush=True)
    return repo, commit


def training_arguments(args):
    command = [args.action, "--workdir", str(args.workdir), "--profile", args.profile,
               "--gpus", str(args.gpus), "--microbatch", str(args.microbatch),
               "--precision", args.precision, "--session-steps", str(args.session_steps),
               "--session-hours", str(args.session_hours), "--checkpoint-every", str(args.checkpoint_every)]
    if args.pair:
        command += ["--pair", args.pair]
    for name in ("restore", "data_source", "data_dir", "packet_cache", "data_revision"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if args.benchmark_microbatches:
        command += ["--benchmark-microbatches", str(args.benchmark_microbatches)]
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--ref", default="main", help="Branch/tag used for a new experiment.")
    parser.add_argument("--commit", help="Optional full SHA; saved runs automatically pin their original SHA.")
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/LatComm"))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--profile", choices=("kaggle", "full", "smoke"), default="kaggle")
    parser.add_argument("--pair", help="Omit to use the profile's model pair")
    parser.add_argument("--action", choices=("setup", "prepare", "stage1", "encode", "stage2", "evaluate", "all"), default="all")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")
    parser.add_argument("--session-steps", type=int, default=0)
    parser.add_argument("--session-hours", type=float, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--benchmark-microbatches", type=int, default=0)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--data-source", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--packet-cache", type=Path)
    parser.add_argument("--data-revision")
    parser.add_argument("--skip-setup", action="store_true", help="Reuse an environment already installed in this session.")
    args = parser.parse_args(argv)
    args.workdir = (args.workdir or Path(f"/kaggle/working/latcomm-{args.profile}")).resolve()
    if args.restore:
        args.restore = args.restore.resolve()
    if args.gpus < 0 or args.microbatch < 1 or args.checkpoint_every < 1 or args.session_steps < 0 or args.session_hours < 0 or args.benchmark_microbatches < 0:
        parser.error("GPU count and session limits must be nonnegative; microbatch/checkpoint interval must be positive.")
    repo, commit = checkout(args)
    script = repo / "kaggle/run.py"
    env = dict(os.environ)
    env["LATCOMM_SOURCE_COMMIT"] = commit
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "kaggle"), env.get("PYTHONPATH", "")])
    if not args.skip_setup or args.action == "setup":
        call([sys.executable, script, "setup", "--workdir", args.workdir], cwd=repo, env=env)
    if args.action == "setup":
        return 0
    command = [sys.executable, str(script), *training_arguments(args)]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=repo, env=env)
    if result.returncode == 75:
        print("Session state saved. Resume with the same workdir or --restore in a later session.", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
