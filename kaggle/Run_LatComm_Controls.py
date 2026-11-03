# Paste this complete file into a NEW cell in the existing Kaggle notebook.
# The trained run, models and prepared test data must still be available.
from pathlib import Path
import os
import subprocess
import sys
import json

RUN_CONTROLS = True
COMPARE_STAGE1 = False
TRAINED_RUN = "/kaggle/working/latcomm-kaggle"
TRAINING_REPO = "/kaggle/working/LatComm"
PREPARED_DATA = "/kaggle/temp/latcomm-data-kaggle"
CONTROLS_WORKDIR = "/kaggle/working/latcomm-controls"
CONTROLS_REPO = "/kaggle/working/LatComm-controls"
CONTROLS_REF = "main"
CONTROLS_GPUS = 0                    # all visible GPUs
CONTROLS_SKIP_SETUP = True           # current session already has the training environment

if RUN_CONTROLS:
    os.environ.setdefault("HF_HOME", "/kaggle/temp/latcomm-hf")
    controls_bootstrap = Path("/kaggle/working/latcomm_controls_bootstrap.py")
    controls_bootstrap.write_text('"""Clone the user\'s LatComm fork, install the environment, run kaggle/run.py.\n\nThis entry point uses only Python\'s standard library and git. It can be copied\ninto a Kaggle cell or used by the accompanying lightweight notebook.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport subprocess\nimport sys\n\nREPO_URL = "https://github.com/SalavatFayzullin/LatComm.git"\nREQUIRED = ("kaggle/run.py", "kaggle/runtime.py", "kaggle/data_utils.py", "kaggle/progress.py", "latcomm/model.py")\n\n\ndef call(command, **kwargs):\n    print("+", " ".join(map(str, command)), flush=True)\n    return subprocess.run(list(map(str, command)), check=True, **kwargs)\n\n\ndef git(repo: Path, *arguments: str) -> str:\n    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()\n\n\ndef canonical_url(value: str) -> str:\n    return value.rstrip("/").removesuffix(".git")\n\n\ndef pinned_commit(args) -> str | None:\n    """A resumed experiment uses its saved SHA even when main has advanced."""\n    pins = []\n    for directory in (args.workdir, args.restore):\n        if directory is None:\n            continue\n        path = Path(directory) / "manifest.json"\n        if args.restore == directory and not path.is_file():\n            raise FileNotFoundError(f"Restore directory has no manifest.json: {directory}")\n        if not path.is_file():\n            continue\n        state = json.loads(path.read_text(encoding="utf-8"))\n        if canonical_url(state["source_url"]) != canonical_url(args.repo_url):\n            raise ValueError("Saved run belongs to another repository; choose another workdir.")\n        pins.append(state["source_commit"])\n    if args.commit:\n        pins.append(args.commit)\n    if len(set(pins)) > 1:\n        raise ValueError("Requested commit differs from the saved run; use its original commit or a new workdir.")\n    commit = pins[0] if pins else None\n    if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", commit):\n        raise ValueError("A pinned commit must be a full 40-character Git SHA.")\n    return commit\n\n\ndef checkout(args) -> tuple[Path, str]:\n    repo = args.repo_dir.resolve()\n    pin = pinned_commit(args)\n    fresh = False\n    if not args.ref or args.ref.startswith("-") or args.repo_url.startswith("-"):\n        raise ValueError("Invalid repository URL or ref.")\n    if not (repo / ".git").exists():\n        if repo.exists() and any(repo.iterdir()):\n            raise RuntimeError(f"Refusing to overwrite a nonempty directory: {repo}")\n        repo.parent.mkdir(parents=True, exist_ok=True)\n        call(["git", "clone", "--no-checkout", "--", args.repo_url, repo])\n        fresh = True\n    if canonical_url(git(repo, "remote", "get-url", "origin")) != canonical_url(args.repo_url):\n        raise RuntimeError(f"{repo} belongs to another repository; choose another --repo-dir.")\n    # Do not reset or overwrite local modifications in an existing Kaggle session.\n    if not fresh and git(repo, "status", "--porcelain", "--untracked-files=no"):\n        raise RuntimeError(f"Checkout has modified tracked files: {repo}. Keep it or choose another --repo-dir.")\n    if pin:\n        available = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{pin}^{{commit}}"],\n                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n        if available.returncode:\n            call(["git", "-C", repo, "fetch", "--no-tags", "origin", pin])\n        revision = pin\n    else:\n        call(["git", "-C", repo, "fetch", "--no-tags", "origin", args.ref])\n        revision = git(repo, "rev-parse", "FETCH_HEAD^{commit}")\n    call(["git", "-C", repo, "checkout", "--detach", revision])\n    commit = git(repo, "rev-parse", "HEAD")\n    missing = [name for name in REQUIRED if not (repo / name).is_file()]\n    if missing:\n        raise RuntimeError(\n            f"Fork revision {commit} is missing {\', \'.join(missing)}. "\n            "Commit and push the kaggle/ scripts to your fork, then select that branch or commit."\n        )\n    print(f"LatComm fork: {args.repo_url}\\nSource commit: {commit}", flush=True)\n    return repo, commit\n\n\ndef training_arguments(args):\n    command = [args.action, "--workdir", str(args.workdir), "--profile", args.profile,\n               "--gpus", str(args.gpus), "--microbatch", str(args.microbatch),\n               "--precision", args.precision, "--session-steps", str(args.session_steps),\n               "--session-hours", str(args.session_hours), "--checkpoint-every", str(args.checkpoint_every)]\n    if args.pair:\n        command += ["--pair", args.pair]\n    for name in ("restore", "data_source", "data_dir", "packet_cache", "data_revision"):\n        value = getattr(args, name)\n        if value is not None:\n            command += ["--" + name.replace("_", "-"), str(value)]\n    for name in ("audit_run", "training_repo", "audit_output"):\n        value = getattr(args, name, None)\n        if value is not None:\n            command += ["--" + name.replace("_", "-"), str(value)]\n    if getattr(args, "compare_stage1", False):\n        command += ["--compare-stage1"]\n    if args.benchmark_microbatches:\n        command += ["--benchmark-microbatches", str(args.benchmark_microbatches)]\n    if getattr(args, "verbose", False):\n        command += ["--verbose"]\n    return command\n\n\ndef main(argv=None) -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--repo-url", default=REPO_URL)\n    parser.add_argument("--ref", default="main", help="Branch/tag used for a new experiment.")\n    parser.add_argument("--commit", help="Optional full SHA; saved runs automatically pin their original SHA.")\n    parser.add_argument("--repo-dir", type=Path)\n    parser.add_argument("--workdir", type=Path)\n    parser.add_argument("--profile", choices=("kaggle", "overnight", "full", "smoke"), default="kaggle")\n    parser.add_argument("--pair", help="Omit to use the profile\'s model pair")\n    parser.add_argument("--action", choices=("setup", "prepare", "stage1", "encode", "stage2", "evaluate", "all", "audit"), default="all")\n    parser.add_argument("--gpus", type=int, default=0)\n    parser.add_argument("--microbatch", type=int, default=1)\n    parser.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")\n    parser.add_argument("--session-steps", type=int, default=0)\n    parser.add_argument("--session-hours", type=float, default=0)\n    parser.add_argument("--checkpoint-every", type=int, default=5)\n    parser.add_argument("--benchmark-microbatches", type=int, default=0)\n    parser.add_argument("--restore", type=Path)\n    parser.add_argument("--data-source", type=Path)\n    parser.add_argument("--data-dir", type=Path)\n    parser.add_argument("--packet-cache", type=Path)\n    parser.add_argument("--data-revision")\n    parser.add_argument("--skip-setup", action="store_true", help="Reuse an environment already installed in this session.")\n    parser.add_argument("--verbose", action="store_true", help="Show detailed per-worker training logs")\n    parser.add_argument("--audit-run", type=Path, help="Completed training workdir to evaluate without training")\n    parser.add_argument("--training-repo", type=Path, help="Optional unchanged source checkout of the completed run")\n    parser.add_argument("--audit-output", type=Path, help="New directory for control evaluation results")\n    parser.add_argument("--compare-stage1", action="store_true",\n                        help="Compare saved Stage 1 and Stage 2 bridges on identical MCQ controls")\n    args = parser.parse_args(argv)\n    default_workdir = "/kaggle/working/latcomm-controls" if args.action == "audit" else f"/kaggle/working/latcomm-{args.profile}"\n    args.workdir = (args.workdir or Path(default_workdir)).resolve()\n    default_repo = "/kaggle/working/LatComm-controls" if args.action == "audit" else "/kaggle/working/LatComm"\n    args.repo_dir = (args.repo_dir or Path(default_repo)).resolve()\n    if args.restore:\n        args.restore = args.restore.resolve()\n    if args.gpus < 0 or args.microbatch < 1 or args.checkpoint_every < 1 or args.session_steps < 0 or args.session_hours < 0 or args.benchmark_microbatches < 0:\n        parser.error("GPU count and session limits must be nonnegative; microbatch/checkpoint interval must be positive.")\n    if args.action == "audit" and (args.audit_run is None or args.restore is not None):\n        parser.error("audit requires --audit-run; use the completed run directly instead of --restore")\n    if args.compare_stage1 and args.action != "audit":\n        parser.error("--compare-stage1 requires --action audit")\n    if args.action == "audit":\n        if args.workdir == args.audit_run.resolve() or args.audit_run.resolve() in args.workdir.parents:\n            parser.error("Choose a separate --workdir for the control results")\n        if args.training_repo and args.repo_dir == args.training_repo.resolve():\n            parser.error("Choose a separate --repo-dir for the audit tools; preserve the training checkout")\n    repo, commit = checkout(args)\n    if args.action == "audit" and not (repo / "kaggle/audit.py").is_file():\n        raise RuntimeError(f"Fork revision {commit} has no control evaluator; commit and push kaggle/audit.py first")\n    script = repo / "kaggle/run.py"\n    env = dict(os.environ)\n    env["LATCOMM_SOURCE_COMMIT"] = commit\n    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "kaggle"), env.get("PYTHONPATH", "")])\n    if not args.skip_setup or args.action == "setup":\n        call([sys.executable, script, "setup", "--workdir", args.workdir], cwd=repo, env=env)\n    if args.action == "setup":\n        return 0\n    command = [sys.executable, str(script), *training_arguments(args)]\n    print("+", " ".join(command), flush=True)\n    result = subprocess.run(command, cwd=repo, env=env)\n    if result.returncode == 75:\n        print("Session state saved. Resume with the same workdir or --restore in a later session.", flush=True)\n    return result.returncode\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n', encoding="utf-8")
    command = [sys.executable, str(controls_bootstrap),
               "--repo-url", "https://github.com/SalavatFayzullin/LatComm.git",
               "--ref", CONTROLS_REF, "--repo-dir", CONTROLS_REPO,
               "--action", "audit", "--workdir", CONTROLS_WORKDIR,
               "--audit-run", TRAINED_RUN, "--training-repo", TRAINING_REPO,
               "--data-dir", PREPARED_DATA, "--packet-cache", str(Path(TRAINED_RUN) / "packets"),
               "--gpus", str(CONTROLS_GPUS)]
    if CONTROLS_SKIP_SETUP:
        command += ["--skip-setup"]
    if COMPARE_STAGE1:
        command += ["--compare-stage1"]
    subprocess.run(command, check=True)
    summary_file = Path(CONTROLS_WORKDIR) / (
        "stage-comparison-results/stage_comparison.json" if COMPARE_STAGE1
        else "audit-results/audit_summary.json")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    names = {"actual_packet": "Правильный пакет", "shuffled_packet": "Чужой пакет",
             "query_only": "Без контекста", "raw_text": "Полный текст"}
    print(f"Сравнение на {summary['examples']} одинаковых вопросах")
    if COMPARE_STAGE1:
        stages = summary["stages"]
        print(f"{'Режим':20} {'Stage 1 acc':>12} {'Stage 2 acc':>12} {'Изменение':>11} {'Loss 1':>10} {'Loss 2':>10}")
        for mode in names:
            first = stages["stage1"]["modes"][mode]
            final = stages["stage2"]["modes"][mode]
            delta = 100 * (final["accuracy"] - first["accuracy"])
            print(f"{names[mode]:20} {first['accuracy']:>11.2%} {final['accuracy']:>11.2%} {delta:>+8.2f} п.п. {first['loss']:>10.4f} {final['loss']:>10.4f}")
        print("Воспроизведение прежнего Stage 2:", stages["stage2"].get("reproduction_check", {}).get("status", "не проверено"))
        print("\nПреимущество правильного пакета над чужим, п.п.:")
        print(f"{'Задача':15} {'Stage 1':>12} {'Stage 2':>12}")
        for task in sorted(stages["stage2"]["chance_accuracy"]["per_task"]):
            gains = [100 * (stages[stage]["modes"]["actual_packet"]["accuracy/" + task]
                            - stages[stage]["modes"]["shuffled_packet"]["accuracy/" + task])
                     for stage in ("stage1", "stage2")]
            print(f"{task:15} {gains[0]:>+12.2f} {gains[1]:>+12.2f}")
        print("\nИтог сравнения:", summary_file)
        print("Подробные ответы и примеры CLOTH:", summary_file.parent / "stage1", "и", summary_file.parent / "stage2")
    else:
        print(f"{'Режим':20} {'Верно':>8} {'Accuracy':>10} {'Loss':>10}")
        for mode, metrics in summary["modes"].items():
            correct = round(metrics["accuracy"] * summary["examples"])
            print(f"{names[mode]:20} {correct:>8} {metrics['accuracy']:>9.2%} {metrics['loss']:>10.4f}")
        print("Ожидаемая accuracy равномерного случайного выбора:", f"{summary['chance_accuracy']['overall']:.2%}")
        print("Воспроизведение прежнего результата:", summary.get("reproduction_check", {}).get("status", "не проверено"))
        print("\nТочность по задачам:")
        modes = list(summary["modes"])
        print(f"{'Задача':15}" + "".join(f"{names[mode]:>19}" for mode in modes))
        for task in sorted(summary["chance_accuracy"]["per_task"]):
            print(f"{task:15}" + "".join(f"{summary['modes'][mode]['accuracy/' + task]:>18.1%}" for mode in modes))
        print("\nПодробные результаты:", summary_file.parent)
        print("Примеры CLOTH:", summary_file.parent / "cloth_examples.jsonl")
else:
    print("Для оценки готового интерфейса задайте RUN_CONTROLS = True.")
