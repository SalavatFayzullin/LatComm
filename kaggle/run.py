"""Run LatComm from this checkout of the user's fork (Python 3.10+).

The Kaggle bootstrap clones the fork and calls `python kaggle/run.py setup`,
then `python kaggle/run.py all`. Source, configs and runtime all come from the
same checkout. Training state and data live in separate working directories.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PAIRS = ("qwen4b_to_qwen0.8b", "qwen0.8b_to_qwen0.8b")
HERE = Path(__file__).resolve().parent


def global_batch_size(profile: str) -> int:
    return 32 if profile == "kaggle" else 384


def call(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def repository() -> Path:
    repo = HERE.parent
    if not (repo / "latcomm/model.py").is_file() or not (repo / "configs/stage1").is_dir():
        raise RuntimeError("Run kaggle/run.py inside the cloned LatComm repository.")
    return repo


def source_identity(repo: Path) -> dict[str, str]:
    """Record the fork and actual revision, including newly committed Kaggle files."""
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    url = subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"], text=True,
    ).strip()
    return {"source_url": url, "source_commit": commit}


def install(repo: Path):
    # cu126 retains Pascal (P100, sm60) and Turing (T4, sm75). Newer default
    # CUDA wheels may drop these architectures. Do not replace this with -U torch.
    call([sys.executable, "-m", "pip", "install", "--upgrade",
          "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
          "--quiet", "--disable-pip-version-check",
          "--index-url", "https://download.pytorch.org/whl/cu126"])
    call([sys.executable, "-m", "pip", "install",
          "transformers==5.15.1", "pydantic>=2.10,<3", "safetensors>=0.5",
          "numpy<3", "PyYAML>=6", "huggingface_hub>=0.34", "sentencepiece", "accelerate>=1.0",
          "--quiet", "--disable-pip-version-check"])
    call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-deps", "-e", repo])


def hardware(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in Kaggle Notebook settings.")
    count = torch.cuda.device_count()
    processes = count if args.gpus == 0 else args.gpus
    batch = global_batch_size(args.profile)
    if processes > count or processes < 1 or batch % (processes * args.microbatch):
        raise ValueError(f"GPU count * microbatch must divide the global batch size {batch}.")
    devices = []
    for i in range(processes):
        prop = torch.cuda.get_device_properties(i)
        # Run a CUDA operation too: this detects wheels without the GPU's arch.
        with torch.cuda.device(i):
            x = torch.ones((16, 16), device=f"cuda:{i}", dtype=torch.float16)
            (x @ x).sum().item()
        devices.append({"name": prop.name, "capability": [prop.major, prop.minor],
                        "memory_gib": prop.total_memory / 1024**3})
    bf16 = all(d["capability"][0] >= 8 for d in devices)
    precision = ("bf16" if bf16 else "fp16") if args.precision == "auto" else args.precision
    if precision == "bf16" and not bf16:
        raise RuntimeError("Native BF16 needs an Ampere-or-newer GPU. Use --precision fp16 on T4/P100.")
    print(f"Hardware: {processes} GPU ({', '.join(d['name'] for d in devices)}) | {precision.upper()} | "
          f"global batch {batch} | microbatch {args.microbatch} | "
          f"accumulation {batch // (processes * args.microbatch)}", flush=True)
    if precision == "fp16":
        print("Numerical deviation from the paper: frozen LLMs FP16, interface/AdamW FP32, "
              "dynamic loss scaling. Stored packets remain BF16. Architecture and objective are unchanged.", flush=True)
    else:
        print("Frozen LLMs BF16; interface computes in FP32, as in the published code. "
              "This differs from the all-BF16 wording in the paper.", flush=True)
    return processes, precision, devices


def manifest(args, repo: Path, source: dict[str, str]):
    import yaml
    from huggingface_hub import HfApi

    path = args.workdir / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key, value in (("source_commit", source["source_commit"]),
                           ("pair", args.pair), ("profile", args.profile)):
            if saved[key] != value:
                raise RuntimeError(f"{key} changed; choose another --workdir for another experiment.")
        if args.data_revision and args.data_revision != saved["dataset_revision"]:
            raise RuntimeError("Dataset revision changed; choose a separate workdir.")
        return saved
    api = HfApi()
    config = yaml.safe_load((repo / "configs" / "stage1" / f"{args.pair}.yaml").read_text())
    models = {}
    for side in ("encoder", "receiver"):
        name = config[side]["model_name_or_path"]
        models[side] = {"name": name, "revision": api.model_info(name).sha}
    result = {**source,
              "pair": args.pair, "profile": args.profile,
              "global_batch_size": global_batch_size(args.profile),
              "dataset": "huaweilin/LatComm",
              "dataset_revision": api.dataset_info("huaweilin/LatComm", revision=args.data_revision).sha,
              "models": models}
    atomic_json(path, result)
    return result


def configure(args, repo: Path, state, data, processes: int):
    import yaml

    configs = {}
    batch = global_batch_size(args.profile)
    for stage in ("stage1", "stage2"):
        config = yaml.safe_load((repo / "configs" / stage / f"{args.pair}.yaml").read_text())
        for side in ("encoder", "receiver"):
            config[side]["revision"] = state["models"][side]["revision"]
        config["data"].update(dataset_root=str(args.data_dir / stage), num_workers=0)
        config["training"].update(output_dir=str(args.workdir / "outputs" / stage),
                                  per_device_batch_size=args.microbatch,
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=batch // (processes * args.microbatch),
                                  log_every=1, eval_every=100, save_every=args.checkpoint_every)
        if stage == "stage2":
            config["data"]["packet_cache"] = str(args.packet_cache)
            config["training"]["init_checkpoint"] = str(args.workdir / "outputs/stage1/checkpoints/final")
        if args.profile == "smoke":
            # This is a distinct, explicitly labelled experiment, not paper results.
            # Keep the two training stages, schedule shape and global batch 384.
            config["training"]["warmup_steps"] = 0
        elif args.profile == "kaggle":
            from data_utils import kaggle_count
            examples = data.get("files", {}).get(f"{stage}/train.jsonl", {}).get(
                "records", kaggle_count(stage, "train"))
            updates = (examples + batch - 1) // batch * config["training"]["num_epochs"]
            original_updates = 5209 if stage == "stage1" else 2208
            config["training"]["warmup_steps"] = max(
                1, round(config["training"]["warmup_steps"] * updates / original_updates))
        path = args.workdir / "configs" / f"{stage}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[stage] = path
    return configs


def environment(repo: Path, source_commit: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(HERE), env.get("PYTHONPATH", "")])
    env["LATCOMM_SOURCE_COMMIT"] = source_commit
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def runtime_command(processes: int, command: list):
    if processes > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(processes), str(HERE / "runtime.py"), *command]
    return [sys.executable, str(HERE / "runtime.py"), *command]


def launch(args, repo, configs, processes, precision, stage, source_commit):
    if stage in ("stage1", "stage2"):
        # Even completed stages go through runtime fingerprint validation. Merely
        # finding a directory called final is not evidence of a compatible run.
        (args.workdir / "outputs" / stage / "session-paused.json").unlink(missing_ok=True)
        command = ["train", "--config", str(configs[stage]), "--precision", precision,
                   "--resume", "auto", "--session-steps", str(args.session_steps),
                   "--session-hours", str(args.session_hours), "--checkpoint-every", str(args.checkpoint_every),
                   "--source-commit", source_commit]
        if args.benchmark_microbatches:
            command += ["--benchmark-microbatches", str(args.benchmark_microbatches)]
    elif stage == "encode":
        command = ["encode", "--config", str(configs["stage2"]), "--precision", precision,
                   "--checkpoint", str(args.workdir / "outputs/stage1/checkpoints/final"),
                   "--output", str(args.packet_cache), "--batch-size", "1"]
    else:
        command = ["evaluate", "--config", str(configs["stage2"]), "--precision", precision,
                   "--checkpoint", str(args.workdir / "outputs/stage2/checkpoints/final"),
                   "--data", str(args.data_dir / "stage2/test.jsonl"),
                   "--output", str(args.workdir / "results")]
    if getattr(args, "verbose", False):
        command += ["--verbose"]
    # torchrun turns the worker's planned 75 into its own nonzero return code.
    result = subprocess.run(runtime_command(processes, command), env=environment(repo, source_commit))
    if result.returncode and stage in ("stage1", "stage2"):
        paused = args.workdir / "outputs" / stage / "session-paused.json"
        if paused.exists():
            print(f"Session boundary reached. Save notebook outputs and attach them to the next session. "
                  f"Re-run the same command to continue {stage}.", flush=True)
            return 75
    return result.returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("setup", "prepare", "stage1", "encode", "stage2", "evaluate", "all"))
    p.add_argument("--workdir", type=Path)
    p.add_argument("--data-dir", type=Path, help="Prepared JSONL data directory (stage1/ and stage2/).")
    p.add_argument("--data-source", type=Path, help="Local copy of the HF repository or prepared data; avoids download.")
    p.add_argument("--packet-cache", type=Path, help="Can point at a persisted /kaggle/input packet directory.")
    p.add_argument("--restore", type=Path, help="Previous saved workdir; restore outputs/manifest, do not copy packet shards.")
    p.add_argument("--pair", choices=PAIRS, help="Default: 0.8B -> 0.8B for kaggle, 4B -> 0.8B otherwise")
    p.add_argument("--profile", choices=("kaggle", "full", "smoke"), default="kaggle")
    p.add_argument("--data-revision", help="Optional HF commit; resolved and pinned on first run.")
    p.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")
    p.add_argument("--gpus", type=int, default=0, help="0: all visible GPUs; 1 or 2: explicit count.")
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--session-steps", type=int, default=0, help="New optimizer updates per invocation; 0: no step cap.")
    p.add_argument("--session-hours", type=float, default=0, help="Graceful stop at next completed update; 0: no time cap.")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--benchmark-microbatches", type=int, default=0,
                   help="Time N Stage 1 microbatches without optimizer updates; use a fresh workdir")
    p.add_argument("--verbose", action="store_true", help="Show detailed per-worker training logs")
    args = p.parse_args(argv)
    if args.pair is None:
        args.pair = PAIRS[1] if args.profile == "kaggle" else PAIRS[0]
    if args.microbatch < 1 or args.checkpoint_every < 1 or args.session_steps < 0 or args.session_hours < 0 or args.benchmark_microbatches < 0:
        p.error("microbatch/checkpoint interval must be positive; session limits cannot be negative")
    if args.benchmark_microbatches and args.action not in ("all", "stage1"):
        p.error("--benchmark-microbatches is supported with all or stage1 only")
    args.workdir = (args.workdir or Path(f"/kaggle/working/latcomm-{args.profile}")).resolve()
    args.data_dir = (args.data_dir or args.workdir / "data").resolve()
    args.packet_cache = (args.packet_cache or args.workdir / "packets").resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.environ["LATCOMM_INDEX_DIR"] = str(args.workdir / "indexes")
    if args.restore:
        restore(args.restore.resolve(), args.workdir)
    repo = repository()
    if args.action == "setup":
        install(repo)
        return 0
    sys.path[:0] = [str(repo), str(HERE)]
    source = source_identity(repo)
    state = manifest(args, repo, source)
    from data_utils import prepare_data
    print("Preparing/verifying dataset files...", flush=True)
    data = prepare_data(args.data_dir, revision=state["dataset_revision"], profile=args.profile,
                        local_source=args.data_source)
    if args.action == "prepare":
        print(json.dumps(data, indent=2, default=str))
        return 0
    processes, precision, devices = hardware(args)
    configs = configure(args, repo, state, data, processes)
    atomic_json(args.workdir / "hardware.json", {"devices": devices, "precision": precision,
                                                "torch": __import__("torch").__version__})
    stages = (("stage1",) if args.benchmark_microbatches else
              (("stage1", "encode", "stage2", "evaluate") if args.action == "all" else (args.action,)))
    names = {"stage1": "Stage 1: context reconstruction", "encode": "Create context packets",
             "stage2": "Stage 2: question answering", "evaluate": "Final test"}
    for position, stage in enumerate(stages, 1):
        print(f"\n[{position}/{len(stages)}] {names[stage]}", flush=True)
        status = launch(args, repo, configs, processes, precision, stage, source["source_commit"])
        if status:
            return status
    return 0


def restore(source: Path, destination: Path):
    """Copy only relatively small training state, leaving large data/cache in Input."""
    if source == destination or not (source / "manifest.json").is_file():
        raise ValueError("--restore must name a different saved latcomm workdir containing manifest.json")
    for name in ("manifest.json", "outputs"):
        src, dst = source / name, destination / name
        if not src.exists() or dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
