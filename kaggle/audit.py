"""Evaluate an existing LatComm run without training or changing its artifacts.

Run this script from the audit checkout, but import the model and scoring code
from --training-repo, at the exact commit and library versions of the old run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

MODES = ("actual_packet", "shuffled_packet", "query_only", "raw_text")
RAW_CONTEXT_PREFIX = "Context:\n{context}\n\n{prompt}"
LOGGER = logging.getLogger("latcomm.audit")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def check_output(output, protected=()):
    output = Path(output).resolve()
    for root in protected:
        if output.is_relative_to(Path(root).resolve()):
            raise ValueError(f"Audit output must be separate from original artifacts: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Audit output already exists and is nonempty: {output}")
    return output


def import_training_code(repo, expected_commit):
    """Import only the pinned training checkout; never write its __pycache__."""
    repo = Path(repo).resolve()
    if git_head(repo) != expected_commit:
        raise ValueError("Training checkout HEAD differs from the run's source commit")
    changed = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if changed:
        raise ValueError("Training checkout has modified tracked files")
    sys.dont_write_bytecode = True
    for name in ("latcomm", "runtime", "data_utils", "progress"):
        existing = sys.modules.get(name)
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(repo):
            raise ValueError(f"Already imported {name} from outside the training checkout")
    os.environ["LATCOMM_REPO"] = str(repo)
    sys.path[:0] = [str(repo), str(repo / "kaggle")]
    runtime = importlib.import_module("runtime")
    data_utils = importlib.import_module("data_utils")
    scoring = importlib.import_module("latcomm.evaluate")
    for module in (runtime, data_utils, scoring, importlib.import_module("latcomm")):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise ValueError(f"Imported code outside the original training checkout: {module.__file__}")
    return runtime, data_utils, scoring


def load_examples(path, parse_example, data):
    examples, sources = [], {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            source = json.loads(line)
            example = parse_example(source, data, f"{path}:{line_number}")
            if example.sample_id in sources:
                raise ValueError(f"Duplicate sample_id: {example.sample_id}")
            examples.append(example)
            sources[example.sample_id] = source
    if not examples:
        raise ValueError("Audit test set is empty")
    return examples, sources


def shuffled_contexts(examples, context_key, seed):
    """One seeded cycle over unique contexts within each task, with no fixed points."""
    groups = {}
    for example in sorted(examples, key=lambda item: (item.dataset, item.sample_id)):
        groups.setdefault(example.dataset, {}).setdefault(context_key(example.context), example)
    donors = {}
    for task, unique in sorted(groups.items()):
        keys = sorted(unique)
        if len(keys) < 2:
            raise ValueError(f"Shuffled-packet control needs two distinct contexts in task {task!r}")
        task_seed = int(sha256(f"{seed}:{task}".encode()).hexdigest(), 16)
        random.Random(task_seed).shuffle(keys)
        for own, other in zip(keys, keys[1:] + keys[:1]):
            if own == other:
                raise AssertionError("Shuffled context must differ from its own context")
            donors[(task, own)] = unique[other]
    return donors


def _model_window(model):
    config = model.receiver.model.config
    config = getattr(config, "text_config", None) or config
    window = getattr(config, "max_position_embeddings", None)
    if not isinstance(window, int) or window < 1:
        raise ValueError("Receiver config must declare a positive max_position_embeddings")
    return window


def prepare_inputs(examples, tokenizer, data, window, modes):
    """Validate all inputs before inference, including the full raw-text prompt."""
    option_tokens = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:max(e.num_choices for e in examples)]:
        ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Original MCQ scorer requires a single token for option {letter!r}")
        option_tokens[letter] = ids[0]
    if len(set(option_tokens.values())) != len(option_tokens):
        raise ValueError("Option letters must have distinct token IDs")
    descriptions, raw_examples = {}, {}
    raw_limit = data.max_prompt_length
    for example in examples:
        if example.num_choices < 2 or example.response not in option_tokens:
            raise ValueError(f"Invalid MCQ example: {example.sample_id}")
        response = tokenizer(example.response, add_special_tokens=False)["input_ids"]
        prompt = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
        if len(response) != 1 or not prompt:
            raise ValueError(f"Invalid MCQ tokenization: {example.sample_id}")
        if len(prompt) > data.max_prompt_length:
            raise ValueError(f"{example.sample_id}: original prompt exceeds its saved length limit")
        if len(prompt) + len(response) > window:
            raise ValueError(f"{example.sample_id}: original input exceeds receiver window {window}")
        item = {"prompt_tokens": len(prompt), "response_tokens": len(response),
                "prediction_position": len(prompt) - 1, "context_characters": len(example.context),
                "prompt_sha256": sha256(example.prompt.encode()).hexdigest()}
        if "raw_text" in modes:
            text = RAW_CONTEXT_PREFIX.format(context=example.context, prompt=example.prompt)
            raw = tokenizer(text, add_special_tokens=True)["input_ids"]
            if len(raw) + len(response) > window:
                raise ValueError(
                    f"{example.sample_id}: full raw-text input needs {len(raw) + len(response)} tokens, "
                    f"receiver window is {window}; no truncation or sample skipping is allowed"
                )
            raw_limit = max(raw_limit, len(raw))
            raw_examples[example.sample_id] = replace(example, prompt=text)
            item.update(raw_prompt_tokens=len(raw), raw_prediction_position=len(raw) - 1,
                        raw_prompt_sha256=sha256(text.encode()).hexdigest())
        descriptions[example.sample_id] = item
    return descriptions, raw_examples, raw_limit


def _hook_counts(model):
    return tuple(len(module._forward_hooks) for module in model.receiver.model.modules())


def evaluate_controls(runtime, model, examples, packets, config, distributed, *,
                      modes=MODES, seed=42, tokenizer=None, scoring=None):
    """Evaluate each control on identical rows, using the original MCQ scorer.

    Dependency injection keeps CPU tests independent of HF downloads and CUDA.
    No encoder forward pass, training step, or mutable data index is used.
    """
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError(f"Modes must be a nonempty distinct selection from {MODES}")
    if config.data.response_format != "mcq":
        raise ValueError("Audit controls require MCQ data")
    ids = [example.sample_id for example in examples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Audit requires nonempty examples with unique sample IDs")
    scoring = scoring or importlib.import_module("latcomm.evaluate")
    tokenizer = tokenizer or runtime.load_tokenizer(config.receiver)
    donors = shuffled_contexts(examples, runtime.context_key, seed) if "shuffled_packet" in modes else {}
    descriptions, raw_examples, raw_limit = prepare_inputs(
        examples, tokenizer, config.data, _model_window(model), modes,
    )
    collator = runtime.Collator(tokenizer, tokenizer, config.data, with_context=False)
    raw_data = config.data.model_copy(update={"max_prompt_length": raw_limit})
    raw_collator = runtime.Collator(tokenizer, tokenizer, raw_data, with_context=False)
    model.requires_grad_(False)
    model.receiver.model.requires_grad_(False)
    model.eval()
    model.receiver.model.eval()
    parameters = list(model.bridge.parameters()) + list(model.receiver.model.parameters())
    if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
        raise ValueError("Audit requires frozen parameters without accumulated gradients")
    hooks = _hook_counts(model)
    local = {mode: [] for mode in modes}
    rows = examples[distributed.rank::distributed.world_size]
    progress = runtime.CounterProgress("Audit examples", len(rows), enabled=distributed.is_main)
    with runtime.torch.inference_mode():
        for index, example in enumerate(rows, 1):
            own_key = runtime.context_key(example.context)
            donor = donors.get((example.dataset, own_key))
            for mode in modes:
                selected = raw_examples[example.sample_id] if mode == "raw_text" else example
                batch = (raw_collator if mode == "raw_text" else collator)([selected])
                expected = descriptions[example.sample_id][
                    "raw_prediction_position" if mode == "raw_text" else "prediction_position"]
                if int(batch["prompt"]["prediction_positions"][0, 0]) != expected:
                    raise AssertionError("MCQ prediction position changed during collation")
                context = nullcontext()
                if mode in ("actual_packet", "shuffled_packet"):
                    packet_example = example if mode == "actual_packet" else donor
                    packet = packets([packet_example])
                    if not bool(runtime.torch.isfinite(packet).all()):
                        raise ValueError("Cached packet contains non-finite values")
                    context = model.reading(packet, batch["prompt"])
                with context:
                    logits = runtime.response_logits(model.receiver, batch["prompt"])
                if _hook_counts(model) != hooks:
                    raise AssertionError("An injection hook remained active after its control")
                if not bool(runtime.torch.isfinite(logits).all()):
                    raise FloatingPointError(f"Non-finite audit logits: {mode}, {example.sample_id}")
                score = scoring.score_responses(logits, batch)
                row = scoring.records(batch, score)[0]
                option_ids = batch["option_ids"].to(logits.device)
                option_logits = logits[0, 0].index_select(-1, option_ids).float()
                log_probabilities = runtime.torch.log_softmax(logits[0, 0].float(), dim=-1)
                conditional = runtime.torch.softmax(option_logits, dim=-1)
                row.update(mode=mode, own_context_key=own_key,
                           shuffled_context_key=runtime.context_key(donor.context) if donor else None,
                           packet_context_key=own_key if mode == "actual_packet" else
                           runtime.context_key(donor.context) if mode == "shuffled_packet" else None,
                           input_tokens=int(batch["prompt"]["attention_mask"].sum()),
                           prediction_position=expected,
                           option_scores={letter: {"logit": float(option_logits[i]),
                               "vocabulary_log_probability": float(log_probabilities[token]),
                               "conditional_probability": float(conditional[i])}
                               for i, (letter, token) in enumerate(zip(
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ", option_ids.tolist()))})
                local[mode].append(row)
            progress.update(index)
    progress.update(len(rows), force=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
        raise AssertionError("Audit unexpectedly enabled training or accumulated gradients")
    shards = distributed.all_gather(local)
    results = {}
    for mode in modes:
        merged = sorted((row for shard in shards for row in shard[mode]), key=lambda row: row["sample_id"])
        if [row["sample_id"] for row in merged] != sorted(ids):
            raise ValueError(f"Control {mode} did not score the identical sample set exactly once")
        results[mode] = merged
    return results, descriptions


def validate_run(runtime, args, manifest):
    """Bind the old checkpoint, exact code/libraries, prepared test and cache."""
    run = Path(args.run_dir)
    checkpoint = run / "outputs/stage2/checkpoints/final"
    info = read_json(checkpoint / "checkpoint.json")
    status = read_json(checkpoint / "runtime.json")
    environment = read_json(run / "outputs/stage2/environment.json")
    if not status.get("complete") or status["progress"]["step"] != info["step"]:
        raise ValueError("Stage 2 final checkpoint is not complete or its step differs")
    config = runtime.LatCommConfig.model_validate(info["config"])
    if config.training.trainable != "readers_and_gates" or config.data.response_format != "mcq":
        raise ValueError("Expected a completed Stage 2 MCQ bridge")
    for side in ("encoder", "receiver"):
        original = manifest["models"][side]
        model = getattr(config, side)
        if not model.revision or model.revision != original["revision"] or model.model_name_or_path != original["name"]:
            raise ValueError(f"Checkpoint {side} differs from the pinned run model")
    code = runtime.code_identity(manifest["source_commit"])
    if code != environment["source"]:
        raise ValueError("Original training code or library versions differ from the saved run")
    precision = environment["precision"]
    if precision not in ("fp16", "bf16") or status.get("precision") != precision:
        raise ValueError("Checkpoint backbone precision differs from the saved environment")
    if environment.get("bridge_dtype") != "float32" or environment.get("packet_storage") != "bfloat16":
        raise ValueError("Unexpected original bridge or packet dtype")
    if status.get("world_size") != environment.get("world_size"):
        raise ValueError("Checkpoint training world size differs from its environment")
    # Read the original full training fingerprint, without restoring optimizer/RNG.
    state = runtime.torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    fingerprint = state["fingerprint"]
    if (state.get("format") != runtime.FORMAT_VERSION or not state.get("complete")
            or state["progress"] != status["progress"]
            or runtime.canonical_hash(fingerprint) != status["fingerprint_sha256"]
            or fingerprint["code"] != code or fingerprint["precision"] != precision):
        raise ValueError("Checkpoint training state/provenance differs from the completed run")
    expected_settings = config.model_dump(mode="json")
    expected_settings["training"].pop("output_dir")
    expected_settings["training"].pop("init_checkpoint")
    expected_settings["data"].pop("dataset_root")
    expected_settings["data"].pop("packet_cache")
    if expected_settings != fingerprint["settings"]:
        raise ValueError("Checkpoint config differs from its original training fingerprint")
    del state
    test = Path(args.data_dir) / "stage2/test.jsonl"
    receipt = read_json(Path(args.data_dir) / "preparation.json")
    test_hash = file_hash(test)
    if (receipt["revision"] != manifest["dataset_revision"] or receipt["profile"] != manifest["profile"]
            or receipt["files"]["stage2/test.jsonl"]["sha256"] != test_hash):
        raise ValueError("Prepared test data differs from the original run")
    identity = runtime.packet_fingerprint(args.packet_cache)
    meta, provenance = identity["meta"], identity["meta"]["provenance"]
    shape = [len(info["layer_pairs"]), config.bridge.num_slots, config.bridge.context_dim]
    if (provenance["code"] != code or provenance["encoder"] != config.encoder.model_dump(mode="json")
            or provenance["precision"] != precision or meta["dtype"] != "bfloat16"
            or meta["packet_shape"] != shape or provenance["packet_shape"] != shape
            or provenance["layer_pairs"] != info["layer_pairs"]
            or provenance["sources"].get("test.jsonl") != test_hash
            or provenance["max_context_length"] != config.data.max_context_length):
        raise ValueError("Original packet cache provenance differs from checkpoint or test data")
    if fingerprint["packets"] != identity:
        raise ValueError("Packet cache differs from the cache used for Stage 2 training")
    compressor = runtime.checkpoint_compressor_hash(checkpoint)
    if provenance["compressor"] != compressor:
        raise ValueError("Final bridge and cached compressors differ")
    bridge_hash = file_hash(checkpoint / "bridge.safetensors")
    previous = run / "results/provenance.json"
    if previous.exists():
        saved = read_json(previous)
        if (saved["data_sha256"] != test_hash or saved["bridge_sha256"] != bridge_hash
                or saved["precision"] != precision or saved["source"] != code):
            raise ValueError("Original final-test provenance differs from these audit inputs")
    return config, checkpoint, test, precision, identity, {
        "training_source_commit": manifest["source_commit"], "training_code": code,
        "checkpoint_sha256": bridge_hash, "test_sha256": test_hash,
        "cache_identity_sha256": runtime.canonical_hash(identity),
        "precision": precision, "packet_storage": meta["dtype"],
        "checkpoint_step": info["step"], "checkpoint_selection": "original_final",
        "training_world_size": environment["world_size"],
        "original_final_test_metrics": read_json(run / "results/metrics.json")
            if (run / "results/metrics.json").exists() else None,
    }


def paired_comparison(reference, control):
    if [row["sample_id"] for row in reference] != [row["sample_id"] for row in control]:
        raise ValueError("Paired comparisons require identical sample IDs and order")
    count = len(reference)
    return {"examples": count,
            "accuracy_delta": sum(b["correct"] - a["correct"] for a, b in zip(reference, control)) / count,
            "loss_delta": sum(b["nll"] - a["nll"] for a, b in zip(reference, control)) / count,
            "prediction_flips": sum(a["prediction"] != b["prediction"] for a, b in zip(reference, control)),
            "correct_to_wrong": sum(bool(a["correct"]) and not b["correct"] for a, b in zip(reference, control)),
            "wrong_to_correct": sum(not a["correct"] and bool(b["correct"]) for a, b in zip(reference, control))}


def reproduction_check(results, run, scoring):
    """Report reproduction differences explicitly, without hiding CUDA variation."""
    if "actual_packet" not in results:
        return {"status": "unavailable", "reason": "actual_packet mode was not evaluated"}
    rows = results["actual_packet"]
    actual_metrics = scoring.summarize(rows)
    run = Path(run)
    metrics_path, predictions_path = run / "results/metrics.json", run / "results/predictions.jsonl"
    report = {"status": "unavailable", "prediction_difference_count": None,
              "prediction_difference_ids": [], "metrics": {"status": "unavailable"},
              "predictions": {"status": "unavailable"}}
    if metrics_path.is_file():
        reference_metrics = read_json(metrics_path)
        differences = {}
        for key in ("accuracy", "loss"):
            if key in reference_metrics:
                reference, actual = reference_metrics[key], actual_metrics[key]
                absolute_tolerance = 1e-12 if key == "accuracy" else 1e-5
                relative_tolerance = 0.0 if key == "accuracy" else 1e-4
                differences[key] = {"reference": reference, "audit": actual, "delta": actual - reference,
                    "matches": math.isclose(actual, reference, rel_tol=relative_tolerance,
                                             abs_tol=absolute_tolerance),
                    "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance}
        report["metrics"] = {"status": "match" if len(differences) == 2 and all(
            item["matches"] for item in differences.values()) else "different" if any(
            not item["matches"] for item in differences.values()) else "unavailable",
            "values": differences, "reference_sha256": file_hash(metrics_path)}
    if predictions_path.is_file():
        with predictions_path.open("r", encoding="utf-8") as handle:
            reference_rows = [json.loads(line) for line in handle]
        reference, duplicates = {}, set()
        for row in reference_rows:
            if row["sample_id"] in reference:
                duplicates.add(row["sample_id"])
            reference[row["sample_id"]] = row
        actual = {row["sample_id"]: row for row in rows}
        common = set(reference) & set(actual)
        gold = sorted(key for key in common if reference[key].get("gold") != actual[key]["gold"])
        predictions = sorted(key for key in common if reference[key].get("prediction") != actual[key]["prediction"])
        missing = sorted(set(reference) - set(actual))
        extra = sorted(set(actual) - set(reference))
        changed = sorted(set(gold) | set(predictions) | set(missing) | set(extra) | duplicates)
        report["prediction_difference_count"] = len(changed)
        report["prediction_difference_ids"] = changed
        report["predictions"] = {"status": "different" if changed else "match",
            "reference_examples": len(reference_rows), "audit_examples": len(rows),
            "gold_mismatch_ids": gold, "prediction_mismatch_ids": predictions,
            "missing_from_audit": missing, "missing_from_reference": extra,
            "duplicate_reference_ids": sorted(duplicates), "reference_sha256": file_hash(predictions_path)}
    statuses = (report["metrics"]["status"], report["predictions"]["status"])
    report["status"] = "different" if "different" in statuses else "match" if all(
        status == "match" for status in statuses) else "unavailable"
    if report["status"] == "unavailable":
        report["reason"] = "Original final-test metrics and predictions are both required for a full reproduction check"
    return report


def make_summary(results, examples, scoring, provenance):
    metrics = {mode: scoring.summarize(rows) for mode, rows in results.items()}
    tasks = sorted({example.dataset for example in examples})
    chance = {"overall": sum(1 / e.num_choices for e in examples) / len(examples),
              "per_task": {task: sum(1 / e.num_choices for e in examples if e.dataset == task)
                  / sum(e.dataset == task for e in examples) for task in tasks}}
    comparisons = {}
    if "actual_packet" in results:
        for mode, rows in results.items():
            if mode != "actual_packet":
                comparisons[mode] = paired_comparison(results["actual_packet"], rows)
                comparisons[mode]["per_task"] = {task: paired_comparison(
                    [r for r in results["actual_packet"] if r["dataset"] == task],
                    [r for r in rows if r["dataset"] == task]) for task in tasks}
    return {"format": 1, "examples": len(examples), "modes": metrics,
            "chance_accuracy": chance, "comparisons_to_actual_packet": comparisons,
            "reproduction_check": {"status": "unavailable", "reason": "Reference results not supplied"},
            "provenance": provenance,
            "protocol": {"raw_prompt_wrapper": RAW_CONTEXT_PREFIX, "batch_size": 1,
                "same_sample_set": True, "truncation": False, "training": False,
                "gradient_tracking": False, "shuffle": "seeded cycle of unique contexts within each task",
                "checkpoint_selection": "original_final; no claim about an epoch-2 checkpoint"}}


def publish_results(output, summary, results, examples, sources, descriptions):
    """Publish a complete directory atomically; never merge with existing results."""
    output = check_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    def write_json(name, value):
        (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_json("audit_summary.json", summary)
    write_json("input_lengths.json", descriptions)
    with (staging / "test_source_records.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in sorted(sources):
            handle.write(json.dumps(sources[sample_id], ensure_ascii=False, allow_nan=False) + "\n")
    for mode in results:
        write_json(f"metrics_{mode}.json", summary["modes"][mode])
    with (staging / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in sorted(sources):
            for mode, rows in results.items():
                row = next(row for row in rows if row["sample_id"] == sample_id)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    indexed = {mode: {row["sample_id"]: row for row in rows} for mode, rows in results.items()}
    with (staging / "cloth_examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in sorted(examples, key=lambda e: e.sample_id):
            if "cloth" in example.dataset.lower():
                row = {"sample_id": example.sample_id, "source_record": sources[example.sample_id],
                       "actual_prompt": example.prompt, "context": example.context, "gold": example.response,
                       "raw_text_prompt": RAW_CONTEXT_PREFIX.format(context=example.context, prompt=example.prompt),
                       "modes": {mode: rows[example.sample_id] for mode, rows in indexed.items()}}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    # Check again after writing, to avoid overwriting a competing audit.
    check_output(output)
    if output.exists():
        output.rmdir()  # Only an empty output directory is allowed here.
    os.rename(staging, output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True, type=Path)
    result.add_argument("--training-repo", required=True, type=Path)
    result.add_argument("--data-dir", required=True, type=Path)
    result.add_argument("--packet-cache", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--verbose", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)  # Deliberately before importing torch/transformers.
    check_output(args.output, (args.run_dir, args.training_repo, args.data_dir, args.packet_cache))
    manifest = read_json(args.run_dir / "manifest.json")
    runtime, data_utils, scoring = import_training_code(args.training_repo, manifest["source_commit"])
    runtime.configure_logging(args.verbose)
    distributed = runtime.initialize_distributed()
    try:
        config, checkpoint, test, precision, identity, provenance = validate_run(runtime, args, manifest)
        examples, sources = load_examples(test, data_utils.parse_example, config.data)
        receipt = read_json(args.data_dir / "preparation.json")
        if len(examples) != receipt["files"]["stage2/test.jsonl"]["records"]:
            raise ValueError("Audit test sample count differs from the prepared data receipt")
        LOGGER.info("Audit | verified original run; %d examples; loading frozen receiver only", len(examples))
        model = runtime.build_runtime_model(config, precision, distributed.device, load_encoder=False)
        runtime.load_bridge(checkpoint, model.bridge)
        if (list(model.bridge.packet_shape) != identity["meta"]["packet_shape"]
                or runtime.compressor_hash(model.bridge) != identity["meta"]["provenance"]["compressor"]):
            raise ValueError("Loaded bridge differs from the original packet cache")
        results, descriptions = evaluate_controls(runtime, model, examples, runtime.PacketCache(args.packet_cache),
            config, distributed, modes=tuple(args.modes), seed=args.seed, scoring=scoring)
        if distributed.is_main:
            try:
                audit_commit = git_head(Path(__file__).resolve().parent.parent)
            except subprocess.CalledProcessError:
                audit_commit = None
            provenance.update(audit_source_commit=audit_commit, audit_script_sha256=file_hash(__file__),
                              seed=args.seed, evaluation_world_size=distributed.world_size)
            summary = make_summary(results, examples, scoring, provenance)
            summary["reproduction_check"] = reproduction_check(results, args.run_dir, scoring)
            publish_results(args.output, summary, results, examples, sources, descriptions)
            reproduction = summary["reproduction_check"]
            if reproduction["status"] != "match":
                LOGGER.warning("Audit | original-final reproduction: %s | differing prediction rows: %s | details in audit_summary.json",
                               reproduction["status"], reproduction.get("prediction_difference_count"))
            else:
                LOGGER.info("Audit | original-final reproduction: match (accuracy, loss and per-sample gold/prediction)")
            for mode, metrics in summary["modes"].items():
                LOGGER.info("Audit | %s | %s", mode, runtime.format_metrics(metrics))
            LOGGER.info("Audit complete | results: %s", args.output)
        distributed.barrier()
        return 0
    finally:
        distributed.close()


if __name__ == "__main__":
    raise SystemExit(main())
