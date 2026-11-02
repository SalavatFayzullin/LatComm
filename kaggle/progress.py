"""Compact console output; no model state, RNG, CUDA sync or collectives."""
from __future__ import annotations

from collections import deque
import logging
import os
import time

LOGGER = logging.getLogger("latcomm.kaggle")
INTERVAL = 30.0


class WorkerFilter(logging.Filter):
    def __init__(self, rank: int, verbose: bool):
        super().__init__()
        self.rank, self.verbose = rank, verbose

    def filter(self, record):
        record.worker = f"[rank {self.rank}] " if self.verbose or self.rank else ""
        if not self.verbose and "unauthenticated requests to the HF Hub" in record.getMessage():
            return False
        return self.verbose or self.rank == 0 or record.levelno >= logging.WARNING


def configure_logging(verbose=False):
    rank = int(os.environ.get("RANK", "0"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(worker)s%(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in logging.getLogger().handlers:
        for previous in list(handler.filters):
            if isinstance(previous, WorkerFilter):
                handler.removeFilter(previous)
        handler.addFilter(WorkerFilter(rank, verbose))
    # HTTP status probes (including optional-file 404s) aren't training events.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.WARNING)
    from transformers.utils import logging as transformers_logging
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
    # Use the same handler for library warnings; their default handlers would
    # otherwise duplicate messages and bypass the worker/noise filter.
    transformers_logging.disable_default_handler()
    transformers_logging.enable_propagation()
    hub_logger = logging.getLogger("huggingface_hub")
    hub_logger.handlers.clear()
    hub_logger.propagate = True
    if verbose:
        transformers_logging.enable_progress_bar()
        enable_progress_bars()
    else:
        transformers_logging.disable_progress_bar()
        disable_progress_bars()


def duration(seconds):
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_metrics(metrics):
    fields = []
    if "loss" in metrics:
        fields.append(f"loss={metrics['loss']:.4f}")
    for key, label in (("accuracy", "acc"), ("token_accuracy", "token_acc")):
        if key in metrics:
            fields.append(f"{label}={metrics[key]:.1%}")
            break
    if "macro_accuracy" in metrics:
        fields.append(f"macro_acc={metrics['macro_accuracy']:.1%}")
    return " | ".join(fields)


class CounterProgress:
    """Periodically report completed items, with a rate local to this session."""
    def __init__(self, label, total, *, initial=0, interval=INTERVAL, enabled=True):
        self.label, self.total, self.initial = label, total, initial
        self.interval, self.enabled = interval, enabled
        self.started = self.last_report = time.monotonic()

    def due(self):
        return self.enabled and time.monotonic() - self.last_report >= self.interval

    def update(self, completed, *, force=False):
        if not self.enabled or not (force or self.due()):
            return
        now = time.monotonic()
        percentage = completed / self.total if self.total else 1.0
        message = f"{self.label} | {completed:,}/{self.total:,} ({percentage:.0%})"
        advanced = completed - self.initial
        if advanced > 0 and completed < self.total:
            message += f" | ETA~{duration((now - self.started) / advanced * (self.total - completed))}"
        LOGGER.info("%s", message)
        self.last_report = now


class ProgressLoader:
    """Wrap evaluation without changing its batches or distributed operations."""
    def __init__(self, loader, label, *, enabled=True):
        self.loader, self.label, self.enabled = loader, label, enabled

    def __iter__(self):
        progress = CounterProgress(self.label, len(self.loader), enabled=self.enabled)
        for completed, batch in enumerate(self.loader, 1):
            yield batch
            progress.update(completed)


class TrainingProgress:
    def __init__(self, label, total, epochs, steps_per_epoch, *, enabled=True,
                 interval=INTERVAL, memory=None):
        self.label, self.total, self.epochs = label, total, epochs
        self.steps_per_epoch, self.enabled = steps_per_epoch, enabled
        self.interval, self.memory = interval, memory
        self.last_report = self.last_update_report = time.monotonic()
        self.seconds = deque(maxlen=10)
        self.metrics = {}

    def prefix(self, step):
        epoch = (step - 1) // self.steps_per_epoch + 1
        return f"{self.label} | epoch {epoch}/{self.epochs} | step {step}/{self.total}"

    def microbatch(self, step, completed, target, elapsed):
        if not self.enabled or time.monotonic() - self.last_report < self.interval:
            return
        message = f"{self.prefix(step)} | microbatch {completed}/{target} | elapsed {duration(elapsed)}"
        if self.metrics:
            message += " | last " + format_metrics(self.metrics)
        LOGGER.info("%s", message)
        self.last_report = time.monotonic()

    def update(self, step, seconds, metrics=None, *, force=False):
        self.seconds.append(seconds)
        if metrics:
            self.metrics = metrics
        if not self.enabled or not (force or len(self.seconds) == 1
                                   or time.monotonic() - self.last_update_report >= self.interval):
            return
        fields = [f"{self.prefix(step)} ({step / self.total:.1%})"]
        if self.metrics:
            fields.append(format_metrics(self.metrics))
        fields.append(f"step_time={duration(seconds)}")
        if step < self.total:
            fields.append(f"ETA~{duration(sum(self.seconds) / len(self.seconds) * (self.total - step))}")
        if self.memory:
            memory = self.memory()
            if "container_ram_limit_gib" in memory:
                fields.append(f"RAM={memory['container_ram_used_gib']:.1f}/{memory['container_ram_limit_gib']:.0f}GiB")
        LOGGER.info("%s", " | ".join(fields))
        self.last_report = self.last_update_report = time.monotonic()
