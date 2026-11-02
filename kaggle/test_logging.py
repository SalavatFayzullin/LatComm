"""Console regressions: periodic progress, truthful resume ETA, visible errors."""
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress
import runtime


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock.now)
    return clock


def test_compact_mode_reports_a_long_step_without_phase_spam_or_memory_queries(monkeypatch, caplog, clock):
    monkeypatch.setattr(runtime, "memory_snapshot", lambda: pytest.fail("sampled detailed debug memory"))
    summary = progress.TrainingProgress("Stage 2", 144, 3, 48)
    reporter = runtime.MicrobatchProgress(0, summary=summary)
    with caplog.at_level(logging.INFO, logger=runtime.LOGGER.name):
        reporter.start(97, 16)
        for phase in ("packet_read", "receiver_forward", "backward"):
            reporter.phase(phase)
        reporter.begin_microbatch()
        reporter.finish_microbatch()
        assert not caplog.records
        clock.now = 31.0
        reporter.finish_microbatch()
        reporter.repeat(1, 1024, 512)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "epoch 3/3 | step 97/144 | microbatch 2/16" in messages[0]
    assert "FP16 overflow" in messages[1] and "1024 -> 512" in messages[1]
    assert not any("microbatch_phase" in message or "gpu_reserved" in message for message in messages)


def test_completed_updates_keep_reporting_metrics_after_microbatch_heartbeats(caplog, clock):
    summary = progress.TrainingProgress("Stage 2", 144, 3, 48)
    with caplog.at_level(logging.INFO, logger=progress.LOGGER.name):
        clock.now = 40
        summary.update(95, 40, {"loss": 1.2, "accuracy": 0.5})
        clock.now = 71
        summary.microbatch(96, 12, 16, 31)
        clock.now = 80
        summary.update(96, 40, {"loss": 1.1, "accuracy": 0.6})
        clock.now = 81
        summary.update(97, 1, {"loss": 1.0, "accuracy": 0.7})
        clock.now = 82
        summary.update(144, 1, {"loss": 0.9, "accuracy": 0.8}, force=True)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 4  # Step 97 is throttled; final step is always shown.
    assert "last loss=1.2000 | acc=50.0%" in messages[1]
    assert "step 96/144 (66.7%)" in messages[2] and "acc=60.0%" in messages[2]
    assert "epoch 3/3 | step 144/144 (100.0%)" in messages[3]
    assert "ETA" not in messages[3]


def test_worker_filter_removes_duplicates_and_http_noise_without_hiding_errors():
    def record(name, level, message):
        return logging.LogRecord(name, level, "test", 1, message, (), None)
    secondary = progress.WorkerFilter(1, False)
    assert not secondary.filter(record("latcomm.kaggle", logging.INFO, "epoch started"))
    failure = record("latcomm.kaggle", logging.ERROR, "checkpoint write failed")
    assert secondary.filter(failure) and failure.worker == "[rank 1] "
    assert secondary.filter(record("transformers", logging.WARNING, "model weights are missing"))
    token_notice = record("huggingface_hub", logging.WARNING, "You are sending unauthenticated requests to the HF Hub")
    assert not progress.WorkerFilter(0, False).filter(token_notice)
    assert progress.WorkerFilter(0, True).filter(token_notice)
    assert progress.WorkerFilter(1, True).filter(record("latcomm.kaggle", logging.DEBUG, "backward"))


def test_counter_resume_eta_excludes_previously_cached_work(caplog, clock):
    counter = progress.CounterProgress("Packets saved", 10, initial=8)
    with caplog.at_level(logging.INFO, logger=progress.LOGGER.name):
        clock.now = 30
        counter.update(8)
        clock.now = 60
        counter.update(9)
        counter.update(10, force=True)
    messages = [record.getMessage() for record in caplog.records]
    assert "ETA" not in messages[0]
    assert "9/10 (90%) | ETA~1m 00s" in messages[1]
    assert "10/10 (100%)" in messages[2] and "ETA" not in messages[2]


def test_evaluation_progress_preserves_batches_exactly(caplog, clock):
    batches = [object(), object(), object()]
    wrapped = iter(progress.ProgressLoader(batches, "Validation"))
    with caplog.at_level(logging.INFO, logger=progress.LOGGER.name):
        assert next(wrapped) is batches[0]
        clock.now = 31
        assert next(wrapped) is batches[1]
        assert next(wrapped) is batches[2]
        with pytest.raises(StopIteration):
            next(wrapped)
    assert len(caplog.records) == 1
    assert "Validation | 1/3" in caplog.records[0].getMessage()
