"""Build the single-session Kaggle launcher for the user's fork."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTRO = """# LatComm за одну ночь

В Kaggle выберите **GPU T4 ×2** и **Internet On**, затем **Save Version → Save & Run All**.
Ноутбук клонирует [ваш форк LatComm](https://github.com/SalavatFayzullin/LatComm)
и запускает `kaggle/run.py` из его ветки `main`. Датасет и веса загрузятся автоматически.
В `main` должна быть опубликована версия с профилем `nightly` и `kaggle/night_report.py`.

Stage 1 использует 28 672 текста (896 обновлений). Stage 2 — 1 536 вопросов
и три эпохи. Финальный тест содержит 192 вопроса. Ориентир на двух T4 — около
10 часов вычислений плюс подготовка. Запас до 12-часового лимита нужен для
сохранения Output. При срабатывании лимита ноутбук выведет доступные метрики
и сохранённый прогресс.

Итог напечатается в конце. В Output сохраняются `latcomm-ready/night_report.txt`,
чекпоинты, результаты теста и ответы, а также `latcomm-ready-controls` со сравнением
правильных, чужих и отсутствующих пакетов и полного текста.
"""

RUN = '''import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

started = time.monotonic()
training_deadline = started + 11 * 3600
process_deadline = started + (11 * 60 + 25) * 60
workdir = Path("/kaggle/working/latcomm-ready")
controls = workdir.with_name("latcomm-ready-controls")
scratch = Path("/kaggle/temp")
workdir.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(scratch / "latcomm-ready-hf")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["LATCOMM_DEADLINE_MONOTONIC"] = str(training_deadline)


def run_bounded(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 124
    with subprocess.Popen(command, start_new_session=True) as process:
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                # Let torchrun forward termination to its workers first.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            print("Достигнут резервный лимит процесса. Сохраняю доступные результаты.", flush=True)
            return 124


with TemporaryDirectory(prefix="latcomm-source-", dir=scratch) as temp:
    repo = Path(temp) / "LatComm"
    runner = [sys.executable, str(repo / "kaggle/run.py")]
    reporter = repo / "kaggle/night_report.py"
    run_status, audit_status = 1, None
    try:
        devices = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True, timeout=15)
        if len(devices.strip().splitlines()) < 2:
            raise RuntimeError("Для этого прогона выберите GPU T4 ×2 в Kaggle.")
        print(devices.strip(), flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", "--",
                        "https://github.com/SalavatFayzullin/LatComm.git", str(repo)],
                       check=True, timeout=180)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        print("Коммит форка", revision, flush=True)
        help_text = subprocess.check_output(runner + ["--help"], text=True, timeout=15)
        if not reporter.is_file() or "nightly" not in help_text:
            raise RuntimeError("В main форка ещё нет профиля nightly и отчёта. Опубликуйте изменения в kaggle/.")
        print("Подготовка окружения…", flush=True)
        run_status = run_bounded(runner + ["setup", "--workdir", str(workdir)], training_deadline)
        if run_status == 0:
            run_status = run_bounded(runner + ["all", "--profile", "nightly", "--workdir", str(workdir),
                                               "--gpus", "2"], process_deadline)
        if run_status == 0 and not (controls / "audit-results/audit_summary.json").is_file():
            print("Контрольная оценка финальной модели…", flush=True)
            audit_status = run_bounded(runner + ["audit", "--audit-run", str(workdir), "--workdir", str(controls),
                                                "--training-repo", str(repo), "--data-dir", str(workdir / "data"),
                                                "--gpus", "2"], process_deadline)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if not reporter.is_file():
            raise
        print(f"Ошибка запуска: {error}", flush=True)
    finally:
        if reporter.is_file():
            report = [sys.executable, str(reporter), "--workdir", str(workdir),
                      "--controls-dir", str(controls), "--elapsed-seconds", str(time.monotonic() - started),
                      "--run-exit-code", str(run_status)]
            if audit_status is not None:
                report += ["--audit-exit-code", str(audit_status)]
            subprocess.run(report, check=True)
'''


def main():
    compile(RUN, "LatComm_Ready.ipynb:run", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": RUN.splitlines(keepends=True)},
    ]}
    target = ROOT / "kaggle/LatComm_Ready.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
