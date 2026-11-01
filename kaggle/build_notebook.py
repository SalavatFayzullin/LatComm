"""Build the lightweight Kaggle bootstrap notebook and source bundle."""
from pathlib import Path
from copy import deepcopy
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [cell("markdown", """# LatComm: запуск из своего форка

Этот ноутбук клонирует **[SalavatFayzullin/LatComm](https://github.com/SalavatFayzullin/LatComm)**,
устанавливает среду и запускает `kaggle/run.py` из клонированного репозитория.
Перед первым запуском **закоммитьте и отправьте папку `kaggle/` в свой форк**.

В Kaggle включите **Internet** и **GPU T4 ×2** или **P100**.
Данные авторов загружаются автоматически с Hugging Face.
`PROFILE = "full"` использует полный набор; `"smoke"` проверяет всю цепочку на малой выборке.
Даже в smoke исходный архив Stage 1 (около 2,8 GB) скачивается целиком.

Новый запуск берёт `REPO_REF`. Для продолжения обучения используется SHA из сохранённого
`manifest.json`, чтобы обновление ветки main не изменило код посреди эксперимента.
"""), cell("code", '''from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/SalavatFayzullin/LatComm.git"
REPO_REF = "main"                       # branch/tag for a new experiment
SOURCE_COMMIT = None                    # optional full SHA for a fixed version
REPO_DIR = "/kaggle/working/LatComm"     # source code from your fork
PROFILE = "full"                       # "smoke" for a small pipeline check
WORKDIR = f"/kaggle/working/latcomm-{PROFILE}"  # checkpoints/results, separate from source
PAIR = "qwen4b_to_qwen0.8b"             # alternative: qwen0.8b_to_qwen0.8b
ACTION = "all"                         # setup, prepare, stage1, encode, stage2, evaluate, all
NUM_GPUS = 0                           # 0 = all visible GPUs
SESSION_STEPS = 25                     # new updates per session, full schedule is retained
SESSION_HOURS = 8.0                    # graceful stop at the next optimizer boundary
BENCHMARK_MICROBATCHES = 0              # 3 = time only 3 microbatches; use a fresh WORKDIR
RESTORE = None                        # e.g. /kaggle/input/previous-output/latcomm-full
DATA_SOURCE = None                    # optional local copy of published data
DATA_DIR = f"/kaggle/temp/latcomm-data-{PROFILE}"
PACKET_CACHE = WORKDIR + "/packets"    # full cache can require tens of GB
SKIP_SETUP = False                    # True only if this session's environment is installed
BOOTSTRAP_FILE = Path("/kaggle/working/latcomm_bootstrap.py")

os.environ["HF_HOME"] = "/kaggle/temp/latcomm-hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
if not os.environ.get("HF_TOKEN"):
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass
''')]

bootstrap = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
cells.append(cell("code", f"# Bootstrap only; training code will be loaded from your fork.\nBOOTSTRAP_FILE.write_text({bootstrap!r}, encoding='utf-8')\n"))

cells += [cell("markdown", """## Клонирование, установка и запуск

Следующая ячейка клонирует форк, выводит фактический SHA, выполняет
`python kaggle/run.py setup`, затем выбранный `ACTION` из того же checkout.
Повторное клонирование репозитория авторов не выполняется.
Установка и обучение идут в отдельных Python-процессах.

PyTorch закреплён на **2.11.0 cu126** для T4/P100; Transformers — **5.15.1**.
На этих GPU LLM работают в FP16, обучаемый интерфейс — в FP32, пакеты хранятся в BF16.
Это сохраняет алгоритм и отличается по вычислительной точности от H100 авторов.
"""), cell("code", '''command = [sys.executable, str(BOOTSTRAP_FILE),
           "--repo-url", REPO_URL, "--ref", REPO_REF, "--repo-dir", REPO_DIR,
           "--workdir", WORKDIR, "--profile", PROFILE, "--pair", PAIR,
           "--action", ACTION, "--gpus", str(NUM_GPUS),
           "--session-steps", str(SESSION_STEPS), "--session-hours", str(SESSION_HOURS),
           "--benchmark-microbatches", str(BENCHMARK_MICROBATCHES),
           "--data-dir", DATA_DIR, "--packet-cache", PACKET_CACHE]
if SOURCE_COMMIT:
    command += ["--commit", SOURCE_COMMIT]
if RESTORE:
    command += ["--restore", RESTORE]
if DATA_SOURCE:
    command += ["--data-source", DATA_SOURCE]
if SKIP_SETUP:
    command += ["--skip-setup"]
result = subprocess.run(command)
if result.returncode == 75:
    print("Сессия завершена штатно; сохраните Output и продолжите с RESTORE.")
elif result.returncode:
    raise RuntimeError(f"LatComm failed, exit code {result.returncode}. See logs above.")
'''), cell("markdown", """## Продолжение и результаты

Код 75 означает штатную паузу: сохранены optimizer, scheduler, scaler, RNG и позиция в данных.
Для следующей сессии сохраните Output, подключите его через Input и укажите `RESTORE`
на прежний `WORKDIR`, содержащий `manifest.json` и `outputs`.
Число GPU, микробатч, профиль и пару моделей при точном продолжении сохраняйте.

`/kaggle/temp` не сохраняется; JSONL и модели можно скачать повторно с закреплённых ревизий.
Кэш пакетов сохраняется отдельно. Его полный объём может превышать лимит сохраняемого output Kaggle.
Готовый совместимый кэш можно подключить из `/kaggle/input/...` через `PACKET_CACHE`.
Полный эксперимент требует многих сессий; `SESSION_STEPS` ограничивает текущую сессию.
"""), cell("code", '''import json
benchmark = Path(WORKDIR) / "outputs/stage1/benchmark.json"
metrics = Path(WORKDIR) / "results/metrics.json"
if benchmark.exists():
    print(json.dumps(json.loads(benchmark.read_text()), indent=2, ensure_ascii=False))
elif metrics.exists():
    print(json.dumps(json.loads(metrics.read_text()), indent=2, ensure_ascii=False))
else:
    print("Оценка ещё не завершена. Checkpoints:", str(Path(WORKDIR) / "outputs"))
'''), cell("markdown", (ROOT / "README_RU.md").read_text(encoding="utf-8"))]

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"}}, "nbformat": 4, "nbformat_minor": 4}
target = ROOT / "LatComm_Kaggle.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for index, item in enumerate(cells):
    if item["cell_type"] == "code":
        compile("".join(item["source"]), f"notebook-cell-{index}", "exec")
try:
    import nbformat
except ImportError:
    pass
else:
    nbformat.validate(notebook)
diagnostic = deepcopy(notebook)
diagnostic["cells"][0]["source"] = [
    "# LatComm: диагностика скорости\n",
    "\nСначала отправьте обновлённую папку `kaggle/` в свой форк.\n",
    "Этот запуск измеряет **3 микробатча** Stage 1 без обновления весов и без начальной оценки на 256 примерах.\n",
    "Используется отдельный WORKDIR; существующие checkpoints сохраняются.\n",
    "Вывод показывает rank, фазу encoder/receiver/backward и время каждого микробатча.\n",
    "Оценка времени полного шага приблизительная; данные и модели всё равно должны быть загружены.\n",
]
settings = "".join(diagnostic["cells"][1]["source"])
settings = settings.replace('WORKDIR = f"/kaggle/working/latcomm-{PROFILE}"',
                            'WORKDIR = "/kaggle/working/latcomm-benchmark"')
settings = settings.replace('ACTION = "all"', 'ACTION = "stage1"')
settings = settings.replace('BENCHMARK_MICROBATCHES = 0', 'BENCHMARK_MICROBATCHES = 3')
diagnostic["cells"][1]["source"] = settings.splitlines(keepends=True)
diagnostic_path = ROOT / "LatComm_Diagnostic.ipynb"
diagnostic_path.write_text(json.dumps(diagnostic, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for item in diagnostic["cells"]:
    if item["cell_type"] == "code":
        compile("".join(item["source"]), "diagnostic-notebook-cell", "exec")
with zipfile.ZipFile(ROOT / "LatComm_Kaggle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ("bootstrap.py", "run.py", "data_utils.py", "runtime.py", "README_RU.md", target.name, diagnostic_path.name):
        archive.write(ROOT / name, "kaggle/" + name)
print(target)
