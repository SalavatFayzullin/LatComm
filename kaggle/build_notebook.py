"""Build the lightweight Kaggle bootstrap notebook and source bundle."""
from pathlib import Path
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
По умолчанию выбран уменьшенный эксперимент **`PROFILE = "kaggle"`**:

- Qwen3.5-0.8B → Qwen3.5-0.8B; обе LLM заморожены.
- Stage 1: 2048 текстов, одна эпоха, 64 обновления.
- Stage 2: по 128 вопросов из каждого из 12 наборов, три эпохи, 144 обновления.
- Global batch 32, microbatch 1; на двух GPU накопление составляет 16 микробатчей.
- Тест: по 16 вопросов из каждого набора, всего 192.

Архитектура, два этапа и функция потерь соответствуют методу статьи. Объём данных,
batch и warmup уменьшены; метрики такого эксперимента не воспроизводят таблицы статьи.
Исходный архив Stage 1 (около 2,8 GB) всё равно скачивается и просматривается потоком.
`PROFILE = "full"` явно включает исходный полный бюджет, `"smoke"` — короткую проверку цепочки.

Для ночного эксперимента выберите **`PROFILE = "overnight"`**. Он увеличивает Stage 1 до
45 056 текстов (1 408 обновлений), сохраняя объём Stage 2 и теста. По темпу прошлого запуска
на T4 ×2 это около 15 часов вычислений. В одной GPU-сессии Kaggle доступно до 12 часов.

Первую часть запустите через **Save Version → Save & Run All**, оставив `RESTORE = None`.
Ноутбук запустит только Stage 1 и сохранит чекпоинт примерно через 9,5 часа обучения.
Затем подключите Output сохранённой версии как Input в чистой сессии и укажите `RESTORE`
на каталог `latcomm-overnight` из этого Input. С тем же профилем и T4 ×2 снова выберите
**Save & Run All**. Ноутбук продолжит обучение и после финального теста сравнит правильные
и подменённые пакеты. Подготовленные данные сохраняются рядом с весами и метриками.

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
PROFILE = "kaggle"                     # choose "overnight" for ~15h across Kaggle sessions
WORKDIR = f"/kaggle/working/latcomm-{PROFILE}"  # checkpoints/results, separate from source
PAIR = None                           # kaggle/overnight: 0.8B -> 0.8B; full/smoke: 4B -> 0.8B
RESTORE = None                        # set to the saved WORKDIR under /kaggle/input/... to continue
ACTION = "stage1" if PROFILE == "overnight" and RESTORE is None else "all"
NUM_GPUS = 2 if PROFILE == "overnight" else 0  # 0 = all visible GPUs
SESSION_STEPS = 0                      # 0: finish training; optional per-session update cap
SESSION_HOURS = 9.5 if PROFILE == "overnight" else 0.0  # training time per stage
BENCHMARK_MICROBATCHES = 0              # 3 = time only 3 microbatches; use a fresh WORKDIR
VERBOSE = False                       # True: detailed per-worker diagnostics
DATA_SOURCE = None                    # optional local copy of published data
DATA_DIR = WORKDIR + "/data" if PROFILE == "overnight" else f"/kaggle/temp/latcomm-data-{PROFILE}"
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

Установщик использует PyTorch **2.11.0** и индекс cu126; Transformers — **5.15.1**.
Совместимая установленная сборка PyTorch может быть сохранена pip (например, cu128 на T4).
На этих GPU LLM работают в FP16, обучаемый интерфейс — в FP32, пакеты хранятся в BF16.
L2-нормализация query/key резервной реализации Qwen3.5 считается в FP32,
чтобы избежать NaN в её обратном проходе для малых FP16-векторов.
Это сохраняет алгоритм и отличается по вычислительной точности от H100 авторов.
"""), cell("code", '''command = [sys.executable, str(BOOTSTRAP_FILE),
           "--repo-url", REPO_URL, "--ref", REPO_REF, "--repo-dir", REPO_DIR,
           "--workdir", WORKDIR, "--profile", PROFILE,
           "--action", ACTION, "--gpus", str(NUM_GPUS),
           "--session-steps", str(SESSION_STEPS), "--session-hours", str(SESSION_HOURS),
           "--benchmark-microbatches", str(BENCHMARK_MICROBATCHES),
           "--data-dir", DATA_DIR, "--packet-cache", PACKET_CACHE]
if SOURCE_COMMIT:
    command += ["--commit", SOURCE_COMMIT]
if PAIR:
    command += ["--pair", PAIR]
if RESTORE:
    command += ["--restore", RESTORE]
if DATA_SOURCE:
    command += ["--data-source", DATA_SOURCE]
if SKIP_SETUP:
    command += ["--skip-setup"]
if VERBOSE:
    command += ["--verbose"]
result = subprocess.run(command)
if result.returncode == 75:
    print("Сессия завершена штатно; сохраните Output и продолжите с RESTORE.")
elif result.returncode:
    raise RuntimeError(f"LatComm failed, exit code {result.returncode}. See logs above.")
elif PROFILE == "overnight" and ACTION == "all" and (Path(WORKDIR) / "results/metrics.json").is_file():
    controls_dir = Path(WORKDIR + "-controls")
    summary = controls_dir / "audit-results/audit_summary.json"
    if not summary.is_file():
        audit = subprocess.run([sys.executable, str(Path(REPO_DIR) / "kaggle/run.py"), "audit",
                                "--audit-run", WORKDIR, "--training-repo", REPO_DIR,
                                "--workdir", str(controls_dir), "--data-dir", DATA_DIR,
                                "--packet-cache", PACKET_CACHE, "--gpus", str(NUM_GPUS)])
        if audit.returncode:
            print("Контрольная оценка завершилась с ошибкой. Веса и основной тест сохранены в", WORKDIR)
    if summary.is_file():
        print("Сравнение контрольных режимов:", summary)
'''), cell("markdown", """## Продолжение и результаты

Код 75 означает штатную паузу: сохранены optimizer, scheduler, scaler, RNG и позиция в данных.
Для следующей сессии сохраните Output, подключите его через Input и укажите `RESTORE`
на прежний `WORKDIR`, содержащий `manifest.json` и `outputs`.
Число GPU, микробатч, профиль и пару моделей при точном продолжении сохраняйте.

`/kaggle/temp` не сохраняется; JSONL и модели можно скачать повторно с закреплённых ревизий.
Кэш пакетов сохраняется отдельно. Его полный объём может превышать лимит сохраняемого output Kaggle.
Готовый совместимый кэш можно подключить из `/kaggle/input/...` через `PACKET_CACHE`.
Полный эксперимент требует многих сессий. Уменьшенный профиль выполняет 208 обновлений;
фактическое время и достаточность памяти нужно оценивать по текущему запуску.
"""), cell("code", '''import json
benchmark = Path(WORKDIR) / "outputs/stage1/benchmark.json"
metrics = Path(WORKDIR) / "results/metrics.json"
if BENCHMARK_MICROBATCHES and benchmark.exists():
    print(json.dumps(json.loads(benchmark.read_text()), indent=2, ensure_ascii=False))
elif metrics.exists():
    print(json.dumps(json.loads(metrics.read_text()), indent=2, ensure_ascii=False))
else:
    print("Оценка ещё не завершена. Checkpoints:", str(Path(WORKDIR) / "outputs"))
''')]

cells.append(cell("markdown", """## Контрольные оценки уже обученного интерфейса

Эта проверка использует готовый запуск. Повторного обучения нет.
Для текущей сессии оставьте `TRAINED_RUN`, `TRAINING_REPO` и `PREPARED_DATA`
как указано ниже. Измените `RUN_CONTROLS` на `True` и выполните **только следующую ячейку**.
Она самостоятельна: ячейки обучения повторно запускать не требуется.

На тех же тестовых вопросах будут сравнены:
обученный интерфейс с правильным пакетом; пакет другого контекста;
получатель без интерфейса и контекста; получатель с полным текстом контекста.
Во всех вариантах сохраняются вопросы, варианты и правило выбора ответа.
Подробные примеры CLOTH также сохраняются для проверки формата и ошибок.

Новый код проверок загружается в отдельный checkout. Исходный код обучения берётся
из SHA сохранённого запуска; модель, пакеты и прежние результаты не перезаписываются.
Результаты появятся в `CONTROLS_WORKDIR/audit-results`.
"""))
controls_code = '''from pathlib import Path
import os
import subprocess
import sys
import json

RUN_CONTROLS = False
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
    controls_bootstrap.write_text(BOOTSTRAP_SOURCE_PLACEHOLDER, encoding="utf-8")
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
        print("\\nПреимущество правильного пакета над чужим, п.п.:")
        print(f"{'Задача':15} {'Stage 1':>12} {'Stage 2':>12}")
        for task in sorted(stages["stage2"]["chance_accuracy"]["per_task"]):
            gains = [100 * (stages[stage]["modes"]["actual_packet"]["accuracy/" + task]
                            - stages[stage]["modes"]["shuffled_packet"]["accuracy/" + task])
                     for stage in ("stage1", "stage2")]
            print(f"{task:15} {gains[0]:>+12.2f} {gains[1]:>+12.2f}")
        print("\\nИтог сравнения:", summary_file)
        print("Подробные ответы и примеры CLOTH:", summary_file.parent / "stage1", "и", summary_file.parent / "stage2")
    else:
        print(f"{'Режим':20} {'Верно':>8} {'Accuracy':>10} {'Loss':>10}")
        for mode, metrics in summary["modes"].items():
            correct = round(metrics["accuracy"] * summary["examples"])
            print(f"{names[mode]:20} {correct:>8} {metrics['accuracy']:>9.2%} {metrics['loss']:>10.4f}")
        print("Ожидаемая accuracy равномерного случайного выбора:", f"{summary['chance_accuracy']['overall']:.2%}")
        print("Воспроизведение прежнего результата:", summary.get("reproduction_check", {}).get("status", "не проверено"))
        print("\\nТочность по задачам:")
        modes = list(summary["modes"])
        print(f"{'Задача':15}" + "".join(f"{names[mode]:>19}" for mode in modes))
        for task in sorted(summary["chance_accuracy"]["per_task"]):
            print(f"{task:15}" + "".join(f"{summary['modes'][mode]['accuracy/' + task]:>18.1%}" for mode in modes))
        print("\\nПодробные результаты:", summary_file.parent)
        print("Примеры CLOTH:", summary_file.parent / "cloth_examples.jsonl")
else:
    print("Для оценки готового интерфейса задайте RUN_CONTROLS = True.")
'''.replace("BOOTSTRAP_SOURCE_PLACEHOLDER", repr(bootstrap))
cells.append(cell("code", controls_code))
controls_target = ROOT / "Run_LatComm_Controls.py"
controls_target.write_text(
    "# Paste this complete file into a NEW cell in the existing Kaggle notebook.\n"
    "# The trained run, models and prepared test data must still be available.\n"
    + controls_code.replace("RUN_CONTROLS = False", "RUN_CONTROLS = True", 1), encoding="utf-8")
cells.append(cell("markdown", """## Сравнение сохранённых Stage 1 и Stage 2

Для локализации слабой зависимости от пакета эта самостоятельная ячейка проверяет
интерфейс после Stage 1 и финальный интерфейс после Stage 2 на одних тестовых вопросах.
Задайте `RUN_STAGE_COMPARISON = True` и выполните **только следующую ячейку**.
Повторного обучения и установки среды нет; используются сохранённые checkpoints и пакеты.

Для обоих интерфейсов выполняются те же четыре режима. В таблице будут точность,
loss и преимущество правильного пакета над чужим по задачам. Общий протокол:
`CONTROLS_WORKDIR/stage-comparison-results/stage_comparison.json`.
Полные ответы и примеры CLOTH сохраняются отдельно в подпапках `stage1` и `stage2`.
Прежняя папка `audit-results` сохраняется. Проверка Stage 1 на MCQ здесь диагностическая:
этот интерфейс обучался реконструкции текста, а не выбору буквы ответа.
"""))
comparison_code = controls_code.replace("RUN_CONTROLS", "RUN_STAGE_COMPARISON").replace(
    "COMPARE_STAGE1 = False", "COMPARE_STAGE1 = True", 1).replace(
    "Для оценки готового интерфейса задайте RUN_STAGE_COMPARISON = True.",
    "Для сравнения сохранённых Stage 1 и Stage 2 задайте RUN_STAGE_COMPARISON = True.")
cells.append(cell("code", comparison_code))
comparison_target = ROOT / "Run_LatComm_Stage_Comparison.py"
comparison_target.write_text(
    "# Paste this complete file into a NEW cell in the existing Kaggle notebook.\n"
    "# Uses saved Stage 1 / Stage 2 checkpoints and the original test data; no training.\n"
    + comparison_code.replace("RUN_STAGE_COMPARISON = False", "RUN_STAGE_COMPARISON = True", 1),
    encoding="utf-8")
cells.append(cell("markdown", (ROOT / "README_RU.md").read_text(encoding="utf-8")))

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
with zipfile.ZipFile(ROOT / "LatComm_Kaggle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ("bootstrap.py", "run.py", "data_utils.py", "runtime.py", "progress.py", "audit.py", "README_RU.md", controls_target.name, comparison_target.name, target.name):
        archive.write(ROOT / name, "kaggle/" + name)
    archive.write(ROOT.parent / "latcomm/evaluate.py", "latcomm/evaluate.py")
print(target)
