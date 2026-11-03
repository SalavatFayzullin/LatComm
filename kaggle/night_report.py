"""Print and save a self-contained summary of the overnight run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = {
    "actual_packet": "Правильный пакет",
    "shuffled_packet": "Чужой пакет",
    "query_only": "Только вопрос",
    "raw_text": "Полный текст",
}


def number(value):
    if type(value) in (float, int) and math.isfinite(value):
        return value
    return None


def percentage(value):
    value = number(value)
    return f"{100 * value:.2f}%" if value is not None else "—"


def decimal(value):
    value = number(value)
    return f"{value:.4f}" if value is not None else "—"


def read_object(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


def validation_rows(path):
    rows = {}
    skipped = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(row, dict) and row.get("type") == "eval" and type(row.get("step")) is int:
                    rows[row["step"]] = row
    return [rows[step] for step in sorted(rows)], skipped


def describe_training(stage_dir, label, accuracy_key):
    latest = read_object(stage_dir / "latest.json")
    rows, skipped = validation_rows(stage_dir / "metrics.jsonl")
    lines = []
    if latest:
        status = "завершён" if latest.get("complete") is True else "есть промежуточный чекпоинт"
        lines.append(f"{label} | {status} | сохранённый шаг {latest.get('step', '—')}")
    else:
        lines.append(f"{label} | чекпоинт не найден")
    if not rows:
        lines.append("  Валидационные метрики ещё не записаны.")
    elif accuracy_key == "token_accuracy":
        scored = [row for row in rows if number(row.get("loss")) is not None]
        selected = [("первая", rows[0])]
        if scored:
            best = min(scored, key=lambda row: row["loss"])
            if best["step"] not in (rows[0]["step"], rows[-1]["step"]):
                selected.append(("лучшая по loss", best))
        if len(rows) > 1:
            selected.append(("последняя", rows[-1]))
        for name, row in selected:
            lines.append(f"  Валидация ({name}) | шаг {row['step']:>4} | "
                         f"loss {decimal(row.get('loss'))} | "
                         f"точность по токенам {percentage(row.get(accuracy_key))}")
    else:
        for row in rows:
            lines.append(f"  Валидация | шаг {row['step']:>4} | loss {decimal(row.get('loss'))} | "
                         f"точность {percentage(row.get(accuracy_key))}")
    if skipped:
        lines.append(f"  Неполных строк журнала пропущено {skipped}.")
    return lines


def describe_controls(summary):
    modes = summary.get("modes", {})
    examples = summary.get("examples")
    if not modes:
        return ["Контрольная оценка ещё не сохранена."]
    lines = [
        f"Контрольная оценка | {examples if type(examples) is int else '—'} вопросов",
        f"  {'Режим':22} {'Точность':>9} {'Верно':>9} {'Loss':>9}",
    ]
    for name, label in MODES.items():
        metrics = modes.get(name, {})
        accuracy = number(metrics.get("accuracy"))
        count = f"{round(accuracy * examples)}/{examples}" if accuracy is not None and type(examples) is int else "—"
        lines.append(f"  {label:22} {percentage(accuracy):>9} {count:>9} {decimal(metrics.get('loss')):>9}")
    chance = summary.get("chance_accuracy", {}).get("overall")
    if number(chance) is not None:
        lines.append(f"  Случайный выбор из доступных вариантов | {percentage(chance)}")
    paired = summary.get("comparisons_to_actual_packet", {}).get("shuffled_packet", {})
    delta = number(paired.get("accuracy_delta"))
    if delta is not None:
        # Audit stores shuffled minus actual; the report gives actual minus shuffled.
        actual_gain = -100 * delta if delta else 0
        actual_only = paired.get("correct_to_wrong")
        shuffled_only = paired.get("wrong_to_correct")
        lines.append(f"  Правильный минус чужой пакет | {actual_gain:+.2f} п. п.")
        lines.append(
            f"  Верен только с правильным пакетом {paired.get('correct_to_wrong', '—')} | "
            f"верен только с чужим {paired.get('wrong_to_correct', '—')} | "
            f"предсказание изменилось {paired.get('prediction_flips', '—')}"
        )
        if all(type(count) is int and count >= 0 for count in (actual_only, shuffled_only)):
            discordant = actual_only + shuffled_only
            lower_tail = sum(math.comb(discordant, k) for k in range(min(actual_only, shuffled_only) + 1))
            p_value = min(1.0, (2 * lower_tail) / (2 ** discordant))
            outcome = "достигнут" if actual_gain >= 5 and p_value < 0.05 else "не достигнут"
            lines.append(f"  Точный парный тест Мак-Немара | двустороннее p={p_value:.4g}")
            lines.append(f"  Исследовательский порог пилота (+5 п. п. или больше и p<0,05) | {outcome}")
    check = summary.get("reproduction_check", {})
    status = {"match": "совпало", "different": "есть расхождения",
              "unavailable": "недоступно"}.get(check.get("status"), "недоступно")
    lines.append(f"  Сверка с финальным тестом | {status}")
    if check.get("status") == "different" and type(check.get("prediction_difference_count")) is int:
        lines.append(f"  Расхождения по отдельным вопросам | {check['prediction_difference_count']}")

    task_prefix = "accuracy/"
    tasks = sorted({key[len(task_prefix):] for metrics in modes.values()
                    for key in metrics if key.startswith(task_prefix)})
    if tasks:
        lines.append("")
        lines.append(f"  {'Набор':20} {'Правильный':>10} {'Чужой':>10} {'Вопрос':>10} {'Текст':>10}")
        for task in tasks:
            values = [percentage(modes.get(name, {}).get(task_prefix + task)) for name in MODES]
            lines.append(f"  {task:20}" + "".join(f" {value:>10}" for value in values))
    return lines


def make_report(workdir, controls_dir, *, elapsed_seconds=None, run_exit_code=None, audit_exit_code=None):
    lines = ["Итоги ночного запуска LatComm"]
    if elapsed_seconds is not None:
        lines.append(f"Время с начала запуска | {elapsed_seconds / 3600:.2f} ч")
    manifest = read_object(workdir / "manifest.json")
    hardware = read_object(workdir / "hardware.json")
    if manifest.get("profile"):
        lines.append(f"Профиль {manifest['profile']} | global batch {manifest.get('global_batch_size', '—')} | "
                     f"GPU {len(hardware.get('devices', []))} | точность вычислений {hardware.get('precision', '—')}")
    prepared = read_object(workdir / "data/preparation.json").get("files", {})
    if prepared:
        counts = [prepared.get(f"{name}.jsonl", {}).get("records", "—")
                  for name in ("stage1/train", "stage2/train", "stage2/test")]
        lines.append(f"Stage 1 train {counts[0]} | Stage 2 train {counts[1]} | тест {counts[2]}")
    if run_exit_code == 75:
        lines.append("Штатная пауза обучения. Ниже последний сохранённый прогресс.")
    elif run_exit_code == 124:
        lines.append("Достигнут предел времени процесса. Ниже последний сохранённый прогресс.")
    elif run_exit_code not in (None, 0):
        lines.append(f"Процесс обучения завершился с ошибкой (код {run_exit_code}).")
    lines.append("")
    lines.extend(describe_training(workdir / "outputs/stage1", "Stage 1", "token_accuracy"))
    lines.extend(describe_training(workdir / "outputs/stage2", "Stage 2", "accuracy"))
    lines.append("")
    metrics_path = workdir / "results/metrics.json"
    metrics = read_object(metrics_path)
    if metrics:
        lines.append(f"Финальный тест | точность {percentage(metrics.get('accuracy'))} | "
                     f"macro accuracy {percentage(metrics.get('macro_accuracy'))} | loss {decimal(metrics.get('loss'))}")
    else:
        lines.append("Финальный тест ещё не сохранён.")
    if audit_exit_code not in (None, 0):
        lines.append(f"Процесс контрольной оценки завершился с ошибкой (код {audit_exit_code}).")
    summary_path = controls_dir / "audit-results/audit_summary.json"
    if not summary_path.is_file():
        summary_path = controls_dir / "audit_summary.json"
    summary = read_object(summary_path)
    lines.extend(describe_controls(summary))
    lines.append("")
    lines.append(f"Данные, чекпоинты и журнал обучения | {workdir}")
    if metrics:
        lines.append(f"Финальные метрики | {metrics_path}")
    if summary:
        lines.append(f"Контрольные метрики и ответы | {summary_path.parent}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--controls-dir", required=True, type=Path)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--run-exit-code", type=int)
    parser.add_argument("--audit-exit-code", type=int)
    args = parser.parse_args()
    report = make_report(args.workdir, args.controls_dir, elapsed_seconds=args.elapsed_seconds,
                         run_exit_code=args.run_exit_code, audit_exit_code=args.audit_exit_code)
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "night_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
