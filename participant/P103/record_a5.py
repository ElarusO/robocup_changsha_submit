"""把已核查的 A5 复测结果追加到 P103 实验总表；拒绝重复记录。"""

import csv
from datetime import date
import io
import json
from pathlib import Path


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
CSV_PATH = PARTICIPANT_DIR / "experiments.csv"
SUMMARY_PATH = REPO_ROOT / "outputs" / "P103" / "a5-summary-001" / "summary.json"
IDS = ("E001", "E008", "E009")


def main():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        rows = list(reader)
    if columns is None:
        raise ValueError("实验总表缺少表头")
    records = {row["experiment_id"]: row for row in rows}
    if any(f"A5-{model_id}" in records for model_id in IDS):
        raise FileExistsError("A5 复测记录已存在，不重复追加")
    generated = io.StringIO(newline="")
    writer = csv.DictWriter(generated, fieldnames=columns, lineterminator="\n")
    for model_id in IDS:
        model = summary["models"][model_id]
        original = records[model_id]
        entry = {column: "" for column in columns}
        entry.update({
            "experiment_id": f"A5-{model_id}",
            "date": date.today().isoformat(),
            "kind": "validation_v2_recheck",
            "source_commit": model["origin_commit"],
            "model_sha256": model["model_sha256"],
            "suite_id": summary["suite_id"],
            "seed_schedule_id": summary["seed_schedule_id"],
            "run_id": model["run_id"],
            "output_dir": f"outputs/P103/validation-v2-{model_id}-001",
            "provenance": model["provenance"],
            "status": model["status"],
            "episodes_ok": model["episodes_ok"],
            "episodes_total": 200,
            "performance_score": model["performance_score"],
            "basic_mean_coverage_rate": model["basic_mean_coverage_rate"],
            "basic_mean_collision_rate": model["basic_mean_collision_rate"],
            "cooperation_mean_coverage_rate": model["cooperation_mean_coverage_rate"],
            "cooperation_mean_collision_rate": model["cooperation_mean_collision_rate"],
            "notes": "A5 same fixed development v2; historical model recheck; no retraining; final confirmation untouched",
            "train_case": original["train_case"],
            "training_seed": original["training_seed"],
            "training_steps": original["training_steps"],
            "learning_rate": original["learning_rate"],
            "wall_seconds": original["wall_seconds"],
        })
        writer.writerow(entry)
    with CSV_PATH.open("a", encoding="utf-8", newline="") as target:
        target.write(generated.getvalue())
    print("已追加：" + ", ".join(f"A5-{model_id}" for model_id in IDS))


if __name__ == "__main__":
    main()
