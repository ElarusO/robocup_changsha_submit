"""把已核对的 D 阶段本地单因素评测写入 P103 实验总表。"""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
TABLE = HERE / "experiments.csv"


def main():
    summary = json.loads((OUT / "d-single-factor-summary-001" / "summary.json").read_text(encoding="utf-8"))
    with TABLE.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        existing = list(reader)
    ids = {row["experiment_id"] for row in existing}
    new = []
    for item in summary["paired_model_results"]:
        name = f"D-{item['model']}-{item['mode']}"
        if name in ids:
            raise ValueError(f"记录已存在，不重复写入：{name}")
        result_dir = OUT / f"d-v2-{item['model']}-{item['mode']}"
        result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "ok":
            raise ValueError(f"结果未完成：{name}")
        record = dict.fromkeys(fields, "")
        record.update({
            "experiment_id": name, "date": "2026-09-26", "kind": "single_factor_inference",
            "source_commit": result["source_commit"],
            "model_sha256": result["artifact_hashes"]["artifacts/policy.npz"],
            "suite_id": result["suite_id"], "seed_schedule_id": result["seed_schedule_id"],
            "run_id": result["run_id"], "output_dir": result_dir.relative_to(ROOT).as_posix(),
            "provenance": result["provenance"], "status": result["status"],
            "episodes_ok": "200", "episodes_total": "200",
            "performance_score": result["performance_score"],
            "basic_mean_coverage_rate": item["trial_basic_coverage"],
            "basic_mean_collision_rate": item["trial_basic_collision"],
            "cooperation_mean_coverage_rate": item["trial_coop_coverage"],
            "cooperation_mean_collision_rate": item["trial_coop_collision"],
            "notes": f"D inference-only {item['mode']}; C baseline {item['model']}; protocol_sha256={summary['protocol_sha256']}; E009 unchanged; A4 unused",
            "train_case": "mixed-basic0-coop0", "training_steps": "1044480", "learning_rate": "0.0001",
            "training_seed": {"E014": "7101", "E015": "7102", "E016": "7103"}[item["model"]],
        })
        new.append(record)
    with TABLE.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerows(new)
    print(f"新增 {len(new)} 条记录")


if __name__ == "__main__":
    main()
