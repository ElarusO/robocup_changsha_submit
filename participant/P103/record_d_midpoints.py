"""追加六个 D 完整训练中间检查点的真实本地评测记录。"""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
TABLE = HERE / "experiments.csv"


def main():
    with TABLE.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        existing = {item["experiment_id"] for item in reader}
    records = []
    for i, experiment_id in enumerate(("E026", "E027", "E028")):
        seed = 7101 + i
        for step in (313344, 626688):
            label = f"D-{experiment_id}-{step}"
            if label in existing:
                raise ValueError(f"记录已存在：{label}")
            path = OUT / f"d-param-mid-v2-{experiment_id}-{step}"
            result = json.loads((path / "result.json").read_text(encoding="utf-8"))
            if result["status"] != "ok" or result["errors"] or sum(item["num_episodes"] for item in result["group_metrics"].values()) != 200:
                raise ValueError(f"评测不完整：{path}")
            basic, coop = result["group_metrics"]["basic"], result["group_metrics"]["cooperation"]
            row = dict.fromkeys(fields, "")
            row.update({"experiment_id": label, "date": "2026-09-26", "kind": "d_parameter_midpoint",
                "source_commit": result["source_commit"], "model_sha256": result["artifact_hashes"]["artifacts/policy.npz"],
                "suite_id": result["suite_id"], "seed_schedule_id": result["seed_schedule_id"],
                "run_id": result["run_id"], "output_dir": path.relative_to(ROOT).as_posix(),
                "provenance": result["provenance"], "status": result["status"],
                "episodes_ok": "200", "episodes_total": "200", "performance_score": result["performance_score"],
                "basic_mean_coverage_rate": basic["mean_coverage_rate"], "basic_mean_collision_rate": basic["mean_collision_rate"],
                "cooperation_mean_coverage_rate": coop["mean_coverage_rate"], "cooperation_mean_collision_rate": coop["mean_collision_rate"],
                "notes": f"D full-training checkpoint; lr=0.0002, ent=0; source={experiment_id}; E009 unchanged; A4 unused",
                "train_case": "mixed-basic0-coop0", "training_seed": seed, "training_steps": step, "learning_rate": .0002})
            records.append(row)
    with TABLE.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerows(records)
    print(f"追加 {len(records)} 条中间检查点记录")


if __name__ == "__main__":
    main()
