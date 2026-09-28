"""将 D 新场景、组合和有界调参的完成结果追加到实验总表。"""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
TABLE = HERE / "experiments.csv"


def row_for(fields, label, kind, result_dir, notes, train_case, seed, steps, learning_rate, wall_seconds=""):
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["errors"] or sum(item["num_episodes"] for item in result["group_metrics"].values()) != 200:
        raise ValueError(f"评测不完整：{result_dir}")
    basic, coop = result["group_metrics"]["basic"], result["group_metrics"]["cooperation"]
    row = dict.fromkeys(fields, "")
    row.update({"experiment_id": label, "date": "2026-09-26", "kind": kind,
        "source_commit": result["source_commit"], "model_sha256": result["artifact_hashes"]["artifacts/policy.npz"],
        "suite_id": result["suite_id"], "seed_schedule_id": result["seed_schedule_id"],
        "run_id": result["run_id"], "output_dir": result_dir.relative_to(ROOT).as_posix(),
        "provenance": result["provenance"], "status": result["status"], "episodes_ok": "200", "episodes_total": "200",
        "performance_score": result["performance_score"],
        "basic_mean_coverage_rate": basic["mean_coverage_rate"], "basic_mean_collision_rate": basic["mean_collision_rate"],
        "cooperation_mean_coverage_rate": coop["mean_coverage_rate"], "cooperation_mean_collision_rate": coop["mean_collision_rate"],
        "notes": notes, "train_case": train_case, "training_seed": seed, "training_steps": steps,
        "learning_rate": learning_rate, "wall_seconds": wall_seconds})
    return row


def main():
    progress = json.loads((OUT / "d-parameter-control-001" / "progress.json").read_text(encoding="utf-8"))
    if "decision" not in progress:
        raise ValueError("D 参数自动流程尚未形成停止决定")
    with TABLE.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        existing = {row["experiment_id"] for row in reader}
    records = []
    for experiment_id, seed in (("E014", 7101), ("E015", 7102), ("E016", 7103)):
        for mode in ("base", "target", "avoid", "combo"):
            label = f"D-V3-{experiment_id}-{mode}"
            records.append(row_for(fields, label, "d_inference_new_scene", OUT / f"d-v3-{experiment_id}-{mode}",
                f"v3 new development scenes; {mode} inference; A4 unused; E009 unchanged", "mixed-basic0-coop0", seed, 1044480, .0001))
        label = f"D-V2-{experiment_id}-combo"
        records.append(row_for(fields, label, "d_inference_combo", OUT / f"d-v2-{experiment_id}-combo",
            "v2 combo inference; target then avoidance; A4 unused; E009 unchanged", "mixed-basic0-coop0", seed, 1044480, .0001))
    for experiment_id, info in progress["trials"].items():
        records.append(row_for(fields, experiment_id, "d_parameter_train", OUT / f"d-param-v2-{experiment_id}",
            f"D sequential parameter training; ent_coef={info['ent_coef']}; protocol_sha256={progress['protocol_sha256']}; E009 unchanged; A4 unused",
            "mixed-basic0-coop0", info["seed"], info["actual_steps"], info["learning_rate"], info["elapsed_seconds"]))
    if any(item["experiment_id"] in existing for item in records):
        raise ValueError("实验总表已有同名记录，停止追加")
    with TABLE.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerows(records)
    print(f"追加 {len(records)} 条 D 扩展记录")


if __name__ == "__main__":
    main()
