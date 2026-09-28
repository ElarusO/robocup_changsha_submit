"""对新场景 v3 的原动作、单因素及组合动作做成对汇总。"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "P103"
DEST = OUT / "d-v3-summary-001"
IDS = ("E014", "E015", "E016")
MODES = ("base", "target", "avoid", "combo")


def load(experiment_id, mode):
    folder = OUT / f"d-v3-{experiment_id}-{mode}"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    with (folder / "episodes.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    data = {(row["group_id"], row["case_id"], int(row["repeat_index"])): float(row["mean_j"]) for row in rows}
    if result["status"] != "ok" or len(rows) != 200 or len(data) != 200:
        raise ValueError(f"评测不完整：{folder}")
    return result, data


def paired_scene_difference(new, old):
    by_group = {}
    for group, prefix in (("basic", "v3b"), ("cooperation", "v3c")):
        by_group[group] = np.asarray([
            np.mean([new[(group, f"{prefix}-{i:02d}", repeat)] - old[(group, f"{prefix}-{i:02d}", repeat)] for repeat in (0, 1)])
            for i in range(50)
        ])
    return by_group


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    results = {model: {mode: load(model, mode) for mode in MODES} for model in IDS}
    rows = []
    for model in IDS:
        for mode in MODES:
            result = results[model][mode][0]
            baseline = results[model]["base"][0]
            target = results[model]["target"][0]
            rows.append({"model": model, "mode": mode, "score": result["performance_score"],
                "delta_vs_base": result["performance_score"] - baseline["performance_score"],
                "delta_vs_target": result["performance_score"] - target["performance_score"],
                "basic_coverage": result["group_metrics"]["basic"]["mean_coverage_rate"],
                "coop_coverage": result["group_metrics"]["cooperation"]["mean_coverage_rate"],
                "coop_collision": result["group_metrics"]["cooperation"]["mean_collision_rate"]})
    comparisons = (("target", "base"), ("avoid", "base"), ("combo", "target"))
    rng = np.random.default_rng(20260926)
    indices = {group: rng.integers(0, 50, size=(10000, 50)) for group in ("basic", "cooperation")}
    intervals = {}
    for new_mode, old_mode in comparisons:
        arrays = {model: paired_scene_difference(results[model][new_mode][1], results[model][old_mode][1]) for model in IDS}
        sampled = np.mean(np.stack([500 * sum(array[group][indices[group]].mean(axis=1) for group in indices)
                                    for array in arrays.values()]), axis=0)
        point = np.mean([results[model][new_mode][0]["performance_score"] - results[model][old_mode][0]["performance_score"] for model in IDS])
        intervals[f"{new_mode}_vs_{old_mode}"] = {"mean_paired_score_difference": float(point),
            "fixed_model_scene_bootstrap_ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "note": "Fixed three trained models; paired resampling of 50 distinct scenes per group, not training-seed uncertainty."}
    DEST.mkdir(parents=True)
    with (DEST / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DEST / "summary.json").write_text(json.dumps({"scores": rows, "intervals": intervals,
        "suite": "p103-validation-v3", "note": "New v3 scenes are now used for development comparison, not a final holdout."},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(intervals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
