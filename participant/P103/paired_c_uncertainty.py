"""按同一开发场景配对计算 C 与 B3 最终模型的描述性重采样区间。"""

import csv
import json
from pathlib import Path

import numpy as np


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
FINAL_STEPS = 1044480
PAIRS = ((7101, "E011", "E014"), (7102, "E012", "E015"), (7103, "E013", "E016"))


def episodes(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 200 or any(row["status"] != "ok" for row in rows):
        raise ValueError(f"评测回合不完整：{path}")
    result = {(row["group_id"], row["case_id"], int(row["repeat_index"])): float(row["mean_j"]) for row in rows}
    if len(result) != 200:
        raise ValueError(f"评测回合键重复：{path}")
    return result


def main():
    output = OUTPUTS_DIR / "c-summary-001" / "paired-scene-bootstrap.json"
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((OUTPUTS_DIR / "c-summary-001" / "summary.json").read_text(encoding="utf-8"))
    arrays = {}
    points = {}
    for seed, b_id, c_id in PAIRS:
        b = episodes(OUTPUTS_DIR / f"b3-v2-{b_id}-{FINAL_STEPS}" / "episodes.csv")
        c = episodes(OUTPUTS_DIR / f"c-v2-{c_id}-{FINAL_STEPS}" / "episodes.csv")
        if set(b) != set(c):
            raise ValueError(f"种子 {seed} 的 B/C 场景键不同")
        by_group = {}
        for group, prefix in (("basic", "v2b"), ("cooperation", "v2c")):
            by_group[group] = np.array([
                np.mean([c[(group, f"{prefix}-{index:02d}", repeat)] - b[(group, f"{prefix}-{index:02d}", repeat)] for repeat in (0, 1)])
                for index in range(50)
            ], dtype=np.float64)
        point = 500 * (by_group["basic"].mean() + by_group["cooperation"].mean())
        expected = next(item["score_difference"] for item in summary["paired_differences"] if item["training_seed"] == seed and item["checkpoint_steps"] == FINAL_STEPS)
        if not np.isclose(point, expected, atol=1e-9):
            raise ValueError(f"种子 {seed} 的逐场景分差与官方汇总不同")
        arrays[seed] = by_group
        points[seed] = float(point)
    rng = np.random.default_rng(20260926)
    indices = {group: rng.integers(0, 50, size=(10000, 50)) for group in ("basic", "cooperation")}
    simulated = {}
    for seed, values in arrays.items():
        simulated[seed] = 500 * sum(values[group][indices[group]].mean(axis=1) for group in indices)
    average = np.mean(np.stack(list(simulated.values())), axis=0)
    result = {
        "fixed_training_seeds": [seed for seed, _, _ in PAIRS],
        "paired_score_differences": points,
        "mean_paired_difference": float(np.mean(list(points.values()))),
        "per_seed_scene_bootstrap_ci95": {str(seed): [float(x) for x in np.quantile(values, [0.025, 0.975])] for seed, values in simulated.items()},
        "mean_scene_bootstrap_ci95": [float(x) for x in np.quantile(average, [0.025, 0.975])],
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260926,
        "note": "固定三种训练种子，只对相同开发场景成对重采样；区间描述此开发集的场景波动，不包含训练种子总体不确定性，也不是官方成绩。",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
