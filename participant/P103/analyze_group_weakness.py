"""读取 B3/C 最终回合记录，按相同场景分析覆盖与碰撞的薄弱处。"""

import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
PAIRS = ((7101, "E011", "E014"), (7102, "E012", "E015"), (7103, "E013", "E016"))
GROUPS = (("basic", "v2b"), ("cooperation", "v2c"))
STEP = 1044480


def load(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 200:
        raise ValueError(f"回合数不是 200：{path}")
    data = {}
    for row in rows:
        key = (row["group_id"], row["case_id"], int(row["repeat_index"]))
        if key in data or row["status"] != "ok" or row["accepted"] != "True" or int(row["steps_completed"]) != 10:
            raise ValueError(f"回合无效：{path} {key}")
        data[key] = {
            "coverage": float(row["mean_coverage_rate"]),
            "collision": float(row["mean_collision_rate"]),
            "pair_steps": int(row["collision_pairs_sum"]),
            "j": float(row["mean_j"]),
        }
    return data


def ci(values, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def main():
    output = OUTPUTS_DIR / "c-group-diagnostics-001"
    if output.exists():
        raise FileExistsError(output)
    seed_results = {}
    all_deltas = {group: {metric: [] for metric in ("coverage", "collision", "j")} for group, _ in GROUPS}
    flat_rows = []
    for seed, b_id, c_id in PAIRS:
        b = load(OUTPUTS_DIR / f"b3-v2-{b_id}-{STEP}" / "episodes.csv")
        c = load(OUTPUTS_DIR / f"c-v2-{c_id}-{STEP}" / "episodes.csv")
        if set(b) != set(c):
            raise ValueError(f"种子 {seed} 的回合键不同")
        seed_results[str(seed)] = {}
        for group, prefix in GROUPS:
            cases = []
            for index in range(50):
                case_id = f"{prefix}-{index:02d}"
                b_rows = [b[(group, case_id, repeat)] for repeat in (0, 1)]
                c_rows = [c[(group, case_id, repeat)] for repeat in (0, 1)]
                item = {
                    "training_seed": seed,
                    "group": group,
                    "case_id": case_id,
                    "coverage_delta": float(np.mean([new["coverage"] - old["coverage"] for old, new in zip(b_rows, c_rows)])),
                    "collision_delta": float(np.mean([new["collision"] - old["collision"] for old, new in zip(b_rows, c_rows)])),
                    "j_delta": float(np.mean([new["j"] - old["j"] for old, new in zip(b_rows, c_rows)])),
                    "b_pair_steps": sum(row["pair_steps"] for row in b_rows),
                    "c_pair_steps": sum(row["pair_steps"] for row in c_rows),
                    "b_collision_episodes": sum(row["pair_steps"] > 0 for row in b_rows),
                    "c_collision_episodes": sum(row["pair_steps"] > 0 for row in c_rows),
                }
                cases.append(item)
                flat_rows.append(item)
            coverage = np.array([item["coverage_delta"] for item in cases])
            collision = np.array([item["collision_delta"] for item in cases])
            j = np.array([item["j_delta"] for item in cases])
            for name, values in (("coverage", coverage), ("collision", collision), ("j", j)):
                all_deltas[group][name].append(values)
            seed_results[str(seed)][group] = {
                "coverage_delta": float(coverage.mean()),
                "coverage_scene_ci95": ci(coverage, seed + 10),
                "coverage_scene_wins": int(np.sum(coverage > 1e-12)),
                "coverage_scene_losses": int(np.sum(coverage < -1e-12)),
                "coverage_scene_ties": int(np.sum(np.abs(coverage) <= 1e-12)),
                "collision_delta": float(collision.mean()),
                "collision_scene_ci95": ci(collision, seed + 20),
                "j_delta": float(j.mean()),
                "b_collision_pair_steps": int(sum(item["b_pair_steps"] for item in cases)),
                "c_collision_pair_steps": int(sum(item["c_pair_steps"] for item in cases)),
                "b_collision_episodes": int(sum(item["b_collision_episodes"] for item in cases)),
                "c_collision_episodes": int(sum(item["c_collision_episodes"] for item in cases)),
                "collision_cases_c": [item for item in cases if item["c_pair_steps"] > 0],
                "worst_coverage_cases": sorted(cases, key=lambda item: item["coverage_delta"])[:5],
                "best_coverage_cases": sorted(cases, key=lambda item: item["coverage_delta"], reverse=True)[:5],
            }
    across_seeds = {}
    for group, _ in GROUPS:
        across_seeds[group] = {}
        for metric in ("coverage", "collision", "j"):
            averaged_by_case = np.mean(np.stack(all_deltas[group][metric]), axis=0)
            across_seeds[group][f"{metric}_mean_delta"] = float(averaged_by_case.mean())
            across_seeds[group][f"{metric}_scene_ci95"] = ci(averaged_by_case, 20260926 + len(group) + len(metric))
    result = {
        "comparison": "C final E014–E016 minus paired B3 final E011–E013 on identical development v2 cases",
        "steps": STEP,
        "collision_definition": "每步机器人中心距离小于两半径之和；collision rate 为参与碰撞机器人数量 / 3；pair_steps 为 10 步内碰撞机器人对的出现次数之和",
        "score_decomposition": "组 mean_j = coverage - 0.2 × collision；总分 = 500 × (basic mean_j + cooperation mean_j)",
        "by_seed": seed_results,
        "across_three_fixed_seeds": across_seeds,
        "note": "50 个开发场景的配对重采样区间仅描述固定训练种子下的场景差异，不证明碰撞原因或对未见场景的普遍性。",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({
        "by_seed": {seed: {group: {key: value for key, value in metrics.items() if key not in ("collision_cases_c", "worst_coverage_cases", "best_coverage_cases")} for group, metrics in groups.items()} for seed, groups in seed_results.items()},
        "across_three_fixed_seeds": across_seeds,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
