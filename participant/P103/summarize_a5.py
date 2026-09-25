"""核对 A5 三次开发验证 v2 评测，并给出同场景配对差值。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
IDS = ("E001", "E008", "E009")


def load_run(model_id, expected_hash):
    directory = OUTPUTS_DIR / f"validation-v2-{model_id}-001"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["provenance"] != "local_preview" or result["errors"]:
        raise ValueError(f"{model_id} 评测状态不合格")
    if result["artifact_hashes"]["artifacts/policy.npz"] != expected_hash:
        raise ValueError(f"{model_id} 评测权重摘要与历史记录不一致")
    if result["suite_id"] != "p103-validation-v2" or result["seed_schedule_id"] != "p103-validation-seeds-v2":
        raise ValueError(f"{model_id} 没有使用固定开发验证 v2")
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 200:
        raise ValueError(f"{model_id} 逐回合记录数不是 200")
    episodes = {}
    for row in rows:
        key = (row["group_id"], row["case_id"], int(row["repeat_index"]))
        if key in episodes or row["status"] != "ok" or row["accepted"] != "True" or int(row["steps_completed"]) != 10:
            raise ValueError(f"{model_id} 存在重复、失败或不完整回合：{key}")
        episodes[key] = float(row["mean_j"])
    groups = result["group_metrics"]
    if any(groups[name]["num_scenarios"] != 50 or groups[name]["num_episodes"] != 100 for name in ("basic", "cooperation")):
        raise ValueError(f"{model_id} 分组数量不符合 A5 设计")
    computed_score = 500 * (groups["basic"]["mean_j"] + groups["cooperation"]["mean_j"])
    if not np.isclose(result["performance_score"], computed_score, atol=1e-9):
        raise ValueError(f"{model_id} 汇总分与官方两组等权公式不一致")
    return result, episodes


def paired_difference(left, right, seed):
    # 每个场景先平均两次重复，再在两份模型的同一场景上做差。
    group_diffs = {}
    wins = ties = losses = 0
    for group, prefix in (("basic", "v2b"), ("cooperation", "v2c")):
        diffs = []
        for index in range(50):
            case = f"{prefix}-{index:02d}"
            values = [left[(group, case, repeat)] - right[(group, case, repeat)] for repeat in (0, 1)]
            delta = float(np.mean(values))
            diffs.append(delta)
            wins += delta > 1e-12
            losses += delta < -1e-12
            ties += abs(delta) <= 1e-12
        group_diffs[group] = np.asarray(diffs)
    point = 500 * sum(float(values.mean()) for values in group_diffs.values())
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(10000)
    for values in group_diffs.values():
        indices = rng.integers(0, 50, size=(10000, 50))
        bootstrap += 500 * values[indices].mean(axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "score_difference": point,
        "paired_scene_bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "scene_wins": int(wins),
        "scene_ties": int(ties),
        "scene_losses": int(losses),
        "bootstrap_resamples": 10000,
        "bootstrap_seed": seed,
    }


def main(output):
    output = (output if output.is_absolute() else REPO_ROOT / output).resolve()
    if not output.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("A5 汇总目录必须位于 outputs/P103 下")
    if output.exists():
        raise FileExistsError(f"汇总目录已存在，不覆盖：{output}")
    staged = json.loads((OUTPUTS_DIR / "a5-submissions-001" / "stage-manifest.json").read_text(encoding="utf-8"))
    run_data = {model_id: load_run(model_id, staged[model_id]["model_sha256"]) for model_id in IDS}
    suite_hashes = {result["suite_hash"] for result, _ in run_data.values()}
    if len(suite_hashes) != 1:
        raise ValueError("三份模型使用的 suite 文件摘要不同")
    if len({entry["entry_sha256"] for entry in staged.values()}) != 1:
        raise ValueError("三份模型的推理入口不同")
    episode_keys = [set(episodes) for _, episodes in run_data.values()]
    if not all(keys == episode_keys[0] for keys in episode_keys):
        raise ValueError("三份模型的逐回合场景或重复次数不同")

    models = {}
    for model_id, (result, _) in run_data.items():
        basic, cooperation = result["group_metrics"]["basic"], result["group_metrics"]["cooperation"]
        models[model_id] = {
            "run_id": result["run_id"],
            "model_sha256": staged[model_id]["model_sha256"],
            "origin_commit": staged[model_id]["origin_commit"],
            "provenance": result["provenance"],
            "status": result["status"],
            "episodes_ok": len(run_data[model_id][1]),
            "performance_score": result["performance_score"],
            "performance_score_ci95": result["performance_score_ci95"],
            "basic_mean_coverage_rate": basic["mean_coverage_rate"],
            "basic_mean_collision_rate": basic["mean_collision_rate"],
            "cooperation_mean_coverage_rate": cooperation["mean_coverage_rate"],
            "cooperation_mean_collision_rate": cooperation["mean_collision_rate"],
            "result_path": f"outputs/P103/validation-v2-{model_id}-001/result.json",
        }
    comparisons = {}
    for opponent in ("E001", "E008"):
        comparison = paired_difference(run_data["E009"][1], run_data[opponent][1], 20260925)
        if not np.isclose(comparison["score_difference"], models["E009"]["performance_score"] - models[opponent]["performance_score"], atol=1e-9):
            raise ValueError("配对逐场景差值与官方汇总分差不一致")
        comparisons[f"E009_minus_{opponent}"] = comparison
    summary = {
        "suite_id": "p103-validation-v2",
        "suite_hash": next(iter(suite_hashes)),
        "seed_schedule_id": "p103-validation-seeds-v2",
        "entry_sha256": staged["E009"]["entry_sha256"],
        "models": models,
        "comparisons": comparisons,
        "note": "Windows local_preview；配对区间仅描述本开发集上的采样波动，不能替代官方核验或独立最终确认集",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["model_id", "run_id", "model_sha256", "performance_score", "basic_mean_coverage_rate", "basic_mean_collision_rate", "cooperation_mean_coverage_rate", "cooperation_mean_collision_rate"])
        writer.writeheader()
        for model_id, data in models.items():
            writer.writerow({field: model_id if field == "model_id" else data[field] for field in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
