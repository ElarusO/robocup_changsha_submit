"""在固定开发集 v2/v3 上，按相同场景比较 E009 与 E015 候选。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "outputs" / "P103"
SOURCES = {
    "v2": {
        "E009": "validation-v2-E009-001",
        "E015-base": "c-v2-E015-1044480",
        "E015-target": "d-v2-E015-target",
        "E015-combo": "d-v2-E015-combo",
    },
    "v3": {
        "E009": "e2-v3-E009-001",
        "E015-base": "d-v3-E015-base",
        "E015-target": "d-v3-E015-target",
        "E015-combo": "d-v3-E015-combo",
    },
}
COMPARISONS = (
    ("E015-base", "E009"),
    ("E015-target", "E009"),
    ("E015-combo", "E009"),
    ("E015-target", "E015-base"),
    ("E015-combo", "E015-target"),
)
GROUPS = ("basic", "cooperation")
RESAMPLES = 10000
BOOTSTRAP_SEED = 20260928


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluation(folder):
    result_file = folder / "result.json"
    episode_file = folder / "episodes.csv"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    with episode_file.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if result["status"] != "ok" or result["errors"] or result["provenance"] != "local_preview":
        raise ValueError(f"运行状态或来源不合预期：{folder}")
    if len(records) != 200 or file_hash(episode_file) != result["episode_records_hash"]:
        raise ValueError(f"回合数或原始记录摘要错误：{folder}")
    if any(row["status"] != "ok" or row["accepted"] != "True" or row["attempt_index"] != "0"
           for row in records):
        raise ValueError(f"存在失败回合：{folder}")
    episodes = {}
    for row in records:
        key = (row["group_id"], row["case_id"], int(row["repeat_index"]))
        if key in episodes:
            raise ValueError(f"重复的场景键：{folder}: {key}")
        episodes[key] = {field: float(row[field]) for field in
                         ("mean_j", "mean_coverage_rate", "mean_collision_rate")}
    if len(episodes) != 200 or sum(item["num_episodes"] for item in result["group_metrics"].values()) != 200:
        raise ValueError(f"结果不完整：{folder}")
    for group in GROUPS:
        group_rows = [value for (group_id, _, _), value in episodes.items() if group_id == group]
        if len(group_rows) != 100:
            raise ValueError(f"分组回合数错误：{folder}: {group}")
        for field, reported in (("mean_j", "mean_j"),
                                ("mean_coverage_rate", "mean_coverage_rate"),
                                ("mean_collision_rate", "mean_collision_rate")):
            if not np.isclose(np.mean([row[field] for row in group_rows]),
                              result["group_metrics"][group][reported], atol=1e-10):
                raise ValueError(f"逐回合与汇总值不符：{folder}: {group}/{field}")
    calculated_score = 500 * sum(result["group_metrics"][group]["mean_j"] for group in GROUPS)
    if not np.isclose(calculated_score, result["performance_score"], atol=1e-8):
        raise ValueError(f"分组分与总分不符：{folder}")
    return result, episodes, {
        "result_json": (folder / "result.json").relative_to(ROOT).as_posix(),
        "result_sha256": file_hash(result_file),
        "episodes_csv": episode_file.relative_to(ROOT).as_posix(),
        "episodes_sha256": file_hash(episode_file),
    }


def scene_means(episodes):
    grouped = {}
    for (group, case, repeat), fields in episodes.items():
        grouped.setdefault((group, case), {})[repeat] = fields
    scenes = {}
    for key, repeats in grouped.items():
        if set(repeats) != {0, 1}:
            raise ValueError(f"场景重复安排不完整：{key}")
        scenes[key] = {field: (repeats[0][field] + repeats[1][field]) / 2
                       for field in ("mean_j", "mean_coverage_rate", "mean_collision_rate")}
    if len(scenes) != 100 or any(sum(key[0] == group for key in scenes) != 50 for group in GROUPS):
        raise ValueError("场景数不是基础与协作各 50 个")
    return scenes


def main(output):
    output = output.resolve()
    if output.parent != OUTPUTS.resolve() or output.exists():
        raise ValueError("输出必须是 outputs/P103/ 下尚不存在的直接子目录")

    evaluations = {}
    inputs = {}
    for suite, versions in SOURCES.items():
        evaluations[suite] = {}
        inputs[suite] = {}
        for name, folder in versions.items():
            result, episodes, paths = load_evaluation(OUTPUTS / folder)
            evaluations[suite][name] = (result, episodes, scene_means(episodes))
            inputs[suite][name] = {**paths, "run_id": result["run_id"],
                                    "source_commit": result["source_commit"],
                                    "code_tree_hash": result["code_tree_hash"],
                                    "inference_lock_hash": result["inference_lock_hash"],
                                    "model_sha256": result["artifact_hashes"]["artifacts/policy.npz"],
                                    "suite_id": result["suite_id"],
                                    "suite_hash": result["suite_hash"],
                                    "seed_schedule_id": result["seed_schedule_id"],
                                    "performance_score": result["performance_score"]}
        references = [item[0] for item in evaluations[suite].values()]
        for field in ("suite_id", "suite_hash", "seed_schedule_id", "task_version", "score_version",
                      "inference_lock_hash"):
            if len({item[field] for item in references}) != 1:
                raise ValueError(f"不是同一套评测题和计分口径：{suite}/{field}")
        keys = [set(item[1]) for item in evaluations[suite].values()]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError(f"逐回合场景键不一致：{suite}")
        model_hashes = {inputs[suite][name]["model_sha256"] for name in SOURCES[suite] if name.startswith("E015")}
        if len(model_hashes) != 1:
            raise ValueError(f"E015 变体的 PPO 权重不同：{suite}")
    for name in SOURCES["v2"]:
        if inputs["v2"][name]["model_sha256"] != inputs["v3"][name]["model_sha256"]:
            raise ValueError(f"两套题上模型权重不同：{name}")
        if name.startswith("E015") and inputs["v2"][name]["code_tree_hash"] != inputs["v3"][name]["code_tree_hash"]:
            raise ValueError(f"E015 两套题上的推理代码不同：{name}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampling = {(suite, group): rng.integers(0, 50, size=(RESAMPLES, 50))
                  for suite in SOURCES for group in GROUPS}
    rows = []
    summaries = {}
    for suite in SOURCES:
        summaries[suite] = {}
        for newer, older in COMPARISONS:
            new_result, _, new_scenes = evaluations[suite][newer]
            old_result, _, old_scenes = evaluations[suite][older]
            if set(new_scenes) != set(old_scenes):
                raise ValueError(f"场景不一致：{suite}/{newer}/{older}")
            arrays = {}
            group_changes = {}
            for group in GROUPS:
                cases = sorted(case for group_id, case in new_scenes if group_id == group)
                arrays[group] = np.asarray([new_scenes[(group, case)]["mean_j"] -
                                            old_scenes[(group, case)]["mean_j"] for case in cases])
                for case in cases:
                    n, o = new_scenes[(group, case)], old_scenes[(group, case)]
                    rows.append({"suite": suite, "new": newer, "old": older, "group": group,
                                 "case_id": case, "new_mean_j": n["mean_j"], "old_mean_j": o["mean_j"],
                                 "delta_mean_j": n["mean_j"] - o["mean_j"],
                                 "delta_coverage": n["mean_coverage_rate"] - o["mean_coverage_rate"],
                                 "delta_collision": n["mean_collision_rate"] - o["mean_collision_rate"]})
                group_changes[group] = {
                    "mean_j": new_result["group_metrics"][group]["mean_j"] - old_result["group_metrics"][group]["mean_j"],
                    "coverage": new_result["group_metrics"][group]["mean_coverage_rate"] - old_result["group_metrics"][group]["mean_coverage_rate"],
                    "collision": new_result["group_metrics"][group]["mean_collision_rate"] - old_result["group_metrics"][group]["mean_collision_rate"],
                    "scene_wins": int(np.count_nonzero(arrays[group] > 1e-12)),
                    "scene_losses": int(np.count_nonzero(arrays[group] < -1e-12)),
                    "scene_ties": int(np.count_nonzero(np.abs(arrays[group]) <= 1e-12)),
                }
            point = 500 * sum(arrays[group].mean() for group in GROUPS)
            if not np.isclose(point, new_result["performance_score"] - old_result["performance_score"], atol=1e-8):
                raise ValueError(f"逐场景分差与汇总不一致：{suite}/{newer}/{older}")
            sampled = 500 * sum(arrays[group][resampling[(suite, group)]].mean(axis=1) for group in GROUPS)
            summaries[suite][f"{newer}_vs_{older}"] = {
                "score_difference": float(point),
                "fixed_model_paired_scene_ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
                "group_changes": group_changes,
            }

    combined = {}
    for newer, older in COMPARISONS:
        sampled_by_suite = []
        for suite in SOURCES:
            newer_scenes = evaluations[suite][newer][2]
            older_scenes = evaluations[suite][older][2]
            group_arrays = {}
            for group in GROUPS:
                cases = sorted(case for group_id, case in newer_scenes if group_id == group)
                group_arrays[group] = np.asarray([newer_scenes[(group, case)]["mean_j"] -
                                                  older_scenes[(group, case)]["mean_j"] for case in cases])
            sampled_by_suite.append(500 * sum(group_arrays[group][resampling[(suite, group)]].mean(axis=1)
                                              for group in GROUPS))
        sample_average = np.mean(np.stack(sampled_by_suite), axis=0)
        combined[f"{newer}_vs_{older}"] = {
            "equal_suite_mean_score_difference": float(np.mean([summaries[suite][f"{newer}_vs_{older}"]["score_difference"] for suite in SOURCES])),
            "fixed_model_paired_scene_ci95": [float(x) for x in np.quantile(sample_average, [0.025, 0.975])],
        }

    output.mkdir()
    with (output / "paired-scenes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"method": "same suite, same scene and repeat; average two repeats per scene; 50 distinct scenes per group; score = 500 * (basic mean_j + cooperation mean_j)",
              "bootstrap": {"resamples": RESAMPLES, "seed": BOOTSTRAP_SEED,
                            "note": "Fixed trained models and already-used development suites; intervals describe scene variation only, not training-seed uncertainty, selection bias or official hidden-test performance."},
              "inputs": inputs, "by_suite": summaries, "equal_suite_summary": combined,
              "a4_used": False}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "scores": {suite: {name: item[0]["performance_score"]
                       for name, item in models.items()} for suite, models in evaluations.items()},
                      "equal_suite_summary": combined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "e2-summary-001")
    args = parser.parse_args()
    main(args.output)
