"""按运行前冻结的门槛，成对汇总 E3 与旧目标朝向策略。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
PROTOCOL = HERE / "e3-conditional-target-protocol-v1.json"
PROTOCOL_SHA = "8f237872840831559c8682d0d567e9b2054d5d440b12ea6877c9d58363251bf9"
DEST = OUT / "e3-summary-001"
MODELS = ("E014", "E015", "E016")
SUITES = ("v2", "v3")
GROUPS = ("basic", "cooperation")
RESAMPLES = 10000
BOOTSTRAP_SEED = 20260928


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder, expected_model_hash, suite):
    result_path = folder / "result.json"
    episode_path = folder / "episodes.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with episode_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (result["status"] != "ok" or result["errors"] or result["provenance"] != "local_preview" or
            result["suite_id"] != f"p103-validation-{suite}" or
            result["seed_schedule_id"] != f"p103-validation-seeds-{suite}" or
            result["artifact_hashes"]["artifacts/policy.npz"] != expected_model_hash or
            len(rows) != 200 or result["episode_records_hash"] != sha(episode_path) or
            any(row["status"] != "ok" for row in rows)):
        raise ValueError(f"结果文件不完整或来源不符：{folder}")
    episodes = {}
    for row in rows:
        key = (row["group_id"], row["case_id"], int(row["repeat_index"]))
        if key in episodes:
            raise ValueError(f"场景重复：{folder}/{key}")
        episodes[key] = {field: float(row[field]) for field in
                         ("mean_j", "mean_coverage_rate", "mean_collision_rate")}
    if len(episodes) != 200:
        raise ValueError(f"缺少场景：{folder}")
    return result, episodes, {"result_path": result_path.relative_to(ROOT).as_posix(),
                              "result_sha256": sha(result_path),
                              "episodes_path": episode_path.relative_to(ROOT).as_posix(),
                              "episodes_sha256": sha(episode_path)}


def main():
    if sha(PROTOCOL) != PROTOCOL_SHA or DEST.exists():
        raise ValueError("冻结协议变化，或汇总输出已存在")
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fixed = plan["fixed_inputs"]
    data = {}
    sources = {}
    scenes = []
    results = {}
    for suite in SUITES:
        data[suite] = {}
        sources[suite] = {}
        for model in MODELS:
            expected_hash = fixed["models"][model]["sha256"]
            old = load(OUT / f"d-{suite}-{model}-target", expected_hash, suite)
            new = load(OUT / f"e3-{suite}-{model}-target", expected_hash, suite)
            old_result, old_eps, old_source = old
            new_result, new_eps, new_source = new
            if (set(old_eps) != set(new_eps) or
                    any(old_result[field] != new_result[field] for field in
                        ("suite_hash", "seed_schedule_id", "task_version", "score_version", "inference_lock_hash"))):
                raise ValueError(f"不是同题比较：{suite}/{model}")
            if any(sum(key[0] == group for key in old_eps) != 100 for group in GROUPS):
                raise ValueError(f"组内场景数不符：{suite}/{model}")
            data[suite][model] = {"old": old_result, "new": new_result,
                                  "old_eps": old_eps, "new_eps": new_eps}
            sources[suite][model] = {"old": old_source, "new": new_source,
                                     "model_sha256": expected_hash,
                                     "old_code_tree_hash": old_result["code_tree_hash"],
                                     "new_code_tree_hash": new_result["code_tree_hash"]}
            results[f"{suite}-{model}"] = {
                "old_score": old_result["performance_score"],
                "new_score": new_result["performance_score"],
                "score_difference": new_result["performance_score"] - old_result["performance_score"],
                "group_changes": {group: {
                    "coverage": new_result["group_metrics"][group]["mean_coverage_rate"] - old_result["group_metrics"][group]["mean_coverage_rate"],
                    "collision": new_result["group_metrics"][group]["mean_collision_rate"] - old_result["group_metrics"][group]["mean_collision_rate"],
                    "mean_j": new_result["group_metrics"][group]["mean_j"] - old_result["group_metrics"][group]["mean_j"],
                } for group in GROUPS},
            }
            for key in sorted(old_eps):
                old_row, new_row = old_eps[key], new_eps[key]
                scenes.append({"suite": suite, "model": model, "group": key[0],
                               "case_id": key[1], "repeat_index": key[2],
                               "delta_j": new_row["mean_j"] - old_row["mean_j"],
                               "delta_coverage": new_row["mean_coverage_rate"] - old_row["mean_coverage_rate"],
                               "delta_collision": new_row["mean_collision_rate"] - old_row["mean_collision_rate"]})
    for model in MODELS:
        for field in ("old_code_tree_hash", "new_code_tree_hash"):
            if sources["v2"][model][field] != sources["v3"][model][field]:
                raise ValueError(f"两套题上的推理代码摘要不同：{model}/{field}")

    model_equal_deltas = {model: float(np.mean([results[f"{suite}-{model}"]["score_difference"]
                                               for suite in SUITES])) for model in MODELS}
    overall_delta = float(np.mean(list(model_equal_deltas.values())))
    group_mean_changes = {suite: {group: {
        field: float(np.mean([results[f"{suite}-{model}"]["group_changes"][group][field]
                              for model in MODELS])) for field in ("coverage", "collision")
    } for group in GROUPS} for suite in SUITES}
    rule = {
        "score_at_least_3": overall_delta >= 3.0 - 1e-12,
        "at_least_two_models_improve": sum(delta > 0 for delta in model_equal_deltas.values()) >= 2,
        "group_coverage_drop_at_most_0_005": all(
            group_mean_changes[suite][group]["coverage"] >= -0.005 - 1e-12
            for suite in SUITES for group in GROUPS),
        "group_collision_rise_at_most_0_002": all(
            group_mean_changes[suite][group]["collision"] <= 0.002 + 1e-12
            for suite in SUITES for group in GROUPS),
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_by_suite = []
    for suite in SUITES:
        sampled_by_group = []
        for group in GROUPS:
            cases = sorted({key[1] for key in data[suite][MODELS[0]]["old_eps"] if key[0] == group})
            if len(cases) != 50:
                raise ValueError(f"场景数不等于 50：{suite}/{group}")
            model_deltas = []
            for model in MODELS:
                old_eps = data[suite][model]["old_eps"]
                new_eps = data[suite][model]["new_eps"]
                model_deltas.append([np.mean([new_eps[(group, case, repeat)]["mean_j"] -
                                              old_eps[(group, case, repeat)]["mean_j"]
                                              for repeat in (0, 1)]) for case in cases])
            per_scene = np.mean(np.asarray(model_deltas), axis=0)
            sampled_indices = rng.integers(0, 50, size=(RESAMPLES, 50))
            sampled_by_group.append(per_scene[sampled_indices].mean(axis=1))
        sampled_by_suite.append(500 * sum(sampled_by_group))
    overall_samples = np.mean(np.stack(sampled_by_suite), axis=0)
    interval = [float(value) for value in np.quantile(overall_samples, [0.025, 0.975])]

    DEST.mkdir()
    with (DEST / "paired-episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scenes[0]))
        writer.writeheader()
        writer.writerows(scenes)
    report = {"protocol_sha256": PROTOCOL_SHA, "inputs": sources,
              "per_suite_model": results, "model_equal_suite_score_differences": model_equal_deltas,
              "equal_model_equal_suite_mean_score_difference": overall_delta,
              "fixed_model_paired_scene_ci95": interval,
              "group_mean_changes": group_mean_changes,
              "screening": {**rule, "pass": all(rule.values())},
              "bootstrap": {"resamples": RESAMPLES, "seed": BOOTSTRAP_SEED,
                            "note": "Fixed models on previously used development scenes; does not include training seed or selection uncertainty."},
              "a4_used": False}
    (DEST / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"per_model": model_equal_deltas, "overall_delta": overall_delta,
                      "ci95": interval, "group_mean_changes": group_mean_changes,
                      "screening": report["screening"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
