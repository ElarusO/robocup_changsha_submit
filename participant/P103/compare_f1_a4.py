"""按事前 F1 协议比较两份固定 ZIP 的 A4 评测结果。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_e2_candidates import GROUPS, OUTPUTS, ROOT, load_evaluation, scene_means


HERE = Path(__file__).resolve().parent
RUNS = {"E009": OUTPUTS / "f1-a4-E009-001",
        "E015-combo": OUTPUTS / "f1-a4-E015-combo-001"}
EXPECTED_MODELS = {
    "E009": "6b5c8cc0fc0279a86e3218665e4623eeb10dd4a4b5e75c642933cb87fef6ed51",
    "E015-combo": "0bde83fe3983920ee50446f218d54a58e886ee609bc00fcfb5b68efc95c11173",
}
SUMMARY_DIR = OUTPUTS / "f1-a4-comparison-001"
RESAMPLES = 10000
SEED = 20260929


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if SUMMARY_DIR.exists():
        raise FileExistsError(f"比较结果目录已存在，不覆盖：{SUMMARY_DIR}")
    input_manifest = json.loads((OUTPUTS / "f1-a4-inputs-001" / "input-manifest.json").read_text(encoding="utf-8"))
    protocol_hash = sha(HERE / "F1_A4_PRECOMMITTED_PROTOCOL.md")
    if protocol_hash != input_manifest["protocol_sha256_before_a4"]:
        raise ValueError("事前协议在 A4 运行后已改变")

    results, episodes, scenes, provenance = {}, {}, {}, {}
    for role, folder in RUNS.items():
        result, rows, paths = load_evaluation(folder)
        if result["suite_id"] != "p103-final-confirmation-v1":
            raise ValueError(f"不是固定 A4 套件：{role}")
        if result["seed_schedule_id"] != "p103-final-confirmation-seeds-v1":
            raise ValueError(f"不是固定 A4 种子安排：{role}")
        if result["artifact_hashes"]["artifacts/policy.npz"] != EXPECTED_MODELS[role]:
            raise ValueError(f"模型摘要与冻结 ZIP 不符：{role}")
        results[role], episodes[role], scenes[role] = result, rows, scene_means(rows)
        provenance[role] = {**paths, "run_id": result["run_id"],
                            "code_tree_hash": result["code_tree_hash"],
                            "model_sha256": EXPECTED_MODELS[role],
                            "performance_score": result["performance_score"]}
    for field in ("suite_id", "suite_hash", "seed_schedule_id", "task_version",
                  "score_version", "inference_lock_hash", "provenance"):
        if results["E009"][field] != results["E015-combo"][field]:
            raise ValueError(f"两份结果的 {field} 不一致")
    if set(episodes["E009"]) != set(episodes["E015-combo"]):
        raise ValueError("两份结果的逐回合场景键不一致")
    if set(scenes["E009"]) != set(scenes["E015-combo"]):
        raise ValueError("两份结果的独立场景键不一致")

    paired_rows = []
    group_changes = {}
    group_differences = {}
    for group in GROUPS:
        cases = sorted(case for group_id, case in scenes["E009"] if group_id == group)
        if len(cases) != 50:
            raise ValueError(f"场景数不为 50：{group}")
        differences = []
        for case in cases:
            old, new = scenes["E009"][(group, case)], scenes["E015-combo"][(group, case)]
            row = {"group_id": group, "case_id": case,
                   "E009_mean_j": old["mean_j"], "E015_mean_j": new["mean_j"],
                   "delta_mean_j": new["mean_j"] - old["mean_j"],
                   "delta_coverage": new["mean_coverage_rate"] - old["mean_coverage_rate"],
                   "delta_collision": new["mean_collision_rate"] - old["mean_collision_rate"]}
            paired_rows.append(row)
            differences.append(row["delta_mean_j"])
        group_differences[group] = np.asarray(differences)
        old_metric = results["E009"]["group_metrics"][group]
        new_metric = results["E015-combo"]["group_metrics"][group]
        group_changes[group] = {
            "mean_j": new_metric["mean_j"] - old_metric["mean_j"],
            "coverage": new_metric["mean_coverage_rate"] - old_metric["mean_coverage_rate"],
            "collision": new_metric["mean_collision_rate"] - old_metric["mean_collision_rate"],
            "old_mean_j": old_metric["mean_j"], "new_mean_j": new_metric["mean_j"],
            "old_coverage": old_metric["mean_coverage_rate"],
            "new_coverage": new_metric["mean_coverage_rate"],
            "old_collision": old_metric["mean_collision_rate"],
            "new_collision": new_metric["mean_collision_rate"],
            "scene_wins": int(np.count_nonzero(group_differences[group] > 1e-12)),
            "scene_losses": int(np.count_nonzero(group_differences[group] < -1e-12)),
            "scene_ties": int(np.count_nonzero(np.abs(group_differences[group]) <= 1e-12)),
        }
        if not np.isclose(group_differences[group].mean(), group_changes[group]["mean_j"], atol=1e-10):
            raise ValueError(f"逐场景与汇总组指标不一致：{group}")
    score_difference = float(500 * sum(group_differences[group].mean() for group in GROUPS))
    reported_difference = (results["E015-combo"]["performance_score"]
                           - results["E009"]["performance_score"])
    if not np.isclose(score_difference, reported_difference, atol=1e-8):
        raise ValueError("逐场景分差与报告总分差不一致")

    rng = np.random.default_rng(SEED)
    sampled = 500 * sum(group_differences[group][rng.integers(0, 50, size=(RESAMPLES, 50))].mean(axis=1)
                        for group in GROUPS)
    lower, upper = (float(v) for v in np.quantile(sampled, [0.025, 0.975]))
    gates = {"complete_and_paired": True,
             "score_difference_at_least_5": score_difference >= 5.0,
             "paired_scene_ci95_lower_above_0": lower > 0.0,
             "both_group_mean_j_delta_at_least_minus_0_01": all(
                 group_changes[group]["mean_j"] >= -0.01 for group in GROUPS)}
    report = {"protocol_sha256_before_a4": protocol_hash,
              "source": "Windows local_preview, not official hidden evaluation",
              "fixed_zip_sha256": {role: input_manifest["packages"][role]["archive_sha256"] for role in RUNS},
              "suite_sha256": input_manifest["suite_sha256"],
              "seeds_sha256": input_manifest["seeds_sha256"],
              "inputs": provenance,
              "scores": {role: results[role]["performance_score"] for role in RUNS},
              "score_difference_E015_minus_E009": score_difference,
              "paired_scene_bootstrap": {"resamples": RESAMPLES, "seed": SEED,
                                          "ci95": [lower, upper]},
              "group_changes_E015_minus_E009": group_changes,
              "gates": gates,
              "recommend_switch_to_E015": all(gates.values())}
    SUMMARY_DIR.mkdir(parents=True)
    with (SUMMARY_DIR / "paired-scenes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (SUMMARY_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scores": report["scores"],
                      "score_difference_E015_minus_E009": score_difference,
                      "ci95": [lower, upper], "gates": gates,
                      "recommend_switch_to_E015": report["recommend_switch_to_E015"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
