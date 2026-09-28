"""核对 B3/C 同题开发验证结果，汇总三种子均分、波动及配对差值。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
STEPS = (313344, 626688, 1044480)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output, phase):
    output = (output if output.is_absolute() else REPO_ROOT / output).resolve()
    if not output.is_relative_to(OUTPUTS_DIR.resolve()) or output.exists():
        raise ValueError("汇总目录须是 outputs/P103/ 下未使用的新目录")
    if phase not in ("b3", "c"):
        raise ValueError("phase 必须为 b3 或 c")
    protocol_path = PARTICIPANT_DIR / ("b1-protocol-v1.json" if phase == "b3" else "c-protocol-v1.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    experiments = ("E011", "E012", "E013") if phase == "b3" else ("E014", "E015", "E016")
    suite_path = REPO_ROOT / protocol["evaluation"]["suite"]
    seeds_path = REPO_ROOT / protocol["evaluation"]["seeds"]
    if sha256(suite_path) != protocol["evaluation"]["suite_sha256"] or sha256(seeds_path) != protocol["evaluation"]["seeds_sha256"]:
        raise ValueError("固定开发验证文件发生变化")
    rows = []
    episode_keys = None
    suite_hash = None
    entry_hash = None
    for experiment_id in experiments:
        for step in STEPS:
            label = f"{experiment_id}-{step}"
            stage = OUTPUTS_DIR / f"{phase}-stage-{label}"
            evaluation = OUTPUTS_DIR / f"{phase}-v2-{label}"
            recorded = json.loads((stage / "stage-manifest.json").read_text(encoding="utf-8"))
            result = json.loads((evaluation / "result.json").read_text(encoding="utf-8"))
            with (evaluation / "episodes.csv").open(encoding="utf-8", newline="") as source:
                episodes = list(csv.DictReader(source))
            keys = {(row["group_id"], row["case_id"], row["repeat_index"]) for row in episodes}
            if len(episodes) != 200 or len(keys) != 200 or any(row["status"] != "ok" for row in episodes):
                raise ValueError(f"{label} 回合记录不完整")
            if episode_keys is None:
                episode_keys = keys
                suite_hash = result["suite_hash"]
                entry_hash = recorded["entry_sha256"]
            if keys != episode_keys or result["suite_hash"] != suite_hash or recorded["entry_sha256"] != entry_hash:
                raise ValueError(f"{label} 与其他模型的评测口径不同")
            if result["status"] != "ok" or result["errors"] or result["provenance"] != "local_preview":
                raise ValueError(f"{label} 评测状态不合格")
            if result["suite_id"] != "p103-validation-v2" or result["seed_schedule_id"] != "p103-validation-seeds-v2":
                raise ValueError(f"{label} 未使用固定开发验证 v2")
            if result["artifact_hashes"]["artifacts/policy.npz"] != recorded["model_sha256"]:
                raise ValueError(f"{label} 权重摘要不符")
            groups = result["group_metrics"]
            if not np.isclose(result["performance_score"], 500 * (groups["basic"]["mean_j"] + groups["cooperation"]["mean_j"]), atol=1e-9):
                raise ValueError(f"{label} 官方总分公式核对失败")
            rows.append({
                "experiment_id": experiment_id,
                "training_seed": next(item.get("seed", item.get("training_seed")) for item in protocol["experiments"] if item["experiment_id"] == experiment_id),
                "checkpoint_steps": step,
                "run_id": result["run_id"],
                "model_sha256": recorded["model_sha256"],
                "performance_score": result["performance_score"],
                "basic_mean_coverage_rate": groups["basic"]["mean_coverage_rate"],
                "basic_mean_collision_rate": groups["basic"]["mean_collision_rate"],
                "cooperation_mean_coverage_rate": groups["cooperation"]["mean_coverage_rate"],
                "cooperation_mean_collision_rate": groups["cooperation"]["mean_collision_rate"],
                "result_path": (evaluation / "result.json").relative_to(REPO_ROOT).as_posix(),
            })
    by_step = {}
    for step in STEPS:
        scores = [row["performance_score"] for row in rows if row["checkpoint_steps"] == step]
        by_step[str(step)] = {
            "scores": scores,
            "mean": statistics.mean(scores),
            "sample_standard_deviation": statistics.stdev(scores),
            "minimum": min(scores),
            "maximum": max(scores),
        }
    selected_step = max(STEPS, key=lambda step: (by_step[str(step)]["mean"], -step))
    summary = {
        "protocol_sha256": sha256(protocol_path),
        "suite_hash": suite_hash,
        "entry_sha256": entry_hash,
        "episode_keys_count": len(episode_keys),
        "results": rows,
        "by_checkpoint": by_step,
        "selected_shared_checkpoint_steps": selected_step,
        "note": "A2 后训练实现；分数均为 Windows local_preview，最终确认集未使用。",
    }
    if phase == "b3":
        summary["historical_E009_v2_score"] = protocol["historical_reference"]["validation_v2_score"]
    else:
        b_summary = json.loads((OUTPUTS_DIR / "b3-summary-001" / "summary.json").read_text(encoding="utf-8"))
        if sha256(OUTPUTS_DIR / "b3-summary-001" / "summary.json") != protocol["decision_basis"]["b3_summary_sha256"]:
            raise ValueError("B3 汇总摘要与 C 冻结协议不一致")
        if b_summary["suite_hash"] != suite_hash or b_summary["entry_sha256"] != entry_hash or b_summary["episode_keys_count"] != len(episode_keys):
            raise ValueError("C 与 B3 评测口径不同")
        with (OUTPUTS_DIR / "b3-v2-E011-313344" / "episodes.csv").open(encoding="utf-8", newline="") as source:
            b_keys = {(row["group_id"], row["case_id"], row["repeat_index"]) for row in csv.DictReader(source)}
        if b_keys != episode_keys:
            raise ValueError("C 与 B3 的逐回合场景和重复次数不同")
        b_by_seed_step = {(row["training_seed"], row["checkpoint_steps"]): row for row in b_summary["results"]}
        paired = []
        for row in rows:
            old = b_by_seed_step[(row["training_seed"], row["checkpoint_steps"])]
            paired.append({
                "training_seed": row["training_seed"],
                "checkpoint_steps": row["checkpoint_steps"],
                "c_experiment": row["experiment_id"],
                "b_experiment": old["experiment_id"],
                "score_difference": row["performance_score"] - old["performance_score"],
                "basic_coverage_difference": row["basic_mean_coverage_rate"] - old["basic_mean_coverage_rate"],
                "cooperation_coverage_difference": row["cooperation_mean_coverage_rate"] - old["cooperation_mean_coverage_rate"],
                "basic_collision_difference": row["basic_mean_collision_rate"] - old["basic_mean_collision_rate"],
                "cooperation_collision_difference": row["cooperation_mean_collision_rate"] - old["cooperation_mean_collision_rate"],
            })
        final = [item for item in paired if item["checkpoint_steps"] == STEPS[-1]]
        summary["paired_differences"] = paired
        summary["final_paired_score_differences"] = [item["score_difference"] for item in final]
        summary["final_mean_score_difference"] = statistics.mean(item["score_difference"] for item in final)
        summary["final_improved_seed_count"] = sum(item["score_difference"] > 0 for item in final)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"by_checkpoint": by_step, "selected_shared_checkpoint_steps": selected_step, "final_mean_score_difference": summary.get("final_mean_score_difference")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("b3", "c"), default="b3")
    args = parser.parse_args()
    main(args.output, args.phase)
