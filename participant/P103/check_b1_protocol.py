"""B1 只读自检：配置、历史基线、验证集与实验编号，不启动训练。"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys

from make_validation_v2 import validate_all


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
PROTOCOL_PATH = PARTICIPANT_DIR / "b1-protocol-v1.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def check():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    training = protocol["training"]
    experiments = protocol["experiments"]
    evaluation = protocol["evaluation"]
    if git("branch", "--show-current") != "P103":
        raise ValueError("当前分支不是 P103")
    if [entry["experiment_id"] for entry in experiments] != ["E011", "E012", "E013"]:
        raise ValueError("实验编号或顺序与 B1 规划不一致")
    if [entry["seed"] for entry in experiments] != [7101, 7102, 7103]:
        raise ValueError("三种训练种子与 B1 规划不一致")
    if training["case_id"] != "basic-0" or training["learning_rate"] != 0.0001 or not training["from_scratch"]:
        raise ValueError("训练场景、学习率或从零训练条件被修改")
    if training["n_steps"] * training["num_vec_envs"] * training["agents_per_env"] != training["rollout_samples"]:
        raise ValueError("每轮样本数计算不一致")
    per_segment = math.ceil(
        (training["total_steps_requested"] // training["curve_points"]) / training["rollout_samples"]
    ) * training["rollout_samples"]
    if per_segment * training["curve_points"] != training["expected_actual_steps_with_current_loop"]:
        raise ValueError("实际样本预算预计值与现有分段规则不符")
    if [per_segment * index for index in (3, 6, 10)] != training["checkpoint_sample_steps"]:
        raise ValueError("检查点与现有分段规则不符")
    if version("stable-baselines3") != "2.9.0" or version("supersuit") != "3.11.0":
        raise ValueError("训练依赖版本与本次固定规则不一致")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("训练 Python 版本不是 3.12")

    sets = validate_all(json.loads((PARTICIPANT_DIR / "validation-sets-v2-manifest.json").read_text(encoding="utf-8")))
    if evaluation["suite"] != "participant/P103/validation-suite-v2.yaml" or evaluation["seeds"] != "participant/P103/validation-seeds-v2.json":
        raise ValueError("开发验证文件路径变更")
    if evaluation["suite_sha256"] != sets["development"]["suite_sha256"] or evaluation["seeds_sha256"] != sets["development"]["schedule_sha256"]:
        raise ValueError("开发验证文件摘要与 B1 固定配置不符")
    baseline = protocol["historical_reference"]
    result_path = REPO_ROOT / baseline["result"]
    baseline_result = json.loads(result_path.read_text(encoding="utf-8"))
    if baseline_result["artifact_hashes"]["artifacts/policy.npz"] != baseline["model_sha256"]:
        raise ValueError("E009 基线模型摘要不一致")
    if not math.isclose(baseline_result["performance_score"], baseline["validation_v2_score"], abs_tol=1e-10):
        raise ValueError("E009 开发验证 v2 基线分数不一致")

    with (PARTICIPANT_DIR / "experiments.csv").open(encoding="utf-8", newline="") as source:
        prior_ids = {row["experiment_id"] for row in csv.DictReader(source)}
    occupied = []
    for experiment in experiments:
        identifier = experiment["experiment_id"]
        target = (REPO_ROOT / experiment["train_output"]).resolve()
        if not target.is_relative_to(OUTPUTS_DIR.resolve()):
            raise ValueError(f"{identifier} 训练输出不在 outputs/P103 下")
        if identifier in prior_ids or any(OUTPUTS_DIR.glob(f"*{identifier}*")):
            occupied.append(identifier)
    if occupied:
        raise FileExistsError(f"实验编号或输出目录已占用：{occupied}")
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "branch": "P103",
        "base_commit": protocol["base_commit"],
        "current_commit": git("rev-parse", "HEAD"),
        "train_script_sha256_now": sha256(PARTICIPANT_DIR / "train.py"),
        "python": sys.version.split()[0],
        "stable_baselines3": version("stable-baselines3"),
        "supersuit": version("supersuit"),
        "experiment_ids_available": True,
        "training_seeds": [entry["seed"] for entry in experiments],
        "samples_per_rollout": training["rollout_samples"],
        "planned_checkpoints": training["checkpoint_sample_steps"],
        "development_suite_sha256": sets["development"]["suite_sha256"],
        "final_confirmation_hash_intact": True,
        "baseline_E009_score": baseline_result["performance_score"],
        "training_started": False,
    }


def main(output):
    output = (output if output.is_absolute() else REPO_ROOT / output).resolve()
    if not output.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("B1 自检输出必须位于 outputs/P103 下")
    if output.exists():
        raise FileExistsError(f"自检目录已存在，不覆盖：{output}")
    result = check()
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
