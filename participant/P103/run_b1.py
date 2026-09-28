"""按冻结的 B1 协议启动指定种子的训练；--dry-run 只展示配置。"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from train import _REPO_ROOT, train, validate_training_args


PARTICIPANT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PARTICIPANT_DIR / "b1-protocol-v1.json"
FROZEN_PROTOCOL_SHA256 = "901bfc8df3f2db6dceb524048d9c114904c9cc2f81a63a724f31739b73056dc8"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(experiment_id):
    if sha256(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("B1 固定协议文件的 SHA-256 已变化；停止启动训练")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=_REPO_ROOT, text=True).strip()
    if branch != "P103":
        raise ValueError(f"当前分支不是 P103：{branch}")
    if sys.version_info[:2] != (3, 12) or version("stable-baselines3") != "2.9.0" or version("supersuit") != "3.11.0":
        raise ValueError("训练环境版本与 B1 冻结配置不一致")
    evaluation = protocol["evaluation"]
    for path_key, hash_key in (("suite", "suite_sha256"), ("seeds", "seeds_sha256")):
        if sha256(_REPO_ROOT / evaluation[path_key]) != evaluation[hash_key]:
            raise ValueError(f"开发验证文件变化：{evaluation[path_key]}")
    matches = [entry for entry in protocol["experiments"] if entry["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"B1 协议没有实验编号：{experiment_id}")
    experiment = matches[0]
    settings = protocol["training"]
    if not settings["from_scratch"] or settings["case_id"] != "basic-0":
        raise ValueError("B1 必须从零开始，只使用 basic-0 训练场景")
    out_dir = _REPO_ROOT / experiment["train_output"]
    args = argparse.Namespace(
        total_steps=settings["total_steps_requested"],
        curve_points=settings["curve_points"],
        num_vec_envs=settings["num_vec_envs"],
        seed=experiment["seed"],
        train_case=settings["case_id"],
        learning_rate=settings["learning_rate"],
        gamma=settings["gamma"],
        ent_coef=settings["ent_coef"],
        n_steps=settings["n_steps"],
        batch_size=settings["batch_size"],
        checkpoint_steps=settings["checkpoint_sample_steps"],
        out=out_dir,
        protocol_path=PROTOCOL_PATH,
    )
    _, _, expected_actual_steps = validate_training_args(args)
    if expected_actual_steps != settings["expected_actual_steps_with_current_loop"]:
        raise ValueError("当前训练循环的实际样本预算与 B1 固定协议不一致")
    if out_dir.exists():
        raise FileExistsError(f"训练目录已存在，不覆盖：{out_dir}")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E011", "E012", "E013"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="只核对并展示参数，不开始训练")
    parsed = parser.parse_args()
    args = prepare(parsed.experiment_id)
    if parsed.dry_run:
        print(json.dumps({
            "experiment_id": parsed.experiment_id,
            "seed": args.seed,
            "train_case": args.train_case,
            "total_steps_requested": args.total_steps,
            "checkpoint_steps": args.checkpoint_steps,
            "output": args.out.relative_to(_REPO_ROOT).as_posix(),
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "training_started": False,
        }, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
