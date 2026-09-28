"""按预先冻结的 C 协议训练 E014–E016；--dry-run 仅核对配置。"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from train import _REPO_ROOT, train, validate_training_args


PARTICIPANT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = PARTICIPANT_DIR / "c-protocol-v1.json"
FROZEN_PROTOCOL_SHA256 = "37f86ac5669e7449742ac17815d596248ad60ec5b89022d530a51290b5e71131"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(experiment_id):
    if sha256(PROTOCOL_PATH) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("C 协议文件已变化，停止启动训练")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if sha256(_REPO_ROOT / protocol["decision_basis"]["b3_summary"]) != protocol["decision_basis"]["b3_summary_sha256"]:
        raise ValueError("B3 决策依据摘要已变化")
    if sha256(PARTICIPANT_DIR / "b1-protocol-v1.json") != protocol["decision_basis"]["b1_protocol_sha256"]:
        raise ValueError("B1 冻结协议已变化")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=_REPO_ROOT, text=True).strip()
    if branch != "P103":
        raise ValueError(f"当前分支不是 P103：{branch}")
    if sys.version_info[:2] != (3, 12) or version("stable-baselines3") != "2.9.0" or version("supersuit") != "3.11.0":
        raise ValueError("训练环境版本与 B/C 对照不一致")
    settings = protocol["training"]
    if sha256(PARTICIPANT_DIR / "train.py") != settings["train_script_sha256"] or sha256(PARTICIPANT_DIR / "mixed_env.py") != settings["mixed_env_sha256"]:
        raise ValueError("C 训练实现与冻结协议不一致")
    for key, hash_key in (("suite", "suite_sha256"), ("seeds", "seeds_sha256")):
        if sha256(_REPO_ROOT / protocol["evaluation"][key]) != protocol["evaluation"][hash_key]:
            raise ValueError("开发验证配置已变化")
    matches = [item for item in protocol["experiments"] if item["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"C 协议没有编号 {experiment_id}")
    selected = matches[0]
    args = argparse.Namespace(
        total_steps=settings["total_steps_requested"],
        curve_points=settings["curve_points"],
        num_vec_envs=settings["num_vec_envs"],
        seed=selected["training_seed"],
        train_case=settings["train_case"],
        learning_rate=settings["learning_rate"],
        gamma=settings["gamma"],
        ent_coef=settings["ent_coef"],
        n_steps=settings["n_steps"],
        batch_size=settings["batch_size"],
        checkpoint_steps=settings["checkpoint_sample_steps"],
        out=_REPO_ROOT / selected["output"],
        protocol_path=PROTOCOL_PATH,
    )
    _, _, expected = validate_training_args(args)
    if expected != settings["expected_actual_steps"]:
        raise ValueError("C 训练实际样本预算与协议不一致")
    if args.out.exists():
        raise FileExistsError(f"C 训练目录已存在，不覆盖：{args.out}")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E014", "E015", "E016"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    args = prepare(parsed.experiment_id)
    if parsed.dry_run:
        print(json.dumps({
            "experiment_id": parsed.experiment_id,
            "seed": args.seed,
            "train_case": args.train_case,
            "checkpoint_steps": args.checkpoint_steps,
            "output": args.out.relative_to(_REPO_ROOT).as_posix(),
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "training_started": False,
        }, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
