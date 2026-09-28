"""检查一次训练的检查点网络、归一化统计量、元数据及文件哈希。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from postprocess import checked_layers


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    directory = (directory if directory.is_absolute() else REPO_ROOT / directory).resolve()
    if not directory.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("训练目录必须位于 outputs/P103 下")
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or status["status"] != "complete":
        raise ValueError("训练尚未完整结束")
    checkpoints = manifest["checkpoints"]
    if not checkpoints or [item["actual_steps"] for item in checkpoints] != sorted({item["actual_steps"] for item in checkpoints}):
        raise ValueError("检查点缺失、重复或未按样本数排序")
    summaries = []
    for item in checkpoints:
        checkpoint = directory / item["directory"]
        recorded = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        if recorded != item or checkpoint.name != f"checkpoint-{item['actual_steps']}":
            raise ValueError(f"检查点记录不一致：{checkpoint}")
        for name, details in item["files"].items():
            path = checkpoint / name
            if sha256(path) != details["sha256"] or path.stat().st_size != details["size_bytes"]:
                raise ValueError(f"检查点文件摘要或大小不一致：{path}")
        meta = json.loads((checkpoint / "training_meta.json").read_text(encoding="utf-8"))
        if meta["total_steps"] != item["actual_steps"] or meta["requested_checkpoint_targets"] != item["requested_targets"]:
            raise ValueError(f"检查点元数据步数不一致：{checkpoint}")
        model = PPO.load(str(checkpoint / "model.zip"), device="cpu")
        checked_layers(model)
        if model.num_timesteps != item["actual_steps"]:
            raise ValueError(f"网络自身的样本步数不一致：{checkpoint}")
        with np.load(checkpoint / "vecnormalize-stats.npz", allow_pickle=False) as stats:
            if set(stats.files) != {"obs_mean", "obs_var", "obs_eps", "obs_clip"}:
                raise ValueError(f"归一化文件字段不一致：{checkpoint}")
            if stats["obs_mean"].shape != (104,) or stats["obs_var"].shape != (104,):
                raise ValueError(f"归一化文件不是 104 维：{checkpoint}")
            if not np.isfinite(stats["obs_mean"]).all() or not np.isfinite(stats["obs_var"]).all():
                raise ValueError(f"归一化统计量不是有限数：{checkpoint}")
            if np.any(stats["obs_var"] < 0) or float(stats["obs_eps"]) <= 0 or float(stats["obs_clip"]) <= 0:
                raise ValueError(f"归一化统计量数值无效：{checkpoint}")
        summaries.append({
            "actual_steps": item["actual_steps"],
            "targets": item["requested_targets"],
            "model_sha256": item["files"]["model.zip"]["sha256"],
            "normalization_sha256": item["files"]["vecnormalize-stats.npz"]["sha256"],
            "train_script_sha256": meta["train_script_sha256"],
        })
    final = checkpoints[-1]
    for name, details in final["files"].items():
        if sha256(directory / name) != details["sha256"]:
            raise ValueError(f"根目录 {name} 与最终检查点不一致")
    if not manifest["root_model_equals_final_checkpoint"] or manifest["actual_total_steps"] != final["actual_steps"]:
        raise ValueError("最终步数或根目录文件与最终检查点不一致")
    return {
        "directory": directory.relative_to(REPO_ROOT).as_posix(),
        "status": "verified",
        "requested_total_steps": manifest["requested_total_steps"],
        "actual_total_steps": manifest["actual_total_steps"],
        "checkpoints": summaries,
        "root_matches_final": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (args.output if args.output.is_absolute() else REPO_ROOT / args.output).resolve()
    if not output.is_relative_to(OUTPUTS_DIR.resolve()) or output.exists():
        raise ValueError("核查输出必须是 outputs/P103 下尚不存在的文件")
    result = verify(args.train_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
