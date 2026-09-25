"""把一次 P103 PPO 训练转换为可评测的 NumPy 模型。

在仓库根目录、使用 .venv-train 的 Python 执行，例如：
    python participant/P103/postprocess.py export --model-dir outputs/P103/train-001 --experiment-id E002
    python participant/P103/postprocess.py activate --experiment-id E002
    python participant/P103/postprocess.py restore --experiment-id E002

export 不改变当前参赛模型；activate 会备份并切换候选模型；restore 恢复备份。
提交预检与公开评测仍使用仓库提供的脚本和 .venv 评测环境。
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from stable_baselines3 import PPO

from train import check_export


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
MODEL_PATH = PARTICIPANT_DIR / "artifacts" / "policy.npz"
SUBMISSION_PATH = PARTICIPANT_DIR / "submission.yaml"
MODEL_RELATIVE_PATH = "artifacts/policy.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_dir(experiment_id: str) -> Path:
    if re.fullmatch(r"E\d{3,}", experiment_id) is None:
        raise ValueError("实验编号须为 E002 这类格式")
    return OUTPUTS_DIR / f"post-{experiment_id}"


def checked_training_dir(value: str) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    try:
        path.relative_to(OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("训练目录必须位于 outputs/P103/ 内") from exc
    for name in ("model.zip", "vecnormalize-stats.npz", "training_meta.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return path


def checked_layers(model: PPO):
    layers = list(model.policy.mlp_extractor.policy_net)
    action = model.policy.action_net
    names = [layer.__class__.__name__ for layer in layers]
    if names != ["Linear", "Tanh", "Linear", "Tanh"]:
        raise ValueError(f"网络结构与 entry.py 不一致：{names}")
    if action.__class__.__name__ != "Linear":
        raise ValueError("动作输出层与 entry.py 不一致")
    if [tuple(layer.weight.shape) for layer in (layers[0], layers[2], action)] != [
        (64, 104), (64, 64), (2, 64)
    ]:
        raise ValueError("权重形状与 entry.py 不一致")
    return layers[0], layers[2], action


def export_model(args) -> None:
    model_dir = checked_training_dir(args.model_dir)
    out_dir = experiment_dir(args.experiment_id)
    if out_dir.exists():
        raise FileExistsError(f"结果目录已存在，请换一个实验编号：{out_dir}")
    if args.num_obs < 1:
        raise ValueError("--num-obs 必须大于 0")

    training_meta = json.loads((model_dir / "training_meta.json").read_text(encoding="utf-8"))
    model = PPO.load(str(model_dir / "model.zip"), device="cpu")
    first, second, action = checked_layers(model)
    with np.load(model_dir / "vecnormalize-stats.npz", allow_pickle=False) as stats:
        expected = {"obs_mean", "obs_var", "obs_eps", "obs_clip"}
        if set(stats.files) != expected or stats["obs_mean"].shape != (104,) or stats["obs_var"].shape != (104,):
            raise ValueError("归一化统计量与 104 维观测不一致")
        arrays = {
            "W1": first.weight.detach().cpu().numpy().T.astype(np.float64),
            "b1": first.bias.detach().cpu().numpy().astype(np.float64),
            "W2": second.weight.detach().cpu().numpy().T.astype(np.float64),
            "b2": second.bias.detach().cpu().numpy().astype(np.float64),
            "W3": action.weight.detach().cpu().numpy().T.astype(np.float64),
            "b3": action.bias.detach().cpu().numpy().astype(np.float64),
            **{key: np.asarray(stats[key], dtype=np.float64) for key in expected},
        }
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("模型或归一化统计量包含非有限值")
    if np.any(arrays["obs_var"] < 0) or float(arrays["obs_eps"]) <= 0 or float(arrays["obs_clip"]) <= 0:
        raise ValueError("归一化统计量无效")

    out_dir.mkdir(parents=True)
    candidate = out_dir / "policy.npz"
    np.savez(candidate, **arrays)
    check_export(SimpleNamespace(
        model_dir=model_dir,
        npz=candidate,
        num_vec_envs=int(training_meta["num_vec_envs"]),
        seed=int(training_meta["seed"]),
        num_obs=args.num_obs,
    ))
    info = {
        "experiment_id": args.experiment_id,
        "training_dir": model_dir.relative_to(REPO_ROOT).as_posix(),
        "training_model_sha256": sha256(model_dir / "model.zip"),
        "normalization_stats_sha256": sha256(model_dir / "vecnormalize-stats.npz"),
        "training_seed": int(training_meta["seed"]),
        "training_steps": int(training_meta["total_steps"]),
        "candidate_sha256": sha256(candidate),
        "candidate_size_bytes": candidate.stat().st_size,
        "consistency_observations": args.num_obs,
        "consistency_check": "passed",
    }
    (out_dir / "export-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"导出并核对完成：{candidate}")
    print(f"SHA-256: {info['candidate_sha256']}  字节数: {info['candidate_size_bytes']}")
    print(f"下一步：.\\.venv-train\\Scripts\\python.exe participant/P103/postprocess.py activate --experiment-id {args.experiment_id}")


def load_candidate(experiment_id: str):
    out_dir = experiment_dir(experiment_id)
    info = json.loads((out_dir / "export-info.json").read_text(encoding="utf-8"))
    candidate = out_dir / "policy.npz"
    if info["experiment_id"] != experiment_id or info["consistency_check"] != "passed":
        raise ValueError("导出记录未通过一致性检查")
    if sha256(candidate) != info["candidate_sha256"] or candidate.stat().st_size != info["candidate_size_bytes"]:
        raise ValueError("候选模型与导出记录不一致")
    return out_dir, candidate, info


def manifest_values():
    text = SUBMISSION_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data.get("participant_id") != "P103":
        raise ValueError("submission.yaml 的编号不是 P103")
    entries = data.get("checkpoint_manifest", [])
    if len(entries) != 1 or entries[0].get("path") != MODEL_RELATIVE_PATH:
        raise ValueError("脚本仅适用于单个 policy.npz 的当前清单")
    entry = entries[0]
    if sha256(MODEL_PATH) != entry["sha256"] or MODEL_PATH.stat().st_size != entry["size_bytes"]:
        raise ValueError("当前模型与 submission.yaml 不一致，停止替换")
    return text, entry["sha256"], entry["size_bytes"]


def updated_manifest(text: str, old_sha: str, old_size: int, new_sha: str, new_size: int) -> str:
    old_hash_line = f'    sha256: "{old_sha}"'
    old_size_line = f"    size_bytes: {old_size}"
    if text.count(old_hash_line) != 1 or text.count(old_size_line) != 1:
        raise ValueError("无法在 submission.yaml 中唯一定位模型摘要和大小")
    return text.replace(old_hash_line, f'    sha256: "{new_sha}"').replace(
        old_size_line, f"    size_bytes: {new_size}"
    )


def activate(experiment_id: str) -> None:
    out_dir, candidate, info = load_candidate(experiment_id)
    original_text, old_sha, old_size = manifest_values()
    if old_sha == info["candidate_sha256"]:
        raise ValueError("当前模型已经是该候选模型")
    backup_model = out_dir / "original-policy.npz"
    backup_manifest = out_dir / "original-submission.yaml"
    new_text = updated_manifest(
        original_text, old_sha, old_size,
        info["candidate_sha256"], info["candidate_size_bytes"],
    )
    if backup_model.exists() or backup_manifest.exists():
        if not (backup_model.is_file() and backup_manifest.is_file()):
            raise ValueError("原模型备份不完整")
        if sha256(backup_model) != old_sha or backup_manifest.read_text(encoding="utf-8") != original_text:
            raise ValueError("当前模型与该实验的原始备份不同，停止激活")
    else:
        shutil.copyfile(MODEL_PATH, backup_model)
        backup_manifest.write_text(original_text, encoding="utf-8")
    try:
        shutil.copyfile(candidate, MODEL_PATH)
        SUBMISSION_PATH.write_text(new_text, encoding="utf-8")
        manifest_values()
    except Exception:
        shutil.copyfile(backup_model, MODEL_PATH)
        SUBMISSION_PATH.write_text(original_text, encoding="utf-8")
        raise
    (out_dir / "activated-submission.yaml").write_text(new_text, encoding="utf-8")
    print(f"已激活候选模型，原模型和清单备份在 {out_dir}")
    check_dir = OUTPUTS_DIR / f"check-{experiment_id}"
    eval_dir = OUTPUTS_DIR / f"eval-{experiment_id}"
    if check_dir.exists() or eval_dir.exists():
        print("此实验的预检或评测目录已存在；保留已有结果，重新评测请使用新实验编号。")
    else:
        print("请在仓库根目录使用评测环境运行：")
        print(f".\\.venv\\Scripts\\python.exe scripts/check_submission.py --submission participant/P103 --output outputs/P103/check-{experiment_id}")
        print(f".\\.venv\\Scripts\\python.exe scripts/evaluate_one.py --submission participant/P103 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P103/eval-{experiment_id}")
    print(f"如果不保留此模型：.\\.venv-train\\Scripts\\python.exe participant/P103/postprocess.py restore --experiment-id {experiment_id}")


def restore(experiment_id: str) -> None:
    out_dir, _candidate, info = load_candidate(experiment_id)
    backup_model = out_dir / "original-policy.npz"
    backup_manifest = out_dir / "original-submission.yaml"
    activated_manifest = out_dir / "activated-submission.yaml"
    if not all(path.is_file() for path in (backup_model, backup_manifest, activated_manifest)):
        raise FileNotFoundError("缺少激活时保存的原模型或清单")
    if sha256(MODEL_PATH) != info["candidate_sha256"]:
        raise ValueError("当前模型已发生其他改动，停止恢复以免覆盖工作")
    if SUBMISSION_PATH.read_text(encoding="utf-8") != activated_manifest.read_text(encoding="utf-8"):
        raise ValueError("当前清单已发生其他改动，停止恢复以免覆盖工作")
    backup_data = yaml.safe_load(backup_manifest.read_text(encoding="utf-8"))
    backup_entry = backup_data.get("checkpoint_manifest", [{}])[0]
    if backup_data.get("participant_id") != "P103" or backup_entry.get("path") != MODEL_RELATIVE_PATH:
        raise ValueError("原清单备份无效")
    if sha256(backup_model) != backup_entry.get("sha256") or backup_model.stat().st_size != backup_entry.get("size_bytes"):
        raise ValueError("原模型备份与原清单不一致")
    shutil.copyfile(backup_model, MODEL_PATH)
    SUBMISSION_PATH.write_text(backup_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_values()
    print("已恢复激活前的模型和清单；实验输出目录仍保留。")


def main() -> None:
    parser = argparse.ArgumentParser(description="P103 训练模型导出、一致性检查和候选切换")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="导出并验证训练模型，不改变参赛模型")
    export.add_argument("--model-dir", required=True, help="例如 outputs/P103/train-001")
    export.add_argument("--experiment-id", required=True, help="例如 E002")
    export.add_argument("--num-obs", type=int, default=256, help="导出一致性检查的观测样本数")
    for name in ("activate", "restore"):
        sub.add_parser(name).add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_model(args)
    elif args.command == "activate":
        activate(args.experiment_id)
    else:
        restore(args.experiment_id)


if __name__ == "__main__":
    main()
