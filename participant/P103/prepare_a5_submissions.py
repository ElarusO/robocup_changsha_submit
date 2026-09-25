"""为 A5 比较建立独立的本地评测副本；不修改当前提交模型。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
MODELS = {
    "E001": REPO_ROOT / "participant" / "_template" / "artifacts" / "policy.npz",
    "E008": OUTPUTS_DIR / "post-E008" / "policy.npz",
    "E009": PARTICIPANT_DIR / "artifacts" / "policy.npz",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(out_dir):
    out_dir = (out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir).resolve()
    if not out_dir.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("A5 临时副本必须放在 outputs/P103 下")
    if out_dir.exists():
        raise FileExistsError(f"临时副本已存在，不覆盖：{out_dir}")
    with (PARTICIPANT_DIR / "experiments.csv").open(encoding="utf-8", newline="") as source:
        records = {row["experiment_id"]: row for row in csv.DictReader(source)}
    current_manifest = yaml.safe_load((PARTICIPANT_DIR / "submission.yaml").read_text(encoding="utf-8"))
    current_entry = current_manifest["checkpoint_manifest"][0]
    if current_entry["sha256"] != records["E009"]["model_sha256"] or sha256(MODELS["E009"]) != current_entry["sha256"]:
        raise ValueError("当前提交不是已归档的 E009，停止建立对照副本")

    for model_id, path in MODELS.items():
        if not path.is_file() or sha256(path) != records[model_id]["model_sha256"]:
            raise ValueError(f"{model_id} 权重文件与历史实验记录不一致：{path}")

    out_dir.mkdir(parents=True)
    result = {}
    for model_id, source_model in MODELS.items():
        stage = out_dir / model_id / "P103"
        artifacts = stage / "artifacts"
        artifacts.mkdir(parents=True)
        for name in ("entry.py", "requirements-infer.lock", "THIRD_PARTY.md", "LICENSE"):
            shutil.copyfile(PARTICIPANT_DIR / name, stage / name)
        shutil.copyfile(source_model, artifacts / "policy.npz")

        model_sha = sha256(artifacts / "policy.npz")
        stage_manifest = dict(current_manifest)
        stage_manifest["checkpoint_manifest"] = [{
            "path": "artifacts/policy.npz",
            "sha256": model_sha,
            "size_bytes": (artifacts / "policy.npz").stat().st_size,
        }]
        stage_manifest["train_command"] = "python train.py"
        stage_manifest["metadata"] = {
            "description": f"P103 A5 本地对照副本：{model_id} 历史权重；仅用于开发验证 v2",
        }
        (stage / "submission.yaml").write_text(
            yaml.safe_dump(stage_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (stage / "LOG.md").write_text(
            f"# A5 本地对照副本\n\n模型：{model_id}；权重 SHA-256：`{model_sha}`。"
            "本目录只用于同一套开发验证 v2 的本地比较，不是新的训练实验或最终提交。\n",
            encoding="utf-8",
        )
        (stage / "REPORT.md").write_text(
            f"# {model_id} 本地评测副本\n\n权重来自 `{source_model.relative_to(REPO_ROOT).as_posix()}`。"
            "推理入口沿用当前 P103 的 `entry.py`；原始训练与模型出处见 P103 正式 `LOG.md`。"
            "A5 分数以各自独立评测输出目录为准。\n",
            encoding="utf-8",
        )
        (stage / "experiments.csv").write_text(
            "experiment_id,model_sha256,origin_commit\n"
            f"{model_id},{model_sha},{records[model_id]['source_commit']}\n",
            encoding="utf-8",
        )
        result[model_id] = {
            "submission": stage.relative_to(REPO_ROOT).as_posix(),
            "model_source": source_model.relative_to(REPO_ROOT).as_posix(),
            "model_sha256": model_sha,
            "origin_commit": records[model_id]["source_commit"],
            "entry_sha256": sha256(stage / "entry.py"),
        }
    (out_dir / "stage-manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
