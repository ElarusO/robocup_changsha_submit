"""独立评测 D 完整预算训练中的预定中间检查点，检查分数是否提前见顶。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
TRAIN_PYTHON = ROOT / ".venv-train" / "Scripts" / "python.exe"
EVAL_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(experiment_id, step):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("当前分支不是 P103")
    train_dir = OUT / f"train-{experiment_id}"
    run = json.loads((train_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run["status"] != "complete" or run["actual_total_steps"] != 1044480:
        raise ValueError("完整训练尚未完成")
    checkpoint = train_dir / f"checkpoint-{step}"
    record = next(item for item in run["checkpoints"] if item["actual_steps"] == step)
    for name, info in record["files"].items():
        if sha(checkpoint / name) != info["sha256"]:
            raise ValueError(f"检查点文件摘要错误：{name}")
    export_id = f"{experiment_id}{step}"
    export_dir = OUT / f"post-{export_id}"
    parent = OUT / f"d-param-mid-stage-{experiment_id}-{step}"
    check = OUT / f"d-param-mid-check-{experiment_id}-{step}"
    result_dir = OUT / f"d-param-mid-v2-{experiment_id}-{step}"
    if any(path.exists() for path in (export_dir, parent, check, result_dir)):
        raise FileExistsError("中间检查点输出已有文件，不覆盖")
    subprocess.run([TRAIN_PYTHON, HERE / "postprocess.py", "export", "--model-dir", checkpoint,
                    "--experiment-id", export_id, "--num-obs", "256"], cwd=ROOT, check=True)
    info = json.loads((export_dir / "export-info.json").read_text(encoding="utf-8"))
    if info["training_steps"] != step or info["consistency_check"] != "passed":
        raise ValueError("导出动作核对失败")
    stage = parent / "P103"
    (stage / "artifacts").mkdir(parents=True)
    for name in ("entry.py", "requirements-infer.lock", "THIRD_PARTY.md", "LICENSE"):
        shutil.copyfile(HERE / name, stage / name)
    artifact = stage / "artifacts" / "policy.npz"
    shutil.copyfile(export_dir / "policy.npz", artifact)
    manifest = yaml.safe_load((HERE / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{"path": "artifacts/policy.npz", "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}]
    manifest["metadata"] = {"description": f"P103 D 中间检查点 {experiment_id} {step}，开发验证"}
    (stage / "submission.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (stage / "LOG.md").write_text(f"# D 中间检查点 {experiment_id} {step}\n", encoding="utf-8")
    (stage / "REPORT.md").write_text("# 本地开发验证副本\n", encoding="utf-8")
    (stage / "experiments.csv").write_text(f"experiment_id,step\n{experiment_id},{step}\n", encoding="utf-8")
    subprocess.run([EVAL_PYTHON, ROOT / "scripts" / "check_submission.py", "--submission", stage, "--output", check], cwd=ROOT, check=True)
    subprocess.run([EVAL_PYTHON, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
                    "--suite", HERE / "validation-suite-v2.yaml", "--seeds", HERE / "validation-seeds-v2.json",
                    "--output", result_dir], cwd=ROOT, check=True)
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    if (result["status"] != "ok" or result["errors"] or
            result["artifact_hashes"]["artifacts/policy.npz"] != info["candidate_sha256"] or
            sum(item["num_episodes"] for item in result["group_metrics"].values()) != 200):
        raise ValueError("中间检查点评测未通过")
    print(json.dumps({"model": experiment_id, "step": step, "score": result["performance_score"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E026", "E027", "E028"), required=True)
    parser.add_argument("--step", type=int, choices=(313344, 626688), required=True)
    args = parser.parse_args()
    main(args.experiment_id, args.step)
