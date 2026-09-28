"""导出并用固定开发验证 v2 评测一个 B3/C 检查点；不触碰正式提交模型。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from postprocess import export_model
from verify_b2_checkpoints import verify


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
TRAINING_PYTHON = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"
EVALUATION_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
SUITE = PARTICIPANT_DIR / "validation-suite-v2.yaml"
SEEDS = PARTICIPANT_DIR / "validation-seeds-v2.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(experiment_id, step):
    if experiment_id not in ("E011", "E012", "E013", "E014", "E015", "E016"):
        raise ValueError("只接受 B3/C 的 E011–E016")
    phase = "b3" if experiment_id in ("E011", "E012", "E013") else "c"
    if step not in (313344, 626688, 1044480):
        raise ValueError("不是 B1 冻结的检查点步数")
    train_dir = OUTPUTS_DIR / f"train-{experiment_id}"
    verified = verify(train_dir)
    if verified["actual_total_steps"] != 1044480 or [item["actual_steps"] for item in verified["checkpoints"]] != [313344, 626688, 1044480]:
        raise ValueError("训练检查点与 B1 冻结协议不一致")
    export_id = f"E{experiment_id[1:]}{step}"
    export_dir = OUTPUTS_DIR / f"post-{export_id}"
    stage_parent = OUTPUTS_DIR / f"{phase}-stage-{experiment_id}-{step}"
    eval_dir = OUTPUTS_DIR / f"{phase}-v2-{experiment_id}-{step}"
    check_dir = OUTPUTS_DIR / f"{phase}-check-{experiment_id}-{step}"
    for path in (export_dir, stage_parent, eval_dir, check_dir):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    checkpoint_dir = train_dir / f"checkpoint-{step}"
    export_model(argparse.Namespace(model_dir=str(checkpoint_dir), experiment_id=export_id, num_obs=256))
    export_info = json.loads((export_dir / "export-info.json").read_text(encoding="utf-8"))
    if export_info["training_steps"] != step or export_info["consistency_check"] != "passed":
        raise ValueError("导出一致性检查未通过")

    stage = stage_parent / "P103"
    (stage / "artifacts").mkdir(parents=True)
    for name in ("entry.py", "requirements-infer.lock", "THIRD_PARTY.md", "LICENSE"):
        shutil.copyfile(PARTICIPANT_DIR / name, stage / name)
    shutil.copyfile(export_dir / "policy.npz", stage / "artifacts" / "policy.npz")
    model_hash = sha256(stage / "artifacts" / "policy.npz")
    if model_hash != export_info["candidate_sha256"]:
        raise ValueError("评测副本与导出模型摘要不同")
    manifest = yaml.safe_load((PARTICIPANT_DIR / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{
        "path": "artifacts/policy.npz", "sha256": model_hash,
        "size_bytes": (stage / "artifacts" / "policy.npz").stat().st_size,
    }]
    manifest["metadata"] = {"description": f"P103 {phase.upper()} 开发验证副本：{experiment_id}，训练样本 {step}"}
    launcher = "run_b1.py" if phase == "b3" else "run_c.py"
    manifest["train_command"] = f"participant/P103/{launcher} --experiment-id {experiment_id}"
    (stage / "submission.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (stage / "LOG.md").write_text(f"# {phase.upper()} 开发验证副本\n\n{experiment_id}，检查点 {step}。仅用于本地开发验证。\n", encoding="utf-8")
    (stage / "REPORT.md").write_text(f"# {phase.upper()} 开发验证副本\n\n来源：`{checkpoint_dir.relative_to(REPO_ROOT).as_posix()}`。\n", encoding="utf-8")
    (stage / "experiments.csv").write_text("experiment_id,training_steps,model_sha256\n" + f"{experiment_id},{step},{model_hash}\n", encoding="utf-8")
    (stage_parent / "stage-manifest.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "checkpoint_steps": step,
        "checkpoint_path": checkpoint_dir.relative_to(REPO_ROOT).as_posix(),
        "export_info": (export_dir / "export-info.json").relative_to(REPO_ROOT).as_posix(),
        "submission": stage.relative_to(REPO_ROOT).as_posix(),
        "model_sha256": model_hash,
        "entry_sha256": sha256(stage / "entry.py"),
        "suite_sha256": sha256(SUITE),
        "seeds_sha256": sha256(SEEDS),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    subprocess.run([str(EVALUATION_PYTHON), "scripts/check_submission.py", "--submission", str(stage), "--output", str(check_dir)], cwd=REPO_ROOT, check=True)
    subprocess.run([str(EVALUATION_PYTHON), "scripts/evaluate_one.py", "--submission", str(stage), "--suite", str(SUITE), "--seeds", str(SEEDS), "--output", str(eval_dir)], cwd=REPO_ROOT, check=True)
    result = json.loads((eval_dir / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["errors"] or result["artifact_hashes"]["artifacts/policy.npz"] != model_hash:
        raise ValueError("开发验证结果失败或权重摘要不一致")
    if result["suite_id"] != "p103-validation-v2" or result["seed_schedule_id"] != "p103-validation-seeds-v2":
        raise ValueError("开发验证所用套件或种子安排不符合 B1 协议")
    print(json.dumps({"experiment_id": experiment_id, "step": step, "score": result["performance_score"], "result": (eval_dir / "result.json").relative_to(REPO_ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E011", "E012", "E013", "E014", "E015", "E016"), required=True)
    parser.add_argument("--step", type=int, choices=(313344, 626688, 1044480), required=True)
    args = parser.parse_args()
    main(args.experiment_id, args.step)
