"""以三份固定 C 模型运行目标朝向+队友避让组合的本地对照。"""

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
PLAN = json.loads((HERE / "d-single-factor-protocol-v1.json").read_text(encoding="utf-8"))
V3 = json.loads((HERE / "validation-v3-manifest.json").read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(experiment_id):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("分支不是 P103")
    model = next(item for item in PLAN["models"] if item["experiment_id"] == experiment_id)
    for key, hash_key in (("entry_base", "entry_base_sha256"), ("factor_source", "factor_source_sha256")):
        if sha(ROOT / PLAN[key]) != PLAN[hash_key]:
            raise ValueError(f"单因素实现已变化：{key}")
    if sha(ROOT / model["model"]) != model["model_sha256"]:
        raise ValueError("模型权重已变化")
    if sha(HERE / "validation-suite-v3.yaml") != V3["suite_sha256"] or sha(HERE / "validation-seeds-v3.json") != V3["seeds_sha256"]:
        raise ValueError("v3 场景已变化")
    parent = OUT / f"d-stage-{experiment_id}-combo"
    stage = parent / "P103"
    check = OUT / f"d-check-{experiment_id}-combo"
    evaluations = [("v2", HERE / "validation-suite-v2.yaml", HERE / "validation-seeds-v2.json"),
                   ("v3", HERE / "validation-suite-v3.yaml", HERE / "validation-seeds-v3.json")]
    if any(path.exists() for path in (parent, check, *(OUT / f"d-{name}-{experiment_id}-combo" for name, _, _ in evaluations))):
        raise FileExistsError("组合实验已有输出，不覆盖")
    (stage / "artifacts").mkdir(parents=True)
    for name in ("requirements-infer.lock", "THIRD_PARTY.md", "LICENSE", "LOG.md", "REPORT.md", "experiments.csv"):
        shutil.copyfile(HERE / name, stage / name)
    shutil.copyfile(HERE / "entry.py", stage / "entry_base.py")
    shutil.copyfile(HERE / "single_factor_policy.py", stage / "single_factor_policy.py")
    shutil.copyfile(HERE / "combined_policy.py", stage / "combined_policy.py")
    artifact = stage / "artifacts" / "policy.npz"
    shutil.copyfile(ROOT / model["model"], artifact)
    (stage / "entry.py").write_text(
        "from entry_base import build_policy as build_base_policy\n"
        "from combined_policy import CombinedPolicy\n\n"
        "def build_policy(context):\n"
        "    return CombinedPolicy(build_base_policy(context))\n", encoding="utf-8")
    manifest = yaml.safe_load((HERE / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{"path": "artifacts/policy.npz", "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}]
    manifest["metadata"] = {"description": f"P103 组合策略开发验证 {experiment_id}；正式 E009 不变"}
    (stage / "submission.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (parent / "provenance.json").write_text(json.dumps({"experiment_id": experiment_id, "model_sha256": sha(artifact),
        "combo_protocol_sha256": sha(HERE / "d-combo-protocol-v1.json"), "combined_policy_sha256": sha(stage / "combined_policy.py"),
        "factor_source_sha256": sha(stage / "single_factor_policy.py"), "entry_base_sha256": sha(stage / "entry_base.py")},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, ROOT / "scripts" / "check_submission.py", "--submission", stage, "--output", check], cwd=ROOT, check=True)
    for name, suite, seeds in evaluations:
        result_dir = OUT / f"d-{name}-{experiment_id}-combo"
        subprocess.run([sys.executable, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
                        "--suite", suite, "--seeds", seeds, "--output", result_dir], cwd=ROOT, check=True)
        result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
        if (result["status"] != "ok" or result["errors"] or
                result["artifact_hashes"]["artifacts/policy.npz"] != model["model_sha256"] or
                result["suite_id"] != f"p103-validation-{name}" or
                sum(item["num_episodes"] for item in result["group_metrics"].values()) != 200):
            raise ValueError(f"{name} 组合评测未通过核对")
        print(json.dumps({"model": experiment_id, "suite": name, "score": result["performance_score"],
                          "basic_coverage": result["group_metrics"]["basic"]["mean_coverage_rate"],
                          "coop_coverage": result["group_metrics"]["cooperation"]["mean_coverage_rate"],
                          "coop_collision": result["group_metrics"]["cooperation"]["mean_collision_rate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E014", "E015", "E016"), required=True)
    main(parser.parse_args().experiment_id)
