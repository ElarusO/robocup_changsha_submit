"""对固定 C 模型运行冻结的 E3 条件目标策略；仅通过初筛后才用 combo 模式。"""

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
PROTOCOL = HERE / "e3-conditional-target-protocol-v1.json"
PROTOCOL_SHA = "8f237872840831559c8682d0d567e9b2054d5d440b12ea6877c9d58363251bf9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"冻结文件缺失或变化：{path}")


def run(command):
    print("运行：", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main(experiment_id, mode):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("当前分支不是 P103")
    checked(PROTOCOL, PROTOCOL_SHA)
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fixed = plan["fixed_inputs"]
    for path, digest in ((plan["fixed_rule"]["new_rule_source"], plan["fixed_rule"]["new_rule_sha256"]),
                         (fixed["base_entry"], fixed["base_entry_sha256"]),
                         (fixed["old_factor_source"], fixed["old_factor_sha256"])):
        checked(ROOT / path, digest)
    model = fixed["models"][experiment_id]
    checked(ROOT / model["path"], model["sha256"])
    for suite in ("v2", "v3"):
        checked(HERE / f"validation-suite-{suite}.yaml", fixed[f"{suite}_suite_sha256"])
        checked(HERE / f"validation-seeds-{suite}.json", fixed[f"{suite}_seeds_sha256"])

    tag = f"{experiment_id}-{mode}"
    parent = OUT / f"e3-stage-{tag}"
    stage = parent / "P103"
    check = OUT / f"e3-check-{tag}"
    evaluation_dirs = {suite: OUT / f"e3-{suite}-{tag}" for suite in ("v2", "v3")}
    if any(path.exists() for path in (parent, check, *evaluation_dirs.values())):
        raise FileExistsError(f"E3 输出目录已被占用：{tag}")
    (stage / "artifacts").mkdir(parents=True)
    for name in ("requirements-infer.lock", "THIRD_PARTY.md", "LICENSE", "LOG.md", "REPORT.md", "experiments.csv"):
        shutil.copyfile(HERE / name, stage / name)
    shutil.copyfile(HERE / "entry.py", stage / "entry_base.py")
    shutil.copyfile(HERE / "single_factor_policy.py", stage / "single_factor_policy.py")
    shutil.copyfile(HERE / "e3_conditional_target.py", stage / "e3_conditional_target.py")
    artifact = stage / "artifacts" / "policy.npz"
    shutil.copyfile(ROOT / model["path"], artifact)
    (stage / "entry.py").write_text(
        "from entry_base import build_policy as build_base_policy\n"
        "from e3_conditional_target import ConditionalTargetPolicy\n\n"
        "def build_policy(context):\n"
        f"    return ConditionalTargetPolicy(build_base_policy(context), with_avoidance={mode == 'combo'})\n",
        encoding="utf-8",
    )
    manifest = yaml.safe_load((HERE / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{"path": "artifacts/policy.npz", "sha256": sha(artifact),
                                         "size_bytes": artifact.stat().st_size}]
    manifest["metadata"] = {"description": f"P103 E3 本地开发验证 {tag}；正式 E009 不变"}
    (stage / "submission.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (parent / "provenance.json").write_text(json.dumps({
        "protocol_sha256": PROTOCOL_SHA, "experiment_id": experiment_id, "mode": mode,
        "model_sha256": sha(artifact), "entry_base_sha256": sha(stage / "entry_base.py"),
        "old_factor_sha256": sha(stage / "single_factor_policy.py"),
        "new_rule_sha256": sha(stage / "e3_conditional_target.py"),
        "entry_sha256": sha(stage / "entry.py"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run([sys.executable, ROOT / "scripts" / "check_submission.py", "--submission", stage, "--output", check])
    code_hash = None
    for suite, output in evaluation_dirs.items():
        run([sys.executable, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
             "--suite", HERE / f"validation-suite-{suite}.yaml",
             "--seeds", HERE / f"validation-seeds-{suite}.json", "--output", output])
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if (result["status"] != "ok" or result["errors"] or
                result["artifact_hashes"]["artifacts/policy.npz"] != model["sha256"] or
                result["suite_id"] != f"p103-validation-{suite}" or
                result["seed_schedule_id"] != f"p103-validation-seeds-{suite}" or
                sum(group["num_episodes"] for group in result["group_metrics"].values()) != 200):
            raise ValueError(f"E3 评测未完整通过：{tag}/{suite}")
        if code_hash is not None and code_hash != result["code_tree_hash"]:
            raise ValueError(f"两个套件使用了不同策略代码：{tag}")
        code_hash = result["code_tree_hash"]
        print(json.dumps({"tag": tag, "suite": suite, "score": result["performance_score"],
                          "basic": result["group_metrics"]["basic"],
                          "cooperation": result["group_metrics"]["cooperation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E014", "E015", "E016"), required=True)
    parser.add_argument("--mode", choices=("target", "combo"), default="target")
    args = parser.parse_args()
    main(args.experiment_id, args.mode)
