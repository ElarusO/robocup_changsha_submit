"""按冻结方案制作临时提交副本，并对一个模型的一种动作修正做开发验证。"""

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
PROTOCOL = HERE / "d-single-factor-protocol-v1.json"
PROTOCOL_SHA = "4dfa883f2d7390a809784b38ead748f6431b6e51ce5690f3a293961e1878f446"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"冻结文件不一致：{path}")


def run(command):
    print("运行：", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main(experiment_id, mode):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("当前分支不是 P103")
    verified(PROTOCOL, PROTOCOL_SHA)
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = next(item for item in plan["models"] if item["experiment_id"] == experiment_id)
    verified(ROOT / plan["entry_base"], plan["entry_base_sha256"])
    verified(ROOT / plan["factor_source"], plan["factor_source_sha256"])
    verified(ROOT / model["model"], model["model_sha256"])
    verified(ROOT / model["baseline_result"], model["baseline_result_sha256"])
    verified(ROOT / plan["evaluation"]["suite"], plan["evaluation"]["suite_sha256"])
    verified(ROOT / plan["evaluation"]["seeds"], plan["evaluation"]["seeds_sha256"])

    tag = f"{experiment_id}-{mode}"
    stage_parent = OUT / f"d-stage-{tag}"
    check_dir = OUT / f"d-check-{tag}"
    eval_dir = OUT / f"d-v2-{tag}"
    if any(path.exists() for path in (stage_parent, check_dir, eval_dir)):
        raise FileExistsError(f"{tag} 输出已存在；不会覆盖")
    stage = stage_parent / "P103"
    artifact = stage / "artifacts" / "policy.npz"
    artifact.parent.mkdir(parents=True)
    for name in ("requirements-infer.lock", "THIRD_PARTY.md", "LICENSE", "LOG.md", "REPORT.md", "experiments.csv"):
        shutil.copyfile(HERE / name, stage / name)
    shutil.copyfile(HERE / "entry.py", stage / "entry_base.py")
    shutil.copyfile(HERE / "single_factor_policy.py", stage / "single_factor_policy.py")
    shutil.copyfile(ROOT / model["model"], artifact)
    (stage / "entry.py").write_text(
        "from entry_base import build_policy as build_base_policy\n"
        "from single_factor_policy import SingleFactorPolicy\n\n"
        "def build_policy(context):\n"
        f"    return SingleFactorPolicy(build_base_policy(context), {mode!r})\n",
        encoding="utf-8",
    )
    manifest = yaml.safe_load((HERE / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{"path": "artifacts/policy.npz", "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}]
    manifest["metadata"] = {"description": f"P103 D 单因素本地开发验证 {tag}；原始 C 模型不变"}
    (stage / "submission.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (stage_parent / "provenance.json").write_text(json.dumps({
        "protocol_sha256": PROTOCOL_SHA, "experiment_id": experiment_id, "mode": mode,
        "model_sha256": sha(artifact), "factor_sha256": sha(stage / "single_factor_policy.py"),
        "entry_base_sha256": sha(stage / "entry_base.py"), "entry_sha256": sha(stage / "entry.py"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run([sys.executable, ROOT / "scripts" / "check_submission.py", "--submission", stage, "--output", check_dir])
    run([sys.executable, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
         "--suite", ROOT / plan["evaluation"]["suite"], "--seeds", ROOT / plan["evaluation"]["seeds"],
         "--output", eval_dir])
    result = json.loads((eval_dir / "result.json").read_text(encoding="utf-8"))
    if (result["status"] != "ok" or result["errors"] or result["artifact_hashes"]["artifacts/policy.npz"] != model["model_sha256"]
            or sum(group["num_episodes"] for group in result["group_metrics"].values()) != 200
            or result["suite_id"] != "p103-validation-v2" or result["seed_schedule_id"] != "p103-validation-seeds-v2"):
        raise ValueError(f"{tag} 结果未通过核对")
    print(json.dumps({"tag": tag, "score": result["performance_score"], "groups": result["group_metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E014", "E015", "E016"), required=True)
    parser.add_argument("--mode", choices=("target", "avoid"), required=True)
    args = parser.parse_args()
    main(args.experiment_id, args.mode)
