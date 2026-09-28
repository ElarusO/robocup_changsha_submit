"""在新场景 v3 上复测固定 C 模型和已实现的单因素动作策略。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
PLAN = json.loads((HERE / "d-single-factor-protocol-v1.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((HERE / "validation-v3-manifest.json").read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(experiment_id, mode):
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("分支不是 P103")
    suite, seeds = HERE / "validation-suite-v3.yaml", HERE / "validation-seeds-v3.json"
    if sha(suite) != MANIFEST["suite_sha256"] or sha(seeds) != MANIFEST["seeds_sha256"]:
        raise ValueError("v3 验证集被修改")
    model = next(item for item in PLAN["models"] if item["experiment_id"] == experiment_id)
    if mode == "base":
        stage = OUT / f"c-stage-{experiment_id}-1044480" / "P103"
    else:
        stage = OUT / f"d-stage-{experiment_id}-{mode}" / "P103"
    if not stage.is_dir() or sha(stage / "artifacts" / "policy.npz") != model["model_sha256"]:
        raise ValueError("实验副本缺失或模型摘要不一致")
    output = OUT / f"d-v3-{experiment_id}-{mode}"
    if output.exists():
        raise FileExistsError(f"v3 评测已存在，不覆盖：{output}")
    subprocess.run([sys.executable, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
                    "--suite", suite, "--seeds", seeds, "--output", output], cwd=ROOT, check=True)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if (result["status"] != "ok" or result["errors"] or
            result["artifact_hashes"]["artifacts/policy.npz"] != model["model_sha256"] or
            result["suite_id"] != "p103-validation-v3" or result["seed_schedule_id"] != "p103-validation-seeds-v3" or
            sum(group["num_episodes"] for group in result["group_metrics"].values()) != 200):
        raise ValueError("v3 评测未完成")
    print(json.dumps({"model": experiment_id, "mode": mode, "score": result["performance_score"],
                      "basic_coverage": result["group_metrics"]["basic"]["mean_coverage_rate"],
                      "coop_coverage": result["group_metrics"]["cooperation"]["mean_coverage_rate"],
                      "coop_collision": result["group_metrics"]["cooperation"]["mean_collision_rate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", choices=("E014", "E015", "E016"), required=True)
    parser.add_argument("--mode", choices=("base", "target", "avoid"), required=True)
    args = parser.parse_args()
    main(args.experiment_id, args.mode)
