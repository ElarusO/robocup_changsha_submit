"""按冻结 D 协议自动训练、导出、预检、评测并执行有界单变量筛选。"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "P103"
PROTOCOL = HERE / "d-parameter-protocol-v1.json"
PROTOCOL_SHA = "ef3d2f9af85822e76756403c637bcaa053c7ac3ad640d4ef43da90c32690fdbb"
TRAIN_PYTHON = ROOT / ".venv-train" / "Scripts" / "python.exe"
EVAL_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CONTROL = OUT / "d-parameter-control-001"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"冻结输入不一致：{path}")


def result_metrics(result):
    basic, coop = result["group_metrics"]["basic"], result["group_metrics"]["cooperation"]
    return {"score": result["performance_score"], "basic_coverage": basic["mean_coverage_rate"],
            "coop_coverage": coop["mean_coverage_rate"], "basic_collision": basic["mean_collision_rate"],
            "coop_collision": coop["mean_collision_rate"]}


def load_result(path, model_hash=None):
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result["status"] != "ok" or result["errors"] or result["suite_id"] != "p103-validation-v2" or
            result["seed_schedule_id"] != "p103-validation-seeds-v2" or
            sum(group["num_episodes"] for group in result["group_metrics"].values()) != 200):
        raise ValueError(f"不完整评测：{path}")
    if model_hash and result["artifact_hashes"]["artifacts/policy.npz"] != model_hash:
        raise ValueError(f"评测模型摘要不符：{path}")
    return result_metrics(result)


def run_logged(command, log_path):
    print("[command]", " ".join(str(part) for part in command), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        completed = subprocess.run([str(part) for part in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"命令失败 ({completed.returncode})，日志：{log_path}")


def trial(experiment_id, seed, learning_rate, ent_coef, full, plan):
    train_dir = OUT / f"train-{experiment_id}"
    export_dir = OUT / f"post-{experiment_id}"
    stage_parent = OUT / f"d-param-stage-{experiment_id}"
    check_dir = OUT / f"d-param-check-{experiment_id}"
    eval_dir = OUT / f"d-param-v2-{experiment_id}"
    files = [train_dir, export_dir, stage_parent, check_dir, eval_dir]
    if any(path.exists() for path in files):
        raise FileExistsError(f"{experiment_id} 有既存输出；拒绝覆盖或跳过：{[str(p) for p in files if p.exists()]}")
    requested = plan["fixed_training"]["full_requested_steps" if full else "short_requested_steps"]
    expected = plan["fixed_training"]["full_actual_steps" if full else "short_actual_steps"]
    curve = plan["fixed_training"]["full_curve_points" if full else "short_curve_points"]
    checkpoints = [313344, 626688, 1044480] if full else [313344]
    train_cmd = [TRAIN_PYTHON, HERE / "train.py", "--total-steps", requested, "--curve-points", curve,
                 "--num-vec-envs", 4, "--seed", seed, "--train-case", "mixed-basic0-coop0",
                 "--learning-rate", learning_rate, "--ent-coef", ent_coef, "--n-steps", 512,
                 "--batch-size", 256, "--gamma", 0.99, "--checkpoint-steps", *checkpoints,
                 "--protocol-path", PROTOCOL, "--out", train_dir]
    started = time.time()
    run_logged(train_cmd, CONTROL / f"{experiment_id}-train.log")
    run_manifest = json.loads((train_dir / "run_manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((train_dir / "training_meta.json").read_text(encoding="utf-8"))
    if (run_manifest["status"] != "complete" or run_manifest["actual_total_steps"] != expected or
            meta["seed"] != seed or meta["learning_rate"] != learning_rate or meta["ent_coef"] != ent_coef or
            meta["protocol_sha256"] != PROTOCOL_SHA or meta["train_script_sha256"] != plan["fixed_training"]["train_script_sha256"] or
            meta["mixed_env_sha256"] != plan["fixed_training"]["mixed_env_sha256"]):
        raise ValueError(f"{experiment_id} 训练记录与冻结配置不符")
    run_logged([TRAIN_PYTHON, HERE / "postprocess.py", "export", "--model-dir", train_dir,
                "--experiment-id", experiment_id, "--num-obs", 256], CONTROL / f"{experiment_id}-export.log")
    info = json.loads((export_dir / "export-info.json").read_text(encoding="utf-8"))
    artifact = export_dir / "policy.npz"
    if info["consistency_check"] != "passed" or sha(artifact) != info["candidate_sha256"] or info["training_steps"] != expected:
        raise ValueError(f"{experiment_id} 导出校验不通过")
    stage = stage_parent / "P103"
    (stage / "artifacts").mkdir(parents=True)
    for name in ("entry.py", "requirements-infer.lock", "THIRD_PARTY.md", "LICENSE"):
        shutil.copyfile(HERE / name, stage / name)
    shutil.copyfile(artifact, stage / "artifacts" / "policy.npz")
    manifest = yaml.safe_load((HERE / "submission.yaml").read_text(encoding="utf-8"))
    manifest["checkpoint_manifest"] = [{"path": "artifacts/policy.npz", "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}]
    manifest["metadata"] = {"description": f"P103 D 训练参数候选 {experiment_id}，仅开发验证"}
    (stage / "submission.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (stage / "LOG.md").write_text(f"# D 候选 {experiment_id}\n\n训练记录见 {train_dir.relative_to(ROOT).as_posix()}。\n", encoding="utf-8")
    (stage / "REPORT.md").write_text(f"# D 候选 {experiment_id}\n\n此目录仅用于开发验证。\n", encoding="utf-8")
    (stage / "experiments.csv").write_text("experiment_id,model_sha256\n" + f"{experiment_id},{sha(artifact)}\n", encoding="utf-8")
    (stage_parent / "provenance.json").write_text(json.dumps({"experiment_id": experiment_id, "training_seed": seed,
       "learning_rate": learning_rate, "ent_coef": ent_coef, "actual_steps": expected, "model_sha256": sha(artifact),
       "protocol_sha256": PROTOCOL_SHA, "entry_sha256": sha(stage / "entry.py")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_logged([EVAL_PYTHON, ROOT / "scripts" / "check_submission.py", "--submission", stage, "--output", check_dir], CONTROL / f"{experiment_id}-check.log")
    audit = json.loads((check_dir / "audit-report.json").read_text(encoding="utf-8"))
    if audit["rejections"]:
        raise ValueError(f"{experiment_id} 预检拒绝")
    run_logged([EVAL_PYTHON, ROOT / "scripts" / "evaluate_one.py", "--submission", stage,
                "--suite", ROOT / plan["development_evaluation"]["suite"],
                "--seeds", ROOT / plan["development_evaluation"]["seeds"], "--output", eval_dir],
               CONTROL / f"{experiment_id}-eval.log")
    metrics = load_result(eval_dir / "result.json", sha(artifact))
    record = {"experiment_id": experiment_id, "seed": seed, "learning_rate": learning_rate,
              "ent_coef": ent_coef, "actual_steps": expected, "model_sha256": sha(artifact),
              "elapsed_seconds": round(time.time() - started, 1), **metrics}
    print("[result]", json.dumps(record, ensure_ascii=False), flush=True)
    return record


def evaluate_candidate(candidate, baseline):
    gains = [item["score"] - other["score"] for item, other in zip(candidate, baseline)]
    basic_change = sum(item["basic_coverage"] - other["basic_coverage"] for item, other in zip(candidate, baseline)) / 3
    coop_change = sum(item["coop_coverage"] - other["coop_coverage"] for item, other in zip(candidate, baseline)) / 3
    mean_gain = sum(gains) / 3
    return {"paired_score_gains": gains, "mean_gain": mean_gain, "improved_models": sum(x > 0 for x in gains),
            "mean_basic_coverage_change": basic_change, "mean_coop_coverage_change": coop_change,
            "passes": sum(x > 0 for x in gains) >= 2 and mean_gain >= 5 and basic_change >= -0.01 and coop_change >= -0.01}


def save_progress(data):
    (CONTROL / "progress.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "P103":
        raise ValueError("当前分支不是 P103")
    checked(PROTOCOL, PROTOCOL_SHA)
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checked(HERE / "train.py", plan["fixed_training"]["train_script_sha256"])
    checked(HERE / "mixed_env.py", plan["fixed_training"]["mixed_env_sha256"])
    for key, hash_key in (("suite", "suite_sha256"), ("seeds", "seeds_sha256")):
        checked(ROOT / plan["development_evaluation"][key], plan["development_evaluation"][hash_key])
    if not TRAIN_PYTHON.is_file() or not EVAL_PYTHON.is_file():
        raise FileNotFoundError("训练或评测虚拟环境缺失")
    if CONTROL.exists():
        raise FileExistsError(f"自动控制目录已存在，不会重启覆盖：{CONTROL}")
    for entry in plan["short_order"]:
        for experiment_id in entry["experiment_ids"]:
            if any((OUT / f"{prefix}{experiment_id}").exists() for prefix in ("train-", "post-", "d-param-stage-", "d-param-v2-")):
                raise FileExistsError(f"实验编号已占用：{experiment_id}")
    CONTROL.mkdir()
    baseline_short = [load_result(OUT / f"c-v2-E0{14+i}-313344" / "result.json") for i in range(3)]
    baseline_full = [load_result(OUT / f"c-v2-E0{14+i}-1044480" / "result.json") for i in range(3)]
    progress = {"protocol_sha256": PROTOCOL_SHA, "baseline_short": baseline_short, "baseline_full": baseline_full, "trials": {}, "screens": {}}
    save_progress(progress)
    seeds = plan["fixed_training"]["training_seeds"]
    candidates = {}
    for entry in plan["short_order"][:2]:
        label = entry["label"]
        candidates[label] = []
        print(f"[phase] {label}: learning_rate={entry['learning_rate']}", flush=True)
        for experiment_id, seed in zip(entry["experiment_ids"], seeds):
            row = trial(experiment_id, seed, entry["learning_rate"], 0.0, False, plan)
            candidates[label].append(row)
            progress["trials"][experiment_id] = row
            save_progress(progress)
        progress["screens"][label] = evaluate_candidate(candidates[label], baseline_short)
        save_progress(progress)
        print("[screen]", label, json.dumps(progress["screens"][label], ensure_ascii=False), flush=True)
    eligible = [label for label in ("lr-low", "lr-high") if progress["screens"][label]["passes"]]
    selected_lr_label = max(eligible, key=lambda label: progress["screens"][label]["mean_gain"]) if eligible else "baseline"
    selected_lr = {"lr-low": .00005, "lr-high": .0002, "baseline": .0001}[selected_lr_label]
    selected_short = baseline_short if selected_lr_label == "baseline" else candidates[selected_lr_label]
    print(f"[phase] entropy: selected_lr={selected_lr} from {selected_lr_label}", flush=True)
    entry = plan["short_order"][2]
    candidates["entropy"] = []
    for experiment_id, seed in zip(entry["experiment_ids"], seeds):
        row = trial(experiment_id, seed, selected_lr, .005, False, plan)
        candidates["entropy"].append(row)
        progress["trials"][experiment_id] = row
        save_progress(progress)
    progress["screens"]["entropy"] = evaluate_candidate(candidates["entropy"], selected_short)
    save_progress(progress)
    print("[screen] entropy", json.dumps(progress["screens"]["entropy"], ensure_ascii=False), flush=True)
    selected_ent = .005 if progress["screens"]["entropy"]["passes"] else 0.0
    progress["selected_short_configuration"] = {"learning_rate": selected_lr, "ent_coef": selected_ent,
        "learning_rate_basis": selected_lr_label, "entropy_passed": selected_ent > 0}
    save_progress(progress)
    if selected_lr_label == "baseline" and selected_ent == 0:
        progress["decision"] = "short-screen plateau; retain C training configuration; no full retrain"
        save_progress(progress)
        print("[stop]", progress["decision"], flush=True)
        return
    print("[phase] full three-seed verification", flush=True)
    full_rows = []
    for experiment_id, seed in zip(plan["full_experiment_ids"], seeds):
        row = trial(experiment_id, seed, selected_lr, selected_ent, True, plan)
        full_rows.append(row)
        progress["trials"][experiment_id] = row
        save_progress(progress)
    progress["full_comparison"] = evaluate_candidate(full_rows, baseline_full)
    progress["decision"] = ("selected configuration becomes a development candidate" if progress["full_comparison"]["passes"]
        else "full verification did not meet threshold; retain C training configuration; stop")
    save_progress(progress)
    print("[stop]", progress["decision"], json.dumps(progress["full_comparison"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
