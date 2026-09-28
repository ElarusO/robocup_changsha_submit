"""汇总 D 阶段六次单因素评测与原 C 模型的配对结果。"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "P103"
DEST = OUT / "d-single-factor-summary-001"
PLAN = json.loads((ROOT / "participant" / "P103" / "d-single-factor-protocol-v1.json").read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["group_id"], row["case_id"], int(row["repeat_index"])): row for row in rows}
    if len(rows) != 200 or len(result) != 200 or any(row["status"] != "ok" for row in rows):
        raise ValueError(f"非完整配对结果：{path}")
    return result


def metrics(result):
    basic, coop = result["group_metrics"]["basic"], result["group_metrics"]["cooperation"]
    return {
        "score": result["performance_score"],
        "basic_coverage": basic["mean_coverage_rate"],
        "basic_collision": basic["mean_collision_rate"],
        "coop_coverage": coop["mean_coverage_rate"],
        "coop_collision": coop["mean_collision_rate"],
    }


def mean(values):
    return sum(values) / len(values)


def main():
    if DEST.exists():
        raise FileExistsError(f"不覆盖旧汇总：{DEST}")
    rows, cases = [], []
    for model in PLAN["models"]:
        label = model["experiment_id"]
        base_path = ROOT / model["baseline_result"]
        if sha(base_path) != model["baseline_result_sha256"]:
            raise ValueError(f"基线发生变化：{base_path}")
        base_result = json.loads(base_path.read_text(encoding="utf-8"))
        base_rows = records(base_path.parent / "episodes.csv")
        for mode in ("target", "avoid"):
            path = OUT / f"d-v2-{label}-{mode}" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            trial_rows = records(path.parent / "episodes.csv")
            if (result["status"] != "ok" or result["errors"] or result["artifact_hashes"]["artifacts/policy.npz"] != model["model_sha256"]
                    or result["suite_hash"] != base_result["suite_hash"] or result["seed_schedule_id"] != base_result["seed_schedule_id"]
                    or set(base_rows) != set(trial_rows)):
                raise ValueError(f"实验不可配对：{label}-{mode}")
            base, trial = metrics(base_result), metrics(result)
            item = {"model": label, "mode": mode, "baseline_result_sha256": sha(base_path), "trial_result_sha256": sha(path)}
            for key in base:
                item[f"baseline_{key}"] = base[key]
                item[f"trial_{key}"] = trial[key]
                item[f"delta_{key}"] = trial[key] - base[key]
            rows.append(item)
            for group, case_id in (("basic", "v2b-02"), ("cooperation", "v2c-07"), ("cooperation", "v2c-09")):
                keys = [(group, case_id, repeat) for repeat in (0, 1)]
                case = {"model": label, "mode": mode, "group": group, "case_id": case_id}
                for metric in ("mean_coverage_rate", "mean_collision_rate"):
                    bv = mean([float(base_rows[key][metric]) for key in keys])
                    tv = mean([float(trial_rows[key][metric]) for key in keys])
                    case[f"baseline_{metric}"] = bv
                    case[f"trial_{metric}"] = tv
                    case[f"delta_{metric}"] = tv - bv
                cases.append(case)
    by_mode = {}
    for mode in ("target", "avoid"):
        subset = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {f"mean_delta_{key}": mean([row[f"delta_{key}"] for row in subset]) for key in metrics(base_result)}
        by_mode[mode]["basic_coverage_improved_models"] = sum(row["delta_basic_coverage"] > 0 for row in subset)
        by_mode[mode]["coop_collision_reduced_models"] = sum(row["delta_coop_collision"] < 0 for row in subset)
    by_mode["target"]["passes_screen"] = (by_mode["target"]["basic_coverage_improved_models"] >= 2
        and by_mode["target"]["mean_delta_score"] > -5 and by_mode["target"]["mean_delta_coop_collision"] <= .005)
    by_mode["avoid"]["passes_screen"] = (by_mode["avoid"]["coop_collision_reduced_models"] >= 2
        and by_mode["avoid"]["mean_delta_score"] > -5 and by_mode["avoid"]["mean_delta_basic_coverage"] >= -.01)
    DEST.mkdir(parents=True)
    for name, data in (("paired-models.csv", rows), ("focus-cases.csv", cases)):
        with (DEST / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {"protocol_sha256": sha(ROOT / "participant" / "P103" / "d-single-factor-protocol-v1.json"),
               "paired_model_results": rows, "focus_cases": cases, "screening": by_mode,
               "interpretation": "Local development preview; focus cases are explanatory, not an independent holdout."}
    (DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_mode, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
