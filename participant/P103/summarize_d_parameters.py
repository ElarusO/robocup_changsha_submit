"""汇总 D 阶段有界参数搜索的训练记录、三种子对照与停止理由。"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "P103"
SOURCE = OUT / "d-parameter-control-001" / "progress.json"
DEST = OUT / "d-parameter-summary-001"
SEEDS = (7101, 7102, 7103)


def episodes(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["group_id"], row["case_id"], int(row["repeat_index"])): float(row["mean_j"]) for row in rows}
    if len(rows) != 200 or len(result) != 200 or any(row["status"] != "ok" for row in rows):
        raise ValueError(f"回合数据不完整：{path}")
    return result


def bootstrap_full(trials):
    arrays = {}
    for i, seed in enumerate(SEEDS):
        base = episodes(OUT / f"c-v2-E0{14+i}-1044480" / "episodes.csv")
        candidate = episodes(OUT / f"d-param-v2-E0{26+i}" / "episodes.csv")
        if set(base) != set(candidate):
            raise ValueError("完整预算的场景键不一致")
        arrays[seed] = {}
        for group, prefix in (("basic", "v2b"), ("cooperation", "v2c")):
            arrays[seed][group] = np.asarray([np.mean([
                candidate[(group, f"{prefix}-{j:02d}", repeat)] - base[(group, f"{prefix}-{j:02d}", repeat)]
                for repeat in (0, 1)]) for j in range(50)])
    rng = np.random.default_rng(20260926)
    indices = {group: rng.integers(0, 50, size=(10000, 50)) for group in ("basic", "cooperation")}
    sampled = np.mean(np.stack([500 * sum(arrays[seed][group][indices[group]].mean(axis=1) for group in indices)
                                for seed in SEEDS]), axis=0)
    return {"mean_paired_score_difference": float(trials["full_comparison"]["mean_gain"]),
            "fixed_training_seed_scene_bootstrap_ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "resamples": 10000,
            "note": "Fixed three training seeds; paired scene bootstrap does not represent uncertainty over future training seeds."}


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if "decision" not in data:
        raise ValueError("D 参数搜索尚未停止")
    rows = []
    for label, ids in (("lr-low", ("E017", "E018", "E019")),
                       ("lr-high", ("E020", "E021", "E022")),
                       ("entropy", ("E023", "E024", "E025"))):
        cohort = [data["trials"][experiment_id] for experiment_id in ids]
        rows.append({"phase": label, "learning_rate": cohort[0]["learning_rate"], "ent_coef": cohort[0]["ent_coef"],
                     "samples": 313344, "score_7101": cohort[0]["score"], "score_7102": cohort[1]["score"],
                     "score_7103": cohort[2]["score"], "mean_score": float(np.mean([row["score"] for row in cohort])),
                     "mean_gain_vs_selected_control": data["screens"][label]["mean_gain"],
                     "passes": data["screens"][label]["passes"]})
    if "full_comparison" in data:
        cohort = [data["trials"][f"E0{26+i}"] for i in range(3)]
        rows.append({"phase": "full-selected", "learning_rate": cohort[0]["learning_rate"], "ent_coef": cohort[0]["ent_coef"],
                     "samples": 1044480, "score_7101": cohort[0]["score"], "score_7102": cohort[1]["score"],
                     "score_7103": cohort[2]["score"], "mean_score": float(np.mean([row["score"] for row in cohort])),
                     "mean_gain_vs_selected_control": data["full_comparison"]["mean_gain"],
                     "passes": data["full_comparison"]["passes"]})
    summary = {"protocol_sha256": data["protocol_sha256"], "short_baseline_mean": float(np.mean([row["score"] for row in data["baseline_short"]])),
               "full_baseline_mean": float(np.mean([row["score"] for row in data["baseline_full"]])),
               "phases": rows, "screens": data["screens"], "selected_short_configuration": data["selected_short_configuration"],
               "full_comparison": data.get("full_comparison"), "decision": data["decision"],
               "full_paired_scene_bootstrap": bootstrap_full(data) if "full_comparison" in data else None}
    DEST.mkdir(parents=True)
    with (DEST / "phases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
