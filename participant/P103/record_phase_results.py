"""把已核查的 B3 或 C 检查点评测逐项追加到实验总表，拒绝重复编号。"""

import argparse
import csv
from datetime import date
import io
import json
from pathlib import Path


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
CSV_PATH = PARTICIPANT_DIR / "experiments.csv"


def main(phase):
    if phase not in ("b3", "c"):
        raise ValueError("phase 必须为 b3 或 c")
    summary_dir = OUTPUTS_DIR / ("b3-summary-001" if phase == "b3" else "c-summary-001")
    summary = json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
    if len(summary["results"]) != 9:
        raise ValueError("汇总记录必须包含九份模型")
    with CSV_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        existing = list(reader)
    if fields is None:
        raise ValueError("实验总表没有表头")
    existing_ids = {row["experiment_id"] for row in existing}
    generated = io.StringIO(newline="")
    writer = csv.DictWriter(generated, fieldnames=fields, lineterminator="\n")
    new_ids = []
    for result in summary["results"]:
        experiment_id = result["experiment_id"]
        step = result["checkpoint_steps"]
        row_id = experiment_id if step == 1044480 else f"{phase.upper()}-{experiment_id}-{step}"
        if row_id in existing_ids or row_id in new_ids:
            raise FileExistsError(f"实验总表编号已存在：{row_id}")
        meta_path = OUTPUTS_DIR / f"train-{experiment_id}" / f"checkpoint-{step}" / "training_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["total_steps"] != step or meta["seed"] != result["training_seed"] or meta["protocol_sha256"] != summary["protocol_sha256"]:
            raise ValueError(f"{row_id} 训练元数据与评测汇总不一致")
        entry = {key: "" for key in fields}
        entry.update({
            "experiment_id": row_id,
            "date": date.today().isoformat(),
            "kind": f"ppo_{phase}_checkpoint",
            "source_commit": meta["source_commit"],
            "model_sha256": result["model_sha256"],
            "suite_id": "p103-validation-v2",
            "seed_schedule_id": "p103-validation-seeds-v2",
            "run_id": result["run_id"],
            "output_dir": str(Path(result["result_path"]).parent).replace("\\", "/"),
            "provenance": "local_preview",
            "status": "ok",
            "episodes_ok": 200,
            "episodes_total": 200,
            "performance_score": result["performance_score"],
            "basic_mean_coverage_rate": result["basic_mean_coverage_rate"],
            "basic_mean_collision_rate": result["basic_mean_collision_rate"],
            "cooperation_mean_coverage_rate": result["cooperation_mean_coverage_rate"],
            "cooperation_mean_collision_rate": result["cooperation_mean_collision_rate"],
            "notes": f"{phase.upper()} fixed checkpoint; A2-corrected training; protocol_sha256={summary['protocol_sha256']}; current E009 submission unchanged",
            "train_case": meta["train_case"],
            "training_seed": result["training_seed"],
            "training_steps": step,
            "learning_rate": meta["learning_rate"],
            "wall_seconds": meta["wall_seconds"],
        })
        writer.writerow(entry)
        new_ids.append(row_id)
    with CSV_PATH.open("a", encoding="utf-8", newline="") as target:
        target.write(generated.getvalue())
    print("已记录：" + ", ".join(new_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("b3", "c"), required=True)
    main(parser.parse_args().phase)
