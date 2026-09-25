"""生成 P103 固定的本地验证套件；仅供模型选择，不作为官方成绩。"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from coverage_bench.schedules import create_seed_schedule
from coverage_bench.suites import load_suite


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
SUITE_PATH = PARTICIPANT_DIR / "validation-suite-v1.yaml"
SEEDS_PATH = PARTICIPANT_DIR / "validation-seeds-v1.json"


def main() -> None:
    if SUITE_PATH.exists() or SEEDS_PATH.exists():
        raise FileExistsError("本地验证套件已存在；为保证对照一致，不覆盖已有文件")

    source = yaml.safe_load((REPO_ROOT / "configs" / "public-suite-v1.yaml").read_text(encoding="utf-8"))
    source["suite_id"] = "p103-validation-v1"
    source["schedule_id"] = "p103-validation-seeds-v1"
    for group in source["groups"]:
        template = group["cases"][0]
        prefix, seed_base = ("vb", 10000) if group["group_id"] == "basic" else ("vc", 20000)
        group["cases"] = []
        group["policy_repeats"] = 2
        for index in range(10):
            case = deepcopy(template)
            case["case_id"] = f"{prefix}-{index:02d}"
            case["scenario_seed"] = seed_base + index
            group["cases"].append(case)

    SUITE_PATH.write_text(yaml.safe_dump(source, sort_keys=False, allow_unicode=True), encoding="utf-8")
    suite = load_suite(SUITE_PATH)
    schedule = create_seed_schedule(suite, policy_entropy=47318)
    data = {
        "schedule_schema_version": schedule.schedule_schema_version,
        "schedule_id": schedule.schedule_id,
        "schedule_hash": schedule.schedule_hash,
        "records": [asdict(record) for record in schedule.records],
    }
    SEEDS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"生成 {sum(len(group.cases) * group.policy_repeats for group in suite.groups)} 个本地验证回合")
    print(SUITE_PATH)
    print(SEEDS_PATH)


if __name__ == "__main__":
    main()
