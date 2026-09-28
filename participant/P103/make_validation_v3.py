"""生成独立于开发 v2 与 A4 的新场景，用于 D 前的外推检查。"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import yaml

from coverage_bench.schedules import create_seed_schedule, seed_schedule_from_data
from coverage_bench.suites import load_suite


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = HERE / "validation-suite-v3.yaml"
SEEDS = HERE / "validation-seeds-v3.json"
MANIFEST = HERE / "validation-v3-manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def used_seed_sets():
    result = set()
    for prefix in (ROOT / "configs" / "public", HERE / "validation", HERE / "final-confirmation"):
        if prefix.parent == ROOT / "configs":
            suite_path, seeds_path = ROOT / "configs" / "public-suite-v1.yaml", ROOT / "configs" / "public-seeds-v1.json"
        elif prefix.name == "validation":
            suite_path, seeds_path = HERE / "validation-suite-v2.yaml", HERE / "validation-seeds-v2.json"
        else:
            suite_path, seeds_path = HERE / "final-confirmation-suite-v1.yaml", HERE / "final-confirmation-seeds-v1.json"
        suite = load_suite(suite_path)
        schedule = seed_schedule_from_data(json.loads(seeds_path.read_text(encoding="utf-8")), str(seeds_path))
        result.update(case.scenario_seed for group in suite.groups for case in group.cases)
        result.update(seed for record in schedule.records for seed in (record.build_seed, record.policy_seed))
    return result


def main():
    if any(path.exists() for path in (SUITE, SEEDS, MANIFEST)):
        raise FileExistsError("v3 验证文件已经存在，不覆盖")
    source = yaml.safe_load((ROOT / "configs" / "public-suite-v1.yaml").read_text(encoding="utf-8"))
    source["suite_id"] = "p103-validation-v3"
    source["schedule_id"] = "p103-validation-seeds-v3"
    for group in source["groups"]:
        template = group["cases"][0]
        group["policy_repeats"] = 2
        group["cases"] = []
        prefix, base = ("v3b", 71000) if group["group_id"] == "basic" else ("v3c", 81000)
        for index in range(50):
            case = deepcopy(template)
            case["case_id"] = f"{prefix}-{index:02d}"
            case["scenario_seed"] = base + index
            group["cases"].append(case)
    prior = used_seed_sets() | {7101, 7102, 7103}
    scenario = {case["scenario_seed"] for group in source["groups"] for case in group["cases"]}
    if scenario & prior or len(scenario) != 100:
        raise ValueError("新场景种子与已有场景/策略/训练种子冲突")
    SUITE.write_text(yaml.safe_dump(source, sort_keys=False, allow_unicode=True), encoding="utf-8")
    suite = load_suite(SUITE)
    schedule = create_seed_schedule(suite, policy_entropy=67319)
    assigned = [seed for record in schedule.records for seed in (record.build_seed, record.policy_seed)]
    if len(assigned) != 1200 or len(set(assigned)) != 1200 or set(assigned) & (prior | scenario):
        raise ValueError("新策略种子与已有种子冲突")
    SEEDS.write_text(json.dumps({"schedule_schema_version": schedule.schedule_schema_version,
                                 "schedule_id": schedule.schedule_id, "schedule_hash": schedule.schedule_hash,
                                 "records": [asdict(record) for record in schedule.records]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({"purpose": "new-scene check of C and D inference variants; not A4",
                                    "suite_sha256": sha(SUITE), "seeds_sha256": sha(SEEDS),
                                    "basic_scenario_seeds": [71000, 71049], "cooperation_scenario_seeds": [81000, 81049],
                                    "episodes": 200}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
