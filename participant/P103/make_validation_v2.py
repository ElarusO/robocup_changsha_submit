"""生成并核查 P103 的开发验证 v2 与独立最终确认集。不会覆盖现有文件。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import yaml

from coverage_bench.config import validate_task_config
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.schedules import create_seed_schedule, seed_schedule_from_data
from coverage_bench.suites import load_suite


PARTICIPANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARTICIPANT_DIR.parents[1]
PUBLIC_SUITE = REPO_ROOT / "configs" / "public-suite-v1.yaml"
PUBLIC_SCHEDULE = REPO_ROOT / "configs" / "public-seeds-v1.json"
OLD_SUITE = PARTICIPANT_DIR / "validation-suite-v1.yaml"
OLD_SCHEDULE = PARTICIPANT_DIR / "validation-seeds-v1.json"
MANIFEST = PARTICIPANT_DIR / "validation-sets-v2-manifest.json"

SETS = {
    "development": {
        "suite": "validation-suite-v2.yaml",
        "schedule": "validation-seeds-v2.json",
        "suite_id": "p103-validation-v2",
        "schedule_id": "p103-validation-seeds-v2",
        "prefixes": {"basic": "v2b", "cooperation": "v2c"},
        "seed_bases": {"basic": 31000, "cooperation": 41000},
        "policy_entropy": 47319,
    },
    "final_confirmation": {
        "suite": "final-confirmation-suite-v1.yaml",
        "schedule": "final-confirmation-seeds-v1.json",
        "suite_id": "p103-final-confirmation-v1",
        "schedule_id": "p103-final-confirmation-seeds-v1",
        "prefixes": {"basic": "fcb", "cooperation": "fcc"},
        "seed_bases": {"basic": 51000, "cooperation": 61000},
        "policy_entropy": 57319,
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def all_seeds(suite, schedule):
    scenario = {case.scenario_seed for group in suite.groups for case in group.cases}
    policy = {seed for record in schedule.records for seed in (record.build_seed, record.policy_seed)}
    return scenario, policy


def existing_seed_sets():
    result = {}
    for name, suite_path, schedule_path in (
        ("public", PUBLIC_SUITE, PUBLIC_SCHEDULE),
        ("validation_v1", OLD_SUITE, OLD_SCHEDULE),
    ):
        suite = load_suite(suite_path)
        schedule = seed_schedule_from_data(json.loads(schedule_path.read_text(encoding="utf-8")), str(schedule_path))
        result[name] = all_seeds(suite, schedule)
    return result


def make_suite(source, spec):
    data = deepcopy(source)
    data["suite_id"] = spec["suite_id"]
    data["schedule_id"] = spec["schedule_id"]
    for group in data["groups"]:
        group_id = group["group_id"]
        template = group["cases"][0]
        group["policy_repeats"] = 2
        group["cases"] = []
        for index in range(50):
            case = deepcopy(template)
            case["case_id"] = f"{spec['prefixes'][group_id]}-{index:02d}"
            case["scenario_seed"] = spec["seed_bases"][group_id] + index
            group["cases"].append(case)
    return data


def check_set(suite, schedule, spec, used_scenario, used_policy):
    assert suite.suite_id == spec["suite_id"]
    assert suite.schedule_id == schedule.schedule_id == spec["schedule_id"]
    assert {group.group_id for group in suite.groups} == {"basic", "cooperation"}
    assert all(len(group.cases) == 50 and group.policy_repeats == 2 for group in suite.groups)
    cases = [case for group in suite.groups for case in group.cases]
    assert len(cases) == 100 and len({case.case_id for case in cases}) == 100
    assert all(case.task_config.num_agents == 3 and case.task_config.num_targets == 3 for case in cases)
    for case in cases:
        validate_task_config(case.task_config, get_protocol_spec())

    scenario, policy = all_seeds(suite, schedule)
    assert len(scenario) == 100, "本套件内部的场景种子重复"
    assert len(policy) == 1200, "本套件内部的构建/策略种子重复"
    assert not (scenario & used_scenario), "场景种子与已有套件重叠"
    assert not (policy & used_policy), "构建/策略种子与已有套件重叠"
    assert not (scenario & policy), "场景种子与构建/策略种子重叠"
    assert not ({7101, 7102, 7103} & (scenario | policy)), "预留训练种子与验证集重叠"
    expected_keys = {
        (case.case_id, group.group_id, repeat, agent)
        for group in suite.groups for case in group.cases
        for repeat in range(group.policy_repeats)
        for agent in range(case.task_config.num_agents)
    }
    actual_keys = {
        (record.case_id, record.group_id, record.repeat_index, record.agent_index)
        for record in schedule.records
    }
    assert len(schedule.records) == len(expected_keys) == 600
    assert actual_keys == expected_keys, "种子安排没有精确覆盖全部场景、重复和机器人"
    expected_schedule = create_seed_schedule(suite, policy_entropy=spec["policy_entropy"])
    assert schedule.schedule_hash == expected_schedule.schedule_hash
    assert schedule.records == expected_schedule.records
    return scenario, policy


def validate_all(manifest=None):
    used = existing_seed_sets()
    used_scenario = set().union(*(pair[0] for pair in used.values()))
    used_policy = set().union(*(pair[1] for pair in used.values()))
    assert not (used_scenario & used_policy)
    result = {}
    for name, spec in SETS.items():
        suite_path = PARTICIPANT_DIR / spec["suite"]
        schedule_path = PARTICIPANT_DIR / spec["schedule"]
        suite = load_suite(suite_path)
        schedule = seed_schedule_from_data(json.loads(schedule_path.read_text(encoding="utf-8")), str(schedule_path))
        scenario, policy = check_set(suite, schedule, spec, used_scenario, used_policy)
        used_scenario |= scenario
        used_policy |= policy
        result[name] = {
            "suite": spec["suite"],
            "suite_sha256": sha256_bytes(suite_path.read_bytes()),
            "schedule": spec["schedule"],
            "schedule_sha256": sha256_bytes(schedule_path.read_bytes()),
            "schedule_hash": schedule.schedule_hash,
            "policy_entropy": spec["policy_entropy"],
            "groups": {group.group_id: len(group.cases) for group in suite.groups},
            "unique_scenarios": len(scenario),
            "episodes": sum(len(group.cases) * group.policy_repeats for group in suite.groups),
            "schedule_records": len(schedule.records),
        }
    if manifest is not None:
        assert manifest == result, "验证集与封存清单不一致；请检查文件是否被修改"
    return result


def create():
    paths = [MANIFEST] + [PARTICIPANT_DIR / spec[key] for spec in SETS.values() for key in ("suite", "schedule")]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"目标文件已存在，不覆盖：{existing}")
    source = yaml.safe_load(PUBLIC_SUITE.read_text(encoding="utf-8"))
    for spec in SETS.values():
        suite_path = PARTICIPANT_DIR / spec["suite"]
        suite_data = make_suite(source, spec)
        with suite_path.open("x", encoding="utf-8") as output:
            output.write(yaml.safe_dump(suite_data, sort_keys=False, allow_unicode=True))
        suite = load_suite(suite_path)
        schedule = create_seed_schedule(suite, policy_entropy=spec["policy_entropy"])
        schedule_data = {
            "schedule_schema_version": schedule.schedule_schema_version,
            "schedule_id": schedule.schedule_id,
            "schedule_hash": schedule.schedule_hash,
            "records": [asdict(record) for record in schedule.records],
        }
        with (PARTICIPANT_DIR / spec["schedule"]).open("x", encoding="utf-8") as output:
            output.write(json.dumps(schedule_data, ensure_ascii=False, indent=2) + "\n")
    result = validate_all()
    with MANIFEST.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="只检查已生成文件，不生成或覆盖")
    args = parser.parse_args()
    if args.verify:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        result = validate_all(expected)
    else:
        result = create()
    print(json.dumps(result, ensure_ascii=False, indent=2))
