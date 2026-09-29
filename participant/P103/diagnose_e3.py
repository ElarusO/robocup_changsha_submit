"""在已知 v2b-02 场景重放 E016 的旧动作、旧目标规则和 E3 条件规则。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.suites import load_suite
from e3_conditional_target import conditional_target_action
from entry import NumpyMLPPolicy
from single_factor_policy import align_to_visible_target


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "P103"
DEST = OUT / "e3-diagnose-001"
PROTOCOL = HERE / "e3-conditional-target-protocol-v1.json"
PROTOCOL_SHA = "8f237872840831559c8682d0d567e9b2054d5d440b12ea6877c9d58363251bf9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(case, model_path, mode):
    env = make_training_env(case.task_config)
    policies = [NumpyMLPPolicy(model_path.parent) for _ in range(case.task_config.num_agents)]
    timeline = []
    try:
        observations, _ = env.reset(seed=case.scenario_seed)
        agents = list(env.agents)
        for step in range(case.task_config.horizon):
            decisions, actions = [], {}
            for i, agent in enumerate(agents):
                observation = observations[agent]
                original = policies[i].act(observation)
                if mode == "base":
                    action = original
                elif mode == "target":
                    action = align_to_visible_target(original, observation)
                else:
                    action = conditional_target_action(original, observation)
                actions[agent] = action
                visible = np.flatnonzero(np.asarray(observation["target_visible"], dtype=bool))
                chosen = None
                cosine = None
                if len(visible):
                    offsets = np.asarray(observation["targets"], dtype=np.float64)[visible, :2]
                    distances = np.linalg.norm(offsets, axis=1)
                    picked = int(np.argmin(distances))
                    chosen = int(visible[picked])
                    magnitude = float(np.linalg.norm(original))
                    if magnitude > 1e-8 and distances[picked] > 1e-8:
                        cosine = float(np.dot(original, offsets[picked]) / (magnitude * distances[picked]))
                decisions.append({"robot": i, "chosen_target": chosen, "cosine": cosine,
                                  "original_norm": float(np.linalg.norm(original)),
                                  "original_action": original.tolist(), "actual_action": action.tolist(),
                                  "kept_original": bool(np.array_equal(original, action))})
            observations, _, _, _, _ = env.step(actions)
            world = env.unwrapped._current_snapshot
            metric = compute_step_metrics(world)
            positions = np.asarray(world.robot_positions)
            targets = np.asarray(world.target_positions)
            timeline.append({"step": step + 1, "coverage_rate": metric.coverage_rate,
                             "matched_targets": metric.matched_targets,
                             "robot_2_distance_to_target_0": float(np.linalg.norm(positions[2] - targets[0])),
                             "decisions": decisions})
    finally:
        env.close()
    return {"mode": mode, "case_id": case.case_id,
            "mean_coverage": float(np.mean([step["coverage_rate"] for step in timeline])),
            "timeline": timeline}


def historical_coverage(folder):
    with (OUT / folder / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case_id"] == "v2b-02"]
    if len(rows) != 2:
        raise ValueError("旧评测缺少两次重复")
    return [float(row["mean_coverage_rate"]) for row in rows]


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("E3 冻结协议已变化")
    plan = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key, hash_key in (("base_entry", "base_entry_sha256"),
                          ("old_factor_source", "old_factor_sha256")):
        if sha(ROOT / plan["fixed_inputs"][key]) != plan["fixed_inputs"][hash_key]:
            raise ValueError(f"输入文件变动：{key}")
    if sha(ROOT / plan["fixed_rule"]["new_rule_source"]) != plan["fixed_rule"]["new_rule_sha256"]:
        raise ValueError("新规则实现与协议不符")
    model = plan["fixed_inputs"]["models"]["E016"]
    model_path = ROOT / model["path"]
    if sha(model_path) != model["sha256"]:
        raise ValueError("E016 模型变化")
    suite = load_suite(HERE / "validation-suite-v2.yaml")
    case = next(case for group in suite.groups for case in group.cases if case.case_id == "v2b-02")
    traces = [replay(case, model_path, mode) for mode in ("base", "target", "conditional")]
    for trace, folder in zip(traces[:2], ("c-v2-E016-1044480", "d-v2-E016-target")):
        if not all(np.isclose(trace["mean_coverage"], item, atol=1e-9)
                   for item in historical_coverage(folder)):
            raise ValueError(f"旧模式重放与完整评测不符：{folder}")
    DEST.mkdir()
    (DEST / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [{"mode": item["mode"], "mean_coverage": item["mean_coverage"],
                "step5_robot2": item["timeline"][4]["decisions"][2],
                "step5_robot2_distance_to_target0": item["timeline"][4]["robot_2_distance_to_target_0"]}
               for item in traces]
    (DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
