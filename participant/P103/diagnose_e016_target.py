"""重放 E016 原策略与目标朝向策略，记录漏目标场景的逐步动作和覆盖。"""

import csv
import json
from pathlib import Path

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.suites import load_suite
from entry import NumpyMLPPolicy
from single_factor_policy import align_to_visible_target


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "P103"
DEST = OUT / "d-e016-target-diagnosis-001"
MODEL = OUT / "post-E0161044480" / "policy.npz"
CASE_ID = "v2b-02"


def trace(case, mode):
    env = make_training_env(case.task_config)
    policies = [NumpyMLPPolicy(MODEL.parent) for _ in range(case.task_config.num_agents)]
    try:
        observations, _ = env.reset(seed=case.scenario_seed)
        agent_ids = list(env.agents)
        timeline = []
        for step in range(case.task_config.horizon):
            before = env.unwrapped._current_snapshot
            robots = np.asarray(before.robot_positions)
            targets = np.asarray(before.target_positions)
            target_radii = np.asarray(before.target_radii)
            covered_before = (np.linalg.norm(robots[:, None, :] - targets[None, :, :], axis=2) <= target_radii[None, :]).any(axis=0)
            decisions, actions = [], {}
            for i, agent in enumerate(agent_ids):
                obs = observations[agent]
                base = policies[i].act(obs)
                action = align_to_visible_target(base, obs) if mode == "target" else base
                actions[agent] = action
                visible = np.flatnonzero(np.asarray(obs["target_visible"], dtype=bool))
                chosen = None
                if len(visible):
                    offsets = np.asarray(obs["targets"])[visible, :2]
                    chosen = int(visible[np.argmin(np.linalg.norm(offsets, axis=1))])
                decisions.append({"robot": i, "visible_targets": visible.tolist(), "chosen_target": chosen,
                                  "chosen_already_covered": bool(covered_before[chosen]) if chosen is not None else None,
                                  "base_action": base.tolist(), "actual_action": action.tolist(),
                                  "target2_distance_before": float(np.linalg.norm(robots[i] - targets[2]))})
            observations, _, _, _, _ = env.step(actions)
            after = env.unwrapped._current_snapshot
            step_distances = np.linalg.norm(np.asarray(after.robot_positions)[:, None, :] - np.asarray(after.target_positions)[None, :, :], axis=2)
            target_covered = (step_distances <= np.asarray(after.target_radii)[None, :]).any(axis=0)
            metric = compute_step_metrics(after)
            timeline.append({"step": step + 1, "matched_targets": metric.matched_targets,
                             "covered_targets": target_covered.tolist(), "nearest_robot_distance_to_target2": float(step_distances[:, 2].min()),
                             "robots": np.asarray(after.robot_positions).tolist(), "targets": np.asarray(after.target_positions).tolist(),
                             "decisions": decisions})
    finally:
        env.close()
    return {"mode": mode, "case_id": case.case_id, "mean_coverage": float(np.mean([x["matched_targets"] / 3 for x in timeline])), "timeline": timeline}


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    suite = load_suite(ROOT / "participant" / "P103" / "validation-suite-v2.yaml")
    case = next(case for group in suite.groups for case in group.cases if case.case_id == CASE_ID)
    traces = [trace(case, mode) for mode in ("base", "target")]
    with (OUT / "c-v2-E016-1044480" / "episodes.csv").open(encoding="utf-8", newline="") as source:
        base = [row for row in csv.DictReader(source) if row["case_id"] == CASE_ID]
    with (OUT / "d-v2-E016-target" / "episodes.csv").open(encoding="utf-8", newline="") as source:
        target = [row for row in csv.DictReader(source) if row["case_id"] == CASE_ID]
    for trace_data, expected in zip(traces, (base, target)):
        if len(expected) != 2 or not all(np.isclose(trace_data["mean_coverage"], float(row["mean_coverage_rate"])) for row in expected):
            raise ValueError("重放与正式评测不符")
    DEST.mkdir(parents=True)
    (DEST / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in traces:
        print(item["mode"], "coverage", item["mean_coverage"])
        for event in item["timeline"]:
            picks = [d["chosen_target"] for d in event["decisions"]]
            covered_picks = [d["robot"] for d in event["decisions"] if d["chosen_already_covered"]]
            print(event["step"], event["covered_targets"], round(event["nearest_robot_distance_to_target2"], 3), "picks", picks, "already", covered_picks)


if __name__ == "__main__":
    main()
