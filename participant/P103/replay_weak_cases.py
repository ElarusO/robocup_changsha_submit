"""重放少数弱场景，核对逐步覆盖/碰撞，并记录碰撞时的机器人与目标位置。"""

import csv
import json
from pathlib import Path

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.suites import load_suite
from entry import NumpyMLPPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs" / "P103"
CASES = ("v2b-02", "v2b-12", "v2b-31", "v2c-07", "v2c-09")
PAIRS = ((7101, "E011", "E014"), (7102, "E012", "E015"), (7103, "E013", "E016"))
STEP = 1044480


def expected_episode(phase, experiment_id, case_id):
    path = OUTPUTS_DIR / f"{phase}-v2-{experiment_id}-{STEP}" / "episodes.csv"
    with path.open(encoding="utf-8", newline="") as source:
        matches = [row for row in csv.DictReader(source) if row["case_id"] == case_id]
    if len(matches) != 2:
        raise ValueError(f"缺少两次重复的原始记录：{phase} {experiment_id} {case_id}")
    return matches


def replay(case, phase, experiment_id):
    stage = OUTPUTS_DIR / f"{phase}-stage-{experiment_id}-{STEP}" / "P103"
    policies = [NumpyMLPPolicy(stage / "artifacts") for _ in range(case.task_config.num_agents)]
    env = make_training_env(case.task_config)
    try:
        observations, _ = env.reset(seed=case.scenario_seed)
        agent_ids = list(env.agents)
        steps = []
        for step in range(case.task_config.horizon):
            actions = {agent_id: np.asarray(policies[i].act(observations[agent_id]), dtype=np.float32) for i, agent_id in enumerate(agent_ids)}
            observations, _, _, _, _ = env.step(actions)
            world = env.unwrapped._current_snapshot
            metrics = compute_step_metrics(world)
            robots = np.asarray(world.robot_positions, dtype=np.float64)
            targets = np.asarray(world.target_positions, dtype=np.float64)
            distances = np.linalg.norm(robots[:, None, :] - targets[None, :, :], axis=2)
            events = []
            for i in range(len(robots)):
                for j in range(i + 1, len(robots)):
                    pair_distance = float(np.linalg.norm(robots[i] - robots[j]))
                    if pair_distance < float(world.robot_radii[i] + world.robot_radii[j]):
                        events.append({
                            "robots": [i, j],
                            "distance": pair_distance,
                            "contact_threshold": float(world.robot_radii[i] + world.robot_radii[j]),
                            "nearest_targets": [int(np.argmin(distances[i])), int(np.argmin(distances[j]))],
                            "nearest_target_distances": [float(np.min(distances[i])), float(np.min(distances[j]))],
                            "actions": [actions[agent_ids[i]].tolist(), actions[agent_ids[j]].tolist()],
                        })
            steps.append({
                "step": step + 1,
                "matched_targets": metrics.matched_targets,
                "coverage_rate": metrics.coverage_rate,
                "collision_rate": metrics.collision_rate,
                "collision_pairs": metrics.collision_pairs,
                "collision_events": events,
            })
    finally:
        env.close()
    expected = expected_episode(phase, experiment_id, case.case_id)
    coverage = float(np.mean([item["coverage_rate"] for item in steps]))
    collision = float(np.mean([item["collision_rate"] for item in steps]))
    pair_steps = sum(item["collision_pairs"] for item in steps)
    for row in expected:
        if not np.isclose(coverage, float(row["mean_coverage_rate"]), atol=1e-9) or not np.isclose(collision, float(row["mean_collision_rate"]), atol=1e-9) or pair_steps != int(row["collision_pairs_sum"]):
            raise ValueError(f"重放与正式评测不一致：{phase} {experiment_id} {case.case_id} repeat={row['repeat_index']}")
    return {"experiment_id": experiment_id, "phase": phase, "case_id": case.case_id, "coverage": coverage, "collision": collision, "pair_steps": pair_steps, "steps": steps}


def main():
    output = OUTPUTS_DIR / "c-group-replay-001"
    if output.exists():
        raise FileExistsError(output)
    suite = load_suite(REPO_ROOT / "participant" / "P103" / "validation-suite-v2.yaml")
    case_map = {case.case_id: case for group in suite.groups for case in group.cases}
    records = []
    for seed, b_id, c_id in PAIRS:
        for case_id in CASES:
            case = case_map[case_id]
            for phase, experiment_id in (("b3", b_id), ("c", c_id)):
                record = replay(case, phase, experiment_id)
                record["training_seed"] = seed
                records.append(record)
    output.mkdir(parents=True)
    (output / "trajectories.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    concise = [{
        "seed": item["training_seed"], "experiment_id": item["experiment_id"], "case_id": item["case_id"],
        "coverage": item["coverage"], "collision": item["collision"], "pair_steps": item["pair_steps"],
        "collision_steps": [step["step"] for step in item["steps"] if step["collision_pairs"] > 0],
        "matched_targets_by_step": [step["matched_targets"] for step in item["steps"]],
    } for item in records]
    (output / "summary.json").write_text(json.dumps(concise, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([item for item in concise if item["case_id"] in ("v2b-02", "v2c-07", "v2c-09")], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
