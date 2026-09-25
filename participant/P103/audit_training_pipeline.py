"""A2 训练数据通路核查；只读检查，不修改现有模型或评测文件。"""

import json

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from train import _PUBLIC_SUITE, build_env_stack
from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation
from coverage_bench.suites import load_suite


def step_zeros(env, count, slots):
    result = None
    for _ in range(count):
        result = env.step(np.zeros((slots, 2), dtype=np.float32))
    return result


def run(case_id):
    seed = 7101
    copies = 4
    suite = load_suite(_PUBLIC_SUITE)
    case = next(case for group in suite.groups for case in group.cases if case.case_id == case_id)
    horizon = case.task_config.horizon
    spec = get_protocol_spec()
    env_a = build_env_stack(copies, seed, case_id)
    env_b = build_env_stack(copies, seed, case_id)
    env_other = build_env_stack(copies, seed + 1, case_id)
    raw_env = make_training_env(case.task_config)
    norm_env = VecNormalize(build_env_stack(copies, seed, case_id), norm_obs=True, norm_reward=False, clip_obs=10.0)
    try:
        first_a, first_b, first_other = env_a.reset(), env_b.reset(), env_other.reset()
        raw_observations, _ = raw_env.reset(seed=seed)
        expected = flatten_observation(raw_observations["agent_0"], spec)
        assert first_a.shape == (copies * case.task_config.num_agents, 104)
        assert first_a.dtype == np.float32
        assert np.array_equal(first_a[0], expected), "训练展平顺序与官方展平函数不一致"
        assert np.array_equal(first_a, first_b), "同种子的首次重置不一致"
        assert not np.array_equal(first_a, first_other), "不同种子的首次重置相同"
        assert not np.array_equal(first_a[0], first_a[case.task_config.num_agents]), "两个副本场景相同"

        slots = len(first_a)
        after_a, rewards_a, done_a, infos_a = step_zeros(env_a, horizon, slots)
        after_b, rewards_b, done_b, infos_b = step_zeros(env_b, horizon, slots)
        assert np.all(done_a) and np.all(done_b), "到达时限后应结束回合"
        assert np.array_equal(after_a, after_b), "同种子的自动重置不一致"
        assert np.array_equal(rewards_a, rewards_b), "同种子的结束奖励不一致"
        assert np.all(after_a[:, -2] == 0), "下一回合观测的步数应归零"
        for info in infos_a:
            assert info["step_index"] == horizon, "终局步数被重置信息覆盖"
            assert info["metrics"] is not None and info["reward_terms"] is not None
            assert info["TimeLimit.truncated"] is True, "时间截断标记缺失"
            assert isinstance(info["terminal_observation"], np.ndarray)
            assert info["terminal_observation"].shape == (104,)
            assert info["terminal_observation"].dtype == np.float32
            assert info["terminal_observation"][-2] == horizon

        second_a, _, second_done_a, _ = step_zeros(env_a, horizon, slots)
        second_b, _, second_done_b, _ = step_zeros(env_b, horizon, slots)
        assert np.all(second_done_a) and np.all(second_done_b)
        assert np.array_equal(second_a, second_b), "第二次自动重置不一致"

        normalized_first = norm_env.reset()
        assert normalized_first.shape == first_a.shape
        assert norm_env.obs_rms.mean.shape == (104,)
        assert np.array_equal(norm_env.get_original_obs(), first_a)
        norm_after, norm_rewards, norm_done, norm_infos = step_zeros(norm_env, horizon, slots)
        assert np.all(norm_done)
        assert np.array_equal(norm_env.get_original_obs(), after_a)
        assert np.array_equal(norm_rewards, rewards_a), "配置要求奖励不归一化"
        assert norm_env.norm_obs and not norm_env.norm_reward
        assert norm_env.training, "训练阶段必须更新观测统计量"
        assert np.isfinite(norm_env.obs_rms.mean).all() and np.isfinite(norm_env.obs_rms.var).all()
        raw_terminal = infos_a[0]["terminal_observation"]
        expected_normalized_terminal = np.clip(
            (raw_terminal - norm_env.obs_rms.mean) / np.sqrt(norm_env.obs_rms.var + norm_env.epsilon),
            -norm_env.clip_obs, norm_env.clip_obs,
        ).astype(np.float32)
        assert np.allclose(norm_infos[0]["terminal_observation"], expected_normalized_terminal, atol=1e-6)
        assert norm_infos[0]["TimeLimit.truncated"] is True
        assert norm_after.shape == first_a.shape
        return {
            "audit": "A2",
            "case_id": case_id,
            "seed": seed,
            "env_copies": copies,
            "agent_slots": slots,
            "observation_size": first_a.shape[1],
            "horizon": horizon,
            "first_reset_reproducible": True,
            "first_and_second_auto_reset_reproducible": True,
            "different_seed_and_copies_distinct": True,
            "official_flatten_matches_training": True,
            "terminal_info_preserved": True,
            "time_limit_truncated_marked": True,
            "terminal_observation_flattened": True,
            "normalization_matches_sb3": True,
            "reward_normalization_disabled": True,
        }
    finally:
        env_a.close()
        env_b.close()
        env_other.close()
        raw_env.close()
        norm_env.close()


if __name__ == "__main__":
    print(json.dumps([run(case_id) for case_id in ("basic-0", "basic-1", "coop-0", "coop-1")], ensure_ascii=False, indent=2))
