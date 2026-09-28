"""把两个 basic-0 与两个 coop-0 场景副本并成一个 SB3 VecEnv。"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class MixedCoverageVecEnv(VecEnv):
    """按固定 1:1 比例拼接两种任务；每个副本仍保留自己的回合重置。"""

    def __init__(self, basic, cooperation):
        if basic.num_envs != cooperation.num_envs or basic.observation_space != cooperation.observation_space or basic.action_space != cooperation.action_space:
            raise ValueError("基础/协作环境的数量或输入输出空间不一致")
        self.parts = (basic, cooperation)
        self._split = basic.num_envs
        super().__init__(basic.num_envs + cooperation.num_envs, basic.observation_space, basic.action_space)

    def _part_index(self, index):
        if index < 0 or index >= self.num_envs:
            raise IndexError(index)
        return (0, index) if index < self._split else (1, index - self._split)

    def reset(self):
        observations = [part.reset() for part in self.parts]
        self.reset_infos = [info for part in self.parts for info in part.reset_infos]
        return np.concatenate(observations, axis=0)

    def step_async(self, actions):
        actions = np.asarray(actions)
        if len(actions) != self.num_envs:
            raise ValueError("动作数量与混合环境数量不一致")
        self.parts[0].step_async(actions[:self._split])
        self.parts[1].step_async(actions[self._split:])

    def step_wait(self):
        basic = self.parts[0].step_wait()
        cooperation = self.parts[1].step_wait()
        return (
            np.concatenate((basic[0], cooperation[0]), axis=0),
            np.concatenate((basic[1], cooperation[1]), axis=0),
            np.concatenate((basic[2], cooperation[2]), axis=0),
            list(basic[3]) + list(cooperation[3]),
        )

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max - 10000))
        first = self.parts[0].seed(seed)
        second = self.parts[1].seed(seed + 10000)
        return list(first) + list(second)

    def close(self):
        for part in self.parts:
            part.close()

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        return [self.parts[part].get_attr(attr_name, indices=local)[0] for part, local in map(self._part_index, self._get_indices(indices))]

    def set_attr(self, attr_name, value, indices=None):
        for part, local in map(self._part_index, self._get_indices(indices)):
            self.parts[part].set_attr(attr_name, value, indices=local)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [self.parts[part].env_method(method_name, *method_args, indices=local, **method_kwargs)[0] for part, local in map(self._part_index, self._get_indices(indices))]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [self.parts[part].env_is_wrapped(wrapper_class, indices=local)[0] for part, local in map(self._part_index, self._get_indices(indices))]


def build_mixed_env_stack(build_single_case_stack, num_vec_envs, seed):
    if num_vec_envs != 4:
        raise ValueError("C 阶段固定 4 个并行场景：2 个 basic-0 与 2 个 coop-0")
    basic = build_single_case_stack(2, seed, "basic-0")
    try:
        cooperation = build_single_case_stack(2, seed + 10000, "coop-0")
    except BaseException:
        basic.close()
        raise
    return MixedCoverageVecEnv(basic, cooperation)
