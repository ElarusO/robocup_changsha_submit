"""E3: 保留已明显朝向目标的原动作，其余沿用 D 的 40% 目标混合。"""

import numpy as np

from single_factor_policy import align_to_visible_target, avoid_visible_peers


COSINE_KEEP_THRESHOLD = 0.90
MIN_ACTION_NORM_TO_KEEP = 0.50


def conditional_target_action(action, observation):
    """最近可见目标方向与原动作夹角约 25.8° 内且力度足够时不介入。"""
    visible = np.flatnonzero(np.asarray(observation["target_visible"], dtype=bool))
    if len(visible) == 0:
        return action
    offsets = np.asarray(observation["targets"], dtype=np.float64)[visible, :2]
    distances = np.linalg.norm(offsets, axis=1)
    chosen = int(np.argmin(distances))
    if distances[chosen] < 1e-8:
        return action

    original = np.asarray(action, dtype=np.float64)
    magnitude = float(np.linalg.norm(original))
    if magnitude >= MIN_ACTION_NORM_TO_KEEP:
        cosine = float(np.dot(original, offsets[chosen]) / (magnitude * distances[chosen]))
        if cosine >= COSINE_KEEP_THRESHOLD:
            return action
    return align_to_visible_target(action, observation)


class ConditionalTargetPolicy:
    def __init__(self, base_policy, with_avoidance=False):
        self.base_policy = base_policy
        self.with_avoidance = with_avoidance

    def reset(self, context):
        self.base_policy.reset(context)

    def act(self, observation):
        action = conditional_target_action(self.base_policy.act(observation), observation)
        if self.with_avoidance:
            return avoid_visible_peers(action, observation)
        return action

    def close(self):
        self.base_policy.close()
