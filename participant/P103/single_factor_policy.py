"""只依赖本机器人当前观测的两种独立推理动作修正实验。"""

import numpy as np


TARGET_BLEND = 0.4
AVOID_RADIUS = 0.30
AVOID_GAIN = 1.0
CONTACT_DISTANCE = 0.10


def align_to_visible_target(action, observation):
    """把原动作的 40% 换成最近可见目标的方向；无目标时原样返回。"""
    visible = np.flatnonzero(np.asarray(observation["target_visible"], dtype=bool))
    if len(visible) == 0:
        return action
    offsets = np.asarray(observation["targets"], dtype=np.float64)[visible, :2]
    distances = np.linalg.norm(offsets, axis=1)
    chosen = int(np.argmin(distances))
    if distances[chosen] < 1e-8:
        return action
    direction = offsets[chosen] / distances[chosen]
    return np.clip((1.0 - TARGET_BLEND) * action + TARGET_BLEND * direction, -1.0, 1.0).astype(np.float32)


def avoid_visible_peers(action, observation):
    """邻机进入 0.30 距离内时施加斥力；未见邻机时原样返回。"""
    visible = np.flatnonzero(np.asarray(observation["peer_visible"], dtype=bool))
    if len(visible) == 0:
        return action
    offsets = np.asarray(observation["peers"], dtype=np.float64)[visible, :2]
    repulsion = np.zeros(2, dtype=np.float64)
    for offset in offsets:
        distance = float(np.linalg.norm(offset))
        if distance < 1e-8 or distance >= AVOID_RADIUS:
            continue
        strength = AVOID_GAIN * (AVOID_RADIUS - distance) / (AVOID_RADIUS - CONTACT_DISTANCE)
        repulsion -= strength * offset / distance
    return np.clip(np.asarray(action, dtype=np.float64) + repulsion, -1.0, 1.0).astype(np.float32)


class SingleFactorPolicy:
    def __init__(self, base_policy, mode):
        if mode not in ("target", "avoid"):
            raise ValueError(f"未知单因素实验模式：{mode}")
        self.base_policy = base_policy
        self.mode = mode

    def reset(self, context):
        self.base_policy.reset(context)

    def act(self, observation):
        original = self.base_policy.act(observation)
        if self.mode == "target":
            return align_to_visible_target(original, observation)
        return avoid_visible_peers(original, observation)

    def close(self):
        self.base_policy.close()
