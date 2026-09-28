"""组合候选：先调整目标朝向，再对同一个动作执行队友避让。"""

from single_factor_policy import align_to_visible_target, avoid_visible_peers


class CombinedPolicy:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def reset(self, context):
        self.base_policy.reset(context)

    def act(self, observation):
        return avoid_visible_peers(align_to_visible_target(self.base_policy.act(observation), observation), observation)

    def close(self):
        self.base_policy.close()
