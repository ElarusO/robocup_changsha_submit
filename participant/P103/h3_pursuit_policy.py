"""Fixed H3 legal pursuit reference; only own current observations."""
import numpy as np
from coverage_bench.protocol import get_protocol_spec
from single_factor_policy import align_to_visible_target, avoid_visible_peers


class PursuitReference:
    def __init__(self, base_policy):
        self.base_policy = base_policy
        self.spec = get_protocol_spec()

    def reset(self, context):
        self.base_policy.reset(context)

    def act(self, observation):
        action = align_to_visible_target(self.base_policy.act(observation), observation)
        choices = np.flatnonzero(np.asarray(observation['target_visible'], dtype=bool))
        if len(choices):
            rows = np.asarray(observation['targets'], dtype=np.float64)[choices]
            offsets = rows[:, :2] * self.spec.position_scale
            k = int(np.argmin(np.linalg.norm(offsets, axis=1)))
            radius = float(rows[k, 2]) * self.spec.position_scale
            offset = offsets[k]
            if np.linalg.norm(offset) > radius + 0.05:
                action = (offset / np.max(np.abs(offset))).astype(np.float32)
        return avoid_visible_peers(action, observation)

    def close(self):
        self.base_policy.close()
