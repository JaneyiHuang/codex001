from __future__ import annotations

import numpy as np

from temporal.env import MECEnv


class OffloadPolicy:
    name = "offload"

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return np.ones(env.M, dtype=np.int64)
