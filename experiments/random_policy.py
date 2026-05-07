from __future__ import annotations

import numpy as np

from env import MECEnv


class RandomPolicy:
    name = "random"

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 2, size=env.M, dtype=np.int64)

