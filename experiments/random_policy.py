from __future__ import annotations

import numpy as np

from env import MECEnv


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, 2, size=env.M, dtype=np.int64)
