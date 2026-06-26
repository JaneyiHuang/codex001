from __future__ import annotations

import numpy as np

from temporal.env import MECEnv


class LocalPolicy:
    name = "local"

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return np.zeros(env.M, dtype=np.int64)
