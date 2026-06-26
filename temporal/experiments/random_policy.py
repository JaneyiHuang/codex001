from __future__ import annotations

import numpy as np

from temporal.env import MECEnv


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int | None = None, offload_probability: float = 0.5):
        if not 0.0 <= offload_probability <= 1.0:
            raise ValueError("offload_probability must be in [0, 1].")
        self.rng = np.random.default_rng(seed)
        self.offload_probability = float(offload_probability)
        if abs(self.offload_probability - 0.5) > 1e-12:
            self.name = f"random_p{int(round(self.offload_probability * 100))}"

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return (self.rng.random(env.M) < self.offload_probability).astype(np.int64)
