from __future__ import annotations

import os

import numpy as np

from env import MECEnv
from .qmix import QMIX


class QMIXPolicy:
    name = "qmix"

    def __init__(self, cfg, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot find trained QMIX model: {model_path}\n"
                "Please run python -m QMIX.train_qmix first, or pass --qmix-model-path."
            )

        self.agent = QMIX(
            obs_dim=cfg.obs_dim,
            state_dim=cfg.state_dim,
            n_actions=cfg.n_actions,
            n_agents=cfg.M,
            agent_hidden_dims=[64, 64],
            mixer_embed_dim=32,
            mixer_hyper_hidden_dim=64,
            lr=5e-4,
            gamma=0.99,
            target_update_interval=200,
            max_grad_norm=10.0,
            device=device,
        )
        self.agent.load(model_path)
        print(f"Loaded trained QMIX model from: {model_path}")

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return self.agent.select_actions(obs, epsilon=0.0, deterministic=True)
