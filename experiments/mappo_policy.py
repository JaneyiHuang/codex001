from __future__ import annotations

import os

import numpy as np

from env import MECEnv
from mappo import MAPPO


class MAPPOPolicy:
    name = "mappo"

    def __init__(self, cfg, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot find trained MAPPO model: {model_path}\n"
                "Please run train.py first, or pass --model-path."
            )

        self.agent = MAPPO(
            obs_dim=cfg.obs_dim,
            state_dim=cfg.state_dim,
            n_actions=cfg.n_actions,
            n_agents=cfg.M,
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[256, 256],
            actor_lr=3e-4,
            critic_lr=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            critic_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=10,
            minibatch_size=256,
            device=device,
        )
        self.agent.load(model_path)
        print(f"Loaded trained MAPPO model from: {model_path}")

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        act_out = self.agent.select_actions(obs, deterministic=True)
        return act_out["actions"]

