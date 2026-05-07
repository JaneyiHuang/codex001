from __future__ import annotations

import os

import numpy as np
import torch

from env import MECEnv
from models import Actor


class PrunedMAPPOPolicy:
    def __init__(
        self,
        cfg,
        model_path: str,
        device: str = "cpu",
        policy_name: str = "pruned_mappo",
    ):
        self.name = policy_name
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot find compressed MAPPO actor: {model_path}\n"
                "Please run prune_mappo_actor.py first, or pass --pruned-model-path."
            )

        ckpt = torch.load(model_path, map_location=device)
        hidden_dims = ckpt.get("actor_hidden_dims")
        if hidden_dims is None:
            raise ValueError(
                "The pruned checkpoint must contain actor_hidden_dims. "
                "Please generate it with prune_mappo_actor.py."
            )

        self.device = device
        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            n_actions=cfg.n_actions,
            hidden_dims=hidden_dims,
        ).to(device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

        params = ckpt.get("pruned_params", "unknown")
        source = ckpt.get("source_model_path", "unknown")
        print(f"Loaded {self.name} actor from: {model_path}")
        print(f"Compressed hidden dims: {hidden_dims}, params: {params}, source: {source}")

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy()
