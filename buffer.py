# buffer.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for MAPPO.

    It stores one trajectory segment:
        obs, state, action, log_prob, reward, done, value

    Then it computes:
        returns, advantages

    Shapes:
        obs:        (T, M, obs_dim)
        state:      (T, state_dim)
        actions:    (T, M)
        log_probs:  (T, M)
        rewards:    (T,)
        dones:      (T,)
        values:     (T,)
        returns:    (T,)
        advantages: (T,)
    """

    def __init__(
        self,
        episode_limit: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.reset()

    def reset(self) -> None:
        """Clear all stored rollout data."""
        self.ptr = 0

        self.obs = np.zeros(
            (self.episode_limit, self.n_agents, self.obs_dim),
            dtype=np.float32
        )
        self.states = np.zeros(
            (self.episode_limit, self.state_dim),
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.episode_limit, self.n_agents),
            dtype=np.int64
        )
        self.log_probs = np.zeros(
            (self.episode_limit, self.n_agents),
            dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.episode_limit,),
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.episode_limit,),
            dtype=np.float32
        )
        self.values = np.zeros(
            (self.episode_limit,),
            dtype=np.float32
        )

        self.returns = np.zeros(
            (self.episode_limit,),
            dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.episode_limit,),
            dtype=np.float32
        )

    def store(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """
        Store one environment step.

        Args:
            obs:        shape (M, obs_dim)
            state:      shape (state_dim,)
            actions:    shape (M,)
            log_probs:  shape (M,)
            reward:     scalar
            done:       bool
            value:      scalar
        """
        if self.ptr >= self.episode_limit:
            raise IndexError("RolloutBuffer is full. Please reset before storing more data.")

        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """
        Compute GAE advantages and returns.

        Args:
            last_value:
                V(s_T), used for bootstrapping if the episode is not done.
                If episode is done, this usually should be 0.
        """
        gae = 0.0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )

            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Return collected rollout data as torch tensors.

        Only return the valid part [0:self.ptr].
        """
        data = {
            "obs": torch.tensor(self.obs[:self.ptr], dtype=torch.float32, device=self.device),
            "states": torch.tensor(self.states[:self.ptr], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions[:self.ptr], dtype=torch.long, device=self.device),
            "log_probs": torch.tensor(self.log_probs[:self.ptr], dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(self.rewards[:self.ptr], dtype=torch.float32, device=self.device),
            "dones": torch.tensor(self.dones[:self.ptr], dtype=torch.float32, device=self.device),
            "values": torch.tensor(self.values[:self.ptr], dtype=torch.float32, device=self.device),
            "returns": torch.tensor(self.returns[:self.ptr], dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(self.advantages[:self.ptr], dtype=torch.float32, device=self.device),
        }
        return data

    def normalize_advantages(self, eps: float = 1e-8) -> None:
        """
        Normalize advantages for more stable PPO/MAPPO training.
        """
        adv = self.advantages[:self.ptr]
        adv_mean = adv.mean()
        adv_std = adv.std()
        self.advantages[:self.ptr] = (adv - adv_mean) / (adv_std + eps)

    def __len__(self) -> int:
        return self.ptr