from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class QMIXReplayBuffer:
    """
    Transition replay buffer for the feed-forward QMIX trainer.

    Shapes:
        obs:         (capacity, M, obs_dim)
        states:      (capacity, state_dim)
        actions:     (capacity, M)
        rewards:     (capacity,)
        next_obs:    (capacity, M, obs_dim)
        next_states: (capacity, state_dim)
        dones:       (capacity,)
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        device: str = "cpu",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = device

        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; current size is {self.size}."
            )

        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[indices], dtype=torch.float32, device=self.device),
            "states": torch.tensor(self.states[indices], dtype=torch.float32, device=self.device),
            "actions": torch.tensor(self.actions[indices], dtype=torch.long, device=self.device),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=torch.float32, device=self.device),
            "next_states": torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        }

    def __len__(self) -> int:
        return self.size
