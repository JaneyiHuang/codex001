# models.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    A simple multi-layer perceptron block.
    Used by both Actor and Critic.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """
    Actor network for one agent.

    Input:
        obs of shape (..., obs_dim)

    Output:
        logits of shape (..., n_actions)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.mlp = MLPBlock(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=n_actions
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return action logits.
        """
        logits = self.mlp(obs)
        return logits

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        Build a categorical distribution from logits.
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy.

        Returns:
            action: shape (...,)
            log_prob: shape (...,)
        """
        dist = self.get_action_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Choose the action with maximum probability.
        Useful for evaluation/testing.
        """
        logits = self.forward(obs)
        action = torch.argmax(logits, dim=-1)
        return action


class Critic(nn.Module):
    """
    Centralized critic network.

    Input:
        global state of shape (..., state_dim)

    Output:
        state value of shape (..., 1)
    """
    def __init__(self, state_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.mlp = MLPBlock(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self.mlp(state)
        return value