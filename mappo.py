# mappo.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models import Actor, Critic


class MAPPO:
    """
    MAPPO trainer.

    It manages:
        - actor network
        - critic network
        - optimizers
        - PPO-style update
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        critic_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 256,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = device

        self.actor = Actor(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=actor_hidden_dims or [128, 128]
        ).to(device)

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=critic_hidden_dims or [256, 256]
        ).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    # =========================================================
    # Action selection
    # =========================================================
    def select_actions(self, obs: np.ndarray, deterministic: bool = False) -> Dict[str, np.ndarray]:
        """
        Select actions for all agents.

        Args:
            obs: shape (M, obs_dim)
            deterministic:
                False -> sample from policy
                True  -> choose greedy actions

        Returns:
            {
                "actions": np.ndarray,    shape (M,)
                "log_probs": np.ndarray,  shape (M,)
            }
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if deterministic:
                logits = self.actor(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = torch.argmax(logits, dim=-1)
                log_probs = dist.log_prob(actions)
            else:
                dist = self.actor.get_action_dist(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

        return {
            "actions": actions.cpu().numpy(),
            "log_probs": log_probs.cpu().numpy(),
        }

    def get_value(self, state: np.ndarray) -> float:
        """
        Get critic value V(s).

        Args:
            state: shape (state_dim,)

        Returns:
            scalar float
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1).item()

        return value

    # =========================================================
    # PPO/MAPPO update
    # =========================================================
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor and critic using rollout batch.

        batch keys:
            obs:        (T, M, obs_dim)
            states:     (T, state_dim)
            actions:    (T, M)
            log_probs:  (T, M)
            returns:    (T,)
            advantages: (T,)
        """
        obs = batch["obs"].to(self.device)                 # (T, M, obs_dim)
        states = batch["states"].to(self.device)           # (T, state_dim)
        actions = batch["actions"].to(self.device)         # (T, M)
        old_log_probs = batch["log_probs"].to(self.device) # (T, M)
        returns = batch["returns"].to(self.device)         # (T,)
        advantages = batch["advantages"].to(self.device)   # (T,)

        T = obs.shape[0]

        # flatten agent observations/actions for actor update
        flat_obs = obs.reshape(T * self.n_agents, self.obs_dim)                  # (T*M, obs_dim)
        flat_actions = actions.reshape(T * self.n_agents)                        # (T*M,)
        flat_old_log_probs = old_log_probs.reshape(T * self.n_agents)            # (T*M,)

        # each time-step advantage is shared by all agents
        flat_advantages = advantages.unsqueeze(1).repeat(1, self.n_agents).reshape(-1)  # (T*M,)

        # for critic
        critic_states = states                                                    # (T, state_dim)
        critic_returns = returns                                                  # (T,)

        # indices for minibatch sampling
        actor_batch_size = T * self.n_agents
        critic_batch_size = T

        actor_losses = []
        critic_losses = []
        entropies = []
        total_losses = []

        for _ in range(self.update_epochs):
            # -------------------------
            # Actor minibatch update
            # -------------------------
            actor_indices = np.arange(actor_batch_size)
            np.random.shuffle(actor_indices)

            for start in range(0, actor_batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = actor_indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]

                dist = self.actor.get_action_dist(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                # -------------------------
                # Critic minibatch update
                # -------------------------
                critic_indices = np.arange(critic_batch_size)
                np.random.shuffle(critic_indices)

                # to keep code simple, critic is updated once per actor minibatch using a sampled critic minibatch
                critic_mb_size = min(self.minibatch_size, critic_batch_size)
                critic_mb_idx = critic_indices[:critic_mb_size]

                mb_states = critic_states[critic_mb_idx]
                mb_returns = critic_returns[critic_mb_idx]

                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = ((values_pred - mb_returns) ** 2).mean()

                total_loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                total_losses.append(total_loss.item())

        info = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        }
        return info

    # =========================================================
    # Save / Load
    # =========================================================
    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])