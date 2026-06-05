from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class AgentQNetwork(nn.Module):
    """Shared per-agent Q network."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Iterable[int] | None = None,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [64, 64])

        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMixer(nn.Module):
    """
    Monotonic mixing network from QMIX.

    It maps individual agent Q-values and the global state to Q_tot. Hypernetworks
    generate non-negative mixing weights, preserving the monotonic relation between
    each local Q and the joint action-value.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hyper_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, embed_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (batch, n_agents)
            states: (batch, state_dim)

        Returns:
            q_tot: (batch,)
        """
        batch_size = agent_qs.shape[0]

        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.embed_dim, 1)
        v = self.v(states).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + v
        return q_tot.view(batch_size)


class QMIX:
    """
    Feed-forward QMIX trainer for the MEC offloading environment.

    The environment already exposes per-agent observations and a centralized state,
    so this class only handles the value networks, target networks, action selection,
    and TD updates.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        n_agents: int,
        agent_hidden_dims: list[int] | None = None,
        mixer_embed_dim: int = 32,
        mixer_hyper_hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        target_update_interval: int = 200,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.update_steps = 0

        self.agent = AgentQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=agent_hidden_dims or [64, 64],
        ).to(device)
        self.target_agent = AgentQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=agent_hidden_dims or [64, 64],
        ).to(device)

        self.mixer = QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            embed_dim=mixer_embed_dim,
            hyper_hidden_dim=mixer_hyper_hidden_dim,
        ).to(device)
        self.target_mixer = QMixer(
            n_agents=n_agents,
            state_dim=state_dim,
            embed_dim=mixer_embed_dim,
            hyper_hidden_dim=mixer_hyper_hidden_dim,
        ).to(device)

        self.optimizer = Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            lr=lr,
        )
        self.update_targets()

    def select_actions(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.agent(obs_tensor)
            actions = torch.argmax(q_values, dim=-1)

        if (not deterministic) and epsilon > 0.0:
            random_mask = np.random.random(self.n_agents) < epsilon
            actions_np = actions.cpu().numpy()
            random_actions = np.random.randint(0, self.n_actions, size=self.n_agents)
            actions_np[random_mask] = random_actions[random_mask]
            return actions_np.astype(np.int64)

        return actions.cpu().numpy().astype(np.int64)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        batch_size = obs.shape[0]
        flat_obs = obs.reshape(batch_size * self.n_agents, self.obs_dim)
        flat_next_obs = next_obs.reshape(batch_size * self.n_agents, self.obs_dim)

        q_values = self.agent(flat_obs).view(batch_size, self.n_agents, self.n_actions)
        chosen_qs = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_qs, states)

        with torch.no_grad():
            target_q_values = self.target_agent(flat_next_obs).view(
                batch_size,
                self.n_agents,
                self.n_actions,
            )
            target_max_qs = target_q_values.max(dim=2).values
            target_q_tot = self.target_mixer(target_max_qs, next_states)
            targets = rewards + self.gamma * (1.0 - dones) * target_q_tot

        td_error = q_tot - targets
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.update_targets()

        return {
            "loss": float(loss.item()),
            "td_error_abs": float(td_error.abs().mean().item()),
            "q_total_mean": float(q_tot.mean().item()),
            "target_q_total_mean": float(targets.mean().item()),
        }

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_agent": self.target_agent.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_steps": self.update_steps,
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "n_agents": self.n_agents,
                "gamma": self.gamma,
                "target_update_interval": self.target_update_interval,
                "max_grad_norm": self.max_grad_norm,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_agent.load_state_dict(ckpt.get("target_agent", ckpt["agent"]))
        self.target_mixer.load_state_dict(ckpt.get("target_mixer", ckpt["mixer"]))
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_steps = int(ckpt.get("update_steps", 0))
