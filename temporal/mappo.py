# mappo.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from temporal.models import Actor, Critic


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
        # 环境维度
        self.obs_dim = obs_dim  # 单个智能体观测维度
        self.state_dim = state_dim # 全局状态维度
        self.n_actions = n_actions # 动作数量
        self.n_agents = n_agents   # 智能体数量

        # 强化学习超参数
        self.gamma = gamma              # 折扣因子：未来奖励的重要程度
        self.gae_lambda = gae_lambda    # 优势函数计算参数
        self.clip_eps = clip_eps        # PPO 核心：截断范围，防止训练崩
        self.entropy_coef = entropy_coef # 探索鼓励系数
        self.critic_coef = critic_coef  # 评论家损失权重
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.device = device

        # 创建 Actor
        self.actor = Actor(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=actor_hidden_dims or [128, 128]
        ).to(device)
        # 创建 Critic
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=critic_hidden_dims or [256, 256]
        ).to(device)

        # 创建优化器（负责更新网络）
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    # =========================================================
    # Action selection 动作选择
    # =========================================================
    def select_actions(self, obs: np.ndarray, deterministic: bool = False) -> Dict[str, np.ndarray]:
        """
        Select actions for all agents.
        动作选择

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
            if deterministic:# 测试模式：选概率最大的动作（不探索）
                logits = self.actor(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = torch.argmax(logits, dim=-1)
                log_probs = dist.log_prob(actions)
            else:# 训练模式：按概率采样动作（带探索）
                dist = self.actor.get_action_dist(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

        return {# 动作 + 动作对数概率
            "actions": actions.cpu().numpy(),
            "log_probs": log_probs.cpu().numpy(),
        }
    # =========================================================
    # 价值估计
    # =======================================
    def get_value(self, state: np.ndarray) -> float:
        """
        Get critic value V(s).

        Args: 输入：全局状态 state
            state: shape (state_dim,)

        Returns: 输出：Critic 评估的当前局势价值 V (s)
            scalar float
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1).item()

        return value

    # =========================================================
    # PPO/MAPPO update 网络更新
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
        #获取batch数据
        obs = batch["obs"].to(self.device)                 # (T, M, obs_dim)
        states = batch["states"].to(self.device)           # (T, state_dim)
        actions = batch["actions"].to(self.device)         # (T, M) 
        old_log_probs = batch["log_probs"].to(self.device) # (T, M) old_log_probs：旧策略的动作对数概率
        returns = batch["returns"].to(self.device)         # (T,) returns：折扣累计回报（真实价值）
        advantages = batch["advantages"].to(self.device)   # (T,) advantages：优势函数（这个动作好不好，比平均水平高多少）

        T = obs.shape[0]

#T：时间步（比如收集了 128 步经验）
#M：智能体数量（比如 3 个智能体）

        # flatten agent observations/actions for actor update 【铺平 Actor 数据】
        flat_obs = obs.reshape(T * self.n_agents, self.obs_dim)                  # (T*M, obs_dim)
        flat_actions = actions.reshape(T * self.n_agents)                        # (T*M,)
        flat_old_log_probs = old_log_probs.reshape(T * self.n_agents)            # (T*M,)

        # each time-step advantage is shared by all agents 优势函数也铺平
        flat_advantages = advantages.unsqueeze(1).repeat(1, self.n_agents).reshape(-1)  # (T*M,)

        # for critic 【Critic 不需要铺平，因为Critic 看全局状态，一个时间步只需要一个全局状态。
        critic_states = states                                                    # (T, state_dim)
        critic_returns = returns                                                  # (T,)

        # indices for minibatch sampling
        actor_batch_size = T * self.n_agents
        critic_batch_size = T

        actor_losses = []
        critic_losses = []
        entropies = []
        total_losses = []

        for _ in range(self.update_epochs):# 开始多轮训练（update_epochs） ，默认 10 轮  作用：同一批经验学习 10 次，提高数据利用率
            # -------------------------
            # Actor minibatch update【Actor 随机采样 minibatch】
            # -------------------------
            actor_indices = np.arange(actor_batch_size)
            np.random.shuffle(actor_indices)# 打乱所有样本，然后一小批一小批取。

            for start in range(0, actor_batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = actor_indices[start:end]

                # 取一小批数据 mb:minibatch
                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]

                # 新策略概率
                dist = self.actor.get_action_dist(mb_obs)# 把当前这一批观测 mb_obs 输入现在的 Actor 网络，算出每个动作的概率分布，存入 dist 这个变量里。
                new_log_probs = dist.log_prob(mb_actions)# new_log_probs：现在的网络给出的动作概率
                entropy = dist.entropy().mean()# entropy：熵，越大探索越强

                # 重要度采样比例 ratio:新策略和旧策略的差异程度
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO 截断损失(如果优势是正的 → 鼓励这个动作
                                #但更新不能太大，太大训练会崩
                                #所以用 clamp 把 ratio 限制在 [0.8, 1.2]
                                #取最小的那个，保证更新保守、稳定)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # -------------------------
                # Critic minibatch update
                # -------------------------
                # 生成 Critic 数据的索引
                critic_indices = np.arange(critic_batch_size)# critic_batch_size = 总共有多少条全局状态数据（一般 = T）
                np.random.shuffle(critic_indices)
                #np.arange(...) = 生成 [0,1,2,...,T-1] 这样一串序号
                #np.random.shuffle(...) = 把序号随机打乱

                # to keep code simple, critic is updated once per actor minibatch using a sampled critic minibatch（每次更新 Actor 的时候，顺便随机抽一小批数据更新一下 Critic）
                critic_mb_size = min(self.minibatch_size, critic_batch_size)  # 确定 Critic 小批量取多少数据（我们设定的 minibatch 大小是 256，但如果总数据比 256 少，就取全部数据，所以用 min() 保证不会取超了）
                critic_mb_idx = critic_indices[:critic_mb_size]# 取前 N 个打乱后的索引

                # 取出对应的全局状态和真实回报
                mb_states = critic_states[critic_mb_idx]
                mb_returns = critic_returns[critic_mb_idx]

                #Critic 训练:让预测的价值 逼近 真实折扣回报 returns
                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = ((values_pred - mb_returns) ** 2).mean()

                # 总损失 = 策略损失 + 价值损失权重（设为0.5）*价值损失 - 探索权重（设为0.005）*熵
                total_loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

                # 反向传播，更新网络(到info之前都是)
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
    # Save / Load 模型保存&加载
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
