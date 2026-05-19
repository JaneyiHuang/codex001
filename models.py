# models.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    通用的多层感知机模块，被 Actor 和 Critic 共用，简化代码
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()

        layers = []
        last_dim = input_dim # last_dim = 上一层的输出维度 = 下一层的输入维度

        # 循环建隐藏层：线性层 + ReLU激活
        for hidden_dim in hidden_dims:  # hidden_dims：隐藏层维度列表，比如 [128,128] 表示两层 128 神经元的隐藏层
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        # 最后一层：输出层（无激活函数）
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
        if hidden_dims is None: # 如果没有指定 就默认两层128隐藏层
            hidden_dims = [128, 128]

        self.mlp = MLPBlock(
            input_dim=obs_dim, # 输入：智能体自己的局部观测
            hidden_dims=hidden_dims,
            output_dim=n_actions # 输出：每个动作的logits（logits:所有可选动作的未归一化概率）
        )

#下面是Actor的四个方法
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        直接输出动作的 logits（原始分数，未经过 softmax）
        """
        logits = self.mlp(obs)
        return logits

    def get_action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """
        把 logits 变成离散概率分布（分类分布），用于采样动作。
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        这个方法是训练时用的

        Sample actions from policy.

        Returns:
            action: shape (...,)
            log_prob: shape (...,)
        """
        dist = self.get_action_dist(obs)
        action = dist.sample() # 随机采样动作
        log_prob = dist.log_prob(action) # 动作的对数概率
        return action, log_prob # 返回动作 + 动作对数概率（用于策略梯度计算）

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        测试 / 评估时用

        直接选概率最大的动作，不探索（贪心策略）
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
        if hidden_dims is None:# 如果没有指定 就默认两层256隐藏层
            hidden_dims = [256, 256]

        self.mlp = MLPBlock(
            input_dim=state_dim, # 输入：全局状态
            hidden_dims=hidden_dims,
            output_dim=1 # 输出：状态价值（1个数 状态价值 V (s) → 评估当前全局局面好不好）
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = self.mlp(state)
        return value