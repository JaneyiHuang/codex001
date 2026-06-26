'''
1. 加载训练好的大模型（128,128）
2. 计算每个神经元的“重要性”
3. 删掉不重要的 25% 神经元
4. 重建一个更小的模型
5. 测试速度：变小后快多少
6. 保存小模型
'''
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from temporal.config import EnvConfig
from temporal.models import Actor


def get_actor_linear_layers(actor: Actor) -> List[nn.Linear]:
    '''
    # 取出所有线性层
        比如：把 Actor 里的 3 个线性层取出来：
            第一层：128 个神经元
            第二层：128 个神经元
            输出层
    '''    
    return [module for module in actor.mlp.net if isinstance(module, nn.Linear)]


def count_parameters(model: nn.Module) -> int:
    '''
    计算模型参数量
        算模型有多少个参数，用来对比 剪枝前和剪枝后
    '''
    return sum(param.numel() for param in model.parameters())


def compute_keep_indices(
    actor: Actor,
    pruning_percentage: float,
) -> Dict[int, List[int]]:
    '''
    计算哪些神经元要保留
        对每一层神经元，算L1 范数 = 重要性：
        重要性越高 → 贡献越大 → 保留
        重要性越低 → 贡献越小 → 剪掉
    '''
    if not 0.0 <= pruning_percentage < 1.0:
        raise ValueError("pruning_percentage must be in [0.0, 1.0).")# 剪枝比例必须在 0~1 之间
    
    # 取出前两层隐藏层（输出层不剪）
    linear_layers = get_actor_linear_layers(actor)
    hidden_layers = linear_layers[:-1]
    keep_indices: Dict[int, List[int]] = {} # 用来存：每层保留哪些神经元。

    for layer_idx, layer in enumerate(hidden_layers):
        l1_norms = torch.norm(layer.weight.data, p=1, dim=1)# 计算每个神经元的重要性
        num_keep = max(1, int(l1_norms.shape[0] * (1.0 - pruning_percentage)))# 比如剪枝率 0.25 → 保留 75% 神经元，神经元总数 × (1 - 剪枝比例)
        _, indices = torch.topk(l1_norms, num_keep)# 选出最重要的 N 个神经元，记录它们的编号。
        indices = torch.sort(indices).values # 选出分数最高的神经元，记录它们的编号
        keep_indices[layer_idx] = indices.cpu().numpy().astype(int).tolist() # 记下来：这一层要保留哪些神经元。
        print(
            f"hidden layer {layer_idx}: keeping {num_keep} "
            f"out of {l1_norms.shape[0]} neurons"
        )

    return keep_indices


def build_pruned_actor(# 重建剪枝后的小模型
    cfg: EnvConfig,
    original_actor: Actor,
    keep_indices: Dict[int, List[int]],
) -> Actor:
    linear_layers = get_actor_linear_layers(original_actor)
    if len(linear_layers) != 3:
        raise ValueError(
            "This pruning script expects Actor hidden_dims=[h1, h2], "
            f"but found {len(linear_layers) - 1} hidden Linear layers."
        )

    keep_0 = keep_indices[0] # 第一层保留哪些
    keep_1 = keep_indices[1] # 第二层保留哪些
    pruned_actor = Actor(   # 创建小模型，神经元变少
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[len(keep_0), len(keep_1)],
    )

    src_layers = get_actor_linear_layers(original_actor)
    dst_layers = get_actor_linear_layers(pruned_actor)

    with torch.no_grad():
        # First hidden layer: select output neurons, keep all input features.
        dst_layers[0].weight.copy_(src_layers[0].weight[keep_0, :])# 第一层：保留重要神经元的权重（weight:决定输入影响多大）
        dst_layers[0].bias.copy_(src_layers[0].bias[keep_0])# 复制偏置（bias:决定神经元默认激活强度），只复制我们保留下来的那些神经元的偏置
        # src_layers[0]：第一层原来的全部偏置，形状（128，），128个神经元、每个1个偏置
        #keep_0:决定保留的神经元编号,eg:keep_0 = [2,5,7,12,......95] （一共96个）

        # Second hidden layer: select output neurons and the kept first-layer inputs.
        dst_layers[1].weight.copy_(src_layers[1].weight[keep_1, :][:, keep_0])
        # 原来权重形状【H2=128,H1=128】,行：本层每个神经元，列：前一层每个神经元的输入连接
        # weight[keep_1, :]  行：只保留本层要留下的 K2 个神经元，形状变成：[K2, 128]
        # [:, keep_0]  列：只保留前一层留下来的 K1 个神经元的连接,形状最终：[K2, K1]
        dst_layers[1].bias.copy_(src_layers[1].bias[keep_1])

        # Output layer: keep all action logits, select kept second-layer inputs.
        dst_layers[2].weight.copy_(src_layers[2].weight[:, keep_1])
        # 原来的权重：[动作数, H2=128]
        # 所以行不变，因为动作个数不会发生变化
        # 列只取 keep_1：只保留第二层留下来的 K2 个神经元，最终形状：[动作数, K2]
        dst_layers[2].bias.copy_(src_layers[2].bias)
        #输出神经元一个都不剪，所以偏置直接复制。因为只是适配前一层

    return pruned_actor


def load_original_actor(cfg: EnvConfig, model_path: str, device: str) -> Actor:# 加载训练好的大模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find MAPPO checkpoint: {model_path}")

    actor = Actor(
        obs_dim=cfg.obs_dim,
        n_actions=cfg.n_actions,
        hidden_dims=[128, 128],
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def measure_actor_latency(# 测试推理速度（延迟）
    actor: Actor,
    obs_dim: int,
    n_agents: int,
    device: str,
    repeats: int = 2000,
    warmup: int = 200,
) -> float:
    actor.eval()
    obs = torch.randn(n_agents, obs_dim, dtype=torch.float32, device=device)# 造假数据，形状和真数据一样

    with torch.no_grad():# 不计算梯度，只测速
        for _ in range(warmup):# 预热模型：让模型 “先跑两步热热身”，消除第一次加载的慢速度，测出来的速度才准！
            _ = actor(obs)# _:这个py文件里面出现的_都是占位符，虽然会输出一些或者对应一些内容，但是对代码运行没有用，
                            #比如在这里只是用来执行依次预热，下面是用来走依次2000次测试的时间，上面也有一个，_返回的是最大的分数，但是没有什么用，我们只需要记录下标

        start = time.perf_counter()  # 测 2000 次的总时间
        for _ in range(repeats):
            _ = actor(obs)
        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / repeats# *1000是秒变成毫秒


def parse_args() -> argparse.Namespace:# 命令行参数，运行代码的时候起作用，设置值模型路径、剪枝比例、设备、保存路径
    parser = argparse.ArgumentParser(description="Neuron-prune a trained MAPPO actor.")
    parser.add_argument(
        "--model-path",
        default=os.path.join("results", "mappo_checkpoint.pt"),
        help="Path to the trained MAPPO checkpoint.",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.join("temporal", "results", "mappo_actor_pruned_p25.pt"),
        help="Path for the pruned actor checkpoint.",
    )
    parser.add_argument(
        "--pruning-percentage",
        type=float,
        default=0.25,
        help="Fraction of hidden neurons removed from each hidden layer.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig()

    original_actor = load_original_actor(cfg, args.model_path, args.device)# 1、加载大模型
    keep_indices = compute_keep_indices(original_actor, args.pruning_percentage)# 2、计算保留哪些神经元
    pruned_actor = build_pruned_actor(cfg, original_actor, keep_indices).to(args.device) #3、构建小模型
    pruned_actor.eval()

    # 计算剪枝前后参数的变化，然后计算压缩率
    original_params = count_parameters(original_actor)
    pruned_params = count_parameters(pruned_actor)
    compression_ratio = original_params / max(pruned_params, 1)

    # 测速
    original_latency_ms = measure_actor_latency(
        original_actor,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )
    pruned_latency_ms = measure_actor_latency(
        pruned_actor,
        obs_dim=cfg.obs_dim,
        n_agents=cfg.M,
        device=args.device,
        repeats=args.latency_repeats,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(# 保存小模型
        {
            "actor": pruned_actor.state_dict(),
            "actor_hidden_dims": [len(keep_indices[0]), len(keep_indices[1])],
            "obs_dim": cfg.obs_dim,
            "n_actions": cfg.n_actions,
            "n_agents": cfg.M,
            "source_model_path": args.model_path,
            "pruning_percentage": args.pruning_percentage,
            "keep_indices": keep_indices,
            "original_params": original_params,
            "pruned_params": pruned_params,
            "compression_ratio": compression_ratio,
            "original_latency_ms": original_latency_ms,
            "pruned_latency_ms": pruned_latency_ms,
        },
        args.save_path,
    )

    print("\n================ Pruning Summary ================\n")
    print(f"Source model: {args.model_path}")
    print(f"Pruned actor: {args.save_path}")
    print(f"Pruning percentage: {args.pruning_percentage:.2f}")
    print(f"Hidden dims: {[128, 128]} -> {[len(keep_indices[0]), len(keep_indices[1])]}")
    print(f"Parameters: {original_params} -> {pruned_params}")
    print(f"Compression ratio: {compression_ratio:.3f}x")
    print(f"Actor latency: {original_latency_ms:.6f} ms -> {pruned_latency_ms:.6f} ms")
    print("\n=================================================\n")


if __name__ == "__main__":
    main()
