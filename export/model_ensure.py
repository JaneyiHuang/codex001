#确认模型.pt是什么类型
import torch

ckpt = torch.load("temporal/results/mappo_actor_temporal_distilled_p25.pt", map_location="cpu")
print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())