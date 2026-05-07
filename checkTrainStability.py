import pandas as pd
import numpy as np

# 读取训练日志
df = pd.read_csv("results/training_log.csv")

# 你想看最后多少轮
last_n = 200

tail = df.tail(last_n).copy()

for col in ["episode_delay_mean", "episode_drop_rate", "episode_reward"]:
    values = tail[col].values
    mean_val = np.mean(values)
    std_val = np.std(values)

    # 相对波动：标准差 / 平均值
    # reward 可能为负，这里取绝对值避免符号影响
    denom = abs(mean_val) if abs(mean_val) > 1e-8 else 1.0
    cv = std_val / denom

    lower_5 = mean_val * 0.95
    upper_5 = mean_val * 1.15
    lower_10 = mean_val * 0.9
    upper_10 = mean_val * 1.1
    lower_20 = mean_val * 0.8
    upper_20 = mean_val * 1.2

    in_5 = np.mean((values >= min(lower_5, upper_5)) & (values <= max(lower_5, upper_5)))
    in_10 = np.mean((values >= min(lower_10, upper_10)) & (values <= max(lower_10, upper_10)))
    in_20 = np.mean((values >= min(lower_20, upper_20)) & (values <= max(lower_20, upper_20)))

    print("=" * 60)
    print(f"{col}")
    print(f"最后 {last_n} 轮平均值: {mean_val:.6f}")
    print(f"最后 {last_n} 轮标准差: {std_val:.6f}")
    print(f"相对波动(标准差/均值绝对值): {cv:.4f}")
    print(f"落在 ± 5% 内的比例: {in_5:.2%}")
    print(f"落在 ±10% 内的比例: {in_10:.2%}")
    print(f"落在 ±20% 内的比例: {in_20:.2%}")
    

# 再看最后 200 轮的前100 vs 后100
half = last_n // 2
first_half = tail.iloc[:half]
second_half = tail.iloc[half:]

print("\n" + "#" * 60)
print("最后 200 轮前100 vs 后100 对比")
for col in ["episode_delay_mean", "episode_drop_rate", "episode_reward"]:
    m1 = first_half[col].mean()
    m2 = second_half[col].mean()
    diff = m2 - m1
    print(f"{col}: 前100均值={m1:.6f}, 后100均值={m2:.6f}, 差值(后-前)={diff:.6f}")