#!/usr/bin/env python3
"""
Run long-term simulated MEC evaluation on Jetson Nano.

This is different from jetson_infer.py:
  - jetson_infer.py checks one-step deployment inference.
  - jetson_eval_sim.py runs many simulated episodes and reports long-term delay.

It is still a simulation, not a real MEC closed-loop deployment. The simulated
state transition and delay calculation are copied from the training environment.

Example:
    python3 jetson_eval_sim.py --episodes 50 --model-path ./mappo_actor_temporal_distilled_p25.pt
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import time

import numpy as np

from jetson_infer import JetsonTemporalPolicy, resolve_model_path


class EnvConfig(object):
    def __init__(self, load_factor=1.0):
        self.M = 4
        self.episode_limit = 200
        self.delta = 1.0

        self.B = 1e6
        self.sigma2 = 1e-9
        self.p_tx = 0.5

        self.f_local = 5e8
        self.f_edge = 1e10
        self.rho_local = 1200.0
        self.rho_edge = 500.0
        self.kappa = 1e-28

        self.E_max = 5.0
        self.E_init = 2.5
        self.H_min = 0.1
        self.H_max = 0.5

        self.task_min_bits = 1.5e6 * float(load_factor)
        self.task_max_bits = 6e6 * float(load_factor)

        self.Q_loc_max = 1e7
        self.Q_tx_max = 1e7
        self.Q_edge_max = 4e7

        self.psi = 300.0
        self.reward_scale = 200.0
        self.h_norm_clip = 10.0

        self.n_actions = 2
        self.obs_dim = 10
        self.state_dim = 8 * self.M + 2


class MECEnv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = cfg.M
        self.t = 0
        self.episode_limit = cfg.episode_limit

        self.lambda_arr = None
        self.H_arr = None
        self.h_arr = None

        self.E_arr = None
        self.Q_loc = None
        self.Q_tx = None
        self.F_loc = None
        self.F_tx = None
        self.Q_edge = None
        self.F_edge = None

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.t = 0
        self.E_arr = np.full(self.M, self.cfg.E_init, dtype=np.float32)
        self.Q_loc = np.zeros(self.M, dtype=np.float32)
        self.Q_tx = np.zeros(self.M, dtype=np.float32)
        self.F_loc = np.zeros(self.M, dtype=np.float32)
        self.F_tx = np.zeros(self.M, dtype=np.float32)
        self.Q_edge = np.float32(0.0)
        self.F_edge = np.float32(0.0)

        self.lambda_arr = self.sample_tasks()
        self.H_arr = self.sample_energy()
        self.h_arr = self.sample_channel()
        return self.get_obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.M,):
            raise ValueError("actions should have shape ({},), got {}".format(self.M, actions.shape))

        cfg = self.cfg
        curr_lambda = self.lambda_arr.copy()
        curr_H = self.H_arr.copy()
        curr_h = self.h_arr.copy()

        curr_E = self.E_arr.copy()
        curr_Q_loc = self.Q_loc.copy()
        curr_Q_tx = self.Q_tx.copy()
        curr_F_loc = self.F_loc.copy()
        curr_F_tx = self.F_tx.copy()
        curr_Q_edge = float(self.Q_edge)
        curr_F_edge = float(self.F_edge)

        next_Q_loc = curr_Q_loc.copy()
        next_Q_tx = curr_Q_tx.copy()
        next_F_loc = curr_F_loc.copy()
        next_F_tx = curr_F_tx.copy()

        energy_cost = np.zeros(self.M, dtype=np.float32)
        delay_arr = np.zeros(self.M, dtype=np.float32)
        drop_flags = np.zeros(self.M, dtype=np.int64)
        rate_arr = np.zeros(self.M, dtype=np.float32)
        edge_candidates = []

        for m in range(self.M):
            task_bits = float(curr_lambda[m])

            if actions[m] == 0:
                e_loc = self.local_energy_cost()
                enough_energy = curr_E[m] >= e_loc

                if enough_energy:
                    d_loc_bits = self.local_service_bits()
                    energy_cost[m] = e_loc
                else:
                    d_loc_bits = 0.0
                    energy_cost[m] = 0.0

                q_loc_prime = max(curr_Q_loc[m] - d_loc_bits, 0.0)
                if q_loc_prime + task_bits > cfg.Q_loc_max:
                    next_Q_loc[m] = q_loc_prime
                    next_F_loc[m] = curr_F_loc[m]
                    delay_arr[m] = cfg.psi
                    drop_flags[m] = 1
                else:
                    next_Q_loc[m] = q_loc_prime + task_bits
                    if enough_energy and cfg.f_local > 0:
                        t_loc = self.local_processing_slots(task_bits)
                        s_loc = max(float(self.t), float(curr_F_loc[m]))
                        c_loc = s_loc + t_loc - 1
                        next_F_loc[m] = c_loc + 1
                        delay_arr[m] = c_loc - float(self.t)
                    else:
                        nominal_t_loc = max(1, self.local_processing_slots(task_bits))
                        est_start = max(float(self.t) + 1.0, float(curr_F_loc[m]))
                        delay_arr[m] = est_start + nominal_t_loc - 1 - float(self.t)
                        next_F_loc[m] = curr_F_loc[m]
            else:
                e_tx = self.tx_energy_cost()
                enough_energy = curr_E[m] >= e_tx

                if enough_energy:
                    rate = self.uplink_rate(curr_h[m])
                    d_tx_bits = rate * cfg.delta
                    energy_cost[m] = e_tx
                else:
                    rate = 0.0
                    d_tx_bits = 0.0
                    energy_cost[m] = 0.0

                rate_arr[m] = rate
                q_tx_prime = max(curr_Q_tx[m] - d_tx_bits, 0.0)
                if q_tx_prime + task_bits > cfg.Q_tx_max:
                    next_Q_tx[m] = q_tx_prime
                    next_F_tx[m] = curr_F_tx[m]
                    delay_arr[m] = cfg.psi
                    drop_flags[m] = 1
                else:
                    next_Q_tx[m] = q_tx_prime + task_bits
                    if enough_energy and rate > 0:
                        t_tx = self.tx_slots(task_bits, rate)
                        s_tx = max(float(self.t), float(curr_F_tx[m]))
                        c_tx = s_tx + t_tx - 1
                        next_F_tx[m] = c_tx + 1
                        edge_candidates.append({"m": m, "task_bits": task_bits, "C_tx": c_tx})
                    else:
                        nominal_rate = self.uplink_rate(curr_h[m])
                        nominal_t_tx = self.tx_slots(task_bits, nominal_rate)
                        nominal_t_edge = self.edge_processing_slots(task_bits)
                        est_tx_start = max(float(self.t) + 1.0, float(curr_F_tx[m]))
                        est_c_tx = est_tx_start + nominal_t_tx - 1
                        est_s_edge = max(est_c_tx + 1, float(curr_F_edge))
                        delay_arr[m] = est_s_edge + nominal_t_edge - 1 - float(self.t)
                        next_F_tx[m] = curr_F_tx[m]

        edge_candidates.sort(key=lambda item: item["C_tx"])
        q_edge_working = max(curr_Q_edge - self.edge_service_bits(), 0.0)
        f_edge_working = curr_F_edge

        for item in edge_candidates:
            m = item["m"]
            task_bits = item["task_bits"]
            c_tx = item["C_tx"]

            if q_edge_working + task_bits > cfg.Q_edge_max:
                delay_arr[m] = cfg.psi
                drop_flags[m] = 1
            else:
                q_edge_working += task_bits
                t_edge = self.edge_processing_slots(task_bits)
                s_edge = max(c_tx + 1, f_edge_working)
                c_edge = s_edge + t_edge - 1
                f_edge_working = c_edge + 1
                delay_arr[m] = c_edge - float(self.t)

        next_E = np.minimum(cfg.E_max, np.maximum(curr_E - energy_cost + curr_H, 0.0))

        self.Q_loc = next_Q_loc.astype(np.float32)
        self.Q_tx = next_Q_tx.astype(np.float32)
        self.F_loc = next_F_loc.astype(np.float32)
        self.F_tx = next_F_tx.astype(np.float32)
        self.Q_edge = np.float32(q_edge_working)
        self.F_edge = np.float32(f_edge_working)
        self.E_arr = next_E.astype(np.float32)

        reward = -float(np.mean(delay_arr) / cfg.reward_scale)
        info = {
            "delay_mean": float(np.mean(delay_arr)),
            "delay_sum": float(np.sum(delay_arr)),
            "drop_rate": float(np.mean(drop_flags)),
            "drop_num": int(np.sum(drop_flags)),
            "offload_rate": float(np.mean(actions == 1)),
            "energy_mean": float(np.mean(self.E_arr)),
            "rate_mean": float(np.mean(rate_arr)),
        }

        self.t += 1
        done = self.t >= self.episode_limit
        self.lambda_arr = self.sample_tasks()
        self.H_arr = self.sample_energy()
        self.h_arr = self.sample_channel()
        return self.get_obs(), reward, done, info

    def sample_tasks(self):
        return np.random.uniform(self.cfg.task_min_bits, self.cfg.task_max_bits, size=self.M).astype(np.float32)

    def sample_energy(self):
        return np.random.uniform(self.cfg.H_min, self.cfg.H_max, size=self.M).astype(np.float32)

    def sample_channel(self):
        return np.random.exponential(scale=1.0, size=self.M).astype(np.float32)

    def uplink_rate(self, h_gain):
        snr = self.cfg.p_tx * float(h_gain) / self.cfg.sigma2
        return self.cfg.B * np.log2(1.0 + snr)

    def local_service_bits(self):
        return self.cfg.f_local * self.cfg.delta / self.cfg.rho_local

    def edge_service_bits(self):
        return self.cfg.f_edge * self.cfg.delta / self.cfg.rho_edge

    def local_processing_slots(self, task_bits):
        return int(np.ceil(task_bits * self.cfg.rho_local / (self.cfg.f_local * self.cfg.delta)))

    def edge_processing_slots(self, task_bits):
        return int(np.ceil(task_bits * self.cfg.rho_edge / (self.cfg.f_edge * self.cfg.delta)))

    def tx_slots(self, task_bits, rate):
        denom = max(rate * self.cfg.delta, 1e-8)
        return int(np.ceil(task_bits / denom))

    def local_energy_cost(self):
        return self.cfg.kappa * (self.cfg.f_local ** 3) * self.cfg.delta

    def tx_energy_cost(self):
        return self.cfg.p_tx * self.cfg.delta

    def get_obs(self):
        obs = []
        for m in range(self.M):
            obs_m = np.array(
                [
                    self.lambda_arr[m] / self.cfg.task_max_bits,
                    self.E_arr[m] / self.cfg.E_max,
                    self.H_arr[m] / self.cfg.H_max,
                    min(self.h_arr[m], self.cfg.h_norm_clip) / self.cfg.h_norm_clip,
                    self.Q_loc[m] / self.cfg.Q_loc_max,
                    self.Q_tx[m] / self.cfg.Q_tx_max,
                    min(self.F_loc[m], self.cfg.episode_limit) / self.cfg.episode_limit,
                    min(self.F_tx[m], self.cfg.episode_limit) / self.cfg.episode_limit,
                    self.Q_edge / self.cfg.Q_edge_max,
                    min(self.F_edge, self.cfg.episode_limit) / self.cfg.episode_limit,
                ],
                dtype=np.float32,
            )
            obs.append(obs_m)
        return np.stack(obs, axis=0)


def build_agent_states(env):
    agents = []
    for m in range(env.M):
        agents.append(
            {
                "task_bits": float(env.lambda_arr[m]),
                "energy": float(env.E_arr[m]),
                "harvested_energy": float(env.H_arr[m]),
                "channel_gain": float(env.h_arr[m]),
                "q_loc": float(env.Q_loc[m]),
                "q_tx": float(env.Q_tx[m]),
                "f_loc": float(env.F_loc[m]),
                "f_tx": float(env.F_tx[m]),
            }
        )
    return agents


def build_edge_state(env):
    return {"q_edge": float(env.Q_edge), "f_edge": float(env.F_edge)}


class TemporalPolicyAdapter(object):
    name = "temporal_distilled_mappo"

    def __init__(self, model_path, device, safety_enabled):
        self.policy = JetsonTemporalPolicy(
            model_path=model_path,
            device=device,
            safety_enabled=safety_enabled,
        )
        self.reset_episode()

    def reset_episode(self):
        self.policy.reset()
        self.steps = 0
        self.model_call_steps = 0
        self.decision_agent_count = 0
        self.safety_interrupt_count = 0

    def select_actions(self, env):
        result = self.policy.select_actions(build_agent_states(env), build_edge_state(env))
        self.steps += 1
        if result["decision_agents"]:
            self.model_call_steps += 1
            self.decision_agent_count += len(result["decision_agents"])
        self.safety_interrupt_count += len(result["safety_interrupted_agents"])
        return np.asarray(result["actions"], dtype=np.int64)

    def episode_stats(self, n_agents):
        total_agent_steps = max(self.steps * n_agents, 1)
        return {
            "model_call_step_ratio": float(self.model_call_steps / float(max(self.steps, 1))),
            "decision_agent_ratio": float(self.decision_agent_count / float(total_agent_steps)),
            "safety_interrupt_count": float(self.safety_interrupt_count),
        }


class LocalPolicy(object):
    name = "local"

    def reset_episode(self):
        pass

    def select_actions(self, env):
        return np.zeros(env.M, dtype=np.int64)

    def episode_stats(self, n_agents):
        return {}


class OffloadPolicy(object):
    name = "offload"

    def reset_episode(self):
        pass

    def select_actions(self, env):
        return np.ones(env.M, dtype=np.int64)

    def episode_stats(self, n_agents):
        return {}


class RandomPolicy(object):
    name = "random"

    def __init__(self, seed, offload_probability=0.5):
        self.rng = np.random.RandomState(seed)
        self.offload_probability = float(offload_probability)

    def reset_episode(self):
        pass

    def select_actions(self, env):
        return (self.rng.rand(env.M) < self.offload_probability).astype(np.int64)

    def episode_stats(self, n_agents):
        return {}


def evaluate_policy(cfg, policy, episodes, seed):
    episode_records = []
    start_time = time.perf_counter()

    for episode_idx in range(episodes):
        env = MECEnv(cfg)
        env.seed(seed + episode_idx)
        policy.reset_episode()
        env.reset()

        done = False
        reward_sum = 0.0
        delay_values = []
        drop_values = []
        offload_values = []
        energy_values = []

        while not done:
            actions = policy.select_actions(env)
            _, reward, done, info = env.step(actions)
            reward_sum += reward
            delay_values.append(info["delay_mean"])
            drop_values.append(info["drop_rate"])
            offload_values.append(info["offload_rate"])
            energy_values.append(info["energy_mean"])

        record = {
            "reward": float(reward_sum),
            "delay": float(np.mean(delay_values)),
            "drop_rate": float(np.mean(drop_values)),
            "offload_rate": float(np.mean(offload_values)),
            "energy": float(np.mean(energy_values)),
        }
        record.update(policy.episode_stats(cfg.M))
        episode_records.append(record)

    elapsed_s = time.perf_counter() - start_time
    return summarize_records(policy.name, episode_records, elapsed_s)


def mean_std(records, key, default=0.0):
    values = [float(record.get(key, default)) for record in records]
    return float(np.mean(values)), float(np.std(values))


def summarize_records(policy_name, records, elapsed_s):
    summary = {"policy": policy_name, "episodes": len(records), "elapsed_s": float(elapsed_s)}
    for key in [
        "reward",
        "delay",
        "drop_rate",
        "offload_rate",
        "energy",
        "model_call_step_ratio",
        "decision_agent_ratio",
        "safety_interrupt_count",
    ]:
        mean_value, std_value = mean_std(records, key)
        summary[key + "_mean"] = mean_value
        summary[key + "_std"] = std_value
    return summary


def build_policies(names, args, model_path):
    policies = []
    for name in names:
        if name == "temporal":
            policies.append(
                TemporalPolicyAdapter(
                    model_path=model_path,
                    device=args.device,
                    safety_enabled=not args.disable_safety,
                )
            )
        elif name == "local":
            policies.append(LocalPolicy())
        elif name == "offload":
            policies.append(OffloadPolicy())
        elif name == "random":
            policies.append(RandomPolicy(seed=args.seed, offload_probability=args.random_offload_prob))
        else:
            raise ValueError("Unknown policy: {}".format(name))
    return policies


def save_csv(path, rows):
    if not rows:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print("\nLong-term simulated evaluation")
    print("policy                  delay_mean  drop_mean  offload   reward_mean  model_calls")
    print("----------------------  ----------  ---------  --------  -----------  -----------")
    for row in rows:
        print(
            "{:<22}  {:>10.4f}  {:>9.4f}  {:>8.4f}  {:>11.4f}  {:>11.4f}".format(
                row["policy"],
                row["delay_mean"],
                row["drop_rate_mean"],
                row["offload_rate_mean"],
                row["reward_mean"],
                row["model_call_step_ratio_mean"],
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate long-term simulated delay on Jetson Nano.")
    parser.add_argument("--model-path", default=None, help="Path to mappo_actor_temporal_distilled_p25.pt.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of simulated episodes.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--load-factor", type=float, default=1.0, help="Scale task_min_bits/task_max_bits.")
    parser.add_argument(
        "--policies",
        default="temporal,local,offload,random",
        help="Comma-separated policies: temporal,local,offload,random.",
    )
    parser.add_argument("--random-offload-prob", type=float, default=0.5)
    parser.add_argument("--disable-safety", action="store_true", help="Disable temporal safety interrupts.")
    parser.add_argument("--save-csv", default="jetson_eval_summary.csv", help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.load_factor <= 0:
        raise ValueError("--load-factor must be positive.")

    model_path = resolve_model_path(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint not found: {}".format(model_path))

    names = [name.strip().lower() for name in args.policies.split(",") if name.strip()]
    cfg = EnvConfig(load_factor=args.load_factor)
    policies = build_policies(names, args, model_path)

    rows = []
    for policy in policies:
        print("Evaluating {} for {} episodes...".format(policy.name, args.episodes))
        rows.append(evaluate_policy(cfg, policy, args.episodes, args.seed))

    print_table(rows)
    save_csv(args.save_csv, rows)
    print("\nSaved CSV: {}".format(args.save_csv))
    print(json.dumps({"model_path": model_path, "load_factor": args.load_factor, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
