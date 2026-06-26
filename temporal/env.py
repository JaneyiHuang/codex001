# env.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import numpy as np

from temporal.config import EnvConfig


class MECEnv:
    """
    Multi-agent MEC environment for MAPPO.

    Action:
        0 -> Local computing
        1 -> Offloading

    reset() return:
        {
            "obs": np.ndarray,    shape (M, obs_dim)
            "state": np.ndarray,  shape (state_dim,)
        }

    step(actions) return:
        {
            "obs": next_obs,
            "state": next_state,
            "reward": reward,     shared reward
            "done": done,
            "info": info
        }
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.M = cfg.M

        self.t = 0
        self.episode_limit = cfg.episode_limit

        # dynamic variables
        self.lambda_arr = None   # current task arrivals, shape (M,)
        self.H_arr = None        # current harvested energy, shape (M,)
        self.h_arr = None        # current channel gains |h|^2, shape (M,)

        self.E_arr = None        # battery energy, shape (M,)
        self.Q_loc = None        # local queue, shape (M,)
        self.Q_tx = None         # tx queue, shape (M,)
        self.F_loc = None        # local busy-until timestamp, shape (M,)
        self.F_tx = None         # tx busy-until timestamp, shape (M,)

        self.Q_edge = None       # edge queue, scalar
        self.F_edge = None       # edge busy-until timestamp, scalar

    # =========================================================
    # Public API
    # =========================================================
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset one episode."""
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

        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
        }

    def step(self, actions: np.ndarray | List[int]) -> Dict[str, Any]:
        """
        Advance one time slot.

        Args:
            actions: array-like, shape (M,)
                     0 -> local
                     1 -> offload
        """
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.M,), f"actions should have shape ({self.M},)"

        cfg = self.cfg

        # store current state snapshots for this slot
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

        # next state containers
        next_Q_loc = curr_Q_loc.copy()
        next_Q_tx = curr_Q_tx.copy()
        next_F_loc = curr_F_loc.copy()
        next_F_tx = curr_F_tx.copy()

        next_Q_edge = curr_Q_edge
        next_F_edge = curr_F_edge

        energy_cost = np.zeros(self.M, dtype=np.float32)
        delay_arr = np.zeros(self.M, dtype=np.float32)
        drop_flags = np.zeros(self.M, dtype=np.int64)

        # debug statistics
        tx_slot_arr = np.zeros(self.M, dtype=np.float32)
        edge_slot_arr = np.zeros(self.M, dtype=np.float32)
        rate_arr = np.zeros(self.M, dtype=np.float32)

        # first stage: handle local tasks and tx-stage of offloading tasks
        # offloading tasks that pass tx-stage will be collected for edge-stage
        edge_candidates = []

        for m in range(self.M):
            task_bits = float(curr_lambda[m])

            if actions[m] == 0:
                # -----------------------------
                # Local computing path
                # -----------------------------
                e_loc = self.local_energy_cost()
                enough_energy = curr_E[m] >= e_loc

                if enough_energy:
                    d_loc_bits = self.local_service_bits()
                    energy_cost[m] = e_loc
                else:
                    d_loc_bits = 0.0
                    energy_cost[m] = 0.0

                rate_arr[m] = 0

                q_loc_prime = max(curr_Q_loc[m] - d_loc_bits, 0.0)

                # admission control
                if q_loc_prime + task_bits > cfg.Q_loc_max:
                    # drop
                    next_Q_loc[m] = q_loc_prime
                    next_F_loc[m] = curr_F_loc[m]
                    delay_arr[m] = cfg.psi
                    drop_flags[m] = 1
                else:
                    # admitted
                    next_Q_loc[m] = q_loc_prime + task_bits

                    if enough_energy and cfg.f_local > 0:
                        T_loc = self.local_processing_slots(task_bits)
                        S_loc = max(float(self.t), float(curr_F_loc[m]))
                        C_loc = S_loc + T_loc - 1
                        next_F_loc[m] = C_loc + 1
                        delay_arr[m] = C_loc - float(self.t)
                    else:
                        # No energy means no local service in this slot.
                        # Keep the task in the queue and do not reserve future CPU time.
                        # The delay here is only a surrogate estimate because the exact
                        # completion time depends on future harvested energy.
                        nominal_T_loc = max(1, self.local_processing_slots(task_bits))
                        est_start = max(float(self.t) + 1.0, float(curr_F_loc[m]))
                        delay_arr[m] = est_start + nominal_T_loc - 1 - float(self.t)
                        next_F_loc[m] = curr_F_loc[m]

            else:
                # -----------------------------
                # Offloading path - tx stage
                # -----------------------------
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

                # tx admission
                if q_tx_prime + task_bits > cfg.Q_tx_max:
                    # drop at tx buffer
                    next_Q_tx[m] = q_tx_prime
                    next_F_tx[m] = curr_F_tx[m]
                    delay_arr[m] = cfg.psi
                    drop_flags[m] = 1
                else:
                    # admitted to tx queue
                    next_Q_tx[m] = q_tx_prime + task_bits

                    if enough_energy and rate > 0:
                        T_tx = self.tx_slots(task_bits, rate)
                        tx_slot_arr[m] = float(T_tx)

                        S_tx = max(float(self.t), float(curr_F_tx[m]))
                        C_tx = S_tx + T_tx - 1
                        next_F_tx[m] = C_tx + 1

                        edge_candidates.append({
                            "m": m,
                            "task_bits": task_bits,
                            "C_tx": C_tx,
                        })
                    else:
                        # No energy means the task only enters the transmission queue.
                        # Do not reserve future link time and do not push it to the
                        # edge stage before an actual transmission can happen.
                        nominal_rate = self.uplink_rate(curr_h[m])
                        nominal_T_tx = self.tx_slots(task_bits, nominal_rate)
                        nominal_T_edge = self.edge_processing_slots(task_bits)
                        est_tx_start = max(float(self.t) + 1.0, float(curr_F_tx[m]))
                        est_C_tx = est_tx_start + nominal_T_tx - 1
                        est_S_edge = max(est_C_tx + 1, float(curr_F_edge))
                        delay_arr[m] = est_S_edge + nominal_T_edge - 1 - float(self.t)
                        next_F_tx[m] = curr_F_tx[m]

        # second stage: edge admission and edge execution
        # sort by transmission completion time
        edge_candidates.sort(key=lambda x: x["C_tx"])

        # IMPORTANT:
        # use one shared edge queue/timestamp updated in order
        q_edge_working = max(curr_Q_edge - self.edge_service_bits(), 0.0)
        f_edge_working = curr_F_edge

        for item in edge_candidates:
            m = item["m"]
            task_bits = item["task_bits"]
            C_tx = item["C_tx"]

            if q_edge_working + task_bits > cfg.Q_edge_max:
                # dropped at edge
                delay_arr[m] = cfg.psi
                drop_flags[m] = 1
            else:
                # admitted at edge
                q_edge_working = q_edge_working + task_bits

                T_edge = self.edge_processing_slots(task_bits)
                edge_slot_arr[m] = float(T_edge)

                S_edge = max(C_tx + 1, f_edge_working)
                C_edge = S_edge + T_edge - 1
                f_edge_working = C_edge + 1

                delay_arr[m] = C_edge - float(self.t)

        next_Q_edge = q_edge_working
        next_F_edge = f_edge_working

        # update battery
        next_E = np.minimum(cfg.E_max, np.maximum(curr_E - energy_cost + curr_H, 0.0))

        # assign next dynamic states
        self.Q_loc = next_Q_loc.astype(np.float32)
        self.Q_tx = next_Q_tx.astype(np.float32)
        self.F_loc = next_F_loc.astype(np.float32)
        self.F_tx = next_F_tx.astype(np.float32)
        self.Q_edge = np.float32(next_Q_edge)
        self.F_edge = np.float32(next_F_edge)
        self.E_arr = next_E.astype(np.float32)

        # shared reward: normalized negative average delay
        # reward = - float(np.mean(delay_arr / cfg.psi)) # 不用惩罚而改用新设定的reward缩放常数
        # shared reward: negative average delay with fixed normalization
        reward = - float(np.mean(delay_arr) / cfg.reward_scale)# 即使用reward_scale来归一化

#         mean_delay = float(np.mean(delay_arr))
#         mean_drop = float(np.mean(drop_flags))
# # 给一个轻量级、显式的 slot 级丢包反馈
#         reward = - (
#             cfg.w_delay * (mean_delay / cfg.reward_scale)
#             + cfg.w_drop * mean_drop
#         )

        # bookkeeping
        offload_rate = float(np.mean(actions == 1))
        drop_num = int(np.sum(drop_flags))

        # move to next slot
        self.t += 1
        done = self.t >= self.episode_limit

        # sample next-slot random variables
        self.lambda_arr = self.sample_tasks()
        self.H_arr = self.sample_energy()
        self.h_arr = self.sample_channel()

        info = {
            "delay_mean": float(np.mean(delay_arr)),
            "delay_sum": float(np.sum(delay_arr)),
            "drop_num": drop_num,
            "drop_rate": float(drop_num / self.M),
            "offload_rate": offload_rate,
            "energy_mean": float(np.mean(self.E_arr)),
            "q_loc_mean": float(np.mean(self.Q_loc)),
            "q_tx_mean": float(np.mean(self.Q_tx)),
            "q_edge": float(self.Q_edge),
            # hjy
            "delay_arr": delay_arr,
            "drop_flags": drop_flags,
            "energy_cost": energy_cost,

            # debug info
            "tx_slot_mean": float(np.mean(tx_slot_arr)),
            "edge_slot_mean": float(np.mean(edge_slot_arr)),
            "rate_mean": float(np.mean(rate_arr)),
        }

        return {
            "obs": self.get_obs(),
            "state": self.get_state(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    # =========================================================
    # Sampling functions
    # =========================================================
    def sample_tasks(self) -> np.ndarray:
        """Sample current slot task arrivals (bits)."""
        return np.random.uniform(
            self.cfg.task_min_bits,
            self.cfg.task_max_bits,
            size=self.M
        ).astype(np.float32)

    def sample_energy(self) -> np.ndarray:
        """Sample harvested energy (J)."""
        return np.random.uniform(
            self.cfg.H_min,
            self.cfg.H_max,
            size=self.M
        ).astype(np.float32)

    def sample_channel(self) -> np.ndarray:
        """
        Sample channel gains |h|^2.
        Use exponential distribution (Rayleigh fading power gain).
        """
        return np.random.exponential(scale=1.0, size=self.M).astype(np.float32)

    # =========================================================
    # Physics / service helpers
    # =========================================================
    def uplink_rate(self, h_gain: float) -> float:
        """Shannon rate."""
        snr = self.cfg.p_tx * float(h_gain) / self.cfg.sigma2
        return self.cfg.B * np.log2(1.0 + snr)

    def local_service_bits(self) -> float:
        """Bits that can be processed locally in one slot."""
        return self.cfg.f_local * self.cfg.delta / self.cfg.rho_local

    def edge_service_bits(self) -> float:
        """Bits that can be processed at edge in one slot."""
        return self.cfg.f_edge * self.cfg.delta / self.cfg.rho_edge

    def local_processing_slots(self, task_bits: float) -> int:
        return int(np.ceil(task_bits * self.cfg.rho_local / (self.cfg.f_local * self.cfg.delta)))

    def edge_processing_slots(self, task_bits: float) -> int:
        return int(np.ceil(task_bits * self.cfg.rho_edge / (self.cfg.f_edge * self.cfg.delta)))

    def tx_slots(self, task_bits: float, rate: float) -> int:
        denom = max(rate * self.cfg.delta, 1e-8)
        return int(np.ceil(task_bits / denom))

    def local_energy_cost(self) -> float:
        return self.cfg.kappa * (self.cfg.f_local ** 3) * self.cfg.delta

    def tx_energy_cost(self) -> float:
        return self.cfg.p_tx * self.cfg.delta

    # =========================================================
    # Observation / state
    # =========================================================
    def get_obs(self) -> np.ndarray:
        """
        Per-agent local observation, normalized.
        obs_m = [
            lambda_m,
            E_m,
            H_m,
            h_m,
            Q_loc_m,
            Q_tx_m,
            F_loc_m,
            F_tx_m,
            Q_edge,
            F_edge
        ]
        """
        obs = []

        for m in range(self.M):
            obs_m = np.array([
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
            ], dtype=np.float32)
            obs.append(obs_m)

        return np.stack(obs, axis=0)

    def get_state(self) -> np.ndarray:
        """
        Global state for centralized critic.
        state_dim = 8*M + 2
        """
        state = np.concatenate([
            self.lambda_arr / self.cfg.task_max_bits,
            self.E_arr / self.cfg.E_max,
            self.H_arr / self.cfg.H_max,
            np.minimum(self.h_arr, self.cfg.h_norm_clip) / self.cfg.h_norm_clip,
            self.Q_loc / self.cfg.Q_loc_max,
            self.Q_tx / self.cfg.Q_tx_max,
            np.minimum(self.F_loc, self.cfg.episode_limit) / self.cfg.episode_limit,
            np.minimum(self.F_tx, self.cfg.episode_limit) / self.cfg.episode_limit,
            np.array([
                self.Q_edge / self.cfg.Q_edge_max,
                min(self.F_edge, self.cfg.episode_limit) / self.cfg.episode_limit
            ], dtype=np.float32)
        ], axis=0).astype(np.float32)

        return state

    # =========================================================
    # Optional utilities
    # =========================================================
    def get_env_info(self) -> Dict[str, Any]:
        return {
            "n_agents": self.M,
            "n_actions": self.cfg.n_actions,
            "obs_dim": self.cfg.obs_dim,
            "state_dim": self.cfg.state_dim,
            "episode_limit": self.cfg.episode_limit,
        }

    def seed(self, seed: int) -> None:
        np.random.seed(seed)
