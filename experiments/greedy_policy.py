from __future__ import annotations

import numpy as np

from env import MECEnv


def greedy_action_for_agent(env: MECEnv, m: int) -> int:
    cfg = env.cfg

    task_bits = float(env.lambda_arr[m])
    energy = float(env.E_arr[m])
    channel_gain = float(env.h_arr[m])

    e_loc = env.local_energy_cost()
    if energy >= e_loc and cfg.f_local > 0:
        t_loc = env.local_processing_slots(task_bits)
    else:
        t_loc = cfg.psi

    s_loc = max(float(env.t), float(env.F_loc[m]))
    est_local_delay = (s_loc + t_loc - 1) - float(env.t)

    e_tx = env.tx_energy_cost()
    if energy >= e_tx:
        rate = env.uplink_rate(channel_gain)
        if rate > 0:
            t_tx = env.tx_slots(task_bits, rate)
        else:
            t_tx = cfg.psi
    else:
        t_tx = cfg.psi

    t_edge = env.edge_processing_slots(task_bits)
    s_tx = max(float(env.t), float(env.F_tx[m]))
    c_tx = s_tx + t_tx - 1
    s_edge = max(c_tx + 1, float(env.F_edge))
    est_offload_delay = (s_edge + t_edge - 1) - float(env.t)

    return 0 if est_local_delay <= est_offload_delay else 1


class GreedyPolicy:
    name = "greedy"

    def select_actions(self, env: MECEnv, obs: np.ndarray) -> np.ndarray:
        return np.array(
            [greedy_action_for_agent(env, m) for m in range(env.M)],
            dtype=np.int64,
        )

