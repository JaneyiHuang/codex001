# config.py
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # =========================
    # Basic system settings
    # =========================
    M: int = 4                     # number of UEs
    episode_limit: int = 200       # time slots per episode
    delta: float = 1.0             # slot length (s)

    # =========================
    # Communication settings
    # =========================
    B: float = 1e6                 # bandwidth (Hz)
    sigma2: float = 1e-9           # noise power (W)
    p_tx: float = 0.5              # fixed transmission power (W)

    # =========================
    # Computation settings
    # =========================
    f_local: float = 5e8           # local CPU frequency (cycles/s)
    f_edge: float = 1e10           # edge CPU frequency (cycles/s)
    rho_local: float = 1200.0       # cycles per bit at UE
    rho_edge: float = 500.0        # cycles per bit at edge
    kappa: float = 1e-28           # effective capacitance coefficient

    # =========================
    # Energy settings
    # =========================
    E_max: float = 5.0             # battery capacity (J)
    E_init: float = 2.5            # initial battery energy (J)
    H_min: float = 0.1             # min harvested energy per slot (J)
    H_max: float = 0.5             # max harvested energy per slot (J)

    # =========================
    # Task arrival settings
    # =========================
    task_min_bits: float = 1.5e6   # minimum task size (bits)
    task_max_bits: float = 6e6     # maximum task size (bits)

    # =========================
    # Queue buffer settings
    # =========================
    Q_loc_max: float = 1e7         # local queue capacity (bits)
    Q_tx_max: float = 1e7          # transmission queue capacity (bits)
    Q_edge_max: float = 4e7        # edge queue capacity (bits)

    # =========================
    # Penalty
    # =========================
    psi: float = 300.0              # drop penalty delay

    # =========================
    # Reward helpers
    # =========================
    reward_scale: float = 200.0   # fixed reward normalization新增的关于reward里面设定的缩放
    # w_delay: float = 1.0          # weight of mean delay term
    # w_drop: float = 0.3           # light auxiliary penalty for drop rate

    # =========================
    # Normalization helpers
    # =========================
    h_norm_clip: float = 10.0      # clip for channel gain normalization

    # =========================
    # Spaces
    # =========================
    n_actions: int = 2             # 0: local, 1: offload
    obs_dim: int = 10              # per-agent obs dimension
    state_dim: int = 0             # will be set dynamically

    def __post_init__(self):
        self.state_dim = 8 * self.M + 2
