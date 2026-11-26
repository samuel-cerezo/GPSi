"""GNSS-inertial initialization by distance residuals (core method).

This is a modular rewrite of the original experiment script
(``old_scripts/experimentos_gpsFINAL.py``). The order of operations and of
random-number draws is preserved so that, for the same seed and device, the
numerical results are unchanged.

Pipeline of :func:`run_initialization`:

1. Seed the RNGs and draw random initial IMU biases.
2. Simulate GNSS positions by adding noise to the ground truth.
3. Dead-reckon ``N`` states with IMU preintegration (used as initial guess).
4. Optionally evaluate the observability criterion to choose when the
   GNSS-inertial alignment ``T_wb`` is activated.
5. Jointly optimise all states (and ``T_wb``) with Adam on a loss made of
   preintegrated IMU residuals, GNSS inter-epoch *distance* residuals,
   GNSS anchor residuals (only after activation) and bias random walks.
6. Compute the evaluation metrics.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pypose as pp
import torch

from .data import add_gnss_noise
from .gnss import smooth_gps
from .imu import get_imu_between, imu_residual_preint, preintegrate, propagate_preintegrated
from .metrics import compute_ate, compute_rmse, compute_rotation_rmse, traveled_distance
from .observability import find_Twb_activation
from .residuals import bias_acc_residual, bias_gyro_residual, velocity_smoothness_residual


@dataclass
class InitConfig:
    """Hyper-parameters of one initialization run (defaults = paper experiments)."""

    max_gps_measurements: int = 100          # number of GNSS epochs / states in the window
    use_twb: bool = True                     # estimate the GNSS-inertial alignment T_wb
    min_gps_for_alignment: int = 1           # epoch index from which anchor residuals are used
    use_obs_criterion: bool = False          # choose min_gps_for_alignment automatically
    gps_noise_std: float = 0.2               # simulated GNSS noise [m]
    gravity_init: tuple = (0.0, -9.81, 0.0)  # initial gravity guess (first body frame)
    rot_init_noise: float = 0.01             # std of the rotation perturbation of the initial guess [rad]
    lr: float = 1e-2
    max_epochs: int = 1000
    tolerance: float = 1e-3                  # early stopping: |loss - best| < tolerance ...
    patience: int = 10                       # ... during `patience` consecutive epochs
    imu_weight: float = 100.0
    gps_weight: float = 1.0
    bias_weight: float = 1.0
    gyro_bias_std: float = 1.9393e-5 * np.sqrt(200)
    accel_bias_std: float = 3.0e-3 * np.sqrt(200)
    criterion_sigma_gps: float = 0.5
    criterion_window: int = 10
    criterion_threshold: float = 1e-2
    seed: int = 42
    log_every: int = 10                      # print losses every N epochs (0 disables)
    extra: dict = field(default_factory=dict)


@dataclass
class InitResult:
    metrics: dict
    est_states: list
    T_w_b: dict
    criterion_history: list
    loss_history: list


def _seed_everything(seed):
    torch.manual_seed(seed)
    # The original code left NumPy unseeded (initial biases were therefore not
    # reproducible). We seed it to make runs deterministic.
    np.random.seed(seed)


def run_initialization(clean_data, cfg: InitConfig, device="cpu"):
    """Run one GNSS-inertial initialization on a loaded sequence.

    Args:
        clean_data: dict returned by :func:`gpsi.data.load_sequence` (without
            GNSS noise; it is added here so that the RNG order matches the
            original code).
        cfg: :class:`InitConfig`.
        device: torch device string.

    Returns:
        :class:`InitResult`.
    """
    N = cfg.max_gps_measurements
    if len(clean_data["gt_time"]) < N:
        raise ValueError(f"Sequence has {len(clean_data['gt_time'])} epochs, fewer than max_gps_measurements={N}")
    log = cfg.log_every > 0

    # ---- 1. seeds and initial biases (same draw order as the original) ----
    _seed_everything(cfg.seed)
    bias_g_init = torch.tensor(np.random.normal(0, cfg.gyro_bias_std, 3), dtype=torch.float32, device=device)
    bias_a_init = torch.tensor(np.random.normal(0, cfg.accel_bias_std, 3), dtype=torch.float32, device=device)

    # ---- 2. simulated GNSS ----
    data = add_gnss_noise(clean_data, cfg.gps_noise_std)

    # ---- 3. IMU dead reckoning ----
    state = {
        "p": torch.zeros(3, device=device),
        "v": torch.zeros(3, device=device),
        "R": pp.identity_SO3(device=device),
        "bias_g": bias_g_init,
        "bias_a": bias_a_init,
        "g": torch.tensor(list(cfg.gravity_init), device=device),
    }
    states = [state]
    timestamps = [data["gt_time"][0]]
    gps_measurements = [data["gps_p"][0]]
    gt_velocities = [data["gt_v"][0]]

    for k in range(N - 1):
        t_start, t_end = data["gt_time"][k], data["gt_time"][k + 1]
        imu_ij = get_imu_between(t_start, t_end, data, data["imu_time"], state["bias_g"], state["bias_a"])
        delta = preintegrate(imu_ij, state["bias_g"], state["bias_a"], imu_ij["dt"], gravity=state["g"])
        state = propagate_preintegrated(state, delta)

        states.append(state)
        timestamps.append(t_end)
        gps_measurements.append(data["gps_p"][k + 1])
        gt_velocities.append(data["gt_v"][k + 1])

    # Initial guess of the optimised states.
    est_states = []
    for i in range(len(states)):
        R_init = states[i]["R"]
        if cfg.rot_init_noise > 0:
            omega_noise = torch.randn(3, device=device) * cfg.rot_init_noise
            R_init = R_init @ pp.Exp(pp.LieTensor(omega_noise, ltype=pp.so3_type))
        est_states.append({
            "p": (gps_measurements[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
            "v": (gt_velocities[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
            "R": R_init.detach().clone().requires_grad_(),
            "bias_g": (states[i]["bias_g"] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
            "bias_a": (states[i]["bias_a"] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
            "g": (states[i]["g"] + 0.001 * torch.randn(3, device=device)).requires_grad_(),
        })

    T_w_b = {
        "R": pp.identity_SO3(device=device).detach().clone().requires_grad_(),
        "t": torch.zeros(3, device=device, requires_grad=True),
    }

    params = [s[key] for key in ("p", "v", "R", "bias_g", "bias_a", "g") for s in est_states]
    params += [T_w_b["R"], T_w_b["t"]]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    dt_gps = (data["gt_time"][1] - data["gt_time"][0]).item()
    gps_measurements = torch.stack(gps_measurements)
    # The filter does not depend on the optimised variables: compute it once
    # (the original recomputed the same values at every epoch).
    smoothed_gps = smooth_gps(gps_measurements, cfg.gps_noise_std, dt_gps)

    # ---- 4. observability criterion ----
    criterion_history = []
    activation_frame = None
    min_gps = cfg.min_gps_for_alignment
    if cfg.use_obs_criterion:
        activation_frame, criterion_history = find_Twb_activation(
            est_states, gps_measurements, T_w_b, sigma_gps=cfg.criterion_sigma_gps,
            window_size=cfg.criterion_window, rel_threshold=cfg.criterion_threshold, verbose=log)
        if activation_frame is not None:
            min_gps = activation_frame
        elif log:
            print(f"[gpsi] observability criterion never met; using min_gps_for_alignment={min_gps}")

    # ---- 5. optimisation ----
    best_loss = float("inf")
    epochs_no_improve = 0
    loss_history = []
    r_gps_anchor = None
    epoch = -1

    start_time = time.time()
    for epoch in range(cfg.max_epochs):
        optimizer.zero_grad()
        loss = 0.0

        for i in range(len(states) - 1):
            j = i + 1
            imu_ij = get_imu_between(timestamps[i], timestamps[j], data, data["imu_time"],
                                     est_states[i]["bias_g"], est_states[i]["bias_a"])
            delta = preintegrate(imu_ij, est_states[i]["bias_g"], est_states[i]["bias_a"],
                                 imu_ij["dt"], gravity=est_states[i]["g"])
            r_pre = imu_residual_preint(est_states[i], est_states[j], delta, dt_gps)
            loss += cfg.imu_weight * (r_pre ** 2).mean()

        aligned = False
        for k in range(len(est_states) - 1):
            r_vel = velocity_smoothness_residual(est_states[k], est_states[k + 1])
            r_bias_g = bias_gyro_residual(est_states[k], est_states[k + 1])
            r_bias_a = bias_acc_residual(est_states[k], est_states[k + 1])

            aligned = (k >= min_gps) and cfg.use_twb
            if aligned:
                p_i_world = T_w_b["R"].Act(est_states[k]["p"]) + T_w_b["t"]
                p_j_world = T_w_b["R"].Act(est_states[k + 1]["p"]) + T_w_b["t"]
                r_gps_disp = (p_j_world - p_i_world) - (smoothed_gps[k + 1] - smoothed_gps[k])
                r_gps_anchor = T_w_b["R"].Act(est_states[k]["p"]) + T_w_b["t"] - smoothed_gps[k]
                loss += cfg.gps_weight * ((r_gps_disp ** 2).mean() + (r_gps_anchor ** 2).mean())
            else:
                delta_est = est_states[k + 1]["p"] - est_states[k]["p"]
                r_gps_disp = delta_est - (smoothed_gps[k + 1] - smoothed_gps[k])
                loss += cfg.gps_weight * (r_gps_disp ** 2).mean()

            loss += cfg.bias_weight * (r_bias_g ** 2).mean()
            loss += cfg.bias_weight * (r_bias_a ** 2).mean()

        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_history.append(loss_value)

        if log and epoch % cfg.log_every == 0:
            print(f"Epoch {epoch} | Loss: {loss_value:.6f}")
            print(f"  -  IMU preint.: {r_pre.norm().item():.6f}")
            print(f"  -  Vel smooth: {r_vel.norm().item():.6f}")
            print(f"  -  Bias gyro:  {r_bias_g.norm().item():.6f}")
            print(f"  -  Bias accel: {r_bias_a.norm().item():.6f}")
            print(f"  -  GPS disp.:  {r_gps_disp.norm().item():.6f}")
            if aligned:
                print(f"  -  GPS anchor: {r_gps_anchor.norm().item():.6f}")

        if abs(loss_value - best_loss) < cfg.tolerance:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                if log:
                    print(f"Early stopping at epoch {epoch} with loss {loss_value:.6f}")
                break
        else:
            best_loss = loss_value
            epochs_no_improve = 0

    # ---- 6. metrics ----
    est_positions = [s["p"].detach().cpu() for s in est_states]
    gt_positions = [data["gt_p"][i].detach().cpu() for i in range(len(est_states))]
    est_velocities = [s["v"].detach().cpu() for s in est_states]
    gt_velocities = [v.detach().cpu() for v in gt_velocities]

    rmse_pos = compute_rmse(est_positions, gt_positions)
    rmse_vel = compute_rmse(est_velocities, gt_velocities)
    ate = compute_ate(est_positions, gt_positions)
    elapsed = time.time() - start_time
    if data.get("has_gt_rotation", True):
        rot_rmse = compute_rotation_rmse(est_states, data["gt_q"][:len(est_states)])
    else:
        rot_rmse = float("nan")

    T_final = np.eye(4)
    T_final[:3, :3] = T_w_b["R"].matrix().detach().cpu().numpy()
    T_final[:3, 3] = T_w_b["t"].detach().cpu().numpy().flatten()

    metrics = {
        "use_twb": cfg.use_twb,
        "obs_criterion": cfg.use_obs_criterion,
        "min_gps_for_alignment": min_gps,
        "activation_frame": activation_frame,
        "gps_noise_std": cfg.gps_noise_std,
        "max_gps_measurements": N,
        "seed": cfg.seed,
        "epochs": epoch + 1,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "time_s": elapsed,
        "rmse_pos_m": rmse_pos,
        "ate_m": ate,
        "rmse_vel_mps": rmse_vel,
        "rot_rmse_rad": rot_rmse,
        "rot_rmse_deg": float(np.degrees(rot_rmse)),
        "distance_m": traveled_distance(gps_measurements),
        "T_wb": T_final.flatten().tolist(),
    }

    if log:
        print(f"Time: {elapsed:.6f} s | RMSE pos: {rmse_pos:.6f} m | ATE: {ate:.6f} m | "
              f"RMSE vel: {rmse_vel:.6f} m/s | RMSE rot: {rot_rmse:.6f} rad")

    return InitResult(metrics=metrics, est_states=est_states, T_w_b=T_w_b,
                      criterion_history=criterion_history, loss_history=loss_history)


def config_to_dict(cfg: InitConfig):
    return asdict(cfg)
