"""IMU preintegration, state propagation and the preintegrated IMU residual.

The functions are kept numerically identical to the original research code
(``src/models.py``), including its conventions, so that results do not
change. Known quirks are documented inline rather than fixed.
"""

import pypose as pp
import torch


def get_imu_between(t_i, t_j, imu_data, imu_timestamps, bias_g, bias_a):
    """Return the IMU samples with ``t_i <= t < t_j`` (bias-corrected).

    Args:
        t_i, t_j: interval limits (same units as ``imu_timestamps``).
        imu_data: dict with ``'acc'`` and ``'gyro'`` tensors of shape (N, 3).
        imu_timestamps: tensor (N,) of IMU timestamps.
        bias_g, bias_a: gyroscope / accelerometer biases (3,).

    Returns:
        dict ``{'acc', 'gyro', 'dt'}`` where ``dt`` is the list of time
        differences between consecutive selected samples (len = M - 1).

    Note:
        The biases are subtracted here *and* again in :func:`preintegrate`,
        exactly as in the original implementation.
    """
    acc = imu_data["acc"] - bias_a
    gyro = imu_data["gyro"] - bias_g

    indices = (imu_timestamps >= t_i) & (imu_timestamps < t_j)
    selected_idx = indices.nonzero(as_tuple=True)[0]

    ts = imu_timestamps[selected_idx]
    return {
        "acc": acc[selected_idx],
        "gyro": gyro[selected_idx],
        "dt": (ts[1:] - ts[:-1]).tolist(),
    }


def preintegrate(imu_data, bias_g, bias_a, dt_list, gravity):
    """Preintegrate IMU samples between two states.

    Returns:
        dict ``{'dR': SO3, 'dv': (3,), 'dp': (3,), 'dt': float}``.
    """
    device = bias_g.device
    dR = pp.identity_SO3(device=device)
    dv = torch.zeros(3, device=device)
    dp = torch.zeros(3, device=device)
    v = torch.zeros(3, device=device)

    for k in range(len(dt_list)):
        dt = dt_list[k]
        omega = imu_data["gyro"][k] - bias_g
        accel = imu_data["acc"][k] - bias_a

        # Wrapped in so3 (the original passed a plain tensor, which current
        # PyPose rejects; in legacy EuRoC mode this loop never runs because
        # all float32 time differences are zero).
        dR_k = pp.Exp(pp.so3(omega * dt))
        dR = dR @ dR_k
        # NOTE: original formulation (gravity * dt is added to the acceleration).
        a_world = dR @ accel + gravity * dt
        dv += a_world * dt
        dp += v * dt + 0.5 * a_world * dt ** 2
        v += a_world * dt

    return {"dR": dR, "dv": dv, "dp": dp, "dt": sum(dt_list)}


def propagate_preintegrated(state_i, delta):
    """Propagate ``state_i`` with a preintegrated IMU ``delta``."""
    R_i = state_i["R"]
    v_i = state_i["v"]
    p_i = state_i["p"]
    g = state_i["g"]
    dt = delta["dt"]

    return {
        "p": p_i + v_i * dt + 0.5 * g * dt ** 2 + R_i @ delta["dp"],
        "v": v_i + g * dt + R_i @ delta["dv"],
        "R": R_i @ delta["dR"],
        "bias_g": state_i["bias_g"],
        "bias_a": state_i["bias_a"],
        "g": g,
    }


def imu_residual_preint(state_i, state_j, delta, total_dt):
    """Preintegrated IMU residual ``[r_R, r_v, r_p]`` (9,) between two states."""
    R_i, R_j = state_i["R"], state_j["R"]
    v_i, v_j = state_i["v"], state_j["v"]
    p_i, p_j = state_i["p"], state_j["p"]
    g = state_i["g"]

    r_R = (delta["dR"].Inv() @ (R_i.Inv() @ R_j)).Log()
    r_v = R_i.Inv() @ (v_j - v_i - g * total_dt) - delta["dv"]
    r_p = R_i.Inv() @ (p_j - p_i - v_i * total_dt - 0.5 * g * total_dt ** 2) - delta["dp"]
    return torch.cat([r_R, r_v, r_p])
