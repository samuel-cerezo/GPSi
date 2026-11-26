"""Synthetic GNSS-inertial sequence generator (used by the tests and for demos).

The generated dictionary has the same keys as the dataset loaders in
:mod:`gpsi.data`, with timestamps in seconds (``float64``). The trajectory is
expressed in a world frame that coincides with the first body frame, and the
IMU readings are generated from the analytic trajectory:

``acc = R^T (a_world - g)``, ``gyro = omega_body``.
"""

import numpy as np
import pypose as pp
import torch


def _rotation_from_rotvec(rv):
    return pp.so3(torch.as_tensor(rv, dtype=torch.float64)).Exp()


def generate_synthetic_sequence(n_epochs=30, gnss_rate=20.0, imu_rate=200.0,
                                gravity=(0.0, -9.81, 0.0), imu_noise_std=0.0,
                                seed=0, device="cpu"):
    """Generate a smooth 3-D trajectory with IMU and ground truth.

    Args:
        n_epochs: number of ground-truth / GNSS epochs.
        gnss_rate: GNSS (ground-truth) rate [Hz].
        imu_rate: IMU rate [Hz] (must be a multiple of ``gnss_rate``).
        gravity: gravity vector in the world frame [m/s^2].
        imu_noise_std: white noise added to accelerometer and gyroscope.
        seed: NumPy seed for the IMU noise.

    Returns:
        dict with keys ``gt_time, gt_p, gt_q, gt_v, gt_bg, gt_ba, imu_time,
        acc, gyro, has_gt_rotation``.
    """
    rng = np.random.default_rng(seed)
    duration = (n_epochs - 1) / gnss_rate
    t_imu = np.arange(0.0, duration + 1.0 / gnss_rate, 1.0 / imu_rate)
    t_gt = np.arange(n_epochs) / gnss_rate

    # Position: smooth Lissajous-like motion; velocity and acceleration analytic.
    A = np.array([2.0, 1.5, 0.5])
    w = np.array([0.8, 1.1, 0.6])

    def pos(t):
        return np.stack([A[0] * np.sin(w[0] * t), A[1] * (1 - np.cos(w[1] * t)), A[2] * np.sin(w[2] * t)], -1)

    def vel(t):
        return np.stack([A[0] * w[0] * np.cos(w[0] * t), A[1] * w[1] * np.sin(w[1] * t),
                         A[2] * w[2] * np.cos(w[2] * t)], -1)

    def accel(t):
        return np.stack([-A[0] * w[0] ** 2 * np.sin(w[0] * t), A[1] * w[1] ** 2 * np.cos(w[1] * t),
                         -A[2] * w[2] ** 2 * np.sin(w[2] * t)], -1)

    # Orientation: constant body angular velocity -> R(t) = Exp(omega * t), R(0) = I.
    omega_body = np.array([0.05, -0.08, 0.15])

    def rot(t):
        return _rotation_from_rotvec(omega_body[None, :] * np.asarray(t)[:, None])

    g = np.asarray(gravity, dtype=np.float64)
    R_imu = rot(t_imu)
    a_world = torch.as_tensor(accel(t_imu) - g, dtype=torch.float64)
    acc = R_imu.Inv().Act(a_world).numpy()
    gyro = np.tile(omega_body, (len(t_imu), 1))
    if imu_noise_std > 0:
        acc = acc + imu_noise_std * rng.standard_normal(acc.shape)
        gyro = gyro + imu_noise_std * rng.standard_normal(gyro.shape)

    q_xyzw = rot(t_gt).tensor().numpy()
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)

    f32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    return {
        "gt_time": torch.tensor(t_gt, dtype=torch.float64, device=device),
        "gt_p": f32(pos(t_gt)),
        "gt_q": f32(q_wxyz),
        "gt_v": f32(vel(t_gt)),
        "gt_bg": torch.zeros(n_epochs, 3, device=device),
        "gt_ba": torch.zeros(n_epochs, 3, device=device),
        "imu_time": torch.tensor(t_imu, dtype=torch.float64, device=device),
        "acc": f32(acc),
        "gyro": f32(gyro),
        "has_gt_rotation": True,
    }
