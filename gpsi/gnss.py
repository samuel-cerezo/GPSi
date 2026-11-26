"""GNSS pre-processing: constant-velocity Kalman filter used to smooth positions."""

import torch


def smooth_gps(gps_measurements, gps_var, dt=0.01):
    """Constant-velocity Kalman filter over GNSS positions.

    Args:
        gps_measurements: sequence of (3,) tensors (or an (N,3) tensor).
        gps_var: measurement noise passed as ``R = gps_var * I``. The original
            experiments pass the noise *standard deviation* here; this is kept.
        dt: time step between epochs.

    Returns:
        list of filtered (3,) positions.
    """
    device = gps_measurements[0].device

    x = torch.zeros(6, device=device)  # [px, py, pz, vx, vy, vz]
    x[:3] = gps_measurements[0]
    P = torch.eye(6, device=device) * 1.0

    F = torch.eye(6, device=device)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    H = torch.zeros(3, 6, device=device)
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1

    Q = torch.eye(6, device=device) * 1e-4
    R = torch.eye(3, device=device) * gps_var

    smoothed_positions = []
    for z in gps_measurements:
        x = F @ x
        P = F @ P @ F.T + Q

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ torch.linalg.inv(S)

        x = x + K @ y
        P = (torch.eye(6, device=device) - K @ H) @ P
        smoothed_positions.append(x[:3].clone())

    return smoothed_positions


# Name used by the original experiment scripts.
kalman_filter_gps = smooth_gps
