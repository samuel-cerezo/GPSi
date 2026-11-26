"""Dataset loading (EuRoC, GVINS, MARS) and simulated GNSS measurements.

Every loader returns a dictionary with the same keys:

``gt_time``  (N,)   ground-truth / GNSS epoch timestamps
``gt_p``     (N,3)  ground-truth positions [m]
``gt_q``     (N,4)  ground-truth orientations, quaternion ``[qw, qx, qy, qz]``
``gt_v``     (N,3)  ground-truth velocities [m/s]
``gt_bg``    (N,3)  ground-truth gyroscope bias (zeros if unavailable)
``gt_ba``    (N,3)  ground-truth accelerometer bias (zeros if unavailable)
``imu_time`` (M,)   IMU timestamps
``acc``      (M,3)  accelerometer readings [m/s^2]
``gyro``     (M,3)  gyroscope readings [rad/s]
``has_gt_rotation`` bool, False when the dataset provides no orientation GT

GNSS positions are *simulated* by adding isotropic Gaussian noise to the
ground-truth positions (:func:`add_gnss_noise`), exactly as in the original
experiments.

Timestamps
----------
``time_mode="legacy"`` reproduces the original experiment code: raw
nanosecond timestamps are stored as ``float32``. With EuRoC epoch timestamps
(~1.4e18 ns) float32 has a resolution of ~137 s, so consecutive timestamps
collapse to the same value. This is kept as the default for EuRoC so that the
numbers produced match the original code. ``time_mode="seconds"`` converts
timestamps to seconds in ``float64`` (physically meaningful IMU intervals).
"""

import os

import numpy as np
import pandas as pd
import torch

TIME_MODES = ("legacy", "seconds")

# Default sequences per dataset (directory names under the dataset root).
EUROC_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]
GVINS_SEQUENCES = ["sports_field", "complex_environment", "urban_driving"]

DEFAULT_SEQUENCES = {
    "euroc": EUROC_SEQUENCES,
    "gvins": GVINS_SEQUENCES,
    "mars": [],  # auto-detected from the sub-directories of the dataset root
}

# Default time handling per dataset. EuRoC keeps the original (legacy) code
# path; the GVINS loader of the original code did not run, so GVINS/MARS use
# float64 seconds.
DEFAULT_TIME_MODE = {"euroc": "legacy", "gvins": "seconds", "mars": "seconds"}


def _time_tensor(ns_values, time_mode, device):
    if time_mode == "legacy":
        return torch.tensor(ns_values, dtype=torch.float32, device=device)
    if time_mode == "seconds":
        return torch.tensor(np.asarray(ns_values, dtype=np.float64) * 1e-9, dtype=torch.float64, device=device)
    raise ValueError(f"Unknown time_mode '{time_mode}', expected one of {TIME_MODES}")


def _f32(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


def _load_euroc_style_imu(path, time_mode, device):
    """Read ``mav0/imu0/data.csv``: t[ns], w_x, w_y, w_z, a_x, a_y, a_z."""
    imu_df = pd.read_csv(os.path.join(path, "mav0", "imu0", "data.csv"))
    return {
        "imu_time": _time_tensor(imu_df.iloc[:, 0].values, time_mode, device),
        "acc": _f32(imu_df.iloc[:, 4:7].values, device),
        "gyro": _f32(imu_df.iloc[:, 1:4].values, device),
    }


def load_euroc(path, device="cpu", time_mode="legacy"):
    """Load one EuRoC MAV sequence in ASL format.

    Args:
        path: folder that contains ``mav0/`` (e.g. ``datasets/EuRoC/MH_01_easy``).
    """
    gt_df = pd.read_csv(os.path.join(path, "mav0", "state_groundtruth_estimate0", "data.csv"))
    data = {
        "gt_time": _time_tensor(gt_df.iloc[:, 0].values, time_mode, device),
        "gt_p": _f32(gt_df.iloc[:, 1:4].values, device),   # px, py, pz
        "gt_q": _f32(gt_df.iloc[:, 4:8].values, device),   # qw, qx, qy, qz
        "gt_v": _f32(gt_df.iloc[:, 8:11].values, device),  # vx, vy, vz
        "gt_bg": _f32(gt_df.iloc[:, 11:14].values, device),
        "gt_ba": _f32(gt_df.iloc[:, 14:17].values, device),
        "has_gt_rotation": True,
    }
    data.update(_load_euroc_style_imu(path, time_mode, device))
    return data


def _load_gnss_csv_sequence(path, gt_file, device, time_mode):
    """Shared loader for GVINS / MARS sequences converted to CSV.

    Expected layout::

        <path>/<gt_file>          t[ns], x, y, z [, qw, qx, qy, qz [, vx, vy, vz]]
        <path>/mav0/imu0/data.csv EuRoC-style IMU file

    Positions must be expressed in a local metric frame (e.g. UTM/ENU).
    When orientation is missing, identity quaternions are used and
    ``has_gt_rotation`` is False (rotation errors are then reported as NaN).
    When velocity is missing it is computed by finite differences.
    """
    gt_df = pd.read_csv(os.path.join(path, gt_file))
    t_ns = gt_df.iloc[:, 0].values
    gt_p_np = gt_df.iloc[:, 1:4].values.astype(np.float64)
    n = len(gt_df)
    ncols = gt_df.shape[1]

    if ncols >= 8:
        gt_q_np = gt_df.iloc[:, 4:8].values
        has_rot = True
    else:
        gt_q_np = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        has_rot = False

    if ncols >= 11:
        gt_v_np = gt_df.iloc[:, 8:11].values
    else:
        t_s = np.asarray(t_ns, dtype=np.float64) * 1e-9
        gt_v_np = np.gradient(gt_p_np, t_s, axis=0) if n > 1 else np.zeros((n, 3))

    data = {
        "gt_time": _time_tensor(t_ns, time_mode, device),
        "gt_p": _f32(gt_p_np, device),
        "gt_q": _f32(gt_q_np, device),
        "gt_v": _f32(gt_v_np, device),
        "gt_bg": torch.zeros(n, 3, device=device),
        "gt_ba": torch.zeros(n, 3, device=device),
        "has_gt_rotation": has_rot,
    }
    data.update(_load_euroc_style_imu(path, time_mode, device))
    return data


def load_gvins(path, device="cpu", time_mode="seconds"):
    """Load one GVINS sequence converted to CSV (``gps_utm.csv`` + ``mav0/imu0/data.csv``)."""
    return _load_gnss_csv_sequence(path, "gps_utm.csv", device, time_mode)


def load_mars(path, device="cpu", time_mode="seconds"):
    """Load one MARS-LVIG sequence converted to CSV (``gnss_gt.csv`` + ``mav0/imu0/data.csv``)."""
    return _load_gnss_csv_sequence(path, "gnss_gt.csv", device, time_mode)


LOADERS = {"euroc": load_euroc, "gvins": load_gvins, "mars": load_mars}


def load_sequence(dataset, path, device="cpu", time_mode=None):
    """Load a sequence of ``dataset`` (``euroc``, ``gvins`` or ``mars``) located at ``path``."""
    dataset = dataset.lower()
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {sorted(LOADERS)}")
    if time_mode is None:
        time_mode = DEFAULT_TIME_MODE[dataset]
    return LOADERS[dataset](path, device=device, time_mode=time_mode)


def list_sequences(dataset, root):
    """Sequences of ``dataset`` that exist under ``root``."""
    defaults = DEFAULT_SEQUENCES[dataset.lower()]
    if defaults:
        return [s for s in defaults if os.path.isdir(os.path.join(root, s))]
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "mav0")))


def add_gnss_noise(data, gps_noise_std):
    """Simulate GNSS positions: ``gps_p = gt_p + gps_noise_std * N(0, I)``.

    Uses the global torch RNG (one ``randn_like`` call over the whole
    sequence), as in the original code. Returns a shallow copy of ``data``.
    """
    out = dict(data)
    out["gps_p"] = data["gt_p"] + gps_noise_std * torch.randn_like(data["gt_p"])
    return out
