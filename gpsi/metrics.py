"""Evaluation metrics (identical definitions to the original experiments).

Notes on the definitions, kept for consistency with the paper results:

* :func:`compute_rmse` averages the squared error over *all components* of
  all vectors before the square root (per-axis RMSE), i.e.
  ``sqrt(mean_i mean_c (e_ic^2))``.
* :func:`compute_ate` is the *mean* Euclidean position error,
  ``mean_i ||p_i - p_i^gt||``, computed without any trajectory alignment.
* :func:`compute_rotation_rmse` returns the RMSE of the geodesic rotation
  error in **radians**. The original scripts printed (and stored in the
  ``RMSE rotación`` CSV column) this value with a "deg" label, but no
  conversion was applied. :func:`compute_rotation_rmse_deg` gives degrees.
* Rotations are compared without aligning the estimator frame (first body
  frame) to the ground-truth world frame, as in the original code.
"""

import math

import pypose as pp
import torch


def compute_rmse(est_values, gt_values):
    """Per-component RMSE between two sequences of vectors."""
    errors = [((est - gt) ** 2).mean() for est, gt in zip(est_values, gt_values)]
    return torch.sqrt(torch.stack(errors).mean()).item()


def compute_ate(est_values, gt_values):
    """Mean Euclidean error between two sequences of positions [m]."""
    errors = [torch.norm(est - gt) for est, gt in zip(est_values, gt_values)]
    return torch.stack(errors).mean().item()


def quat_wxyz_to_SO3(q_wxyz):
    """Convert ``[qw, qx, qy, qz]`` quaternion(s) to a PyPose ``SO3``."""
    q_xyzw = torch.cat([q_wxyz[..., 1:], q_wxyz[..., 0:1]], dim=-1)
    return pp.SO3(q_xyzw)


def compute_rotation_rmse(est_states, gt_quaternions):
    """Rotation RMSE in **radians** between ``est_states[i]['R']`` and GT quaternions ``[w,x,y,z]``.

    Identical to the original implementation (see module docstring).
    """
    angles = []
    for i in range(len(est_states)):
        R_est = est_states[i]["R"].matrix()
        R_gt = quat_wxyz_to_SO3(gt_quaternions[i].unsqueeze(0)).matrix()[0]
        R_err = R_est @ R_gt.T
        trace = torch.trace(R_err.reshape(3, 3))
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        angles.append(torch.acos(cos_theta))
    errors = torch.stack(angles)
    return torch.sqrt(torch.mean(errors ** 2)).item()


def compute_rotation_rmse_deg(est_states, gt_quaternions):
    """Rotation RMSE in degrees (``compute_rotation_rmse`` converted to degrees)."""
    return math.degrees(compute_rotation_rmse(est_states, gt_quaternions))


def traveled_distance(positions):
    """Length of the polyline through ``positions`` [m]."""
    dist = 0.0
    for i in range(1, len(positions)):
        dist += torch.norm(positions[i] - positions[i - 1]).item()
    return dist
