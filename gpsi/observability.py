"""Observability criterion for activating the GNSS-inertial alignment T_wb.

For every epoch ``k`` the Hessian of the GNSS residuals w.r.t. the 6-DoF
alignment ``T_wb`` is accumulated over a sliding window, and the ratio
``rho_k = sigma_min / sigma_max`` of its singular values is tracked. The
alignment is activated at the first epoch where the relative change
``|rho_k - rho_{k-1}| / rho_{k-1}`` drops below a threshold (1e-2).
"""

import pypose as pp
import torch
from torch.autograd.functional import jacobian


def pack_Twb(T_w_b):
    """Pack ``{'R': SO3, 't': (3,)}`` into a 6-vector ``[log(R), t]``."""
    return torch.cat([T_w_b["R"].Log().squeeze(0), T_w_b["t"]])


def compute_Twb_hessian(est_states, gps_measurements, T_w_b_init, window_size=10, sigma_gps=0.5):
    """Gauss-Newton Hessian of the GNSS residuals w.r.t. ``T_wb``.

    Returns:
        ``(H, sigma_ratio, singular_values)``.
    """
    device = est_states[0]["p"].device
    H = torch.zeros((6, 6), device=device)
    start = max(0, len(est_states) - window_size)

    for k in range(start, len(est_states) - 1):
        def residual_fun(Twb_vec, k=k):
            rotvec = Twb_vec[0:3].reshape(1, 3)
            t = Twb_vec[3:6]
            R = pp.Exp(pp.LieTensor(rotvec, ltype=pp.so3_type))

            r_anchor = R.Act(est_states[k]["p"]) + t - gps_measurements[k]
            p_i_world = R.Act(est_states[k]["p"]) + t
            p_j_world = R.Act(est_states[k + 1]["p"]) + t
            r_disp = (p_j_world - p_i_world) - (gps_measurements[k + 1] - gps_measurements[k])
            return torch.cat([r_disp, r_anchor])

        Twb_vec = pack_Twb(T_w_b_init).detach().clone().requires_grad_(True)
        J = jacobian(residual_fun, Twb_vec).view(-1, 6)
        W = (1.0 / sigma_gps ** 2) * torch.eye(J.shape[0], device=device)
        H += J.T @ (W @ J)

    if torch.count_nonzero(H) > 0:
        sigmas = torch.linalg.svdvals(H)
        nonzero = sigmas[sigmas > 1e-14]
        sigma_min = nonzero.min().item() if len(nonzero) > 0 else 0.0
        sigma_max = sigmas.max().item()
        ratio = sigma_min / sigma_max if sigma_max > 0 else 0.0
        return H, ratio, sigmas

    return H, 0.0, torch.zeros(6, device=device)


def find_Twb_activation(est_states, gps_measurements, T_w_b_init, sigma_gps=0.5,
                        window_size=10, rel_threshold=1e-2, verbose=False):
    """Find the epoch at which ``T_wb`` becomes observable.

    Returns:
        ``(activation_frame, history)`` where ``activation_frame`` is ``None``
        if the criterion is never met and ``history`` is a list of dicts with
        keys ``k``, ``sigma_ratio`` and ``rel_change`` (NaN for the first epoch).
    """
    sigma_ratios = []
    history = []
    activation = None

    for k in range(1, len(est_states)):
        _, ratio, _ = compute_Twb_hessian(est_states[:k + 1], gps_measurements[:k + 1], T_w_b_init,
                                          window_size=window_size, sigma_gps=sigma_gps)
        sigma_ratios.append(ratio)
        if verbose:
            print(f"[T_wb criterion] k={k} | sigma_min/sigma_max = {ratio:.4e}")

        if len(sigma_ratios) < 2:
            rel = float("nan")
        else:
            prev, curr = sigma_ratios[-2], sigma_ratios[-1]
            rel = abs((curr - prev) / max(prev, 1e-12))
            if activation is None and rel < rel_threshold:
                activation = k
                if verbose:
                    print(f"[T_wb criterion] activation suggested at k={k} (ratio={ratio:.4e})")
        history.append({"k": k, "sigma_ratio": ratio, "rel_change": rel})

    return activation, history
