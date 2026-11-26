"""Plotting utilities: per-run diagnostics and the figures of the paper.

All functions save to files (no interactive window is required); pass
``show=True`` to also open a window.
"""

import os

import matplotlib

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Fixed categorical order (never cycled); at most 8 series per axes.
SERIES_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
}

# Column names of the CSVs written by the original scripts -> current names.
LEGACY_COLUMNS = {
    "USE_Twb": "use_twb",
    "Obs-criteria": "obs_criterion",
    "MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT": "min_gps_for_alignment",
    "gps_noise_std": "gps_noise_std",
    "MAX_GPS_MEASUREMENTS": "max_gps_measurements",
    "Tiempo total de ejecución": "time_s",
    "RMSE posición": "rmse_pos_m",
    "ATE": "ate_m",
    "RMSE velocidad": "rmse_vel_mps",
    "RMSE rotación": "rot_rmse_rad",
    "Distancia recorrida": "distance_m",
    "Transformación rígida (flattened)": "T_wb",
}


def _save(fig, path, show=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def read_results(path):
    """Read a results CSV (current or original column names)."""
    df = pd.read_csv(path).rename(columns=LEGACY_COLUMNS)
    for col in ("ate_m", "rmse_pos_m", "rmse_vel_mps", "rot_rmse_rad", "time_s", "min_gps_for_alignment"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rot_rmse_deg" not in df and "rot_rmse_rad" in df:
        df["rot_rmse_deg"] = np.degrees(df["rot_rmse_rad"])
    for col in ("use_twb", "obs_criterion"):
        if col in df and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


# --------------------------------------------------------------------------- #
# Paper figures
# --------------------------------------------------------------------------- #

def plot_error_vs_num_gnss(curves, metric, ylabel, path, groups=None, show=False):
    """Error vs. number of GNSS measurements before activating ``T_wb``.

    Args:
        curves: dict ``{sequence_name: DataFrame}`` from sweep CSVs.
        metric: column to plot (e.g. ``"ate_m"``, ``"rot_rmse_deg"``).
        groups: optional list of ``(title, [sequence names])`` -> one panel each
            (used to keep at most 8 series per panel, e.g. EuRoC MH / V).
    """
    names = sorted(curves)
    if groups is None:
        groups = [(None, names[i:i + 8]) for i in range(0, len(names), 8)] or [(None, [])]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.2), sharey=True, squeeze=False)
        for ax, (title, members) in zip(axes[0], groups):
            for slot, name in enumerate(members):
                df = curves[name]
                df = df[df["use_twb"]] if "use_twb" in df else df
                agg = df.groupby("min_gps_for_alignment")[metric].mean().reset_index()
                ax.plot(agg["min_gps_for_alignment"], agg[metric], color=SERIES_COLORS[slot],
                        marker=SERIES_MARKERS[slot], label=name)
            ax.set_xlabel("# GNSS measurements before $T_{wb}$ optimization")
            if title:
                ax.set_title(title)
            if members:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
        axes[0][0].set_ylabel(ylabel)
        fig.tight_layout()
        _save(fig, path, show)


def smooth_criterion(rel_changes):
    """Smoothing used in the paper figure: average of k and k-1 from the 3rd value on."""
    rel = list(rel_changes)
    out = rel.copy()
    for i in range(2, len(rel)):
        out[i] = (rel[i] + rel[i - 1]) / 2.0
    return np.asarray(out, dtype=float)


def plot_criterion_curve(curves, path, threshold=1e-2, max_k=20, activation=None, show=False):
    """Relative change of the singular-value ratio vs. GNSS index (log10).

    Args:
        curves: dict ``{sequence_name: DataFrame(k, sigma_ratio, rel_change)}``.
        activation: optional dict ``{sequence_name: k}`` to mark the activation epoch.
    """
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for slot, name in enumerate(sorted(curves)[:8]):
            df = curves[name]
            df = df[df["k"] <= max_k]
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log10(smooth_criterion(df["rel_change"].values))
            ax.plot(df["k"], y, color=SERIES_COLORS[slot], marker=SERIES_MARKERS[slot], label=name)
            if activation and activation.get(name) is not None:
                ax.axvline(activation[name], color=SERIES_COLORS[slot], linewidth=1, linestyle=":")
        ax.axhline(np.log10(threshold), color="#555555", linestyle="--", linewidth=1.5,
                   label=r"threshold $\Delta\rho_{th}$")
        ax.set_xlabel("GNSS measurement index $k$")
        ax.set_ylabel(r"$\log_{10}(\Delta\rho_k)$")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        _save(fig, path, show)


# --------------------------------------------------------------------------- #
# Per-run diagnostics
# --------------------------------------------------------------------------- #

def plot_run(est_states, gps_measurements, data, out_prefix, show=False):
    """Trajectory, velocity, gyroscope bias and gravity of one run."""
    n = len(est_states)
    est_p = torch.stack([s["p"].detach() for s in est_states]).cpu().numpy()
    gt_p = data["gt_p"][:n].detach().cpu().numpy()
    gps_p = torch.as_tensor(gps_measurements)[:n].detach().cpu().numpy() if gps_measurements is not None else None

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*gt_p.T, color=SERIES_COLORS[0], label="ground truth")
        ax.plot(*est_p.T, color=SERIES_COLORS[1], label="estimate")
        if gps_p is not None:
            ax.scatter(*gps_p.T, color=SERIES_COLORS[2], s=8, label="GNSS")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend(frameon=False)
        _save(fig, f"{out_prefix}-trajectory.png", show)

        series = {
            "velocity [m/s]": (torch.stack([s["v"].detach().cpu() for s in est_states]).numpy(),
                               data["gt_v"][:n].detach().cpu().numpy()),
            "gyro bias [rad/s]": (torch.stack([s["bias_g"].detach().cpu() for s in est_states]).numpy(), None),
            "gravity [m/s$^2$]": (torch.stack([s["g"].detach().cpu() for s in est_states]).numpy(), None),
        }
        for (label, (est, gt)), tag in zip(series.items(), ["velocity", "gyro-bias", "gravity"]):
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
            for a, ax in enumerate(axes):
                if gt is not None:
                    ax.plot(gt[:, a], color=SERIES_COLORS[0], label="ground truth")
                ax.plot(est[:, a], color=SERIES_COLORS[1], label="estimate")
                ax.set_title("xyz"[a])
                ax.set_xlabel("GNSS index")
            axes[0].set_ylabel(label)
            axes[0].legend(frameon=False)
            fig.tight_layout()
            _save(fig, f"{out_prefix}-{tag}.png", show)
