"""Experiment protocols of the paper and CSV output.

Modes
-----
``sweep``
    For every sequence, run the initialization activating the GNSS-inertial
    alignment after ``k`` GNSS epochs, for ``k in [1, 5, 10, ..., N-5]``
    (``N = max_gps_measurements``). Produces ``<sequence>.csv`` (error vs
    number of GNSS measurements, Figs. "ATE / rotation error vs #GNSS").
``criterion``
    For every sequence, run the baseline (``k = 1``) and the observability
    criterion (``k`` chosen automatically). Produces
    ``<sequence>-obs-criteria.csv`` and ``<sequence>-criterion-curve.csv``
    (evolution of the singular-value ratio used by the criterion).
``single``
    One run per sequence with the given ``k``.
"""

import csv
import json
import os
from dataclasses import replace

from .data import list_sequences, load_sequence
from .method import InitConfig, run_initialization

RESULT_FIELDS = [
    "dataset", "sequence", "use_twb", "obs_criterion", "min_gps_for_alignment", "activation_frame",
    "gps_noise_std", "max_gps_measurements", "seed", "epochs", "final_loss", "time_s",
    "rmse_pos_m", "ate_m", "rmse_vel_mps", "rot_rmse_rad", "rot_rmse_deg", "distance_m", "T_wb",
]


def sweep_values(max_gps_measurements, step=5):
    """``[1, 5, 10, ...]`` below ``max_gps_measurements`` (as in the original sweep)."""
    return [1] + list(range(step, max_gps_measurements, step))


def _format(value):
    if isinstance(value, float):
        return f"{value:.7f}"
    if isinstance(value, (list, tuple)):
        return json.dumps([round(float(x), 7) for x in value])
    if value is None:
        return ""
    return value


def write_results_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format(v) for k, v in row.items()})


def write_criterion_csv(path, history):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "sigma_ratio", "rel_change"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def run_sequence(dataset, seq_path, seq_name, mode, base_cfg: InitConfig, device="cpu",
                 time_mode=None, repetitions=1, sweep_step=5, out_dir="results"):
    """Run one protocol on one sequence and write its CSV files. Returns the rows."""
    data = load_sequence(dataset, seq_path, device=device, time_mode=time_mode)
    rows = []

    def _run(cfg):
        for r in range(repetitions):
            res = run_initialization(data, replace(cfg, seed=cfg.seed + r), device=device)
            rows.append({"dataset": dataset, "sequence": seq_name, **res.metrics})
            print(f"[{dataset}/{seq_name}] k={res.metrics['min_gps_for_alignment']} "
                  f"criterion={cfg.use_obs_criterion} ATE={res.metrics['ate_m']:.4f} m "
                  f"rot={res.metrics['rot_rmse_rad']:.4f} rad ({res.metrics['time_s']:.1f} s)")
        return res

    out_dir = os.path.join(out_dir, dataset)
    if mode == "sweep":
        for k in sweep_values(base_cfg.max_gps_measurements, sweep_step):
            _run(replace(base_cfg, use_twb=True, use_obs_criterion=False, min_gps_for_alignment=k))
        write_results_csv(os.path.join(out_dir, f"{seq_name}.csv"), rows)
    elif mode == "criterion":
        _run(replace(base_cfg, use_twb=True, use_obs_criterion=False, min_gps_for_alignment=1))
        res = _run(replace(base_cfg, use_twb=True, use_obs_criterion=True))
        write_results_csv(os.path.join(out_dir, f"{seq_name}-obs-criteria.csv"), rows)
        write_criterion_csv(os.path.join(out_dir, f"{seq_name}-criterion-curve.csv"), res.criterion_history)
    elif mode == "single":
        _run(base_cfg)
        write_results_csv(os.path.join(out_dir, f"{seq_name}-single.csv"), rows)
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return rows


def resolve_sequences(dataset, root, sequences=None):
    """Return ``[(name, path), ...]`` for the requested (or all available) sequences."""
    names = sequences or list_sequences(dataset, root)
    out = []
    for name in names:
        path = name if os.path.isdir(name) else os.path.join(root, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Sequence '{name}' not found (looked for '{path}')")
        out.append((os.path.basename(os.path.normpath(path)), path))
    return out
