"""End-to-end test: the initialization converges on a synthetic trajectory
with GNSS noise sigma = 20 cm."""

import pytest
import torch

from gpsi.data import add_gnss_noise
from gpsi.method import InitConfig, run_initialization
from gpsi.metrics import compute_ate
from gpsi.synthetic import generate_synthetic_sequence

N = 20
SIGMA = 0.20


@pytest.fixture(scope="module")
def result():
    data = generate_synthetic_sequence(n_epochs=N, gnss_rate=10.0, imu_rate=50.0)
    cfg = InitConfig(max_gps_measurements=N, gps_noise_std=SIGMA, min_gps_for_alignment=5,
                     max_epochs=150, log_every=0, seed=42)
    res = run_initialization(data, cfg, device="cpu")

    # Same GNSS realisation as inside run_initialization (seeded RNG).
    torch.manual_seed(cfg.seed)
    import numpy as np
    np.random.seed(cfg.seed)
    np.random.normal(size=6)
    noisy = add_gnss_noise(data, SIGMA)
    gnss_ate = compute_ate(list(noisy["gps_p"][:N]), list(data["gt_p"][:N]))
    return res, gnss_ate


def test_loss_decreases(result):
    res, _ = result
    assert res.loss_history[-1] < 0.05 * res.loss_history[0]
    assert all(torch.isfinite(torch.tensor(res.loss_history)))


def test_position_better_than_raw_gnss(result):
    res, gnss_ate = result
    assert gnss_ate > 0.25  # sanity: the GNSS noise is really ~20 cm per axis
    assert res.metrics["ate_m"] < 0.6 * gnss_ate
    assert res.metrics["ate_m"] < SIGMA


def test_rotation_and_velocity_converge(result):
    res, _ = result
    assert res.metrics["rot_rmse_deg"] < 10.0
    assert res.metrics["rmse_vel_mps"] < 0.5


def test_deterministic():
    data = generate_synthetic_sequence(n_epochs=8, gnss_rate=10.0, imu_rate=50.0)
    cfg = InitConfig(max_gps_measurements=8, max_epochs=5, log_every=0)
    a = run_initialization(data, cfg).metrics
    b = run_initialization(data, cfg).metrics
    assert a["ate_m"] == b["ate_m"] and a["T_wb"] == b["T_wb"]


def test_observability_criterion_runs():
    data = generate_synthetic_sequence(n_epochs=12, gnss_rate=10.0, imu_rate=50.0)
    cfg = InitConfig(max_gps_measurements=12, max_epochs=3, log_every=0, use_obs_criterion=True)
    res = run_initialization(data, cfg)
    assert len(res.criterion_history) == 11
    k = res.metrics["activation_frame"]
    assert k is None or 1 <= k < 12
