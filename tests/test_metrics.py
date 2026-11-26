import math

import pypose as pp
import pytest
import torch

from gpsi.metrics import (compute_ate, compute_rmse, compute_rotation_rmse, compute_rotation_rmse_deg,
                          quat_wxyz_to_SO3, traveled_distance)


def _q_wxyz_about_z(angle):
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def test_rmse_zero_for_identical():
    v = [torch.randn(3) for _ in range(5)]
    assert compute_rmse(v, v) == 0.0


def test_rmse_is_per_component():
    est = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 2.0, 0.0])]
    gt = [torch.zeros(3), torch.zeros(3)]
    # mean over all 6 components of squared errors = (1 + 4) / 6
    assert compute_rmse(est, gt) == pytest.approx(math.sqrt(5.0 / 6.0), rel=1e-6)


def test_ate_is_mean_euclidean_error():
    est = [torch.tensor([3.0, 4.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
    gt = [torch.zeros(3), torch.zeros(3)]
    assert compute_ate(est, gt) == pytest.approx(3.0)


def test_ate_constant_offset():
    gt = [torch.randn(3) for _ in range(10)]
    est = [g + torch.tensor([0.1, 0.0, 0.0]) for g in gt]
    assert compute_ate(est, gt) == pytest.approx(0.1, abs=1e-6)


def test_quaternion_conversion_wxyz():
    R = quat_wxyz_to_SO3(_q_wxyz_about_z(math.pi / 2).unsqueeze(0))
    x = R.Act(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(x, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_rotation_rmse_zero_when_equal():
    qs = [_q_wxyz_about_z(a) for a in (0.0, 0.3, -1.0)]
    states = [{"R": quat_wxyz_to_SO3(q.unsqueeze(0))[0]} for q in qs]
    assert compute_rotation_rmse(states, torch.stack(qs)) == pytest.approx(0.0, abs=1e-3)


def test_rotation_rmse_known_angle():
    angle = 0.2
    states = [{"R": pp.identity_SO3()} for _ in range(4)]
    gt = torch.stack([_q_wxyz_about_z(angle)] * 4)
    assert compute_rotation_rmse(states, gt) == pytest.approx(angle, rel=1e-4)  # radians
    assert compute_rotation_rmse_deg(states, gt) == pytest.approx(math.degrees(angle), rel=1e-4)


def test_rotation_rmse_mixed_angles():
    angles = [0.1, 0.3]
    states = [{"R": pp.identity_SO3()} for _ in angles]
    gt = torch.stack([_q_wxyz_about_z(a) for a in angles])
    expected = math.sqrt(sum(a * a for a in angles) / len(angles))
    assert compute_rotation_rmse(states, gt) == pytest.approx(expected, rel=1e-4)


def test_traveled_distance():
    pts = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 2.0]])
    assert traveled_distance(pts) == pytest.approx(7.0)
