import pypose as pp
import pytest
import torch

from gpsi.data import add_gnss_noise, load_sequence
from gpsi.device import get_device
from gpsi.gnss import smooth_gps
from gpsi.imu import get_imu_between
from gpsi.experiments import sweep_values
from gpsi.synthetic import generate_synthetic_sequence


def test_device_falls_back_to_cpu():
    dev = get_device()
    assert dev == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        assert get_device("cuda") == "cpu"


def test_sweep_values_match_original_protocol():
    assert sweep_values(100) == [1] + list(range(5, 100, 5))


def test_gnss_noise_statistics():
    data = generate_synthetic_sequence(n_epochs=2000, gnss_rate=100.0, imu_rate=100.0)
    torch.manual_seed(0)
    noisy = add_gnss_noise(data, 0.2)
    err = noisy["gps_p"] - data["gt_p"]
    assert err.std().item() == pytest.approx(0.2, rel=0.05)
    assert "gps_p" not in data  # input not modified


def test_kalman_smoothing_reduces_noise():
    t = torch.linspace(0, 5, 200)
    gt = torch.stack([t, 0.5 * t, torch.zeros_like(t)], 1)
    torch.manual_seed(1)
    meas = gt + 0.2 * torch.randn_like(gt)
    sm = torch.stack(smooth_gps(meas, 0.2, dt=float(t[1] - t[0])))
    assert (sm[50:] - gt[50:]).norm(dim=1).mean() < (meas[50:] - gt[50:]).norm(dim=1).mean()


def test_get_imu_between_selects_half_open_interval():
    ts = torch.arange(10, dtype=torch.float64)
    imu = {"acc": torch.zeros(10, 3), "gyro": torch.zeros(10, 3)}
    out = get_imu_between(2.0, 5.0, imu, ts, torch.zeros(3), torch.zeros(3))
    assert out["acc"].shape[0] == 3
    assert out["dt"] == [1.0, 1.0]


def test_euroc_loader_roundtrip(tmp_path):
    """Write a synthetic sequence in EuRoC ASL format and load it back."""
    d = generate_synthetic_sequence(n_epochs=10)
    seq = tmp_path / "MH_01_easy" / "mav0"
    (seq / "state_groundtruth_estimate0").mkdir(parents=True)
    (seq / "imu0").mkdir(parents=True)
    t0 = 1403636579758555392
    with open(seq / "state_groundtruth_estimate0" / "data.csv", "w") as f:
        f.write("#t,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bwx,bwy,bwz,bax,bay,baz\n")
        for i in range(10):
            vals = list(d["gt_p"][i].tolist()) + d["gt_q"][i].tolist() + d["gt_v"][i].tolist() + [0.0] * 6
            f.write(f"{t0 + int(round(d['gt_time'][i].item() * 1e9))}," + ",".join(map(str, vals)) + "\n")
    with open(seq / "imu0" / "data.csv", "w") as f:
        f.write("#t,wx,wy,wz,ax,ay,az\n")
        for i in range(len(d["imu_time"])):
            vals = d["gyro"][i].tolist() + d["acc"][i].tolist()
            f.write(f"{t0 + int(round(d['imu_time'][i].item() * 1e9))}," + ",".join(map(str, vals)) + "\n")

    legacy = load_sequence("euroc", str(tmp_path / "MH_01_easy"))
    assert legacy["gt_time"].dtype == torch.float32  # original behaviour
    sec = load_sequence("euroc", str(tmp_path / "MH_01_easy"), time_mode="seconds")
    assert sec["gt_time"].dtype == torch.float64
    assert torch.allclose(sec["gt_time"] - sec["gt_time"][0], d["gt_time"], atol=1e-6)
    assert torch.allclose(sec["gt_p"], d["gt_p"], atol=1e-5)
    assert torch.allclose(sec["acc"], d["acc"], atol=1e-5)


def test_synthetic_imu_consistent_with_trajectory():
    """Integrating the synthetic IMU must reproduce the ground-truth velocity."""
    d = generate_synthetic_sequence(n_epochs=11, gnss_rate=10.0, imu_rate=1000.0)
    g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
    t = d["imu_time"]
    v = d["gt_v"][0].double().clone()
    R = pp.identity_SO3(dtype=torch.float64)
    for k in range(len(t) - 1):
        dt = (t[k + 1] - t[k]).item()
        v = v + (R.Act(d["acc"][k].double()) + g) * dt
        R = R @ pp.so3(d["gyro"][k].double() * dt).Exp()
        if k + 1 == 1000:
            break
    assert torch.allclose(v.float(), d["gt_v"][10], atol=2e-2)
