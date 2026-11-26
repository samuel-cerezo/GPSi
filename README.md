# GPSi: GNSS-Inertial State Initialization by Distance Residuals

Reference implementation of

> S. Cerezo and J. Civera, **"GNSS-inertial State Initialization by Distance Residuals"**
> (v2: *"GNSS-Inertial State Initialization Using Inter-Epoch Baseline Residuals"*), arXiv:2506.11534, 2025.
> [arXiv](https://arxiv.org/abs/2506.11534) · [Project page](https://samuel-cerezo.github.io/gpsi.html)

GPSi initializes a GNSS-inertial system (positions, velocities, orientations,
IMU biases, gravity and the GNSS-inertial alignment `T_wb`) **without vision**.
Global GNSS positions are not used from the start: the optimization first relies
only on *inter-epoch GNSS distance (baseline) residuals*, which are invariant to
the unknown alignment, and switches on the absolute GNSS residuals once `T_wb`
becomes observable. The switching time is chosen with a criterion based on the
evolution of the singular values of the Hessian with respect to `T_wb`.

## Repository structure

```
GPSi/
├── gpsi/                       # Python package
│   ├── data.py                 # EuRoC / GVINS / MARS loaders + simulated GNSS noise
│   ├── imu.py                  # IMU preintegration and preintegrated residual
│   ├── residuals.py            # GNSS distance / anchor residuals, bias random walk
│   ├── gnss.py                 # constant-velocity Kalman smoothing of GNSS positions
│   ├── observability.py        # T_wb activation criterion (Hessian singular values)
│   ├── method.py               # the initialization (InitConfig, run_initialization)
│   ├── metrics.py              # ATE, RMSE, rotation RMSE
│   ├── experiments.py          # paper protocols (sweep / criterion) and CSV output
│   ├── plotting.py             # figures
│   └── synthetic.py            # synthetic trajectories (tests and demos)
├── run_experiments.py          # single entry point: --dataset euroc|gvins|mars
├── scripts/make_paper_figures.py   # figures and tables of the paper from the CSVs
├── tests/                      # pytest suite (runs on CPU)
├── requirements.txt / environment.yml
└── datasets/                   # (not versioned) put the datasets here
```

## Installation

Tested with Python 3.13 on Linux. The code runs on **CPU** when CUDA is not
available (detected with `torch.cuda.is_available()`); a GPU is optional.

With conda:

```bash
conda env create -f environment.yml
conda activate gpsi
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install it first following
<https://pytorch.org/get-started/locally/> and then `pip install -r requirements.txt`.

Check the installation (≈ 40 s on a laptop CPU):

```bash
pytest
```

## Datasets

Place the datasets under `datasets/` (or pass `--data-root`). GNSS positions
are **simulated** from the ground truth by adding isotropic Gaussian noise
(`--gps-noise`, 0.2 m by default), as in the paper.

### EuRoC MAV

Download the sequences in **ASL format** from the official page:
<https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>

```
datasets/EuRoC/
├── MH_01_easy/mav0/{imu0/data.csv, state_groundtruth_estimate0/data.csv, ...}
├── MH_02_easy/mav0/...
└── V2_03_difficult/mav0/...
```

### GVINS

Official dataset: <https://github.com/HKUST-Aerial-Robotics/GVINS-Dataset>
(ROS bags). The loader expects each sequence converted to CSV:

```
datasets/GVINS/<sequence>/          # sports_field, complex_environment, urban_driving
├── gps_utm.csv          # t[ns], x, y, z [, qw, qx, qy, qz [, vx, vy, vz]]   (metric frame, e.g. UTM)
└── mav0/imu0/data.csv   # t[ns], w_x, w_y, w_z, a_x, a_y, a_z                 (EuRoC IMU format)
```

If the ground-truth file has no orientation columns, the rotation error is
reported as `NaN`; missing velocities are computed by finite differences.

### MARS-LVIG

Official dataset: <https://mars.hku.hk/dataset.html> (ROS bags). Same CSV
layout as GVINS, with the ground-truth file named `gnss_gt.csv`:

```
datasets/MARS/<sequence>/{gnss_gt.csv, mav0/imu0/data.csv}
```

Sequences are auto-detected (every sub-folder containing `mav0/`).

> The bag → CSV conversion scripts for GVINS and MARS are not part of this
> release yet; any tool that writes the files above works (e.g. `rosbags`).

## Reproducing the experiments

All experiments are launched with `run_experiments.py`; results are written
as CSV files to `results/<dataset>/`.

| Option | Meaning |
|---|---|
| `--dataset euroc\|gvins\|mars\|synthetic` | dataset |
| `--data-root PATH` | dataset root (default `datasets/EuRoC`, `datasets/GVINS`, `datasets/MARS`) |
| `--sequences ...` | subset of sequences (default: all found) |
| `--mode sweep` | error vs. number of GNSS measurements before activating `T_wb` (k = 1, 5, …, 95) → `<seq>.csv` |
| `--mode criterion` | baseline (k = 1) vs. observability criterion → `<seq>-obs-criteria.csv`, `<seq>-criterion-curve.csv` |
| `--mode single --min-gps K [--obs-criterion] [--no-twb]` | one run |
| `--device auto\|cpu\|cuda` | `auto` = CUDA if available, else CPU |
| `--gps-noise 0.2 --max-gps 100 --seed 42 --repetitions 1` | experiment parameters (paper defaults) |
| `--time-mode legacy\|seconds` | timestamp handling (see *Notes on reproducibility*) |

```bash
# EuRoC
python run_experiments.py --dataset euroc --mode sweep
python run_experiments.py --dataset euroc --mode criterion

# GVINS
python run_experiments.py --dataset gvins --mode sweep
python run_experiments.py --dataset gvins --mode criterion

# MARS
python run_experiments.py --dataset mars --mode sweep
python run_experiments.py --dataset mars --mode criterion

# Single sequence, verbose
python run_experiments.py --dataset euroc --sequences MH_03_medium --mode single --min-gps 20 --log-every 10

# No dataset at hand: synthetic trajectory
python run_experiments.py --dataset synthetic --max-gps 20 --max-epochs 150
```

A full sweep runs 20 optimizations per sequence; expect minutes per run on CPU.

### Figures and tables

```bash
python scripts/make_paper_figures.py --results results --out paper_outputs
```

produces, for every dataset with results:

* `<dataset>-ate-vs-gnss.{pdf,png}` – ATE vs. number of GNSS measurements before `T_wb` optimization,
* `<dataset>-rotation-vs-gnss.{pdf,png}` – rotation RMSE vs. number of GNSS measurements,
* `<dataset>-criterion.{pdf,png}` – observability criterion `log10(Δρ_k)` vs. `k`,
* `<dataset>-table.{md,tex}` – ATE / rotation RMSE with k = 1 vs. the criterion, and the best k of the sweep.

The script also reads CSVs produced by the original research scripts
(Spanish column names).

## Notes on reproducibility

* **Metrics.** `ATE` is the *mean* Euclidean position error (no trajectory
  alignment); `rmse_pos_m`/`rmse_vel_mps` are per-component RMSEs.
  `rot_rmse_rad` is the value the original scripts stored as "RMSE rotación"
  (it is in radians); `rot_rmse_deg` is the same value in degrees.
* **Timestamps.** The original code stored EuRoC nanosecond timestamps as
  `float32`, which collapses consecutive timestamps (resolution ≈ 137 s), so
  IMU intervals are empty. To keep the published numbers, this is the default
  for EuRoC (`--time-mode legacy`). `--time-mode seconds` uses `float64`
  seconds (physically consistent IMU preintegration) and is the default for
  GVINS / MARS.
* **Randomness.** Runs are deterministic for a given `--seed` and device
  (PyTorch and NumPy are seeded). Results on GPU differ slightly from CPU.

## Citation

```bibtex
@article{cerezo2025gpsi,
  title         = {{GNSS}-inertial State Initialization by Distance Residuals},
  author        = {Cerezo, Samuel and Civera, Javier},
  journal       = {arXiv preprint arXiv:2506.11534},
  year          = {2025},
  eprint        = {2506.11534},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.11534}
}
```

## Contact

Developed by [Samuel Cerezo](https://samuel-cerezo.github.io/). Questions,
suggestions and collaborations are welcome.
