# GPSi: GNSS-Initialization

**GPSi** is a system for tightly coupled **GNSS-Inertial** initialization that leverages inertial preintegration and GPS measurements to estimate the initial velocity, gravity, sensor biases, and extrinsic transformation between the IMU and the global (GNSS) reference frame. The method is designed and implemented in Python.

🔗 Project page: [samuel-cerezo.github.io/gpsi](https://samuel-cerezo.github.io/gpsi.html)

## 🚀 Features

- Closed-form initialization based on GNSS and IMU only.
- Observability-based strategy to determine when to trigger GPS alignment.
- Accurate estimation of:
  - IMU velocity and gravity
  - Accelerometer and gyroscope biases
  - Extrinsic transformation \( T_{WB} \) from body to world
- Tested on **EuRoC** and **GVINS** datasets.

## 🧠 Motivation

In GNSS-denied or degraded environments, it's critical to initialize a tightly coupled navigation system using the minimal set of sensors available. GPSi aims to provide a fast and reliable state estimate without relying on visual features, enabling safe operation and global localization even before visual odometry is available or reliable.

## 🗂️ Repository Structure

```
GPSi/
│
├── scripts/               # Python experiments and optimizer
├── include/              # C++ headers for IMU preintegration and residuals
├── src/                  # C++ implementation of closed-form solver
├── datasets/             # EuRoC/GVINS loaders
├── results/              # Evaluation results and figures
├── models.py             # PyTorch-based state representation
├── residuals.py          # Residuals for optimization (IMU, GPS, etc.)
├── plot_trajectories.py  # Trajectory visualization utilities
├── compute_errors.py     # RMSE / ATE computation
├── CMakeLists.txt        # Build config for C++ module
└── README.md
```

## 📦 Dependencies

**Python**
- Python ≥ 3.8
- `torch`
- `pypose`
- `numpy`, `matplotlib`, `scipy`, `pandas`


Install Python requirements:
```bash
pip install -r requirements.txt
```

## 📊 Datasets

Supported datasets:
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [GVINS Dataset](https://github.com/HKUST-Aerial-Robotics/GVINS)


## ⚙️ Running the Initialization

To run the initialization:
```bash
python scripts/main.py
```


## 📈 Results

The system is evaluated on the EuRoC and GVINS datasets, showing:
- RMSE of position and velocity
- Rotation RMSE (degrees)
- Accuracy of the estimated extrinsic transformation

A sample output:

```
✅ Time: 0.143 sec
✅ RMSE position: 0.053 m
✅ ATE: 0.048 m
✅ RMSE velocity: 0.027 m/s
✅ RMSE rotation: 0.45 deg
```

## 📝 Citation

If you use GPSi in your research, please cite the corresponding paper (coming soon).

## 📬 Contact

Developed by [Samuel Cerezo](https://samuel-cerezo.github.io/)

For questions, suggestions, or collaborations, feel free to reach out!
