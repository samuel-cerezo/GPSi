# load_data.py
import pandas as pd
import torch
import numpy as np

def load_euroc_data(path, gps_noise_std=0.1, device='cpu'):
    """
    Carga datos de EuRoC y simula mediciones de GPS ruidoso.
    
    path: carpeta base donde está 'mav0'
    gps_noise_std: desviación estándar del ruido de GPS (en metros)
    """
    # Load ground truth (position + orientation)
    gt_df = pd.read_csv(f"{path}/mav0/state_groundtruth_estimate0/data.csv")
    gt_time = torch.tensor(gt_df.iloc[:, 0].values, dtype=torch.float32, device=device)  # ns to s
    gt_p = torch.tensor(gt_df.iloc[:, 1:4].values, dtype=torch.float32, device=device)  # px, py, pz
    gt_q = torch.tensor(gt_df.iloc[:, 4:8].values, dtype=torch.float32, device=device)  # qw, qx, qy, qz
    gt_v = torch.tensor(gt_df.iloc[:, 8:11].values, dtype=torch.float32, device=device)  # vx, vy, vz
    gt_bg = torch.tensor(gt_df.iloc[:, 11:14].values, dtype=torch.float32, device=device)  # bg
    gt_ba = torch.tensor(gt_df.iloc[:, 14:17].values, dtype=torch.float32, device=device)  # ba

    # Load IMU data
    imu_df = pd.read_csv(f"{path}/mav0/imu0/data.csv")
    imu_time = torch.tensor(imu_df.iloc[:, 0].values, dtype=torch.float32, device=device)  # ns to s
    acc = torch.tensor(imu_df.iloc[:, 4:7].values, dtype=torch.float32, device=device)
    gyro = torch.tensor(imu_df.iloc[:, 1:4].values, dtype=torch.float32, device=device)

    # Simulate GPS by adding noise to ground-truth positions
    gps_p = gt_p + gps_noise_std * torch.randn_like(gt_p)

    return {
        'gt_time': gt_time,
        'gt_p': gt_p,
        'gps_p': gps_p,
        'imu_time': imu_time,
        'acc': acc,
        'gyro': gyro,
        'gt_q': gt_q,
        'gt_v': gt_v,
        'gt_bg': gt_bg,
        'gt_ba': gt_ba,
    }

def load_gvins_data(path, gps_noise_std=0.1, device='cpu'):
    """
    Carga datos del dataset GVINS desde archivos CSV (exportados de rosbag).
    Devuelve un dict compatible con la salida de load_euroc_data().
    """
    # === GPS como ground-truth aproximado ===
    gps_df = pd.read_csv(f"{path}/gps_utm.csv")
    gps_time64 = torch.tensor(gps_df.iloc[:, 0].values, dtype=torch.float64, device=device)
    gps_time64 = gps_time64 * 1      # ns -> s
    gps_time = gps_time64.to(dtype=torch.float32)
    gps_p = torch.tensor(gps_df.iloc[:, 1:4].values, dtype=torch.float32, device=device)
    gps_v = torch.tensor(gps_df.iloc[:, 4:7].values, dtype=torch.float32, device=device)

    # Shift all positions to make the origin at the first GPS point
    gps_origin = gps_p[0]
    gps_p = gps_p - gps_origin
    gps_p_noisy = gps_p + gps_noise_std * torch.randn_like(gps_p)

    # === Cargar IMU ===
    imu_df = pd.read_csv(f"{path}/imu.csv")
    imu_time64 = torch.tensor(imu_df.iloc[:, 0].values, dtype=torch.float64, device=device)
    imu_time64 = imu_time64 * 1      # ns -> s
    imu_time = imu_time64.to(dtype=torch.float32)
    gyro = torch.tensor(imu_df.iloc[:, 1:4].values, dtype=torch.float32, device=device)
    acc = torch.tensor(imu_df.iloc[:, 4:7].values, dtype=torch.float32, device=device)

    # === Biases constantes ===
    N = gps_time.shape[0]
    gt_bg = torch.full((N, 3), 0.001, dtype=torch.float32, device=device)
    gt_ba = torch.full((N, 3), 0.001, dtype=torch.float32, device=device)

    # === Cuaterniones identidad ===
    gt_q = torch.zeros((N, 4), dtype=torch.float32, device=device)
    gt_q[:, 0] = 1.0

    return {
        'gt_time': gps_time,
        'gt_p': gps_p,
        'gt_v': gps_v,
        'gt_q': gt_q,
        'gt_bg': gt_bg,
        'gt_ba': gt_ba,
        'gps_p': gps_p_noisy,
        'imu_time': imu_time,
        'gyro': gyro,
        'acc': acc,
    }