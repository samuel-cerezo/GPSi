# load_data.py
import pandas as pd
import torch
import numpy as np

def load_gvins_data(path, gps_noise_std=0.01, device='cpu'):

    # Load ground truth 
    gt_df = pd.read_csv(f"{path}/gps_utm.csv")
    gt_time64 = torch.tensor(gt_df.iloc[:, 0].values, dtype=torch.float64, device=device)  # ns to s
    gt_time = gt_time64 * 1e-9
    gt_p = torch.tensor(gt_df.iloc[:, 1:4].values, dtype=torch.float32, device=device)  # px, py, pz

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
