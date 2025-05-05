import torch
import pypose as pp
import matplotlib.pyplot as plt

def preintegrate(imu_data, bias_g, bias_a, dt_list, gravity):
    dR = pp.identity_SO3(device=bias_g.device)
    dv = torch.zeros(3, device=bias_g.device)
    dp = torch.zeros(3, device=bias_g.device)
    v = torch.zeros(3, device=bias_g.device)

    for k in range(len(dt_list)):
        dt = dt_list[k]
        omega = imu_data['gyro'][k] - bias_g
        accel = imu_data['acc'][k] - bias_a

        dR_k = pp.so3(omega * dt).Exp()
        dR = dR @ dR_k

        a_world = dR @ accel + gravity
        dv += a_world * dt
        dp += v * dt + 0.5 * a_world * dt**2
        v += a_world * dt

    return {'dR': dR, 'dv': dv, 'dp': dp}

device = 'cpu'
dt = 0.01  # 10ms
N = 100    # 1 segundo total
g = torch.tensor([0, 0, -9.81], device=device)

# Simular IMU: rotación constante y aceleración constante
gyro_true = torch.tensor([0.0, 0.0, 1.0], device=device)  # rad/s (rotando en Z)
accel_true = torch.tensor([0.5, 0.0, 0.0], device=device) # m/s²

gyro_data = gyro_true.unsqueeze(0).repeat(N, 1)
accel_data = accel_true.unsqueeze(0).repeat(N, 1)

dt_list = [dt] * N


imu_data = {
    'gyro': gyro_data,
    'acc': accel_data
}

bias_g = torch.zeros(3, device=device)
bias_a = torch.zeros(3, device=device)

delta = preintegrate(imu_data, bias_g, bias_a, dt_list, g)

# Ground truth (fórmulas físicas)
T = N * dt
v_gt = accel_true * T + g * T
p_gt = 0.5 * accel_true * T**2 + 0.5 * g * T**2
angle_gt = torch.norm(gyro_true) * T  # radianes de giro

print("\n=== COMPARACIÓN ===")
print("Δv estimado :", delta['dv'])
print("Δv ground truth:", v_gt)

print("Δp estimado :", delta['dp'])
print("Δp ground truth:", p_gt)

print("Ángulo estimado (Z):", delta['dR'].Log())
print("Ángulo ground truth :", torch.tensor([0, 0, angle_gt]))
