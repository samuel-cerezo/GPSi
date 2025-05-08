import torch
import pypose as pp
from models import *
from residuals import *
from load_data import *
from plot_trajectories import *
from compute_errors import *
from plot import *
import matplotlib.pyplot as plt
import time
import numpy as np

# ========= CONFIGURACIÓN =========
MAX_GPS_MEASUREMENTS = 100
USE_Twb = True
MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT = 20
gps_noise_std = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ========= PARÁMETROS DEL SENSOR IMU =========
gyro_bias_std = 1.9393e-5 * np.sqrt(200)  # 200 Hz
accel_bias_std = 3.0e-3 * np.sqrt(200)    # 200 Hz

bias_g_init = torch.tensor(np.random.normal(0, gyro_bias_std, 3), dtype=torch.float32, device=device)
bias_a_init = torch.tensor(np.random.normal(0, accel_bias_std, 3), dtype=torch.float32, device=device)

# ========= CARGA DE DATOS =========
data = load_euroc_data('/Users/samucerezo/dev/src/repos/GPSi/datasets', gps_noise_std, device=device)

# ========= ESTADO INICIAL =========
state = {
    'p': torch.zeros(3, device=device),
    'v': torch.zeros(3, device=device),
    'R': pp.identity_SO3(device=device),
    'bias_g': bias_g_init,
    'bias_a': bias_a_init,
    'g': torch.tensor([0, 0, -9.81], device=device)
}

states = [state]
timestamps = [data['gt_time'][0]]
gps_measurements = [data['gps_p'][0]]
gt_velocities = [data['gt_v'][0]]

# ========= PROPAGACIÓN =========
for k in range(MAX_GPS_MEASUREMENTS - 1):
    t_start = data['gt_time'][k]
    t_end = data['gt_time'][k + 1]

    imu_ij = get_imu_between(t_start, t_end, data, data['imu_time'], state['bias_g'], state['bias_a'])
    delta = preintegrate(imu_ij, state['bias_g'], state['bias_a'], imu_ij['dt'], gravity=state['g'])
    state = propagate_preintegrated(state, delta)

    states.append(state)
    timestamps.append(t_end)
    gps_measurements.append(data['gps_p'][k + 1])
    gt_velocities.append(data['gt_v'][k + 1])

# ========= ESTADOS ESTIMADOS =========
est_states = []
for i in range(len(states)):

    # Reordenar el quaternion [w, x, y, z] → [x, y, z, w]
    q_xyzw = torch.cat([data['gt_q'][i, 1:], data['gt_q'][i, 0:1]], dim=0)  # [x, y, z, w]

    est_states.append({
        'p': (gps_measurements[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
        'v': (gt_velocities[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
        'R': states[i]['R'].detach().clone().requires_grad_(), #'R': pp.SO3(q_xyzw.unsqueeze(0)).detach().clone().requires_grad_(),
        'bias_g': (states[i]['bias_g'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
        'bias_a': (states[i]['bias_a'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
        'g': (states[i]['g'] + 0.001 * torch.randn(3, device=device)).requires_grad_()
    })


# ========= Inicializacion TRANSFORMACIÓN GLOBAL T_w_b =========
R_init = torch.tensor([
    [ 9.8746e-01,  1.5601e-01,  2.4348e-02],
    [-1.5605e-01,  9.8775e-01, -4.6277e-04],
    [-2.4122e-02, -3.3424e-03,  9.9970e-01]
], dtype=torch.float32, device=device)

t_init = torch.tensor([0.306, 0.701, 0.108], dtype=torch.float32, device=device)

R_init2 = torch.tensor([
    [ 0.987487,  0.155897,  0.023802],
    [-0.156059,  0.987734,  0.005131],
    [-0.022710, -0.008782,  0.999704]
], dtype=torch.float32, device=device)

t_init2 = torch.tensor([-0.2827, -0.7223, -0.0927], dtype=torch.float32, device=device)

T_w_b = {
    'R': pp.mat2SO3(R_init.unsqueeze(0)).detach().clone().requires_grad_(),
    't': t_init.detach().clone().requires_grad_()
}

# ========= TRANSFORMACIÓN GLOBAL T_w_b =========
T_w_b = {
    'R': pp.identity_SO3(device=device).detach().clone().requires_grad_(),
    't': torch.zeros(3, device=device, requires_grad=True)
}


# ========= AGREGAR A PARÁMETROS =========
params = [p['p'] for p in est_states] + [p['v'] for p in est_states] + [p['R'] for p in est_states] + \
         [p['bias_g'] for p in est_states] + [p['bias_a'] for p in est_states] + [p['g'] for p in est_states] + \
         [T_w_b['R'], T_w_b['t']]

optimizer = torch.optim.Adam(params, lr=1e-2)

dt_gps = (data['gt_time'][1] - data['gt_time'][0]).item()
smoothed_gps = kalman_filter_gps(gps_measurements, gps_noise_std, dt_gps)

gps_measurements = torch.stack(gps_measurements)

bias_g_history = []
bias_a_history = []
g_history = []

# ========= EARLY STOPPING =========
tolerance = 1e-3
patience = 10
best_loss = float('inf')
epochs_no_improve = 0

start_time = time.time()

for epoch in range(1000):
    optimizer.zero_grad()
    loss = 0.0
    alpha_gps = 1

    smoothed_gps = kalman_filter_gps(gps_measurements, gps_noise_std, dt_gps)
    
    for k in range(len(states) - 1):
        i, j = k, k + 1
        imu_ij = get_imu_between(timestamps[i], timestamps[j], data, data['imu_time'],
                                 est_states[i]['bias_g'], est_states[i]['bias_a'])
        delta = preintegrate(imu_ij, est_states[i]['bias_g'], est_states[i]['bias_a'],
                             imu_ij['dt'], gravity=est_states[i]['g'])  #Estos deltas estan en coordenadas de mundo
        dt = dt_gps
        r_pre = imu_residual_preint(est_states[i], est_states[j], delta, dt)
        loss += 100 * (r_pre ** 2).mean()

    for k in range(len(est_states) - 1):

        r_vel = velocity_smoothness_residual(est_states[k], est_states[k + 1])
        r_bias_g = biasGYR_residual(est_states[k], est_states[k + 1])
        r_bias_a = biasACC_residual(est_states[k], est_states[k + 1])


        if (k >= MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT) and (USE_Twb):
            p_i_world = T_w_b['R'].Act(est_states[k]['p']) + T_w_b['t']
            p_j_world = T_w_b['R'].Act(est_states[k + 1]['p']) + T_w_b['t']
            delta_est = p_j_world - p_i_world
            delta_gps = smoothed_gps[k + 1] - smoothed_gps[k]
            r_gps_disp = delta_est - delta_gps
            r_gps_anchor = T_w_b['R'].Act(est_states[k]['p']) + T_w_b['t'] - smoothed_gps[k]
            loss += alpha_gps * ((r_gps_disp ** 2).mean() +  (r_gps_anchor ** 2).mean())
        else:
            delta_est = est_states[k + 1]['p'] - est_states[k]['p']
            delta_gps = smoothed_gps[k + 1] - smoothed_gps[k]
            r_gps_disp = delta_est - delta_gps
            loss += alpha_gps * ( (r_gps_disp ** 2).mean() )
        
        #r_gps_anchor = est_states[k]['p'] - smoothed_gps[k]
        #loss += alpha_gps * ((r_gps_disp ** 2).mean() + (r_gps_anchor ** 2).mean())
 
        #loss += 0.1 * (r_vel ** 2).mean()
        loss += 1 * (r_bias_g ** 2).mean()
        loss += 1 * (r_bias_a ** 2).mean()

    loss.backward()
    optimizer.step()

    bias_g_history.append(est_states[0]['bias_g'].detach().cpu().numpy())
    bias_a_history.append(est_states[0]['bias_a'].detach().cpu().numpy())
    g_history.append(est_states[0]['g'].detach().cpu().numpy())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
        print(f"  -  IMU preint.: {r_pre.norm().item():.6f}")
        print(f"  -  Vel smooth: {r_vel.norm().item():.6f}")
        print(f"  -  Bias gyro:  {r_bias_g.norm().item():.6f}")
        print(f"  -  Bias accel: {r_bias_a.norm().item():.6f}")
        print(f"  -  GPS disp.:  {r_gps_disp.norm().item():.6f}")
        print(f"  -  GPS anchor: {r_gps_anchor.norm().item():.6f}")
    

    # Early stopping
    if abs(loss.item() - best_loss) < tolerance:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping en la época {epoch} con loss {loss.item():.6f}")
            break
    else:
        best_loss = loss.item()
        epochs_no_improve = 0

# ========= RESULTADOS =========
plot_results(est_states, gps_measurements, data, MAX_GPS_MEASUREMENTS)

est_positions = [s['p'].detach().cpu() for s in est_states]
gt_positions = [data['gt_p'][i].detach().cpu() for i in range(len(est_states))]

est_velocities = [s['v'].detach().cpu() for s in est_states]
gt_velocities = [v.detach().cpu() for v in gt_velocities]

rmse_pos = compute_rmse(est_positions, gt_positions)
rmse_vel = compute_rmse(est_velocities, gt_velocities)
ate = compute_ate(est_positions, gt_positions)
elapsed = time.time() - start_time
rot_rmse = compute_rotation_rmse(est_states, data['gt_q'][:len(est_states)])

print(f"✅ Tiempo total de ejecución: {elapsed:.6f} sec.")
print(f"✅ RMSE posición: {rmse_pos:.6f} m")
print(f"✅ ATE: {ate:.6f} m")
print(f"✅ RMSE velocidad: {rmse_vel:.6f} m/s")
print(f"✅ RMSE rotación: {rot_rmse:.6f} deg")

# ========= IMPRIMIR TRANSFORMACIÓN RÍGIDA FINAL =========
R_final = T_w_b['R'].matrix().detach().cpu().numpy()
t_final = T_w_b['t'].detach().cpu().numpy()

print("\nTransformación rígida T_w_b final:")
print("Rotación (matriz 3x3):")
print(R_final)
print("Traslación (vector 3x1):")
print(t_final)
