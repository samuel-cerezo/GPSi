import torch
import pypose as pp
from models import *
from residuals import *
from load_data import *
from plot_trajectories import *
from compute_errors import *
from old_scripts.plot import *
import matplotlib.pyplot as plt
import time
import numpy as np

# ========= CONFIGURATION =========
MAX_GPS_MEASUREMENTS = 100  # Maximum number of GPS measurements to use
USE_Twb = True  # Whether to estimate the extrinsic transformation T_w_b
MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT = 20  # Wait until this many GPS points before applying T_w_b
gps_noise_std = 0.1  # Standard deviation of GPS noise

# Set computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ========= IMU NOISE PARAMETERS =========
gyro_bias_std = 1.9393e-5 * np.sqrt(200)  # Gyroscope bias noise std (200 Hz)
accel_bias_std = 3.0e-3 * np.sqrt(200)    # Accelerometer bias noise std (200 Hz)

# Random initialization of IMU biases
bias_g_init = torch.tensor(np.random.normal(0, gyro_bias_std, 3), dtype=torch.float32, device=device)
bias_a_init = torch.tensor(np.random.normal(0, accel_bias_std, 3), dtype=torch.float32, device=device)

# ========= LOAD DATASET =========
data = load_euroc_data('/Users/samucerezo/dev/src/repos/GPSi/datasets', gps_noise_std, device=device)

# ========= INITIAL STATE =========
# Initial pose, velocity, biases, and gravity
state = {
    'p': torch.zeros(3, device=device),  # Position
    'v': torch.zeros(3, device=device),  # Velocity
    'R': pp.identity_SO3(device=device),  # Orientation (identity)
    'bias_g': bias_g_init,
    'bias_a': bias_a_init,
    'g': torch.tensor([0, 0, -9.81], device=device)  # Gravity vector
}

# Lists to store state trajectory
states = [state]
timestamps = [data['gt_time'][0]]
gps_measurements = [data['gps_p'][0]]
gt_velocities = [data['gt_v'][0]]

# ========= PROPAGATE STATES =========
for k in range(MAX_GPS_MEASUREMENTS - 1):
    t_start = data['gt_time'][k]
    t_end = data['gt_time'][k + 1]

    # Get IMU data between timestamps
    imu_ij = get_imu_between(t_start, t_end, data, data['imu_time'], state['bias_g'], state['bias_a'])

    # Preintegrate IMU data to get relative motion
    delta = preintegrate(imu_ij, state['bias_g'], state['bias_a'], imu_ij['dt'], gravity=state['g'])

    # Propagate state using the preintegration result
    state = propagate_preintegrated(state, delta)

    # Save propagated state and measurements
    states.append(state)
    timestamps.append(t_end)
    gps_measurements.append(data['gps_p'][k + 1])
    gt_velocities.append(data['gt_v'][k + 1])

# ========= ESTIMATED STATES =========
est_states = []
for i in range(len(states)):
    q_xyzw = torch.cat([data['gt_q'][i, 1:], data['gt_q'][i, 0:1]], dim=0)  # Convert quaternion format

    # Create estimated state with small noise added, set as differentiable
    est_states.append({
        'p': (gps_measurements[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
        'v': (gt_velocities[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
        'R': states[i]['R'].detach().clone().requires_grad_(),
        'bias_g': (states[i]['bias_g'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
        'bias_a': (states[i]['bias_a'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
        'g': (states[i]['g'] + 0.001 * torch.randn(3, device=device)).requires_grad_()
    })

# ========= INITIAL EXTRINSIC TRANSFORMATION T_w_b =========
# You can uncomment the following to set a fixed initial guess
# Here, we start with identity
T_w_b = {
    'R': pp.identity_SO3(device=device).detach().clone().requires_grad_(),
    't': torch.zeros(3, device=device, requires_grad=True)
}

# ========= PARAMETERS TO OPTIMIZE =========
params = [p['p'] for p in est_states] + [p['v'] for p in est_states] + [p['R'] for p in est_states] + \
         [p['bias_g'] for p in est_states] + [p['bias_a'] for p in est_states] + [p['g'] for p in est_states] + \
         [T_w_b['R'], T_w_b['t']]

optimizer = torch.optim.Adam(params, lr=1e-2)

# Kalman filter for GPS smoothing
dt_gps = (data['gt_time'][1] - data['gt_time'][0]).item()
gps_cleaned = kalman_filter_gps(gps_measurements, gps_noise_std, dt_gps)
gps_measurements = torch.stack(gps_measurements)

# Tracking history of estimated quantities
bias_g_history = []
bias_a_history = []
g_history = []

# ========= EARLY STOPPING CRITERIA =========
tolerance = 1e-3
patience = 10
best_loss = float('inf')
epochs_no_improve = 0

start_time = time.time()

# ========= OPTIMIZATION LOOP =========
for epoch in range(1000):
    optimizer.zero_grad()
    loss = 0.0
    alpha_gps = 1
    gps_cleaned = smooth_gps(gps_measurements, gps_noise_std, dt_gps)

    for k in range(len(states) - 1):
        # IMU residuals
        i, j = k, k + 1
        imu_ij = get_imu_between(timestamps[i], timestamps[j], data, data['imu_time'],
                                 est_states[i]['bias_g'], est_states[i]['bias_a'])
        delta = preintegrate(imu_ij, est_states[i]['bias_g'], est_states[i]['bias_a'],
                             imu_ij['dt'], gravity=est_states[i]['g'])
        dt = dt_gps
        r_pre = imu_residual_preint(est_states[i], est_states[j], delta, dt)
        loss += 100 * (r_pre ** 2).mean()

    for k in range(len(est_states) - 1):
        # Smoothness and bias consistency residuals
        r_vel = velocity_smoothness_residual(est_states[k], est_states[k + 1])
        r_bias_g = biasGYR_residual(est_states[k], est_states[k + 1])
        r_bias_a = biasACC_residual(est_states[k], est_states[k + 1])

        if (k >= MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT) and (USE_Twb):
            # Apply extrinsic transformation
            p_i_world = T_w_b['R'].Act(est_states[k]['p']) + T_w_b['t']
            p_j_world = T_w_b['R'].Act(est_states[k + 1]['p']) + T_w_b['t']
            delta_est = p_j_world - p_i_world
            delta_gps = gps_cleaned[k + 1] - gps_cleaned[k]
            r_gps_disp = delta_est - delta_gps
            r_gps_anchor = p_i_world - gps_cleaned[k]
            loss += alpha_gps * ((r_gps_disp ** 2).mean() +  (r_gps_anchor ** 2).mean())
        else:
            # Use direct difference without transformation
            delta_est = est_states[k + 1]['p'] - est_states[k]['p']
            delta_gps = gps_cleaned[k + 1] - gps_cleaned[k]
            r_gps_disp = delta_est - delta_gps
            loss += alpha_gps * ( (r_gps_disp ** 2).mean() )

        # Add regularization terms
        loss += 1 * (r_bias_g ** 2).mean()
        loss += 1 * (r_bias_a ** 2).mean()

    # Backpropagation and parameter update
    loss.backward()
    optimizer.step()

    # Save histories
    bias_g_history.append(est_states[0]['bias_g'].detach().cpu().numpy())
    bias_a_history.append(est_states[0]['bias_a'].detach().cpu().numpy())
    g_history.append(est_states[0]['g'].detach().cpu().numpy())

    # Print diagnostics
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
            print(f"Early stopping at epoch {epoch} with loss {loss.item():.6f}")
            break
    else:
        best_loss = loss.item()
        epochs_no_improve = 0

# ========= FINAL RESULTS =========
plot_results(est_states, gps_measurements, data, MAX_GPS_MEASUREMENTS)

# Extract estimated and ground truth positions and velocities
est_positions = [s['p'].detach().cpu() for s in est_states]
gt_positions = [data['gt_p'][i].detach().cpu() for i in range(len(est_states))]

est_velocities = [s['v'].detach().cpu() for s in est_states]
gt_velocities = [v.detach().cpu() for v in gt_velocities]

# Compute evaluation metrics
rmse_pos = compute_rmse(est_positions, gt_positions)
rmse_vel = compute_rmse(est_velocities, gt_velocities)
ate = compute_ate(est_positions, gt_positions)
elapsed = time.time() - start_time
rot_rmse = compute_rotation_rmse(est_states, data['gt_q'][:len(est_states)])

# Print evaluation summary
print(f"✅ Time: {elapsed:.6f} sec.")
print(f"✅ RMSE position: {rmse_pos:.6f} m")
print(f"✅ ATE: {ate:.6f} m")
print(f"✅ RMSE velocity: {rmse_vel:.6f} m/s")
print(f"✅ RMSE rotation: {rot_rmse:.6f} deg")

# ========= PRINT FINAL T_w_b =========
R_final = T_w_b['R'].matrix().detach().cpu().numpy()
t_final = T_w_b['t'].detach().cpu().numpy()

print("\nfinal T_w_b :")
print("R (3x3):")
print(R_final)
print("T (3x1):")
print(t_final)
