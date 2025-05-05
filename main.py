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
from utils import *

# ========= CONFIGURACIÓN =========
MAX_GPS_MEASUREMENTS = 900
g_module = 9.81
gps_noise_std = 0.04
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# ========= CARGA DE DATOS =========
data = load_euroc_data('/home/samuel/dev/repos/GPSi/datasets/MH_01_easy', gps_noise_std, device=device)

# ========= ESTADO INICIAL =========
state = {
    'p': torch.zeros(3, device=device),
    'v': torch.zeros(3, device=device),
    'R': pp.identity_SO3(device=device),
    'bias_g': torch.tensor([0.1, 0.1, 0.1], device=device),  # Bias de giroscopio como parámetro optimizable
    'bias_a': torch.tensor([0.1, 0.1, 0.1], device=device),  # Bias de acelerómetro como parámetro optimizable
    'g_dir': torch.tensor([0.0, 0.0], device=device, requires_grad=True)  # perturbación
}

states = [state]
timestamps = [data['gt_time'][0]]
gps_measurements = [data['gps_p'][0]]
gt_velocities = [data['gt_v'][0]]

# ========= PROPAGACIÓN CON PREINTEGRACIÓN =========
for k in range(MAX_GPS_MEASUREMENTS - 1):
    t_start = data['gt_time'][k]
    t_end = data['gt_time'][k + 1]

    imu_ij = get_imu_between(t_start, t_end, data, data['imu_time'], state['bias_g'], state['bias_a'])
    g_vector = expmap_s2(state['g_dir']) * g_module  # escala por el módulo de la gravedad
    delta = preintegrate(imu_ij, state['bias_g'], state['bias_a'], imu_ij['dt'], gravity=g_vector)
    state = propagate_preintegrated(state, delta, g_module=g_module)

    states.append(state)
    timestamps.append(t_end)
    gps_measurements.append(data['gps_p'][k + 1])
    gt_velocities.append(data['gt_v'][k + 1])

print(f"Longitud de estado: {len(states)}")
timestamps_gt = data['gt_time']
timestamps_imu = data['imu_time']

dt_gps = torch.diff(timestamps_gt) * 1e-9
dt_imu = torch.diff(timestamps_imu) * 1e-9

freq_gps = 1 / dt_gps.mean()
freq_imu = 1 / dt_imu.mean()

print(f"Frecuencia GPS: {freq_gps.item():.2f} Hz")
print(f"Frecuencia IMU: {freq_imu.item():.2f} Hz")


# ========= ESTADOS ESTIMADOS =========

est_states = []
for i in range(len(states)):
    est_states.append({
        'p': (gps_measurements[i] + 0 * torch.randn(3, device=device)).requires_grad_(),
        'v': (gt_velocities[i] + 0.1 * torch.randn(3, device=device)).requires_grad_(),
        'R': states[i]['R'].detach().clone().requires_grad_(),
        'bias_g': states[i]['bias_g'].requires_grad_(),  # Se usa el bias_g estimado que es parámetro optimizable
        'bias_a': states[i]['bias_a'].requires_grad_(),  # Se usa el bias_a estimado que es parámetro optimizable
        'g_dir': torch.tensor([0.0, 0.0], device=device, requires_grad=True)        
    })

# ========= OPTIMIZACIÓN =========
params = [p['p'] for p in est_states] + [p['v'] for p in est_states] + [p['R'] for p in est_states] + \
         [p['bias_g'] for p in est_states] + [p['bias_a'] for p in est_states] + [p['g_dir'] for p in est_states]

optimizer = torch.optim.Adam(params, lr=1e-2)

dt_gps = (data['gt_time'][1] - data['gt_time'][0]).item()
print("{dt_gps:.6f}")
smoothed_gps = kalman_filter_gps(gps_measurements, dt_gps, gps_var=0.01)

# Aseguramos que las mediciones GPS sean las mismas que tienes en tu lista original de gps_measurements
gps_measurements = torch.stack(gps_measurements)  # Esto ya debería ser un tensor 2D de tamaño [N, 3]

# Convertimos las posiciones suavizadas a NumPy
smoothed_np = np.vstack([pos.cpu().numpy() for pos in smoothed_gps])  # Apilamos todas las posiciones suavizadas

# Convertimos las mediciones GPS a NumPy
gps_np = gps_measurements.cpu().numpy()

# ---------------------------------  Graficar -------------------------------------
#fig = plt.figure(figsize=(10, 7))
#ax = fig.add_subplot(111, projection='3d')

# Graficamos las mediciones GPS y las suavizadas
#ax.plot(gps_np[:, 0], gps_np[:, 1], gps_np[:, 2], label='GPS Medido', linestyle=':', linewidth=1)
#ax.plot(smoothed_np[:, 0], smoothed_np[:, 1], smoothed_np[:, 2], label='GPS Suavizado (Kalman)', linewidth=2)

#ax.set_title('Comparación: GPS vs Kalman Smoothed')
#ax.set_xlabel('X (m)')
#ax.set_ylabel('Y (m)')
#ax.set_zlabel('Z (m)')
#ax.legend()
#plt.tight_layout()
#plt.show()

# Listas para almacenar la evolución de los parámetros
bias_g_history = []
bias_a_history = []
g_history = []


start_time = time.time()
################################
for epoch in range(200):
    optimizer.zero_grad()
    loss = 0.0
    
    # Aplicamos el filtro de Kalman a las mediciones GPS a medida que las procesamos
    smoothed_gps = kalman_filter_gps(gps_measurements)  # Filtramos todas las mediciones GPS hasta ahora
    
    for k in range(len(states) - 1):
        i, j = k, k + 1
        g_vector = expmap_s2(state['g_dir']) * g_module  # escala por el módulo de la gravedad
        imu_ij = get_imu_between(timestamps[i], timestamps[j], data, data['imu_time'],est_states[i]['bias_g'], est_states[i]['bias_a'])
        delta = preintegrate(imu_ij, est_states[i]['bias_g'], est_states[i]['bias_a'], imu_ij['dt'], gravity=g_vector)
        #dt = timestamps[j] - timestamps[i]
        dt = dt_gps
        r_pre = imu_residual_preint(est_states[i], est_states[j], delta, dt)
        loss += 100*(r_pre ** 2).mean()

    for k in range(len(est_states) - 1):
        r_vel = velocity_smoothness_residual(est_states[k], est_states[k + 1])
        r_bias = biasGYR_residual(est_states[k], est_states[k + 1])
        delta_est = est_states[k + 1]['p'] - est_states[k]['p']
        delta_gps = smoothed_gps[k + 1] - smoothed_gps[k]
        r_gps_disp = delta_est - delta_gps
        # Residuo de ancla suave (ancla cada posición al GPS suavizado)
        r_gps_anchor = est_states[k]['p'] - smoothed_gps[k]

        # Combinamos ambos residuos
        loss += (r_gps_disp ** 2).mean() +  (r_gps_anchor ** 2).mean()
        ######
        loss += 0.1 * (r_vel ** 2).mean()
        loss += 1 * (r_bias ** 2).mean()

    loss.backward()
    optimizer.step()
    #print(f"Gradientes de bias_g: {est_states[0]['bias_g'].grad}")
    #print(f"Gradientes de bias_a: {est_states[0]['bias_a'].grad}")
    #print(f"Gradientes de g: {est_states[0]['g'].grad}")

    # Almacenamos los valores de los parámetros
    bias_g_history.append(est_states[0]['bias_g'].detach().cpu().numpy())  # Guardamos el primer valor de bias_g (para todos)
    bias_a_history.append(est_states[0]['bias_a'].detach().cpu().numpy())  # Guardamos el primer valor de bias_a (para todos)
    g_history.append(est_states[0]['g'].detach().cpu().numpy())  # Guardamos el primer valor de g (para todos)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
######################

# ========= GRÁFICAS DE EVOLUCIÓN DE LOS PARÁMETROS =========

# Convertimos las listas a arrays de numpy para facilitar el graficado
bias_g_history = np.array(bias_g_history)
bias_a_history = np.array(bias_a_history)
g_history = np.array(g_history)

# Graficamos la evolución de los biases
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(bias_g_history[:, 0], label="bias_g X")
plt.plot(bias_g_history[:, 1], label="bias_g Y")
plt.plot(bias_g_history[:, 2], label="bias_g Z")
plt.title("Evolución de los Biases del Giroscopio")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()

plt.subplot(312)
plt.plot(bias_a_history[:, 0], label="bias_a X")
plt.plot(bias_a_history[:, 1], label="bias_a Y")
plt.plot(bias_a_history[:, 2], label="bias_a Z")
plt.title("Evolución de los Biases del Acelerómetro")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()

plt.subplot(313)
plt.plot(g_history[:, 0], label="g X")
plt.plot(g_history[:, 1], label="g Y")
plt.plot(g_history[:, 2], label="g Z")
plt.title("Evolución de la Gravedad Estimada")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()

plt.tight_layout()
plt.show()


# ========= FUNCIÓN PARA GRAFICAS =========
plot_results(est_states, gps_measurements, data, MAX_GPS_MEASUREMENTS)

# ========= MÉTRICAS FINAL =========
est_positions = [s['p'].detach().cpu() for s in est_states]
gt_positions = [data['gt_p'][i].detach().cpu() for i in range(len(est_states))]

est_velocities = [s['v'].detach().cpu() for s in est_states]
gt_velocities = [gt.detach().cpu() for gt in gt_velocities]

rmse_pos = compute_rmse(est_positions, gt_positions)
rmse_vel = compute_rmse(est_velocities, gt_velocities)
ate = compute_ate(est_positions, gt_positions)

elapsed = time.time() - start_time
print(f"✅ Tiempo total de ejecución: {elapsed:.6f} sec.")
print(f"\n✅ RMSE posición: {rmse_pos:.6f} m")
print(f"✅ ATE: {ate:.6f} m")
print(f"✅ RMSE velocidad: {rmse_vel:.6f} m/s")