import torch
import numpy as np
import time
import csv
import sys
import pypose as pp
from models import *
from residuals import *
from load_data import *
from plot_trajectories import *
from compute_errors import *
from plot import *
import matplotlib.pyplot as plt
from observability_utils import *

#dataset_path = '/Users/samucerezo/dev/src/repos/GPSi/datasets'
dataset_path = '/home/samuel/dev/repos/GPSi/datasets/EuRoc/V2_03_difficult'
gps_noise_std = 0.1

# Función para calcular la distancia recorrida
def calcular_distancia(gps_measurements):
    distancia = 0.0
    for i in range(1, len(gps_measurements)):
        distancia += torch.norm(gps_measurements[i] - gps_measurements[i - 1]).item()
    return distancia

# Función principal para ejecutar el experimento
def run_experiment(criteria, dataset_path,USE_Twb, MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT, gps_noise_std, MAX_GPS_MEASUREMENTS,output_filename):
    # Aquí debes pegar el contenido de tu main.py
    # Reemplaza las variables globales USE_Twb, MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT, gps_noise_std y MAX_GPS_MEASUREMENTS
    # por los argumentos de esta función
    # Al final, asegúrate de devolver un diccionario con las métricas como se muestra abajo

    # Ejemplo de retorno (reemplaza con tus métricas reales):
    ###########################
    # ========= CONFIGURACIÓN =========

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    # ========= PARÁMETROS DEL SENSOR IMU =========
    gyro_bias_std = 1.9393e-5 * np.sqrt(200)  # 200 Hz
    accel_bias_std = 3.0e-3 * np.sqrt(200)    # 200 Hz

    bias_g_init = torch.tensor(np.random.normal(0, gyro_bias_std, 3), dtype=torch.float32, device=device)
    bias_a_init = torch.tensor(np.random.normal(0, accel_bias_std, 3), dtype=torch.float32, device=device)

    # ========= CARGA DE DATOS =========
    data = load_euroc_data(dataset_path, gps_noise_std, device=device)
  
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

        omega_noise = torch.randn(3, device=device) * 0.01  # pequeño ruido angular
        omega_lie_noise = pp.LieTensor(omega_noise, ltype=pp.so3_type)
        dR_noise = pp.Exp(omega_lie_noise)
        
        est_states.append({
            'p': (gps_measurements[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
            'v': (gt_velocities[i] + 0.00 * torch.randn(3, device=device)).requires_grad_(),
            'R': (states[i]['R'] @ dR_noise).detach().clone().requires_grad_(),
            'bias_g': (states[i]['bias_g'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
            'bias_a': (states[i]['bias_a'] + 0.01 * torch.randn(3, device=device)).requires_grad_(),
            'g': (states[i]['g'] + 0.001 * torch.randn(3, device=device)).requires_grad_()
        })


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

    # ================== DETECCIÓN AUTOMÁTICA DE OBSERVABILIDAD ==================

    #USE_Twb = False  # Se activa dinámicamente

    #activation_frame = find_observability_activation(est_states, data, timestamps, data['imu_time'])
    activation_frame = find_Twb_activation(output_filename,est_states, gps_measurements, T_w_b, sigma_gps=0.5, threshold=1e-3)

    print(f"Activación sugerida en frame: {activation_frame}")
    if criteria:
        MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT = activation_frame
    else:
        MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT = 1
    
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
            if (k >= MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT) and (USE_Twb):
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
    #plot_results(est_states, gps_measurements, data, MAX_GPS_MEASUREMENTS)

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
    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final.flatten()

    print("\nTransformación rígida T_w_b final:")
    print("Rotación (matriz 3x3):")
    print(R_final)
    print("Traslación (vector 3x1):")
    print(t_final)

    distancia_recorrida = calcular_distancia(gps_measurements)
    ###########################
    
    return {
        "USE_Twb": USE_Twb,
        "Obs-criteria": criteria,
        "MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT": MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT,
        "gps_noise_std": gps_noise_std,
        "MAX_GPS_MEASUREMENTS": MAX_GPS_MEASUREMENTS,
        "Tiempo total de ejecución": elapsed,
        "RMSE posición": rmse_pos,
        "ATE": ate,
        "RMSE velocidad": rmse_vel,
        "RMSE rotación": rot_rmse,
        "Distancia recorrida": distancia_recorrida,
        "Transformación rígida (flattened)": T_final.flatten().tolist()
    }


# Función principal para ejecutar todos los experimentos y guardar los resultados
def main(dataset_path, output_filename):

    MAX_GPS_MEASUREMENTS = 100
    
    results = []
    num_repeticiones = 1
    min_gps = 1
    criteria = False
    
    # Repetir experimento base 
    #for _ in range(num_repeticiones):
    #    results.append(run_experiment(dataset_path, False, 1, gps_noise_std, MAX_GPS_MEASUREMENTS,output_filename))

    # Repetir cada configuración con USE_Twb=True 
    #for min_gps in [1] + list(range(5, MAX_GPS_MEASUREMENTS, 5)):
    #for _ in range(num_repeticiones):
    #    results.append(run_experiment(criteria, dataset_path, True, min_gps, gps_noise_std, MAX_GPS_MEASUREMENTS,output_filename))
            
    results.append(run_experiment(False, dataset_path, True, min_gps, gps_noise_std, MAX_GPS_MEASUREMENTS,output_filename))
    results.append(run_experiment(True, dataset_path, True, min_gps, gps_noise_std, MAX_GPS_MEASUREMENTS,output_filename))
          
    CSV_filename = output_filename + "-obs-criteria.csv"

    with open(CSV_filename, 'w', newline='') as csvfile:
        fieldnames = [
            "USE_Twb", "Obs-criteria","MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT", "gps_noise_std", "MAX_GPS_MEASUREMENTS",
            "Tiempo total de ejecución", "RMSE posición", "ATE", "RMSE velocidad", "RMSE rotación",
            "Distancia recorrida", "Transformación rígida (flattened)"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Formatear valores numéricos como strings con 7 decimales
            for key in result:
                if isinstance(result[key], float):
                    result[key] = f"{result[key]:.7f}"
                elif isinstance(result[key], list):
                    result[key] = "[" + ", ".join(f"{x:.7f}" if isinstance(x, float) else str(x) for x in result[key]) + "]"
            writer.writerow(result)

    print(f"✅ Resultados guardados en '{CSV_filename}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python experimentos_gps.py <nombre_del_archivo_de_salida>")
    else:
        output_filename = sys.argv[1]
        main(dataset_path, output_filename)
