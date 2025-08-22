# models.py
import torch
import pypose as pp
from utils import *
import numpy as np

def propagate_state(state, imu_data, dt):
    """
    Propagates the state given IMU measurements and time step.
    
    state: dict with keys 'p', 'v', 'R', 'bias_g', 'bias_a', 'g'
    imu_data: dict with keys 'accel', 'gyro'
    dt: float
    """
    p = state['p']
    v = state['v']
    R = state['R']
    bg = state['bias_g']
    ba = state['bias_a']
    g = state['g']

    # Corrected IMU measurements
    omega = imu_data['gyro'] - bg  # shape (3,)
    acc = imu_data['accel'] - ba   # shape (3,)

    # Update rotation
    omega_lie = pp.LieTensor(omega, ltype=pp.so3_type)
    dR = pp.Exp(omega_lie * dt)
    R_new = R @ dR               # Update rotation matrix

    # Update velocity
    v_new = v + (R @ acc + g) * dt

    # Update position
    p_new = p + v * dt + 0.5 * (R @ acc + g) * dt * dt

    # Biases assumed constant for now
    new_state = {
        'p': p_new,
        'v': v_new,
        'R': R_new,
        'bias_g': bg,
        'bias_a': ba,
        'g': g
    }
    return new_state

def preintegrate(imu_data, bias_g, bias_a, dt_list, gravity):
    """
    Calcula los deltas preintegrados entre dos estados.
    
    Args:
        imu_data (dict): {'acc': Tensor[M,3], 'gyro': Tensor[M,3]}
        bias_g (Tensor): bias giroscópico (3,)
        bias_a (Tensor): bias acelerómetro (3,)
        dt_list (list of float): intervalos de tiempo entre medidas
        gravity (Tensor): vector gravedad global (3,)
    
    Returns:
        dict: {'dR': SO3, 'dv': Tensor(3), 'dp': Tensor(3)}
    """
    dR = pp.identity_SO3(device=bias_g.device)
    dv = torch.zeros(3, device=bias_g.device)
    dp = torch.zeros(3, device=bias_g.device)
    v = torch.zeros(3, device=bias_g.device)
    
    for k in range(len(dt_list)):
        dt = dt_list[k]
        omega = imu_data['gyro'][k] - bias_g
        accel = imu_data['acc'][k] - bias_a
        if torch.isnan(accel).any() or torch.isnan(omega).any():
            print(f"[!] NaN en aceleración o giroscopio: accel={accel}, omega={omega}")

        omega_lie = pp.LieTensor(omega * dt, ltype=pp.so3_type)
        dR_k = pp.Exp(omega_lie)

        dR = dR @ dR_k
        a_world = dR @ accel + gravity * dt
        dv += a_world * dt
        dp += v * dt + 0.5 * a_world * dt**2
        v += a_world * dt

    total_dt = sum(dt_list)

    return {
        'dR': dR,
        'dv': dv,
        'dp': dp,
        'dt': total_dt  
    }

def propagate_preintegrated(state_i, delta):
    """
    Propaga el estado usando los resultados preintegrados.

    Args:
        state_i: Estado en el tiempo i (diccionario con p, v, R, bias, g).
        delta: Objeto con ΔR, Δv, Δp generados por preintegración.

    Returns:
        state_j: Estado propagado al tiempo j.
    """
    R_i = state_i['R']
    v_i = state_i['v']
    p_i = state_i['p']
    g = state_i['g']
    dt = delta['dt']

    # Rotación
    R_j = R_i @ delta['dR']

    # Velocidad
    v_j = v_i + g * dt + R_i @ delta['dv']

    # Posición
    p_j = p_i + v_i * dt + 0.5 * g * dt**2 + R_i @ delta['dp']

    # Retenemos bias y gravedad del estado previo
    state_j = {
        'p': p_j,
        'v': v_j,
        'R': R_j,
        'bias_g': state_i['bias_g'],
        'bias_a': state_i['bias_a'],
        'g': g
    }

    return state_j


def get_imu_between(t_i, t_j, imu_data, imu_timestamps, bias_g, bias_a):
    """
    Extrae las mediciones IMU entre los tiempos t_i y t_j (sin incluir t_j).
    
    Args:
        t_i (float): tiempo inicial.
        t_j (float): tiempo final.
        imu_data (dict): {'acc': Tensor[N,3], 'gyro': Tensor[N,3]}.
        imu_timestamps (Tensor[N]): timestamps asociados a las lecturas IMU.

    Returns:
        dict: {'acc': Tensor, 'gyro': Tensor, 'dt': list}
    """
    acc = imu_data['acc'] - bias_a
    gyro = imu_data['gyro'] - bias_g
    
    indices = (imu_timestamps >= t_i) & (imu_timestamps < t_j)
    selected_idx = indices.nonzero(as_tuple=True)[0]

    acc_between = acc[selected_idx]
    gyro_between = gyro[selected_idx]
    
    # Calcular deltas de tiempo
    ts = imu_timestamps[selected_idx]
    dt_list = (ts[1:] - ts[:-1]).tolist()
    
    return {
        'acc': acc_between,
        'gyro': gyro_between,
        'dt': dt_list
    }


def imu_residual_preint(state_i, state_j, delta, total_dt):
    """
    Calcula el residual IMU preintegrado entre dos estados.

    Args:
        state_i (dict): estado en tiempo i
        state_j (dict): estado en tiempo j
        delta (dict): {'dR', 'dv', 'dp'} preintegrados
        total_dt (float): tiempo total entre i y j

    Returns:
        Tensor: residual de error IMU preintegrado
    """
    R_i = state_i['R']
    R_j = state_j['R']
    v_i = state_i['v']
    v_j = state_j['v']
    p_i = state_i['p']
    p_j = state_j['p']
    g   = state_i['g']

    r_R = (delta['dR'].Inv() @ (R_i.Inv() @ R_j)).Log()
    r_v = R_i.Inv() @ (v_j - v_i - g * total_dt) - delta['dv']
    r_p = R_i.Inv() @ (p_j - p_i - v_i * total_dt - 0.5 * g * total_dt**2) - delta['dp']

    return torch.cat([r_R, r_v, r_p])

def add_gps_anchor_residual(loss, est_states, gps_measurements, k0, window_size=5, method='mean', weight=1.0):
    """
    Añade un residuo de anclaje a la pérdida, basado en una media o mediana de GPS en una ventana centrada en k0.

    Args:
        loss (torch.Tensor): El escalar acumulado de pérdida.
        est_states (list): Lista de estados estimados.
        gps_measurements (list): Lista de posiciones GPS.
        k0 (int): Índice del estado al que se le aplica el anclaje.
        window_size (int): Tamaño de la ventana para promedio/mediana.
        method (str): 'mean' o 'median'.
        weight (float): Peso del residuo.

    Returns:
        torch.Tensor: Pérdida actualizada.
    """
    assert window_size % 2 == 1, "La ventana debe ser impar."
    half_window = window_size // 2
    N = len(gps_measurements)

    i_start = max(0, k0 - half_window)
    i_end = min(N, k0 + half_window + 1)

    window = torch.stack(gps_measurements[i_start:i_end])
    if method == 'mean':
        anchor = window.mean(dim=0)
    elif method == 'median':
        anchor, _ = window.median(dim=0)
    else:
        raise ValueError("Método inválido: usa 'mean' o 'median'")

    r_anchor = est_states[k0]['p'] - anchor
    loss += weight * (r_anchor ** 2).mean()
    return loss

def smoothing(gps_measurements, gps_var, dt=0.01):
    device = gps_measurements[0].device
    n = len(gps_measurements)

    # Estado: [px, py, pz, vx, vy, vz]
    initial_pos = gps_measurements[0]
    x = torch.zeros(6, device=device)
    x[:3] = initial_pos  # Inicializamos la posición con la primera medición GPS
    P = torch.eye(6, device=device) * 1.0

    # Matriz de transición (modelo constante en velocidad)
    F = torch.eye(6, device=device)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Matriz de observación (solo posición observada)
    H = torch.zeros(3, 6, device=device)
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1

    Q = torch.eye(6, device=device) * 1e-4  # Ruido proceso
    R = torch.eye(3, device=device) * gps_var  # Ruido medición

    smoothed_positions = []

    for z in gps_measurements:
        # --- Predict ---
        x = F @ x
        P = F @ P @ F.T + Q

        # --- Update ---
        y = z - H @ x                       # Residual
        S = H @ P @ H.T + R                 # Covarianza del residual
        K = P @ H.T @ torch.linalg.inv(S)   # Ganancia de Kalman

        x = x + K @ y
        P = (torch.eye(6, device=device) - K @ H) @ P

        smoothed_positions.append(x[:3].clone())

    return smoothed_positions

def compute_rigid_transform_torch(P_tensors, Q_tensors):
    """
    Calcula la transformación rígida (R, t) que alinea dos listas de tensores 3D.
    
    Args:
        P_tensors (list of torch.Tensor): Lista de puntos estimados (Nx3).
        Q_tensors (list of torch.Tensor): Lista de puntos de referencia (Nx3).
    
    Returns:
        R (torch.Tensor): Matriz de rotación 3x3.
        t (torch.Tensor): Vector de traslación 3x1.
    """
    assert len(P_tensors) == len(Q_tensors), "Los conjuntos deben tener la misma longitud."

    P = torch.stack(P_tensors).cpu().numpy()
    Q = torch.stack(Q_tensors).cpu().numpy()

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)

    R_np = Vt.T @ U.T
    if np.linalg.det(R_np) < 0:
        Vt[-1, :] *= -1
        R_np = Vt.T @ U.T

    t_np = centroid_Q - R_np @ centroid_P
    R = torch.tensor(R_np, dtype=torch.float32)
    t = torch.tensor(t_np, dtype=torch.float32)

    return R, t