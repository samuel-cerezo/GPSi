# residuals.py
import torch
from models import propagate_state

def imu_residual(state_i, state_j, imu_data, dt):
    """
    Compute IMU residual between two states.
    """
    # Predict state_j from state_i
    pred_state = propagate_state(state_i, imu_data, dt)

    # Position residual
    r_p = state_j['p'] - pred_state['p']
    
    # Velocity residual
    r_v = state_j['v'] - pred_state['v']
    
    # Rotation residual (Lie algebra error)
    R_err = pred_state['R'].Inv() @ state_j['R']
    r_R = R_err.Log()  # Log map to lie algebra (vector in R^3)

    # Biases and gravity could also have residuals, but here we assume constant
    return torch.cat([r_p, r_v, r_R], dim=-1)

def omega_residual(state_i, state_j, imu_data, dt):

    R_i = state_i['R']
    R_j = state_j['R']  

    # Rotación relativa (de i a j)
    R_ij = R_i.Inv() @ R_j
    # Vector en so(3) (rotación relativa expresada como vector)
    w_ij = R_ij.Log()
    # Medición de giro y bias
    imu_gyr_i = imu_data['gyro']
    bg_i = state_i['bias_g']

    # Residual
    res = w_ij - (imu_gyr_i - bg_i) 

    return res


def velocity_smoothness_residual(state_i, state_j):
    """
    Penaliza cambios bruscos entre velocidades consecutivas.
    """
    v_i = state_i['v']
    v_j = state_j['v']
    res = v_j - v_i
    return res 


def biasGYR_residual(state_i, state_j):
    """
    Penaliza cambios bruscos entre velocidades consecutivas.
    """
    bg_i = state_i['bias_g']
    bg_j = state_j['bias_g']
    res = bg_j - bg_i
    return res

def biasACC_residual(state_i, state_j):
    """
    Penaliza cambios bruscos entre velocidades consecutivas.
    """
    bg_i = state_i['bias_g']
    bg_j = state_j['bias_g']
    res = bg_j - bg_i
    return res 

def gps_residual(state_i, state_j, gps_i, gps_j, k):
    """
    Compute GPS residual (delta position).
    """
    delta_est = state_j['p'] - state_i['p']
    delta_gps = gps_j - gps_i

    res1 = state_i['p'] - gps_i
    res2 = delta_est - delta_gps

    if k % 10 == 0:
        res = res1
    else:
        res = res2
    
    return res2
    
