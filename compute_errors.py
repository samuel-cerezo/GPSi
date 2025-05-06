# compute_errors.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp


def compute_position_errors(estimated_positions, gt_positions):
    """
    Calcula errores de posición frame a frame.
    
    estimated_positions: (N,3) tensor
    gt_positions: (N,3) tensor
    """
    estimated_positions = estimated_positions.cpu().detach()
    gt_positions = gt_positions.cpu()

    errors = torch.linalg.norm(estimated_positions - gt_positions, dim=1)

    mpe = errors.mean().item()  # Mean Position Error
    maxe = errors.max().item()  # Max Position Error
    rmse = torch.sqrt((errors**2).mean()).item()  # Root Mean Squared Error

    print(f"Mean Position Error (MPE): {mpe:.4f} m")
    print(f"Max Position Error: {maxe:.4f} m")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} m")

    return errors

def plot_position_errors(errors):
    """
    Plotea el error de posición a lo largo del tiempo (frame).
    """
    plt.figure(figsize=(10,6))
    plt.plot(errors.numpy(), 'b.-')
    plt.xlabel('Frame')
    plt.ylabel('Position Error [m]')
    plt.title('Position Error Over Time')
    plt.grid(True)
    plt.show()

def compute_rmse(est_values, gt_values):
    """
    Calcula el RMSE entre dos listas de tensores de 3 dimensiones.

    Args:
        est_values (list of torch.Tensor): Estimaciones.
        gt_values (list of torch.Tensor): Verdades de terreno.

    Returns:
        float: RMSE (en el mismo dispositivo que los tensores).
    """
    errors = [((est - gt) ** 2).mean() for est, gt in zip(est_values, gt_values)]
    mse = torch.stack(errors).mean()
    rmse = torch.sqrt(mse)
    return rmse.item()


def compute_ate(est_values, gt_values):
    """
    Calcula el ATE (Absolute Trajectory Error) entre dos listas de tensores de 3 dimensiones.

    Args:
        est_values (list of torch.Tensor): Estimaciones.
        gt_values (list of torch.Tensor): Verdades de terreno.

    Returns:
        float: ATE promedio (en el mismo dispositivo que los tensores).
    """
    errors = [torch.norm(est - gt) for est, gt in zip(est_values, gt_values)]
    ate = torch.stack(errors).mean()
    return ate.item()


def compute_rotation_rmse(est_states, gt_quaternions):
    """
    Calcula el RMSE de rotación angular en grados entre rotaciones estimadas y ground truth (en quaterniones).
    """
    errors = []

    for i in range(len(est_states)):
        # Rotación estimada (PyTorch tensor 3x3)
        R_est = est_states[i]['R'].matrix()[0]

        # Quaternion GT → SO3 → matriz de rotación
        q_gt = gt_quaternions[i]  # (4,)
        R_gt = pp.SO3(q_gt.unsqueeze(0)).matrix()[0]

        # Error de rotación
        R_err = R_est @ R_gt.T
        trace = torch.trace(R_err).clamp(-1.0, 3.0)

        # Ángulo de rotación en grados
        angle_rad = torch.acos((trace - 1) / 2.0)
        angle_deg = torch.rad2deg(angle_rad).item()
        errors.append(angle_deg)

    # RMSE final
    rmse_deg = np.sqrt(np.mean(np.square(errors)))
    return rmse_deg
