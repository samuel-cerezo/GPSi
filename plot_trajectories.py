# plot_trajectories.py
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectories(gps_positions, estimated_positions, gt_positions=None):
    """
    Plotea trayectorias en 2D y 3D: GPS, Estimado y Ground Truth.
    
    gps_positions: (N,3) tensor
    estimated_positions: (N,3) tensor
    gt_positions: (N,3) tensor (opcional)
    """
    
    #gps_positions = gps_positions.numpy()
    estimated_positions = estimated_positions.cpu().detach().numpy()

    # ========== 2D Plot ==========
    plt.figure(figsize=(10,8))
    plt.plot(gps_positions[:,0], gps_positions[:,1], 'r.--', label='GPS (noisy)')
    plt.plot(estimated_positions[:,0], estimated_positions[:,1], 'b.-', label='Estimated')

    if gt_positions is not None:
        gt_positions_np = gt_positions.cpu().numpy()
        plt.plot(gt_positions_np[:,0], gt_positions_np[:,1], 'g.-', label='Ground Truth')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.title('2D Trajectory Comparison')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # ========== 3D Plot ==========
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gps_positions[:,0], gps_positions[:,1], gps_positions[:,2], 'r.--', label='GPS (noisy)')
    ax.plot(estimated_positions[:,0], estimated_positions[:,1], estimated_positions[:,2], 'b.-', label='Estimated')

    if gt_positions is not None:
        ax.plot(gt_positions_np[:,0], gt_positions_np[:,1], gt_positions_np[:,2], 'g.-', label='Ground Truth')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.set_title('3D Trajectory Comparison')
    plt.tight_layout()
    plt.show()


def plot_results(est_states, gps_measurements, data, MAX_GPS_MEASUREMENTS):
    
    device = data['gt_p'][0].device

    # Posiciones (convertimos a NumPy explícitamente)
    estimated_positions = torch.stack([est_states[i]['p'].detach() for i in range(MAX_GPS_MEASUREMENTS)]).cpu().numpy()
    gps_positions = torch.stack([gps_measurements[i].clone().detach().to(device) for i in range(MAX_GPS_MEASUREMENTS)]).cpu().numpy()
    gt_positions = torch.stack([data['gt_p'][i].clone().detach().to(device) for i in range(MAX_GPS_MEASUREMENTS)]).cpu().numpy()

    # Crear figura y ejes 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar trayectorias
    ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='GT', s=5)
    ax.scatter(gps_positions[:, 0], gps_positions[:, 1], gps_positions[:, 2], label='GPS (ruido)', s=5)
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimado', s=5)
    # Etiquetas y estilo
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trayectoria 3D')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Velocidades
    estimated_velocities = torch.stack([est_states[i]['v'].detach().cpu() for i in range(MAX_GPS_MEASUREMENTS)])
    gt_velocities = torch.stack([data['gt_v'][i].clone().detach().cpu() for i in range(MAX_GPS_MEASUREMENTS)])

    plt.figure(figsize=(12, 5))
    for axis in range(3):
        plt.subplot(1, 3, axis + 1)
        plt.plot(gt_velocities[:, axis], label='GT')
        plt.plot(estimated_velocities[:, axis], label='Estimada')
        plt.title(f'Velocidad eje {["x", "y", "z"][axis]}')
        plt.xlabel('Índice')
        plt.ylabel('Velocidad (m/s)')
        plt.legend()
        plt.grid(True)
    plt.suptitle('Comparación de velocidades')
    plt.tight_layout()
    plt.show()

    # Bias giro
    estimated_bias_gyr = torch.stack([est_states[i]['bias_g'].detach().cpu() for i in range(MAX_GPS_MEASUREMENTS)])
    
    plt.figure(figsize=(12, 5))
    for axis in range(3):
        plt.subplot(1, 3, axis + 1)
        plt.plot(estimated_bias_gyr[:, axis], label='Estimado')
        plt.title(f'Bias gyro eje {["x", "y", "z"][axis]}')
        plt.xlabel('Índice')
        plt.ylabel('Bias (rad/s)')
        plt.legend()
        plt.grid(True)
    plt.suptitle('Bias del giroscopio')
    plt.tight_layout()
    plt.show()

    # Vector gravedad
    estimated_g = torch.stack([est_states[i]['g'].detach().cpu() for i in range(MAX_GPS_MEASUREMENTS)])

    plt.figure(figsize=(12, 5))
    for axis in range(3):
        plt.subplot(1, 3, axis + 1)
        plt.plot(estimated_g[:, axis], label='Estimado')
        plt.title(f'Gravedad eje {["x", "y", "z"][axis]}')
        plt.xlabel('Índice')
        plt.ylabel('m/s²')
        plt.legend()
        plt.grid(True)
    plt.suptitle('Vector gravedad estimado')
    plt.tight_layout()
    plt.show()
