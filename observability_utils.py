import torch
import pypose as pp
from torch.autograd.functional import jacobian
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16
})

def pack_Twb(T_w_b):
    return torch.cat([
        T_w_b['R'].Log().squeeze(0),
        T_w_b['t']
    ])

def compute_Twb_hessian(est_states, gps_measurements, T_w_b_init, window_size=10, sigma_gps=0.5, plot=False, save_path=None, all_sigmas=None):
    device = est_states[0]['p'].device
    H = torch.zeros((6, 6), device=device)
    start = max(0, len(est_states) - window_size)

    for k in range(start, len(est_states) - 1):
        def residual_fun(Twb_vec):
            rotvec = Twb_vec[0:3].reshape(1, 3)
            t = Twb_vec[3:6]
            R = pp.Exp(pp.LieTensor(rotvec, ltype=pp.so3_type))

            r_anchor = R.Act(est_states[k]['p']) + t - gps_measurements[k]
            p_i_world = R.Act(est_states[k]['p']) + t
            p_j_world = R.Act(est_states[k + 1]['p']) + t
            delta_est = p_j_world - p_i_world
            delta_gps = gps_measurements[k + 1] - gps_measurements[k]
            r_disp = delta_est - delta_gps

            return torch.cat([r_disp, r_anchor])

        Twb_vec = pack_Twb(T_w_b_init).detach().clone().requires_grad_(True)
        J = jacobian(residual_fun, Twb_vec)
        J = J.view(-1, 6)
        W = (1.0 / sigma_gps**2) * torch.eye(J.shape[0], device=device)
        H += J.T @ (W @ J)

    if torch.count_nonzero(H) > 0:
        sigmas = torch.linalg.svdvals(H)
        nonzero_sigmas = sigmas[sigmas > 1e-14]
        sigma_min_nonzero = nonzero_sigmas.min().item() if len(nonzero_sigmas) > 0 else 0.0
        sigma_max = sigmas.max().item()
        sigma_ratio = sigma_min_nonzero / sigma_max if sigma_max > 0 else 0.0

        if all_sigmas is not None:
            all_sigmas.append(sigma_ratio)

        if plot:
            plot_Twb_singular_values(sigmas, save_path)

        return H, sigma_ratio, sigmas

    return H, 0.0, torch.zeros(6, device=device)

def find_Twb_activation(output_filename,est_states, gps_measurements, T_w_b_init, sigma_gps=0.5, threshold=None, window_size=10):
    all_sigmas = []
    sigma_ratios = []
    derivative_history = []
    activated = False
    activat_frame = 0

    for k in range(1, len(est_states)):
        sub_states = est_states[:k+1]
        sub_gps = gps_measurements[:k+1]

        _, ratio, sigmas = compute_Twb_hessian(sub_states, sub_gps, T_w_b_init,
                                               window_size=window_size, sigma_gps=sigma_gps,
                                               all_sigmas=all_sigmas)

        sigma_ratios.append(ratio)
        print(f"[TWB TEST] Frame {k} | σ_min/σ_max = {ratio:.4e}")

        if len(sigma_ratios) < 2:
            derivative_history.append(float('nan'))
        else:
            prev, curr = sigma_ratios[-2], sigma_ratios[-1]
            delta_rel = abs((curr - prev) / max(prev, 1e-12))
            derivative_history.append(delta_rel)

            if not activated and delta_rel < 1e-2:
                print(f"\n✅ Activation of T_w_b suggested at frame {k} (ratio = {ratio:.4e})")
                activat_frame = k
                activated = True
    #plot_Twb_relative_derivative(output_filename, derivative_history, activation_frame=activat_frame if activated else None, window_size=window_size)
    return activat_frame if activated  else None 
    

def plot_Twb_singular_values(sigmas, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.semilogy(sigmas.cpu().numpy(), marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value (log scale)')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_Twb_relative_derivative(output_filename, derivative_history, activation_frame=None, window_size=3):
    # Configuración global de fuente y tamaños
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    plt.figure(figsize=(5, 4))

    
    smoothed_derivative = derivative_history.copy()

    # Suavizar desde k = 3 (índice 2) en adelante, usando promedio entre k y k-1
    for i in range(2, len(derivative_history)):
        smoothed_derivative[i] = (derivative_history[i] + derivative_history[i - 1]) / 2.0

    
    # Datos
    k_values = list(range(1, 1 + len(smoothed_derivative)))
    log_deriv = np.log10(smoothed_derivative)

    # Plot log10
    plt.plot(k_values[:10], log_deriv[:10], color='red', linewidth=2.5, label=r'${\Delta\rho_k}$ (seq. '+ output_filename+')')
    plt.axhline(y=np.log10(1e-2), color='blue', linestyle=':', linewidth=2.5, label = r'${\Delta{\rho}}_{th}$')

    # Ejes
    plt.xlim(1, 21)
    plt.xticks(range(2, 21, 2))  # ticks en 2, 4, 6, ..., 30
    plt.ylim(-3, 6)
    plt.yticks(range(-3, 6), labels=[str(i) for i in range(-3, 6)])

    # Etiquetas
    plt.xlabel(r'GPS measurement index k')
    plt.ylabel(r'$log_{10}({\Delta\rho_k})$ [a.u.]')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Guardar figura
    plt.savefig(f"rel-sigma-{output_filename}.png", dpi=600, bbox_inches='tight')
    plt.show()