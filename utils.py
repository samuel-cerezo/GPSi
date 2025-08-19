
import torch
import pypose as pp
import numpy as np
from scipy.spatial.transform import Rotation as R

def make_SO3_from_numpy(R_np, device='cpu'):
    """
    Convierte una matriz de rotación 3x3 a un LieTensor SO3, listo para optimización.
    """
    # Convertir a vector de rotación (log SO(3))
    r = R.from_matrix(R_np)
    rotvec_np = r.as_rotvec()

    # Crear LieTensor en so(3)
    omega = torch.tensor(rotvec_np, dtype=torch.float32, device=device)
    omega_lie = pp.LieTensor(omega, ltype=pp.so3_type)

    # Mapear al grupo con Exp
    R_so3 = pp.Exp(omega_lie).detach().clone().requires_grad_()

    return R_so3

def expmap_s2(delta):
    """
    Exponential map on S^2 using the perturbation delta ∈ R^2.
    Returns a unit vector in R^3 on S^2.
    """
    theta = delta.norm()
    if theta < 1e-8:
        return torch.tensor([0.0, 0.0, 1.0], device=delta.device)
    
    axis = torch.cat((delta, torch.tensor([0.0], device=delta.device)))
    axis = axis / axis.norm()
    R = pp.SO3.Exp(axis * theta)
    return R(torch.tensor([0.0, 0.0, 1.0], device=delta.device))  # z-axis nominal gravity
