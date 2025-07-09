
import torch
import pypose as pp


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
