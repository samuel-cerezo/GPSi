"""Regularisation residuals and GNSS distance / anchor residuals."""


def velocity_smoothness_residual(state_i, state_j):
    """Difference between consecutive velocities."""
    return state_j["v"] - state_i["v"]


def bias_gyro_residual(state_i, state_j):
    """Random-walk residual of the gyroscope bias."""
    return state_j["bias_g"] - state_i["bias_g"]


def bias_acc_residual(state_i, state_j):
    """Random-walk residual of the accelerometer bias.

    NOTE: kept identical to the original code, which uses the *gyroscope*
    bias here (``bias_g``). Changing it to ``bias_a`` would change the
    published numbers.
    """
    return state_j["bias_g"] - state_i["bias_g"]


def gnss_distance_residual(p_i, p_j, gnss_i, gnss_j):
    """Inter-epoch GNSS displacement residual (frame-invariant to translation)."""
    return (p_j - p_i) - (gnss_j - gnss_i)


def gnss_anchor_residual(p_i_world, gnss_i):
    """Absolute GNSS position residual (requires the GNSS-inertial alignment)."""
    return p_i_world - gnss_i
