### Utility functions for the various models, modifified from LR's original utils
import numpy as np
from utils import kl_divergence, jeffreys_divergence


def cost_stack(Xi, params):
    """
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """
    # Unpack parameters
    Ws = params["Ws"]  # (2, NUM_STACKS, N)
    # Kramers-Moyal coefficients
    A_KMs = params["A_KMs"]  # (NUM_STACKS, N)
    C_KMs = params["C_KMs"]  # (NUM_STACKS, N)

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff  # (N, )
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]  # (N, )

    A_KMs[A_KMs == 0] = np.nan
    C_KMs[C_KMs == 0] = np.nan

    V = np.nansum(
        Ws[0] * ((A_vals - A_KMs) / A_KMs) ** 2
        + Ws[1] * ((C_vals - C_KMs) / C_KMs) ** 2
    )
    V /= np.prod(A_KMs.shape)

    return V


def cost_KL_stack(Xi, params):
    """
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """
    # Unpack parameters
    Ws = params["Ws"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KMs = params["A_KMs"]
    C_KMs = params["C_KMs"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdfs = params["pdfs"]
    sfp = params["sfp"]
    kl_reg = params["kl_reg"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    A_KMs[A_KMs == 0] = np.nan
    C_KMs[C_KMs == 0] = np.nan

    V = np.nansum(
        Ws[0] * ((A_vals - A_KMs) / A_KMs) ** 2
        + Ws[1] * ((C_vals - C_KMs) / C_KMs) ** 2
    )
    V /= np.prod(A_KMs.shape)

    sfp_pdf = sfp.solve(A_vals, C_vals)
    kl_divs = kl_divergence(data_pdfs, sfp_pdf, sfp.dx, tol=1e-6)
    kl_div = np.sum(kl_divs[kl_divs > 0])

    # Penalise an equation with non-negative coefficient on largest term
    return V * (1 - kl_reg) + kl_div * kl_reg


def cost_jef_stack(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """

    # Unpack parameters
    Ws = params["Ws"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KMs = params["A_KMs"]
    C_KMs = params["C_KMs"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdfs = params["pdfs"]
    sfp = params["sfp"]
    kl_reg = params["kl_reg"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    A_KMs[A_KMs == 0] = np.nan
    C_KMs[C_KMs == 0] = np.nan

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(
        Ws[0] * ((A_vals - A_KMs) / A_KMs) ** 2
        + Ws[1] * ((C_vals - C_KMs) / C_KMs) ** 2
    )

    V /= np.prod(A_KMs.shape)

    sfp_pdf = sfp.solve(A_vals, C_vals)
    jef_divs = jeffreys_divergence(data_pdfs, sfp_pdf, sfp.dx, tol=1e-6)
    jef_div = np.sum(jef_divs[jef_divs > 0])

    # Penalise an equation with non-negative coefficient on largest term
    return V * (1 - kl_reg) + jef_div * kl_reg


def cost_alpha_stack(alpha, params):
    # Unpack parameters
    Ws = params["Ws"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KMs = params["A_KMs"]
    C_KMs = params["C_KMs"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    Xi = params["Xi0"] * alpha

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    # Penalise an equation with non-negative coefficient on largest term
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(
        Ws[0] * np.abs((A_vals - A_KMs) / A_KMs) ** 2
        + Ws[1] * np.abs((C_vals - C_KMs) / C_KMs) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    return V


def cost_just_jef_stack(Xi, params):
    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdfs = params["pdfs"]
    sfp = params["sfp"]

    A_coeff = Xi[: lib_A.shape[0]]
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    sfp_pdf = sfp.solve(A_vals, C_vals)
    jef_divs = jeffreys_divergence(data_pdfs, sfp_pdf, sfp.dx, tol=1e-6)
    jef_div = np.sum(jef_divs[jef_divs > 0], initial=0.0)

    return jef_div
