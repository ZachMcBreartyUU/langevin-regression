import numpy as np

from utils import kl_divergence

THRESHOLD = 1e-8


def _square_diff(W, A_vals, A_km, C_vals, C_km):
    return np.nansum(
        W[0] * ((A_vals - A_km) / A_vals) ** 2 + W[1] * ((C_vals - C_km) / C_vals) ** 2
    ) / len(A_vals)


def _kl_reg(sfp, A_vals, C_vals, data_pdf):
    sfp_pdf = sfp.solve(A_vals, C_vals)
    return max(kl_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0) + np.abs(
        data_pdf - sfp_pdf
    )


def _lasso(Xi):
    return np.nansum(np.abs(Xi))


def cost(Xi, params):
    Xi[Xi < THRESHOLD] = 0

    lasso = params["lasso"]
    W = params["W"]

    A_km = params["A_KM"]
    C_km = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    A_coeff = Xi[: lib_A.shape[0]]
    C_coeff = Xi[lib_A.shape[0] :]
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ C_coeff

    square_diff = _square_diff(W, A_vals, A_km, C_vals, C_km)
    lasso_val = _lasso(Xi)

    # nonzero_A = np.nonzero(A_coeff)[0]
    # nonzero_C = np.nonzero(C_coeff)[0]
    # if len(nonzero_A) == 0 or len(nonzero_C) == 0:
    #     return np.inf
    # elif A_coeff[nonzero_A[-1]] > 0:
    #     # Penalise having a positive final coefficient
    #     return np.inf

    return (1 - lasso) * square_diff + lasso * lasso_val


def cost_KL(Xi, params):
    Xi[Xi < THRESHOLD] = 0

    lasso = params["lasso"]
    W = params["W"]

    A_km = params["A_KM"]
    C_km = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdf = params["pdf"]
    sfp = params["sfp"]
    kl_reg = params["kl_reg"]

    A_coeff = Xi[: lib_A.shape[0]]
    C_coeff = Xi[lib_A.shape[0] :]
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ C_coeff

    square_diff = _square_diff(W, A_vals, A_km, C_vals, C_km)
    kl_val = _kl_reg(sfp, A_vals, C_vals, data_pdf)
    lasso_val = _lasso(Xi)

    # nonzero_A = np.nonzero(A_coeff)[0]
    # nonzero_C = np.nonzero(C_coeff)[0]
    # if len(nonzero_A) == 0 or len(nonzero_C) == 0:
    #     return np.inf
    # elif A_coeff[nonzero_A[-1]] > 0:
    #     # Penalise having a positive final coefficient
    #     return np.inf

    return (
        (1 - lasso) * (1 - kl_reg) * square_diff
        + (1 - lasso) * (kl_reg) * kl_val
        + lasso * lasso_val
    )
