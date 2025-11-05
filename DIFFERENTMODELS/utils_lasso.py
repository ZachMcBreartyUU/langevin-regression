from typing import Optional
from time import time
from functools import reduce, partial

import numpy as np
from numpy.fft import fft, fftn, fftfreq, ifftn
from scipy.optimize import minimize

from utils import kl_divergence


def _square_diff(W, A_vals, A_km, C_vals, C_km):
    return np.nansum(W[0] * (A_vals - A_km) ** 2 + W[1] * (C_vals - C_km) ** 2) / len(
        A_vals
    )


def _kl_reg(sfp, A_vals, C_vals, data_pdf):
    sfp_pdf = sfp.solve(A_vals, C_vals)
    return max(kl_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0)


def _lasso(Xi):
    return np.nansum(np.abs(Xi))


def cost(Xi, params):
    lasso = params["lasso"]
    W = params["W"]

    A_km = params["A_KM"]
    C_km = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    A_coeff = Xi[: lib_A.shape[0]]
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    square_diff = _square_diff(W, A_vals, A_km, C_vals, C_km)
    lasso_val = _lasso(Xi)

    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        # Penalise having a positive final coefficient
        return np.inf

    return (1 - lasso) * square_diff + lasso * lasso_val


def cost_KL(Xi, params):
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
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    square_diff = _square_diff(W, A_vals, A_km, C_vals, C_km)
    kl_val = _kl_reg(sfp, A_vals, C_vals, data_pdf)
    lasso_val = _lasso(Xi)

    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        # Penalise having a positive final coefficient
        return np.inf

    return (
        (1 - lasso) * (1 - kl_reg) * square_diff
        + (1 - lasso) * (kl_reg) * kl_val
        + lasso * lasso_val
    )
