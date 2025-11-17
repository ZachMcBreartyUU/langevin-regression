### Utility functions for the various models, modifified from LR's original utils
from typing import Optional
from time import time
from functools import reduce, partial

import numpy as np
from numpy.fft import fft, fftn, fftfreq, ifftn
from scipy.optimize import minimize


def sindy_model(Xi, expr_list):
    return reduce(
        lambda a, b: a + b,
        (Xi_i * expr_list_i for Xi_i, expr_list_i in zip(Xi, expr_list)),
    )


def round_to_SF(val, SF):
    assert SF >= 1
    return np.round(val, -np.astype(np.floor(np.log10(val)), int) + (SF - 1))


def round_array_to_SF(vals, SF):
    assert SF >= 1
    roundings: np.ndarray = np.zeros_like(vals)
    avals = np.abs(vals)
    roundings[avals > 0] = -np.floor(np.log10(avals[avals > 0])) + (SF - 1)
    roundings = roundings.astype(int)

    return np.fromiter(
        (np.round(val, rounding) for val, rounding in zip(vals, roundings)),
        float,
        len(vals),
    )


def cost(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """

    # Unpack parameters
    W = params["W"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KM = params["A_KM"]
    C_KM = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_vals = lib_A.T @ Xi[: lib_A.shape[0]]
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    return V


def cost_KL(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """

    # Unpack parameters
    W = params["W"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KM = params["A_KM"]
    C_KM = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdf = params["pdf"]
    sfp = params["sfp"]
    kl_reg = params["kl_reg"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    # Penalise an equation with non-negative coefficient on largest term
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    sfp_pdf = sfp.solve(A_vals, C_vals)
    kl_div = max(kl_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0)

    return V * (1 - kl_reg) + kl_div * kl_reg


def cost_jef(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem:
        W, A_KM, C_KM, A_expr, C_expr
    """

    # Unpack parameters
    W = params["W"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KM = params["A_KM"]
    C_KM = params["C_KM"]

    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdf = params["pdf"]
    sfp = params["sfp"]
    kl_reg = params["kl_reg"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_coeff = Xi[: lib_A.shape[0]]
    # Penalise an equation with non-negative coefficient on largest term
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    sfp_pdf = sfp.solve(A_vals, C_vals)
    jef_div = max(jeffreys_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0)

    return V * (1 - kl_reg) + jef_div * kl_reg


def cost_alpha(alpha, params):
    # Unpack parameters
    W = params["W"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KM = params["A_KM"]
    C_KM = params["C_KM"]

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
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    return V


def cost_just_jef(Xi, params):
    lib_A = params["lib_A"]
    lib_C = params["lib_C"]

    data_pdf = params["pdf"]
    sfp = params["sfp"]

    A_coeff = Xi[: lib_A.shape[0]]
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    sfp_pdf = sfp.solve(A_vals, C_vals)
    jef_div = max(jeffreys_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0)

    return jef_div


def optimise_function(cost, params, maxfev=1e5):
    Xi0 = params["Xi0"]

    opt_fun = partial(cost, params=params)
    res = minimize(
        opt_fun,
        Xi0,
        method="nelder-mead",
        options={"disp": False, "maxfev": int(maxfev), "adaptive": True},
    )
    return res.x, res.fun


def optimise_functions(cost1, cost2, params: dict, maxfev=1e5):
    """optimise cost1, then cost2,
    return the parameter of cost1 divided by the parameter of cost2, and cost1 * cost2
    """
    Xi0 = params["Xi0"]
    chi0 = params["chi0"]

    opt_fun = partial(cost1, params=params)
    res_kl = minimize(
        opt_fun,
        Xi0,
        method="nelder-mead",
        options={"disp": False, "maxfev": int(maxfev), "adaptive": True},
    )
    params["Xi0"] = res_kl.x
    opt_fun = partial(cost2, params=params)
    res_moment = minimize(
        opt_fun,
        chi0,
        method="nelder-mead",
        options={"disp": False, "maxfev": int(maxfev), "adaptive": True},
    )
    xi = res_kl.x / res_moment.x
    v = res_kl.fun * res_moment.fun
    print(xi, v)
    return xi, v


def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """

    # Lists of candidate expressions... coefficients are optimized
    A_expr = params["A_expr"].copy()
    C_expr = params["C_expr"].copy()
    lib_A = params["lib_A"].copy()
    lib_C = params["lib_C"].copy()
    Xi0 = params["Xi0"].copy()

    n_terms = len(A_expr) + len(C_expr)

    Xi = np.zeros((n_terms, n_terms - 1), dtype=Xi0.dtype)  # Output results
    V = np.full((n_terms - 1), np.inf)  # Minimum cost at each step

    # Full regression problem as baseline
    Xi[:, 0], V[0] = opt_fun(params)

    # Start with all candidates
    active = np.array([i for i in range(n_terms)])

    # history of each step
    active_history = [active]
    cost_history = [[V[0]]]

    # Iterate and threshold
    for k in range(1, n_terms - 1):
        start = time()
        # Loop through remaining terms and find the one that increases the cost function the least
        min_idx = None
        min_Xi = None
        costs = []
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term

            # Break off masks for drift/diffusion
            A_active = tmp_active[tmp_active < len(A_expr)]
            C_active = tmp_active[tmp_active >= len(A_expr)] - len(A_expr)

            params["A_expr"] = A_expr[A_active]
            params["C_expr"] = C_expr[C_active]
            params["lib_A"] = lib_A[A_active]
            params["lib_C"] = lib_C[C_active]
            tmp_xi0 = Xi0
            tmp_xi0[A_active[-1]] = -abs(tmp_xi0[-1])
            params["Xi0"] = tmp_xi0

            # Ensure that there is at least one drift and diffusion term left
            if len(C_active) > 0 and len(A_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)
                costs.append(tmp_V)

                # Keep minimum cost
                if tmp_V < V[k]:
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi
            else:
                costs.append(np.inf)
        if min_idx is None:
            raise RuntimeError("Cost function returned NaN / inf for all costs")
        end = time()
        cost_history.append(costs)
        print(f"{k}th regression took {end-start} seconds")
        print(f"Cost: {V[k]}")
        # Delete least important term
        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # Re-initialize with best results from previous
        Xi[active, k] = min_Xi
        active_history.append(active)
        A_active = active[active < len(A_expr)]
        C_active = active[active >= len(A_expr)] - len(A_expr)
        print(f"Active A: {A_expr[A_active]}")
        print(f"Active C: {C_expr[C_active]}", flush=True)

    return Xi, V, active_history, cost_history


def ntrapz(I, dx):
    if isinstance(dx, int) or isinstance(dx, float) or len(dx) == 1:
        return np.trapezoid(I, dx=dx, axis=0)
    else:
        return np.trapezoid(ntrapz(I, dx[1:]), dx=dx[0])


def kl_divergence(
    p_in: np.ndarray,
    q_in: np.ndarray,
    dx: float | list[float] = 1.0,
    tol: Optional[float] = None,
):
    """
    Approximate Kullback-Leibler divergence for arbitrary dimensionality
    """
    if tol is None:
        tol = max(min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q < tol] = tol
    p[p < tol] = tol
    return np.abs(ntrapz(p * np.log(p / q), dx))


def jeffreys_divergence(
    p_in: np.ndarray,
    q_in: np.ndarray,
    dx: float | list[float] = 1.0,
    tol: Optional[float] = None,
):
    if tol is None:
        tol = max(min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q < tol] = tol
    p[p < tol] = tol
    return np.abs(ntrapz(p * np.log(p / q), dx) + ntrapz(q * np.log(q / p), dx))


class SteadyFP:
    """
    Solver object for steady-state Fokker-Planck equation

    Initializing this independently avoids having to re-initialize all of the indexing arrays
      for repeated loops with different drift and diffusion

    Jared Callaham (2020)
    """

    def __init__(self, N: int, dx: float):
        """
        N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
        dx - grid spacing (array of floats)
        """

        self.ndim = 1
        self.N = N
        self.dx = dx

        self.k = 2 * np.pi * fftfreq(self.N, dx)
        self.idx = np.zeros((self.N, self.N), dtype=np.int32)
        for i in range(self.N):
            self.idx[i, :] = i - np.arange(self.N)

    def precompute_operator(self, f, a):
        """
        f - array of drift coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        a - array of diffusion coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        NOTE: To generalize to covariate noise, would need to add a dimension to a
        """
        f_hat = self.dx * fftn(f)
        a_hat = self.dx * fftn(a)

        # Set up spectral projection operator
        self.A = np.einsum("i,ij->ij", -1j * self.k, f_hat[self.idx]) + np.einsum(
            "i,ij->ij", -self.k**2, a_hat[self.idx]
        )

    def solve(self, f, a):
        """
        Solve Fokker-Planck equation from input drift coefficients
        """
        self.precompute_operator(f, a)

        q_hat = np.linalg.lstsq(self.A[1:, 1:], -self.A[1:, 0], rcond=1e-6)[0]
        q_hat = np.append([1], q_hat)
        hist = np.real(ifftn(np.reshape(q_hat, self.N))) / np.prod(self.dx)
        hist /= np.nansum(hist * self.dx)
        return hist
