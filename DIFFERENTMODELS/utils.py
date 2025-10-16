### Utility functions for the various models, modifified from LR's original utils
import sys
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
    A_vals = lib_A.T @ A_coeff
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )
    V /= len(A_vals)  # Norm based on number of bins?

    sfp_pdf = sfp.solve(A_vals, C_vals)
    KL_val = max(kl_divergence(data_pdf, sfp_pdf, sfp.dx, tol=1e-6), 0.0) * kl_reg
    # TEMP TESTING
    if KL_val == 0:
        print("GOT NEGATIVE KL VAL", file=sys.stderr)
        print("A_KM:", A_KM, file=sys.stderr)
        print("C_KM:", C_KM, file=sys.stderr)

    # Penalise an equation with non-negative coefficient on largest term
    if A_coeff[np.nonzero(A_coeff)[0][-1]] > 0:
        return np.inf
    return V + KL_val


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
            f_active = tmp_active[tmp_active < len(A_expr)]
            a_active = tmp_active[tmp_active >= len(A_expr)] - len(A_expr)

            params["A_expr"] = A_expr[f_active]
            params["C_expr"] = C_expr[a_active]
            params["lib_A"] = lib_A[f_active]
            params["lib_C"] = lib_C[a_active]
            params["Xi0"] = Xi0[tmp_active]

            # Ensure that there is at least one drift and diffusion term left
            if len(a_active) > 0 and len(f_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)
                costs.append(tmp_V)

                # Keep minimum cost
                if tmp_V < V[k]:
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi
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
        f_active = active[active < len(A_expr)]
        a_active = active[active >= len(A_expr)] - len(A_expr)
        print(f"Active f: {A_expr[f_active]}")
        print(f"Active a: {C_expr[a_active]}", flush=True)

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
