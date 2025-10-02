"""Code adapted from pitchfork.ipynb, utils.py, fpsolve.py by Jared Callaham
This work done by Zach McBrearty

NOTE: The size of the libraries scales as (N_bins x N_functions), so the initial size of the dataset only matters for the initial dataplotting (optional) and then the binning / moment generation (Kramers Moyal KDE).
The size of the libraries matters _greatly_ as they are constantly inverted, this size can be controlled by the -N or --num-bins flag
"""

import argparse
import json
from time import time

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # typing for 3D plots
from scipy.optimize import minimize
import sympy

import utils
import data_loader as dl

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--plot-intermediate", action="store_true")
args = ap.parse_args()
print(args)

folder_path = dl.SCRATCH_PATH / args.folder
print(folder_path)

with open(folder_path / "metadata.json") as f:
    metadata = json.load(f)

dt = metadata["dt"] * args.step
R = metadata["R"]
ep0 = metadata["epsilon_0"]
ep1 = metadata["epsilon_1"]
lb = metadata["lb"]
U = metadata["U"]
tau = metadata["tau"]
m = lambda y: (y - 1) / np.sqrt(y) if y > 1 else 0.0
A_f = lambda x, y: -2 * R * x + 2 * (m(y) - np.sqrt(x)) * x**2 + ep0 + ep1 * x
A_s = lambda x, y: (U - (1 - lb) * x * y - lb * (m(y) * y)) / tau
B_f = lambda x, y: 2 * np.sqrt(ep0 * x + ep1 * x**2)
B_s = lambda x, y: 0

with np.load(folder_path / "km_log_f_s.npz") as file:
    file: dict[str, np.ndarray]
    pdf = file["pdf"]  # (M, N)
    A_f_km = file["moment1_f"]  # (M, N)
    C_f_km = file["moment2_f"]  # (M, N)
    A_s_km = file["moment1_s"]  # (M, N)
    C_s_km = file["moment2_s"]  # (M, N)

    edges_f = file["fluidity_edges"]  # (M + 1,)
    centers_f = file["fluidity_centers"]  # (M,)
    widths_f = file["fluidity_widths"]  # (M,)
    edges_s = file["sigma_edges"]  # (N + 1,)
    centers_s = file["sigma_centers"]  # (N,)
    widths_s = file["sigma_widths"]  # (N,)

nan_mask = (
    np.isfinite(pdf.flatten())
    * ~np.isclose(A_f_km.flatten(), 0)
    * ~np.isclose(C_f_km.flatten(), 0)
    * ~np.isclose(A_s_km.flatten(), 0)
).astype(bool)
pdf_masked = pdf.flatten()[nan_mask]
A_f_km_masked = A_f_km.flatten()[nan_mask]
C_f_km_masked = C_f_km.flatten()[nan_mask]
A_s_km_masked = A_s_km.flatten()[nan_mask]
C_s_km_masked = C_s_km.flatten()[nan_mask]

f_mesh, s_mesh = np.meshgrid(centers_f, centers_s)

N = len(centers_f)
M = len(centers_s)
print(f"Fluidity shape: {N=}")
print(f"Stress shape: {M=}")
print(f"{pdf.shape=}")
print(f"{A_f_km.shape=}")
print(f"{C_f_km.shape=}")
print(f"{A_s_km.shape=}")
print(f"{C_s_km.shape=}")
print(f"{A_f_km_masked.shape=}")
print(f"{C_f_km_masked.shape=}")
print(f"{A_s_km_masked.shape=}")
print(f"{C_s_km_masked.shape=}")
print(f"{edges_f.shape=}")
print(f"{centers_f.shape=}")
print(f"{widths_f.shape=}")
print(f"{edges_s.shape=}")
print(f"{centers_s.shape=}")
print(f"{widths_s.shape=}")
print(f"{f_mesh.shape=}")
print(f"{s_mesh.shape=}", flush=True)

if args.plot_intermediate:
    fig_pdf, axes_pdf = plt.subplots(
        ncols=2, subplot_kw={"projection": "3d"}, figsize=(11, 6)
    )
    axes_pdf: list[Axes3D]
    axes_pdf[0].plot_wireframe(np.log(f_mesh), s_mesh, pdf.T)
    axes_pdf[0].set_zlabel(r"PDF($f, \sigma$)")
    axes_pdf[0].set_ylabel(r"Stress, $\sigma$)")
    axes_pdf[0].set_xlabel(r"$\log$ Fluidity, $\log f$")

    axes_pdf[1].plot_wireframe(np.log(f_mesh), s_mesh, np.log(pdf.T))
    axes_pdf[1].set_zlabel(r"$\log$ PDF($f, \sigma$)")
    axes_pdf[1].set_ylabel(r"Stress, $\sigma$")
    axes_pdf[1].set_xlabel(r"$\log$ Fluidity, $\log f$")

    fig_pdf.tight_layout()
    fig_pdf.savefig(folder_path / "pdf_log_f_s.png")
    plt.close(fig_pdf)

    fig_moments, axes_moments = plt.subplots(
        nrows=2, ncols=2, subplot_kw={"projection": "3d"}, figsize=(11, 11)
    )
    axes_moments: list[list[Axes3D]]
    axes_moments[0][0].plot_wireframe(np.log(f_mesh), s_mesh, A_f_km.T)
    axes_moments[0][0].set_zlabel(r"$m^{(1, 0)}(f, \sigma)$")
    axes_moments[0][0].set_title(r"$m^{(1, 0)}(f, \sigma)$")
    axes_moments[0][0].set_ylabel(r"Stress, $\sigma$)")
    axes_moments[0][0].set_xlabel(r"$\log$ Fluidity, $\log f$")

    axes_moments[0][1].plot_wireframe(np.log(f_mesh), s_mesh, C_f_km.T)
    axes_moments[0][1].set_zlabel(r"$m^{(2, 0)}(f, \sigma)$")
    axes_moments[0][1].set_title(r"$m^{(2, 0)}(f, \sigma)$")
    axes_moments[0][1].set_ylabel(r"Stress, $\sigma$)")
    axes_moments[0][1].set_xlabel(r"$\log$ Fluidity, $\log f$")

    axes_moments[1][0].plot_wireframe(np.log(f_mesh), s_mesh, A_s_km.T)
    axes_moments[1][0].set_zlabel(r"$m^{(0, 1)}(f, \sigma)$")
    axes_moments[1][0].set_title(r"$m^{(0, 1)}(f, \sigma)$")
    axes_moments[1][0].set_ylabel(r"Stress, $\sigma$)")
    axes_moments[1][0].set_xlabel(r"$\log$ Fluidity, $f$")

    axes_moments[1][1].plot_wireframe(np.log(f_mesh), s_mesh, C_s_km.T)
    axes_moments[1][1].set_zlabel(r"$m^{(0, 2)}(f, \sigma)$")
    axes_moments[1][1].set_title(r"$m^{(0, 2)}(f, \sigma)$")
    axes_moments[1][1].set_ylabel(r"Stress, $\sigma$)")
    axes_moments[1][1].set_xlabel(r"$\log$ Fluidity, $\log f$")

    # fig_moments.tight_layout()
    fig_moments.savefig(folder_path / "moments_log_f_s.png")
    plt.close(fig_moments)

### Build SINDy libraries with sympy
f_sym = sympy.symbols("f")
s_sym = sympy.symbols("σ")

# A_f library: [1, f^1/2, f^2/2, f^3/2, f^4/2, f^5/2] * [σ^-1, σ^-1/2, 1, σ^1/2, σ]
A_f_expr = np.array(
    [f_sym ** (i / 2) * s_sym ** (j / 2) for j in range(-2, 3) for i in range(6)]
)
print("A_f library:", A_f_expr)
num_A_f = len(A_f_expr)

# C_f library:    [1,   f, f^2, f^3, f^4]
# expected coeff: [0, ep0, ep1,   0,   0]
C_f_expr = np.array([f_sym ** (i) for i in range(5)])
print("C_f library:", C_f_expr)
num_C_f = len(C_f_expr)

# A_s library: [1, f, f^2] * [1, σ, σ^2]
A_s_expr = np.array([f_sym**i * s_sym**j for j in range(3) for i in range(3)])
print("A_s library:", A_s_expr)
num_A_s = len(A_s_expr)

print(f"Regressing on library of N={num_A_f+num_C_f+num_A_s} terms")

# C_s library: should be 0, so skip since we would only learn the higher order finite time terms

# Convert sympy expressions into library matrices
lib_A_f = np.empty((num_A_f, N, M))
for k in range(num_A_f):
    lamb_expr = sympy.lambdify([f_sym, s_sym], A_f_expr[k])
    val = lamb_expr(f_mesh, s_mesh)
    if np.ndim(val) == 0:  # f^0 == 1.0, so result is 0 dim
        val = np.full_like(f_mesh, val)
    lib_A_f[k] = val.T
lib_A_f = lib_A_f.reshape(-1, N * M)[..., nan_mask]

lib_C_f = np.empty((num_C_f, N, M))
for k in range(num_C_f):
    lamb_expr = sympy.lambdify([f_sym, s_sym], C_f_expr[k])
    val = lamb_expr(f_mesh, s_mesh)
    if np.ndim(val) == 0:
        val = np.full_like(f_mesh, val)
    lib_C_f[k] = val.T
lib_C_f = lib_C_f.reshape(-1, N * M)[..., nan_mask]

lib_A_s = np.empty((num_A_s, N, M))
for k in range(num_A_s):
    lamb_expr = sympy.lambdify([f_sym, s_sym], A_s_expr[k])
    val = lamb_expr(f_mesh, s_mesh)
    if np.ndim(val) == 0:
        val = np.full_like(f_mesh, val)
    lib_A_s[k] = val.T
lib_A_s = lib_A_s.reshape(-1, N * M)[..., nan_mask]
# All libraries are (num_lib, Q)

# Initialize Xi with least squares regression (no finite-time corrections)
Xi0 = np.random.random(num_A_f + num_C_f + num_A_s) * 5 - 2.5
# Xi0 = np.empty((num_A_f + num_C_f + num_A_s,))
# mask = np.nonzero(A_f_km)[0]
# Xi0[:num_A_f] = lstsq(lib_A_f[:, mask].T, A_f_km[mask])[0]
# mask = np.nonzero(C_f_km)[0]  # This mask may actually be the same as previous
# Xi0[num_A_f : num_A_f + num_C_f] = lstsq(lib_C_f[:, mask].T, C_f_km[mask])[0]
# mask = np.nonzero(A_s_km)[0]  # This mask may actually be the same as previous
# Xi0[num_A_f + num_C_f :] = lstsq(lib_A_s[:, mask].T, A_s_km[mask])[0]
# print("Xi0 =", Xi0)
# NOTE: are these initial conditions good enough for highly non-linear systems?


# FUNCTIONS ADAPTED TO 2D:
def AFP_opt(cost, params):
    ### RUN OPTIMIZATION PROBLEM
    start_time = time()
    Xi0 = params["Xi0"]

    opt_fun = lambda Xi: cost(Xi, params)

    res = minimize(
        opt_fun,
        Xi0,
        method="nelder-mead",
        options={
            "disp": False,
            "maxfev": int(1e4),
            "adaptive": True,
        },
    )
    print(
        "%%%% Optimization time: {0} seconds,   Cost: {1} %%%%".format(
            time() - start_time, res.fun
        )
    )

    # Return coefficients and cost function
    return res.x, res.fun


def cost(Xi, params):
    """
    Least-squares cost function for optimization
    Xi - current coefficient estimates
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, f_KM, a_KM, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    """

    # Unpack parameters
    W = params["W"]  # Optimization weights, (3, Q)

    # Kramers-Moyal coefficients
    A_x_KM = params["A_x_KM"]  # (Q,)
    A_y_KM = params["A_y_KM"]  # (Q,)
    C_x_KM = params["C_x_KM"]  # (Q,)

    lib_A_x = params["lib_A_x"]  # (n_A_x, Q)
    lib_A_y = params["lib_A_y"]  # (n_A_y, Q)
    lib_C_x = params["lib_C_x"]  # (n_C_x, Q)

    n_A_x = lib_A_x.shape[0]
    n_A_y = lib_A_y.shape[0]
    n_C_x = lib_C_x.shape[0]  # unused but symmetry

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_x_vals = lib_A_x.T @ Xi[:n_A_x]  # (n_A_x, Q).T @ (n_A_x,) = (Q,)
    A_y_vals = lib_A_y.T @ Xi[n_A_x : n_A_x + n_A_y]  # (Q,)
    C_x_vals = lib_C_x.T @ Xi[n_A_x + n_A_y :]  # (Q,)

    V = (
        np.sum(W[0] * np.abs((A_x_vals - A_x_KM) / A_x_KM) ** 2)
        + np.sum(W[1] * np.abs((A_y_vals - A_y_KM) / A_y_KM) ** 2)
        + np.sum(W[2] * np.abs((C_x_vals - C_x_KM) / C_x_KM) ** 2)
    )

    return V


def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """

    # Lists of candidate expressions... coefficients are optimized
    A_x_expr = params["A_x_expr"].copy()
    A_y_expr = params["A_y_expr"].copy()
    C_x_expr = params["C_x_expr"].copy()

    lib_A_x = params["lib_A_x"].copy()
    lib_A_y = params["lib_A_y"].copy()
    lib_C_x = params["lib_C_x"].copy()

    num_A_x = len(A_x_expr)
    num_A_y = len(A_y_expr)
    num_C_x = len(C_x_expr)

    Xi0 = params["Xi0"].copy()

    n_terms = num_A_x + num_A_y + num_C_x

    Xi = np.zeros((n_terms, n_terms - 2), dtype=Xi0.dtype)  # Output results
    cost_values = []  # All costs at each step
    V = np.full((n_terms - 2), np.inf)  # minimum cost at each step

    # Full regression problem as baseline
    Xi[:, 0], V_1 = opt_fun(params)  # usually huge
    V[0] = V_1
    cost_values.append([V_1])

    # Start with all candidates
    active = np.array([i for i in range(n_terms)])
    active_history = [active]

    # Iterate and threshold
    for k in range(1, n_terms - 2):
        # Loop through remaining terms and find the one that increases the cost function the least
        costs = []
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting j'th term

            # Break off masks for drift/diffusion
            A_x_active = tmp_active[tmp_active < num_A_x]
            A_y_active = (
                tmp_active[(tmp_active >= num_A_x) * (tmp_active < num_A_x + num_A_y)]
                - num_A_x
            )
            C_x_active = tmp_active[tmp_active >= num_A_x + num_A_y] - (
                num_A_x + num_A_y
            )
            params["A_x_expr"] = A_x_expr[A_x_active]
            params["A_y_expr"] = A_y_expr[A_y_active]
            params["C_x_expr"] = C_x_expr[C_x_active]
            params["lib_A_x"] = lib_A_x[A_x_active]
            params["lib_A_y"] = lib_A_y[A_y_active]
            params["lib_C_x"] = lib_C_x[C_x_active]
            params["Xi0"] = Xi0[tmp_active]  # Z: shouldn't this is found by LSTSQ?

            # Ensure that there is at least one drift x, one drift y, and one diffusion term left
            if len(A_x_active) > 0 and len(A_y_active) > 0 and len(C_x_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)
                costs.append(tmp_V)

                # Keep minimum cost
                if tmp_V < V[k]:
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi
        if not np.isfinite(V[k]):
            raise ValueError("Cost function returned NaN / INF for all iterates")
        cost_values.append(costs)

        print("Cost: {0}".format(V[k]))
        # Delete least important term
        active = np.delete(active, min_idx)  # type: ignore # Remove inactive index
        Xi0[active] = min_Xi  # type: ignore # Re-initialize with best results from previous
        Xi[active, k] = min_Xi  # type: ignore
        active_history.append(active)
        A_x_active = active[active < num_A_x]
        A_y_active = (
            active[(active >= num_A_x) * (active < num_A_x + num_A_y)] - num_A_x
        )
        C_x_active = active[active >= num_A_x + num_A_y] - (num_A_x + num_A_y)
        print(f"Active f x: {A_x_expr[A_x_active]}")
        print(f"Active f y: {A_y_expr[A_y_active]}")
        print(f"Active a x: {C_x_expr[C_x_active]}", flush=True)

    return Xi, V, active_history, cost_values


# Optimization parameters
weight = np.ones_like(pdf_masked)
weight /= np.nansum(weight)
W = np.array([weight, weight, weight])  # Weights from pdf values
params = {
    "W": W,  # (3, Q)
    "A_x_KM": A_f_km_masked,  # (Q,)
    "C_x_KM": C_f_km_masked,  # (Q,)
    "A_y_KM": A_s_km_masked,  # (Q,)
    "Xi0": Xi0,  # (n_A_f + n_A_s + n_C_f,)
    "A_x_expr": A_f_expr,  # (n_A_f,)
    "A_y_expr": A_s_expr,  # (n_A_s,)
    "C_x_expr": C_f_expr,  # (n_C_f,)
    "lib_A_x": lib_A_f,  # (n_A_f, Q)
    "lib_A_y": lib_A_s,  # (n_A_s, Q)
    "lib_C_x": lib_C_f,  # (n_C_f, Q)
}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: AFP_opt(cost, params)
Xi, V, active_history, cost_values = SSR_loop(opt_fun, params)

####################
# SSR cost function
####################

labels = [
    rf"${sympy.latex(t)}$" for t in np.concatenate((A_f_expr, A_s_expr, C_f_expr))
]

n_terms = len(labels)

fig_SSR, (ax_full_cost, ax_dcost, ax_history) = plt.subplots(nrows=3, figsize=(12, 20))
ax_full_cost: plt.Axes  # type: ignore
ax_dcost: plt.Axes  # type: ignore
ax_history: plt.Axes  # type: ignore
# ignore the first point as it is usually very large
for x, y in zip(np.arange(len(V))[1:], cost_values[1:]):
    for z in y:
        ax_full_cost.scatter(x, np.log(z), c="k", alpha=0.2)
ax_full_cost.scatter(np.arange(len(V))[1:], np.log(V)[1:], c="k")
ax_full_cost.set_xticks(np.arange(n_terms - 2))
ax_full_cost.set_xticklabels(np.arange(n_terms, 2, -1))
ax_full_cost.set_xlim(-0.5, n_terms - 2.5)
ax_full_cost.set_xlabel("Sparsity")
ax_full_cost.set_ylabel(r"Cost, $\log V$")

difs = np.log(V)[1:] - np.log(V)[:-1]
ax_dcost.scatter(np.arange(len(V))[:-1], difs, c="k")
ax_dcost.set_xticks(np.arange(n_terms - 2))
ax_dcost.set_xticklabels(np.arange(n_terms, 2, -1))
ax_dcost.set_xlim(-0.5, n_terms - 2.5)
bot = np.max(0, np.min(difs) * 0.9)
ax_dcost.set_ylim(bottom=0)
ax_dcost.set_xlabel("Sparsity")
ax_dcost.set_ylabel(r"dCost, $d \log V$")

square = np.zeros_like(Xi)
for i, hist in enumerate(active_history):
    square[hist, i] = 1
square = square.astype(bool)

# histories
ax_history.pcolor(square, cmap="bone_r", edgecolors="gray")
# drift / diffusion delimiters
ax_history.axhline(y=num_A_f, color="red")
ax_history.axhline(y=num_A_f + num_A_s, color="red")
ax_history.set_yticks(0.5 + np.arange(n_terms))
ax_history.set_yticklabels(labels)
ax_history.set_xticks(0.5 + np.arange(n_terms - 2))
ax_history.set_xticklabels(np.arange(n_terms, 2, -1))
ax_history.set_xlabel("Sparsity")
ax_history.set_ylabel("Active terms")

fig_SSR.tight_layout()
fig_SSR.savefig(folder_path / "SSR_sparsity_log_f_s.png")
plt.close(fig_SSR)

n_terms = len(labels)

fig_SSR_reduced, (ax_full_cost, ax_dcost, ax_history) = plt.subplots(
    nrows=3, figsize=(12, 20)
)
ax_full_cost: plt.Axes  # type: ignore
ax_dcost: plt.Axes  # type: ignore
ax_history: plt.Axes  # type: ignore
# ignore the first point as it is usually very large
skip = 10
for x, y in zip(np.arange(len(V))[skip:], cost_values[skip:]):
    for z in y:
        ax_full_cost.scatter(x, np.log(z), c="k", alpha=0.2)
ax_full_cost.scatter(np.arange(len(V))[skip:], np.log(V)[skip:], c="k")
ax_full_cost.set_xticks(np.arange(n_terms - 2))
ax_full_cost.set_xticklabels(np.arange(n_terms, 2, -1))
ax_full_cost.set_xlim(-0.5, n_terms - 2.5)
ax_full_cost.set_xlabel("Sparsity")
ax_full_cost.set_ylabel(r"Cost, $\log V$")

ax_dcost.scatter(np.arange(len(V))[skip:], np.log(V)[skip:], c="k")
ax_dcost.set_xticks(np.arange(n_terms - 2))
ax_dcost.set_xticklabels(np.arange(n_terms, 2, -1))
ax_dcost.set_xlim(-0.5, n_terms - 2.5)
ax_dcost.set_xlabel("Sparsity")
ax_dcost.set_ylabel(r"Cost, $\log V$")

square = np.zeros_like(Xi)
for i, hist in enumerate(active_history):
    square[hist, i] = 1
square = square.astype(bool)

# histories
ax_history.pcolor(square, cmap="bone_r", edgecolors="gray")
# drift / diffusion delimiters
ax_history.axhline(y=num_A_f, color="red")
ax_history.axhline(y=num_A_f + num_A_s, color="red")
ax_history.set_yticks(0.5 + np.arange(n_terms))
ax_history.set_yticklabels(labels)
ax_history.set_xticks(0.5 + np.arange(n_terms - 2))
ax_history.set_xticklabels(np.arange(n_terms, 2, -1))
ax_history.set_xlabel("Sparsity")
ax_history.set_ylabel("Active terms")

fig_SSR_reduced.tight_layout()
fig_SSR_reduced.savefig(folder_path / "SSR_sparsity_log_f_s.png")
plt.close(fig_SSR_reduced)

# Select model with the fewest terms before the cost function spikes
dlogV = np.log(V[1:]) - np.log(V[:-1])
model_selected = np.argmax(dlogV)  # take the first accepted spike
print(f"Number of terms: {n_terms - model_selected}")
print("With KL regularisation")
print(f"Xi = {Xi[:, model_selected]}")
print(f"Cost = {V[model_selected]:.1e}")

Xi_A_f = Xi[:num_A_f, model_selected]
Xi_A_s = Xi[num_A_f : num_A_f + num_A_s, model_selected]
Xi_C_f = Xi[num_A_f + num_A_s :, model_selected]

# Functions from the expressions
A_f_sym = utils.sindy_model(Xi_A_f, A_f_expr)
A_f_sindy = sympy.lambdify((f_sym, s_sym), A_f_sym)
A_s_sym = utils.sindy_model(Xi_A_s, A_s_expr)
A_s_sindy = sympy.lambdify((f_sym, s_sym), A_s_sym)
C_f_sym = utils.sindy_model(Xi_C_f, C_f_expr)
C_f_sindy = sympy.lambdify((f_sym, s_sym), C_f_sym)

print(f"df = ({A_f_sym}) dt + ({sympy.sqrt(2*C_f_sym)}) dbeta")
print(f"dσ = ({A_s_sym}) dt", flush=True)

A_f_vals = A_f_sindy(f_mesh, s_mesh)
A_s_vals = A_s_sindy(f_mesh, s_mesh)
C_f_vals = C_f_sindy(f_mesh, s_mesh)

# Check if a scalar (happens when library is a constant)
if np.ndim(A_f_vals) == 0:
    A_f_vals = A_f_vals + 0 * f_mesh
if np.ndim(A_s_vals) == 0:
    A_s_vals = A_s_vals + 0 * f_mesh
if np.ndim(C_f_vals) == 0:
    C_f_vals = C_f_vals + 0 * f_mesh

A_f_km = A_f_km.reshape(N, M)
C_f_km = C_f_km.reshape(N, M)
A_s_km = A_s_km.reshape(N, M)

fig_km_vals, axes_km_vals = plt.subplots(
    nrows=2, ncols=2, figsize=(12, 12), subplot_kw={"projection": "3d"}
)
axes_km_vals: list[list[Axes3D]]
# axes_km_vals[0][0].plot_wireframe(centers_f, centers_s, A_f_km, label="KM")
axes_km_vals[0][0].plot_wireframe(f_mesh, s_mesh, A_f_vals)  # , label="SINDy")
axes_km_vals[0][0].set_zlabel(r"$A_{f}(f, \sigma)$")

# axes_km_vals[0][1].plot_wireframe(centers_f, centers_s, A_s_km, label="KM")
axes_km_vals[0][1].plot_wireframe(f_mesh, s_mesh, A_s_vals)  # , label="SINDy")
axes_km_vals[0][1].set_zlabel(r"$A_{\sigma}(f, \sigma)$")

# axes_km_vals[1][0].plot_wireframe(centers_f, centers_s, C_f_km, label="KM")
axes_km_vals[1][0].plot_wireframe(f_mesh, s_mesh, C_f_vals)  # , label="SINDy")
axes_km_vals[1][0].set_zlabel(r"$C_{f}(f, \sigma) = B_{f}^2 / 2$")

# axes_km_vals[1][1].plot_wireframe(centers_f, centers_s, C_s_km, label="KM")
axes_km_vals[1][1].plot_wireframe(
    f_mesh, s_mesh, np.zeros_like(f_mesh)
)  # , label="SINDy")
axes_km_vals[1][1].set_zlabel(r"$C_{\sigma}(f, \sigma) = B_{\sigma}^2 / 2$")

for a in axes_km_vals:
    for b in a:
        # b.legend()
        b.set_xlabel(r"$f$")
        b.set_ylabel(r"$\sigma$")

# fig_km_vals.tight_layout()
fig_km_vals.savefig(folder_path / "final_graph_log_f_s.png")

for a in axes_km_vals:
    for b in a:
        b.set_xscale("log")

# fig_km_vals.tight_layout()
fig_km_vals.savefig(folder_path / "final_graph_log_f_s_logscale.png")
plt.close(fig_km_vals)
