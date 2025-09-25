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
from mpl_toolkits.axisartist import Axes
import sympy
from scipy.optimize import minimize

from utils import sindy_model
import data_loader as dl

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--kl-reg", type=float, default=1)
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
m = lambda y: (y - 1) / np.sqrt(y) if y > 1 else 0.0
A = lambda x, y: -2 * R * x + 2 * (m(y) - np.sqrt(x)) * x**2 + ep0 + ep1 * x
B = lambda x, y: 2 * np.sqrt(ep0 * x + ep1 * x**2)

with np.load(folder_path / "km_log_spacing.npz") as file:
    pdf = file["pdf"]
    A_km = file["moment1"]
    C_km = file["moment2"]
    edges = file["edges"]
    centers = file["centers"]
    widths = file["widths"]
    sigma_data = file["sigma_data"]
sigma_min, sigma_max, sigma_avg, sigma_std = sigma_data

N = len(centers)

if args.plot_intermediate:
    fig, axes = plt.subplots(2, figsize=(12, 12))
    axes: list[plt.Axes]  # type: ignore
    axes[0].plot(centers, pdf)
    axes[0].set_ylabel(r"PDF($f$)")
    axes[0].set_xlabel(r"fluidity, $f$")

    axes[1].plot(np.log(centers), np.log(pdf))
    axes[1].set_ylabel(r"log PDF($f$)")
    axes[1].set_xlabel(r"log fluidity, $\log f$")

    # axes[1].plot(centers, A_km, label=rf"$\tau={dt}$")
    # axes[1].plot(centers, A(centers, sigma_avg), label="$A(f)$")
    # axes[1].set_ylabel(r"First moment, $m^{(1)}(f)$")
    # axes[1].set_xlabel(r"fluidity, $f$")
    # axes[1].legend()

    # axes[2].plot(centers, C_km, label=rf"$\tau={dt}$")
    # axes[2].plot(centers, B(centers, sigma_avg) ** 2 / 2, label="$B(f)^2/2$")
    # axes[2].set_ylabel(r"Second moment, $m^{(2)}(f)$")
    # axes[2].set_xlabel(r"fluidity, $f$")
    # axes[2].legend()

    fig.tight_layout()

    fig.savefig(folder_path / "pdf_moments_log_f.png")

### Build SINDy libraries with sympy
f_sym = sympy.symbols("f")

# A library:      [  1, f^1/2,     f^2/2, f^3/2, f^4/2, f^5/2, f^6/2, f^7/2]
# expected coeff: [ep0,     0, -2R + ep1,     0,    2m,    -2,     0,     0]
A_f_expr = np.array([f_sym ** (i / 2) for i in range(8)])
print("A_f library:", A_f_expr)
num_A_f = len(A_f_expr)

# a library:      [1,   f, f^2, f^3, f^4]
# expected coeff: [0, ep0, ep1,   0,   0]
C_f_expr = np.array([f_sym ** (i) for i in range(5)])
print("a_f library:", C_f_expr)
num_C_f = len(C_f_expr)

# Convert sympy expressions into library matrices
lib_A_f = np.empty([num_A_f, N])
for k in range(num_A_f):
    lamb_expr = sympy.lambdify(f_sym, A_f_expr[k])
    lib_A_f[k] = lamb_expr(centers)

lib_C_f = np.empty([num_C_f, N])
for k in range(num_C_f):
    lamb_expr = sympy.lambdify(f_sym, C_f_expr[k])
    lib_C_f[k] = lamb_expr(centers)

# Initialize Xi with least squares regression (no finite-time corrections)
Xi0 = np.empty((num_A_f + num_C_f))
mask = np.nonzero(np.isfinite(A_km))[0]  # Z: why only mask on f_KM?
Xi0[:num_A_f] = lstsq(lib_A_f[:, mask].T, A_km[mask])[0]
Xi0[num_A_f:] = lstsq(lib_C_f[:, mask].T, C_km[mask])[0]
print("Xi0 =", Xi0)

# Optimization parameters
weight = np.ones_like(pdf)
weight /= np.nansum(weight)
W = np.array([weight, weight])
params = {
    "W": W,
    "f_KM": A_km,
    "a_KM": C_km,
    "Xi0": Xi0,
    "f_expr": A_f_expr,
    "a_expr": C_f_expr,
    "lib_f": lib_A_f,
    "lib_a": lib_C_f,
}


def cost(Xi, params):
    """
    Least-squares cost function for optimization
    This version is only good in 1D, but could be extended pretty easily
    Xi - current coefficient estimates
    param - inputs to optimization problem: grid points, list of candidate expressions, regularizations
        W, f_KM, a_KM, x_pts, y_pts, x_msh, y_msh, f_expr, a_expr, l1_reg, l2_reg, kl_reg, p_hist, etc
    """

    # Unpack parameters
    W = params["W"]  # Optimization weights

    # Kramers-Moyal coefficients
    A_KM = params["f_KM"]
    C_KM = params["a_KM"]

    lib_A = params["lib_f"]
    lib_C = params["lib_a"]

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    A_vals = lib_A.T @ Xi[: lib_A.shape[0]]
    C_vals = lib_C.T @ Xi[lib_A.shape[0] :]

    # Histogram points without data have NaN values in K-M average - ignore these in the average
    V = np.nansum(W[0] * np.abs((A_vals - A_KM) / A_KM) ** 2) + np.nansum(
        W[1] * np.abs((C_vals - C_KM) / C_KM) ** 2
    )

    return V


def AFP_opt(cost, params):
    ### RUN OPTIMIZATION PROBLEM
    start_time = time()
    Xi0 = params["Xi0"]

    opt_fun = lambda Xi: cost(Xi, params)

    res = minimize(
        opt_fun, Xi0, method="nelder-mead", options={"disp": False, "maxfev": int(1e4)}
    )
    print(
        "%%%% Optimization time: {0} seconds,   Cost: {1} %%%%".format(
            time() - start_time, res.fun
        )
    )
    # found constants, cost
    return res.x, res.fun


def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """

    # Lists of candidate expressions... coefficients are optimized
    f_expr = params["f_expr"].copy()
    a_expr = params["a_expr"].copy()
    lib_f = params["lib_f"].copy()
    lib_a = params["lib_a"].copy()
    Xi0 = params["Xi0"].copy()

    n_terms = len(f_expr) + len(a_expr)

    Xi = np.zeros((n_terms, n_terms - 1), dtype=Xi0.dtype)  # Output results
    V = np.full((n_terms - 1), np.inf)  # Cost at each step

    # Full regression problem as baseline
    Xi[:, 0], V[0] = opt_fun(params)

    # Start with all candidates
    active = np.array([i for i in range(n_terms)])
    active_history = [active]
    cost_values = [[V[0]]]

    # Iterate and threshold
    for k in range(1, n_terms - 1):
        # Loop through remaining terms and find the one that increases the cost function the least
        min_idx = -1
        costs = []
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term

            # Break off masks for drift/diffusion
            f_active = tmp_active[tmp_active < len(f_expr)]
            a_active = tmp_active[tmp_active >= len(f_expr)] - len(f_expr)

            params["f_expr"] = f_expr[f_active]
            params["a_expr"] = a_expr[a_active]
            params["lib_f"] = lib_f[f_active]
            params["lib_a"] = lib_a[a_active]
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
        cost_values.append(costs)

        print("Cost: {0}".format(V[k]))
        # Delete least important term
        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # type: ignore # Re-initialize with best results from previous
        Xi[active, k] = min_Xi  # type: ignore
        active_history.append(active)
        f_active = active[active < len(f_expr)]
        a_active = active[active >= len(f_expr)] - len(f_expr)
        print(f"Active f: {f_expr[f_active]}")
        print(f"Active a: {a_expr[a_active]}", flush=True)

    return Xi, V, active_history, cost_values


# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: AFP_opt(cost, params)
Xi, V, active_history, cost_values = SSR_loop(opt_fun, params)

####################
# SSR cost function
####################

labels = [f"${sympy.latex(t)}$" for t in np.concatenate((A_f_expr, C_f_expr))]

n_terms = len(labels)

fig, (ax_all_costs, ax, ax2) = plt.subplots(nrows=3, figsize=(6, 12))
ax_all_costs: Axes
ax: Axes
ax2: Axes
skip = 0
for x, y in zip(np.arange(len(V))[skip:], cost_values[skip:]):
    for z in y:
        ax_all_costs.scatter(x, np.log(z), c="k", alpha=0.2)
ax_all_costs.scatter(np.arange(len(V))[skip:], np.log(V)[skip:], c="k", marker="x")
ax_all_costs.set_xticks(np.arange(n_terms - 1))
ax_all_costs.set_xticklabels(np.arange(n_terms, 1, -1))
ax_all_costs.set_xlim(-0.5, n_terms - 1.5)
ax_all_costs.set_xlabel("Sparsity")
ax_all_costs.set_ylabel(r"Cost, $\log V$")

ax.scatter(np.arange(len(V))[skip:], np.log(V)[skip:], c="k")
ax.set_xticks(np.arange(n_terms - 1))
ax.set_xticklabels(np.arange(n_terms, 1, -1))
ax.set_xlim(-0.5, n_terms - 1.5)
ax.set_xlabel("Sparsity")
ax.set_ylabel(r"Cost, $\log V$")

square = np.zeros_like(Xi)
for i, hist in enumerate(active_history):
    square[hist, i] = 1
square = square.astype(bool)

ax2.pcolor(square, cmap="bone_r", edgecolors="gray")
ax2.axhline(y=num_A_f, color="red")
ax2.set_yticks(0.5 + np.arange(n_terms))
ax2.set_yticklabels(labels)
ax2.set_xticks(0.5 + np.arange(n_terms - 1))
ax2.set_xticklabels(np.arange(n_terms, 1, -1))
ax2.set_xlabel("Sparsity")
ax2.set_ylabel("Active terms")

fig.tight_layout()
fig.savefig(folder_path / "SSR_sparsity_log_f.png")

# Select model with the fewest terms before the cost function spikes
threshold = 1e6
n_terms_selected = np.arange(n_terms, 1, -1)[
    np.nonzero((V[1:] - V[:-1]) > threshold * V[:-1])[0]
]
while n_terms_selected.size == 0:
    threshold /= 10
    if threshold <= 1:
        print(
            f"WARNING: threshold dropped to {threshold} before finding jump", flush=True
        )
    n_terms_selected = np.arange(n_terms, 1, -1)[
        np.nonzero((V[1:] - V[:-1]) > threshold * V[:-1])[0]
    ]
n_terms_selected = n_terms_selected[0]  # take the first accepted spike
print(f"n_terms_selected={n_terms_selected}")
print("With KL regularisation")
print(f"Xi = {Xi[:, 1 - n_terms_selected]}")
print(f"Cost = {V[1 - n_terms_selected]:.1e}")

active = active_history[1 - n_terms_selected]
f_active = active[active < num_A_f]
a_active = active[active >= num_A_f] - num_A_f
Xi_f = Xi[:num_A_f, 1 - n_terms_selected]
Xi_a = Xi[num_A_f:, 1 - n_terms_selected]
# Functions from the expressions
A_sym = sindy_model(Xi_f, A_f_expr)
A_sindy = sympy.lambdify(f_sym, A_sym)
C_sym = sindy_model(Xi_a, C_f_expr)
C_sindy = sympy.lambdify(f_sym, C_sym)

print(f"df = ({A_sym}) dt + ({sympy.sqrt(2*C_sym)}) dbeta")

A_vals = A_sindy(centers)
C_vals = C_sindy(centers)

# Check if a scalar (happens when library is a constant)
if np.ndim(A_vals) == 0:
    A_vals = A_vals + 0 * centers
if np.ndim(C_vals) == 0:
    C_vals = C_vals + 0 * centers

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax: Axes
ax2: Axes
ax.plot(centers, A(centers, sigma_avg), c="gray", lw=2, label="True: average")
ax.plot(centers, A_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax.plot(centers, A_vals, "r", lw=2, label="SINDy")
ax.legend()
ax.set_title("Drift")
ax.set_xlabel(r"$f$")
ax.set_ylabel(r"$A(f)$")

ax2.plot(centers, B(centers, sigma_avg) ** 2 / 2, c="gray", lw=2, label="True diff")
ax2.plot(centers, C_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax2.plot(centers, C_vals, "r", lw=2, label="SINDy")
ax2.legend()
ax2.set_title("Diffusion")
ax2.set_xlabel(r"$f$")
ax2.set_ylabel(r"$C(f) = B^2(f) / 2$")

fig.tight_layout()
fig.savefig(folder_path / "final_graph_f.png")
print(f"Saved fig: {folder_path / "final_graph_f.png"}")

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax: Axes
ax2: Axes
ax.plot(centers, A(centers, sigma_avg), c="gray", lw=2, label="True drift")
ax.plot(centers, A_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax.plot(centers, A_vals, "r", lw=2, label="SINDy")
ax.legend()
ax.set_title("Drift")
ax.set_xlabel(r"$f$")
ax.set_ylabel(r"$A(f)$")

ax2.plot(centers, B(centers, sigma_avg) ** 2 / 2, c="gray", lw=2, label="True diff")
ax2.plot(centers, C_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax2.plot(centers, C_vals, "r", lw=2, label="SINDy")
ax2.legend()
ax2.set_title("Diffusion")
ax2.set_xlabel(r"$f$")
ax2.set_ylabel(r"$C(f) = B^2(f) / 2$")

ax.set_xscale("log")
ax2.set_xscale("log")

fig.tight_layout()
fig.savefig(folder_path / "final_graph_log_f.png")
print(f"Saved fig: {folder_path / "final_graph_log_f.png"}")
