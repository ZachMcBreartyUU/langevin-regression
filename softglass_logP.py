"""Code adapted from pitchfork.ipynb, utils.py, fpsolve.py by Jared Callaham
This work done by Zach McBrearty

NOTE: The size of the libraries scales as (N_bins x N_functions), so the initial size of the dataset only matters for the initial dataplotting (optional) and then the binning / moment generation (Kramers Moyal KDE).
The size of the libraries matters _greatly_ as they are constantly inverted, this size can be controlled by the -N or --num-bins flag
"""

import argparse
import json

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import sympy

import utils
import fpsolve
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

# A library:      [  1, f^1/2,     f^2/2, f^3/2, f^4/2, f^5/2]
# expected coeff: [ep0,     0, -2R + ep1,     0,    2m,    -2]
A_f_expr = np.array([f_sym ** (i / 2) for i in np.arange(6)])
print("A_f library:", A_f_expr)
num_A_f = len(A_f_expr)

# a library:      [1,   f, f^2]
# expected coeff: [0, ep0, ep1]
C_f_expr = np.array([f_sym ** (i) for i in np.arange(0, 3)])
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
Xi0[:num_A_f] = lstsq(lib_A_f[:, mask].T, A_km[mask], rcond=None)[0]
Xi0[num_A_f:] = lstsq(lib_C_f[:, mask].T, C_km[mask], rcond=None)[0]
print("Xi0 =", Xi0)

# Initialize adjoint solver
afp = fpsolve.AdjFP(centers)

# Initialize forward steady-state solver
dx = widths
fp = fpsolve.SteadyFP(N, dx)

# Optimization parameters
weight = np.ones_like(pdf)  # 1 / pdf
weight /= np.nansum(weight)
W = np.array([weight, weight])  # Weights from pdf values
params = {
    "W": W,
    "f_KM": A_km,
    "a_KM": C_km,
    "Xi0": Xi0,
    "f_expr": A_f_expr,  # dictionary keys are legacy
    "a_expr": C_f_expr,  # dictionary keys are legacy
    "lib_f": lib_A_f.T,  # dictionary keys are legacy
    "lib_a": lib_C_f.T,  # dictionary keys are legacy
    "N": N,
    "kl_reg": args.kl_reg,
    "fp": fp,
    "afp": afp,
    "p_hist": pdf,
    "tau": dt,
    "radial": False,
}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: utils.AFP_opt(utils.cost, params)
Xi, V, active_history = utils.SSR_loop(opt_fun, params)

####################
# SSR cost function
####################

labels = [r"${0}$".format(sympy.latex(t)) for t in np.concatenate((A_f_expr, C_f_expr))]

n_terms = len(labels)

fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(6, 8))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
ax.scatter(np.arange(len(V))[1:], np.log(V)[1:], c="k")
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

if args.kl_reg > 0:
    params_cpy = params.copy()
    params_cpy["f_expr"] = A_f_expr[f_active]
    params_cpy["a_expr"] = C_f_expr[a_active]
    params_cpy["lib_f"] = lib_A_f.T[:, f_active]
    params_cpy["lib_a"] = lib_C_f.T[:, a_active]
    params_cpy["Xi0"] = Xi0[active]
    params_cpy["kl_reg"] = 0

    unreg_Xi, unreg_cost = opt_fun(params_cpy)
    unreg_Xi_expanded = np.zeros_like(Xi0)
    unreg_Xi_expanded[active] = unreg_Xi
    print("Without KL regularisation")
    print(f"Xi = {unreg_Xi_expanded}")
    print(f"Cost = {unreg_cost:.1e}", flush=True)

    Xi_f = unreg_Xi_expanded[:num_A_f]
    Xi_a = unreg_Xi_expanded[num_A_f:]
else:
    Xi_f = Xi[:num_A_f, 1 - n_terms_selected]
    Xi_a = Xi[num_A_f:, 1 - n_terms_selected]
# Functions from the expressions
A_sym = utils.sindy_model(Xi_f, A_f_expr)
A_sindy = sympy.lambdify(f_sym, A_sym)
C_sym = utils.sindy_model(Xi_a, C_f_expr)
C_sindy = sympy.lambdify(f_sym, C_sym)

print(f"df = ({A_sym}) dt + ({sympy.sqrt(2*C_sym)}) dbeta")

A_vals = A_sindy(centers)
C_vals = C_sindy(centers)

# Check if a scalar (happens when library is a constant)
if np.ndim(A_vals) == 0:
    A_vals = A_vals + 0 * centers
if np.ndim(C_vals) == 0:
    C_vals = C_vals + 0 * centers

# Compare PDFs: empirical vs Fokker-Planck solution with model
# p_fit = fp.solve(A_vals, C_vals)
# kl_div_val = utils.kl_divergence(pdf, p_fit, dx=dx, tol=1e-6)
# print(f"KL divergence: {kl_div_val:.1e}")

# fig, ax = plt.subplots(figsize=(10, 10))
# ax: plt.Axes  # type: ignore
# ax.plot(centers, pdf, "k", label="Data", lw=3)
# ax.plot(centers, p_fit, "--", c="r", label="Model", lw=3)
# ax.legend()
# ax.set_xlabel(r"$f$")
# ax.set_ylabel(r"$P(f)$")

# fig.savefig(folder_path / "pdf_comparision_log_f.png")

# afp.precompute_operator(A_vals, C_vals)
# q = afp.solve(dt)
# if q is None:
#     raise RuntimeError("Failed to solve adjoint focker planck system")
# f_tau, a_tau = q

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
# ax.plot(
#     centers, A(centers, sigma_min), c="gray", lw=1, alpha=0.5, label="True: min sigma"
# )
# ax.plot(
#     centers, A(centers, sigma_max), c="gray", lw=1, alpha=0.5, label="True: max sigma"
# )
# ax.plot(
#     centers,
#     A(centers, sigma_avg + sigma_std),
#     c="gray",
#     lw=2,
#     alpha=0.5,
#     label="True: +1 std",
# )
# ax.plot(
#     centers,
#     A(centers, sigma_avg - sigma_std),
#     c="gray",
#     lw=2,
#     alpha=0.5,
#     label="True: -1 std",
# )
# ax.plot(centers, A(centers, sigma_avg), c="gray", lw=2, label="True: average")
ax.plot(centers, A_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax.plot(centers, A_vals, "r", lw=2, label="SINDy")
# ax.plot(centers, f_tau, "g:", lw=2, label=rf"$\tau = {dt}$")
ax.legend()
ax.set_title("Drift")
ax.set_xlabel(r"$f$")
ax.set_ylabel(r"$A(f)$")

ax2.plot(centers, B(centers, sigma_avg) ** 2 / 2, c="gray", lw=2, label="True diff")
ax2.plot(centers, C_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax2.plot(centers, C_vals, "r", lw=2, label="SINDy")
# ax2.plot(centers, a_tau, "g:", lw=2, label=rf"$\tau={dt}$")
ax2.legend()
ax2.set_title("Diffusion")
ax2.set_xlabel(r"$f$")
ax2.set_ylabel(r"$C(f) = B^2(f) / 2$")

fig.tight_layout()
fig.savefig(folder_path / "final_graph_f.png")

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
ax.plot(centers, A(centers, sigma_avg), c="gray", lw=2, label="True drift")
ax.plot(centers, A_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax.plot(centers, A_vals, "r", lw=2, label="SINDy")
# ax.plot(centers, f_tau, "g:", lw=2, label=rf"$\tau = {dt}$")
ax.legend()
ax.set_title("Drift")
ax.set_xlabel(r"$f$")
ax.set_ylabel(r"$A(f)$")

ax2.plot(centers, B(centers, sigma_avg) ** 2 / 2, c="gray", lw=2, label="True diff")
ax2.plot(centers, C_km, ls="", marker=".", markersize=8, c="b", label="KM")
ax2.plot(centers, C_vals, "r", lw=2, label="SINDy")
# ax2.plot(centers, a_tau, "g:", lw=2, label=rf"$\tau={dt}$")
ax2.legend()
ax2.set_title("Diffusion")
ax2.set_xlabel(r"$f$")
ax2.set_ylabel(r"$C(f) = B^2(f) / 2$")

ax.set_xscale("log")
ax2.set_xscale("log")

fig.tight_layout()
fig.savefig(folder_path / "final_graph_log_f.png")
