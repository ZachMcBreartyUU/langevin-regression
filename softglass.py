"""Code adapted from pitchfork.ipynb, utils.py, fpsolve.py by Jared Callaham
This work done by Zach McBrearty

NOTE: The size of the libraries scales as (N_bins x N_functions), so the initial size of the dataset only matters for the initial dataplotting (optional) and then the binning / moment generation (Kramers Moyal KDE).
The size of the libraries matters _greatly_ as they are constantly inverted, this size can be controlled by the -N or --num-bins flag
"""

import argparse

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import sympy
from kramersmoyal import km

import utils
import fpsolve
import data_loader as dl

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("-c", "--cutoff", type=float, default=250.0)
ap.add_argument("--kl-reg", type=float, default=10.0)
ap.add_argument("-N", "--num-bins", type=int, default=100)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=-1)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--plot-intermediate", action="store_true")
args = ap.parse_args()
print(args)

folder_path = dl.SCRATCH_PATH / args.folder
print(folder_path)

metadata, times, phi, sigma = dl.get_data(
    folder_path, start=args.start, stop=args.stop, step=args.step
)

dt = metadata["dt"] * args.step
R = metadata["R"]
ep0 = metadata["epsilon_0"]
ep1 = metadata["epsilon_1"]
lb = metadata["lb"]
m = lambda y: (y - 1) / np.sqrt(y) if y > 1 else 0.0
f = lambda x, y: -R * x + (m(y) - np.abs(x)) * x**3
g = lambda x, y: np.sqrt(ep0 + ep1 * x**2)

sigma_avg = np.average(sigma)

# Plot solution
if args.plot_intermediate:
    fig, (ax, ax2) = plt.subplots(2)
    # type checking for ax, though matplotlib technically
    # doesn't export Axes, so include the type ignore
    ax: plt.Axes  # type: ignore
    ax2: plt.Axes  # type: ignore
    ax.plot(times, phi)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"Phi, $\phi$")

    ax2.plot(times, sigma)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"Stress, $\sigma$")

    fig.tight_layout()

    fig.savefig(folder_path / "system_graph_just_phi.png")


def _km_find_range(data: np.ndarray, pdf_cutoff: float, N: int = 100) -> np.ndarray:
    """data: data to find the pdf range for
    pdf_cutoff: data smaller than max(pdf) / pdf_cutoff will be ignored, and thus the range shortened
    N: number of datapoints in the range"""
    kmc, centers = km(data, bins=N, powers=0)  # type: ignore
    pdf: np.ndarray = kmc[0]
    centers: np.ndarray = centers[0]
    threshold = pdf.max() / pdf_cutoff
    # find point where pdf_phi passes above the threshold for the first time
    ascending = ((pdf[1:] > threshold) * (pdf[:-1] < threshold)).nonzero()[0]
    if ascending.size > 0:
        min_ = centers[ascending[0]]
    else:
        min_ = centers[0]
    # find point where pdf_phi passes below the threshold for the last time
    descending = ((pdf[1:] < threshold) * (pdf[:-1] > threshold)).nonzero()[0]
    if descending.size > 0:
        max_ = centers[descending[-1]]
    else:
        max_ = centers[-1]

    return np.linspace(min_, max_, N)


def km_avg(
    data: np.ndarray,
    tau: float,
    pdf_cutoff: float = np.e**3,
    N: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_range = _km_find_range(data, pdf_cutoff, N)
    kmc, centers = km(data, bins=(phi_range,), powers=2)  # type: ignore

    pdf = kmc[0]
    pdf /= np.sum(pdf)

    centers = centers[0]
    moment1, moment2 = kmc[1:] / tau
    return centers, pdf, moment1, moment2


# TODO: KM could be saved in a temp file and recovered to skip the data processing step
centers, pdf, f_KM, a_KM = km_avg(phi, tau=dt, pdf_cutoff=args.cutoff, N=args.num_bins)
del times, phi, sigma  # large arrays which are no longer needed

N = len(centers)  # == args.num_bins

if args.plot_intermediate:
    fig, axes = plt.subplots(3, figsize=(12, 12))
    axes: list[plt.Axes]  # type: ignore
    axes[0].plot(centers, pdf, label=rf"$\tau={dt}$")
    axes[0].set_ylabel(r"PDF($\phi$)")
    axes[0].set_xlabel(r"Phi, $\phi$")
    axes[0].legend()

    axes[1].plot(centers, f_KM, label=rf"$\tau={dt}$")
    axes[1].plot(centers, f(centers, sigma_avg), label="$A(f)$")
    axes[1].set_ylabel(r"First moment, $m^{(1)}(\phi)$")
    axes[1].set_xlabel(r"Phi, $\phi$")
    axes[1].legend()

    axes[2].plot(centers, a_KM, label=rf"$\tau={dt}$")
    axes[2].plot(centers, g(centers, sigma_avg) ** 2 / 2, label="$B(f)^2/2$")
    axes[2].set_ylabel(r"Second moment, $m^{(2)}(\phi)$")
    axes[2].set_xlabel(r"Phi, $\phi$")
    axes[2].legend()

    fig.tight_layout()

    fig.savefig(folder_path / "pdf_moments_just_phi.png")

### Build SINDy libraries with sympy
phi_sym = sympy.symbols("phi")

# f library:      [1,  φ,  φ^3, φ^5, φ|φ|, φ^3|φ|]
# expected coeff: [0, -R, m(σ),   0,    0,      1]
f_phi_expr = np.array(
    [phi_sym**0]
    + [phi_sym ** (2 * i + 1) for i in np.arange(0, 3)]
    + [sympy.Abs(phi_sym) * phi_sym ** (2 * i + 1) for i in np.arange(0, 2)]
)
print("f_phi library:", f_phi_expr)
num_f_phi = len(f_phi_expr)

# a library:      [  1, φ^2, φ^4]
# expected coeff: [ep0, ep1,   0]
a_phi_expr = np.array([phi_sym ** (2 * i) for i in np.arange(0, 3)])
print("a_phi library:", a_phi_expr)
num_a_phi = len(a_phi_expr)

# Convert sympy expressions into library matrices
lib_f_phi = np.empty([num_f_phi, N])
for k in range(num_f_phi):
    lamb_expr = sympy.lambdify(phi_sym, f_phi_expr[k])
    lib_f_phi[k] = lamb_expr(centers)

lib_a_phi = np.empty([num_a_phi, N])
for k in range(num_a_phi):
    lamb_expr = sympy.lambdify(phi_sym, a_phi_expr[k])
    lib_a_phi[k] = lamb_expr(centers)

# Initialize Xi with least squares regression (no finite-time corrections)
Xi0 = np.empty((num_f_phi + num_a_phi))
mask = np.nonzero(np.isfinite(f_KM))[0]  # Z: why only mask on f_KM?
Xi0[:num_f_phi] = lstsq(lib_f_phi[:, mask].T, f_KM[mask], rcond=None)[0]
Xi0[num_f_phi:] = lstsq(lib_a_phi[:, mask].T, a_KM[mask], rcond=None)[0]
print("Xi0 =", Xi0)

# Initialize adjoint solver
afp = fpsolve.AdjFP(centers)

# Initialize forward steady-state solver
dx = centers[1] - centers[0]
fp = fpsolve.SteadyFP(N, dx)

# Optimization parameters
weight = 1 / pdf
weight /= np.nansum(weight)
W = np.array([weight, weight])  # Weights from pdf values
params = {
    "W": W,
    "f_KM": f_KM,
    "a_KM": a_KM,
    "Xi0": Xi0,
    "f_expr": f_phi_expr,
    "a_expr": a_phi_expr,
    "lib_f": lib_f_phi.T,
    "lib_a": lib_a_phi.T,
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

labels = [
    r"${0}$".format(sympy.latex(t)) for t in np.concatenate((f_phi_expr, a_phi_expr))
]

n_terms = len(labels)

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
ax.scatter(np.arange(len(V)), V, c="k")
ax.set_xticks(np.arange(n_terms - 1))
ax.set_xticklabels(np.arange(n_terms, 1, -1))
ax.set_xlabel("Sparsity")
ax.set_ylabel(r"Cost")

square = np.zeros_like(Xi)
for i, hist in enumerate(active_history):
    square[hist, i] = 1
square = square.astype(bool)

ax2.pcolor(square, cmap="bone_r", edgecolors="gray")
ax2.axhline(y=num_f_phi, color="red")
ax2.set_yticks(0.5 + np.arange(n_terms))
ax2.set_yticklabels(labels)
ax2.set_xticks(0.5 + np.arange(n_terms - 1))
ax2.set_xticklabels(np.arange(n_terms, 1, -1))
ax2.set_xlabel("Sparsity")
ax2.set_ylabel("Active terms")

fig.savefig(folder_path / "SSR_sparsity_just_phi.png")

# Select model with the fewest terms before the cost function spikes
threshold = 1e6
n_terms_selected = np.arange(n_terms, 1, -1)[
    np.nonzero(V[1:] - V[:-1] > threshold * V[:-1])[0]
]
while n_terms_selected.size == 0:
    threshold /= 10
    if threshold <= 1:
        print(
            f"WARNING: threshold dropped to {threshold} before finding jump", flush=True
        )
    n_terms_selected = np.arange(n_terms, 1, -1)[
        np.nonzero(V[1:] - V[:-1] > threshold * V[:-1])[0]
    ]
n_terms_selected = n_terms_selected[0]  # take the first accepted spike
print(f"n_terms_selected={n_terms_selected}")
print("With KL regularisation")
print(f"Xi = {Xi[:, 1 - n_terms_selected]}")
print(f"Cost = {V[1 - n_terms_selected]:.1e}")

active = active_history[1 - n_terms_selected]
f_active = active[active < num_f_phi]
a_active = active[active >= num_f_phi] - num_f_phi

params_cpy = params.copy()
params_cpy["f_expr"] = f_phi_expr[f_active]
params_cpy["a_expr"] = a_phi_expr[a_active]
params_cpy["lib_f"] = lib_f_phi.T[:, f_active]
params_cpy["lib_a"] = lib_a_phi.T[:, a_active]
params_cpy["Xi0"] = Xi0[active]
params_cpy["kl_reg"] = 0

unreg_Xi, unreg_cost = opt_fun(params_cpy)
unreg_Xi_expanded = np.zeros_like(Xi0)
unreg_Xi_expanded[active] = unreg_Xi
print("Without KL regularisation")
print(f"Xi = {unreg_Xi_expanded}")
print(f"Cost = {unreg_cost:.1e}", flush=True)

Xi_f = unreg_Xi_expanded[:num_f_phi]
Xi_a = unreg_Xi_expanded[num_f_phi:]

# Functions from the expressions
f_sym = utils.sindy_model(Xi_f, f_phi_expr)
f_sindy = sympy.lambdify(phi_sym, f_sym)
a_sym = utils.sindy_model(Xi_a, a_phi_expr)
a_sindy = sympy.lambdify(phi_sym, a_sym)

print(f"dphi = ({f_sym}) dt + ({sympy.sqrt(2*a_sym)}) dbeta")

f_vals = f_sindy(centers)
a_vals = a_sindy(centers)

# Check if a scalar (happens when library is a constant)
if np.ndim(f_vals) == 0:
    f_vals = f_vals + 0 * centers
if np.ndim(a_vals) == 0:
    a_vals = a_vals + 0 * centers

# Compare PDFs: empirical vs Fokker-Planck solution with model
p_fit = fp.solve(f_vals, a_vals)
kl_div_val = utils.kl_divergence(pdf, p_fit, dx=dx, tol=1e-6)
print(f"KL divergence: {kl_div_val:.1e}")

fig, ax = plt.subplots(figsize=(10, 10))
ax: plt.Axes  # type: ignore
ax.plot(centers, pdf, "k", label="Data", lw=3)
ax.plot(centers, p_fit, "--", c="r", label="Model", lw=3)
ax.legend()
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$P(\phi)$")

fig.savefig(folder_path / "pdf_comparision_just_phi.png")

afp.precompute_operator(f_vals, a_vals)
q = afp.solve(dt)
if q is None:
    raise RuntimeError("Failed to solve adjoint focker planck system")
f_tau, a_tau = q

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
ax.plot(centers, f(centers, sigma_avg), c="gray", lw=2, label="True drift")
ax.plot(centers, f_KM, ls="", marker=".", markersize=8, c="b", label="KM")
ax.plot(centers, f_vals, "r", lw=2, label="Finite Time Corrected")
ax.plot(centers, f_tau, "g:", lw=2, label=rf"$\tau = {dt}$")
ax.legend()
ax.set_title("Drift")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$f(\phi)$")

ax2.plot(centers, g(centers, sigma_avg) ** 2 / 2, c="gray", lw=2, label="True diff")
ax2.plot(centers, a_KM, ls="", marker=".", markersize=8, c="b", label="KM")
ax2.plot(centers, a_vals, "r", lw=2, label="Finite Time Corrected")
ax2.plot(centers, a_tau, "g:", lw=2, label=rf"$\tau={dt}$")
ax2.legend()
ax2.set_title("Diffusion")
ax2.set_xlabel(r"$\phi$")
ax2.set_ylabel(r"$a(\phi)$")

fig.tight_layout()
fig.savefig(folder_path / "final_graph_just_phi.png")
