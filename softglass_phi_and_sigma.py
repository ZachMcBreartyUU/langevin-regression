"""Code adapted from pitchfork.ipynb, utils.py, fpsolve.py by Jared Callaham
This work done by Zach McBrearty

NOTE: The size of the libraries scales as (N_bins x M_bins x N_functions), so the initial size of the dataset only matters for the initial dataplotting (optional) and then the binning / moment generation (Kramers Moyal KDE).
The size of the libraries matters _greatly_ as they are constantly inverted, this size can be controlled by the -N or --num-bins flag
"""

import argparse
from time import time

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # used to type hint 3D plots
import sympy
from kramersmoyal import km

import utils
import fpsolve
import data_loader as dl

print("REDUCED LIBRARY, ADAPTIVE NM")

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("-c", "--cutoff", type=float, default=250.0)
ap.add_argument("--kl-reg", type=float, default=10.0)
ap.add_argument("-N", "--num-x-bins", type=int, default=25)
ap.add_argument("-M", "--num-y-bins", type=int, default=25)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=-1)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--plot-intermediate", action="store_true")
args = ap.parse_args()
print(args)

folder_path = dl.SCRATCH_PATH / args.folder
print(folder_path, flush=True)
step = 1
metadata, times, phi, sigma = dl.get_data(
    folder_path, start=args.start, stop=args.stop, step=args.step
)
timeseries = np.concat([phi, sigma], axis=1)

dt = metadata["dt"] * args.step
R = metadata["R"]
ep0 = metadata["epsilon_0"]
ep1 = metadata["epsilon_1"]
lb = metadata["lb"]
U = metadata["U"]
tau = metadata["tau"]
m = lambda y: (y - 1) / np.sqrt(y) if y > 1 else 0.0


def m_proper(sigma):
    ret = np.zeros_like(sigma)
    mask = sigma > 1
    ret[mask] = (sigma[mask] - 1) / np.sqrt(sigma[mask])
    return ret


if args.plot_intermediate:
    # Plot solution
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

    fig.savefig(folder_path / "system_graph_phi_and_sigma.png")

    # Plot phi against sigma to show where the data is
    fig, ax = plt.subplots(figsize=(12, 12))
    ax: plt.Axes  # type: ignore
    ax.scatter(sigma, phi, marker="x", alpha=0.5)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\phi$")

    fig.tight_layout()

    fig.savefig(folder_path / "phi_sigma_graph.png")


def _km_find_range(data: np.ndarray, threshold: float, N: int = 25) -> np.ndarray:
    """data: data to find the pdf range for
    pdf_cutoff: data smaller than max(pdf) / pdf_cutoff will be ignored, and thus the range shortened
    N: number of datapoints in the range"""
    kmc, centers = km(data, bins=N, powers=0)  # type: ignore
    pdf: np.ndarray = kmc[0]
    centers: np.ndarray = centers[0]
    threshold = pdf.max() / threshold
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


def km_avg_2D(
    data: np.ndarray,
    tau: float,
    pdf_cutoff: float = np.e**3,
    N: int = 25,
    M: int = 25,
    full: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """full: return all powers, or only the moments which should be nonzero"""
    assert data.shape[1] == 2
    # establish ranges of phi and sigma from their individual pdfs
    phi_range = _km_find_range(data[:, 0], pdf_cutoff, N)
    sigma_range = _km_find_range(data[:, 1], pdf_cutoff, M)

    if full:
        kmc, centers, _, powers = km(
            data,
            bins=(phi_range, sigma_range),  # type: ignore
            powers=2,
            full=True,
        )
    else:
        kmc, centers, _, powers = km(
            data,
            bins=(phi_range, sigma_range),  # type: ignore
            powers=np.array([[0, 0], [0, 1], [1, 0], [2, 0]]),  # type: ignore
            full=True,
        )
    assert powers[0][0] == 0 and powers[0][1] == 0
    pdf = kmc[0]
    pdf /= np.sum(pdf)

    moments = kmc[1:] / tau
    powers = powers[1:].astype(int)

    return centers, pdf, moments, powers


print("Generating pdf and moments")
full = False
centers, pdf, moments, powers = km_avg_2D(
    np.concat([phi, sigma], axis=1),
    tau=dt,
    pdf_cutoff=args.cutoff,
    N=args.num_x_bins,
    M=args.num_y_bins,
    full=full,
)
centers_phi, centers_sigma = np.meshgrid(*centers)
del times, phi, sigma  # large arrays which are no longer needed
print("Finished generating pdf and moments", flush=True)

if args.plot_intermediate:
    if not full:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            subplot_kw={"projection": "3d"},
            figsize=(12, 12),
        )
        axes: list[list[Axes3D]]

        # Plot pdf, no 'true' / analytic value
        axes[0][0].plot_wireframe(centers_phi, centers_sigma, pdf.T, axlim_clip=True)
        axes[0][0].set_title("PDF")

        # plot first moment in sigma/stress, with f_sigma
        axes[0][1].plot_wireframe(
            centers_phi, centers_sigma, moments[0].T, axlim_clip=True
        )
        f_sigma = (
            U
            - (1 - lb) * centers_phi**2 * centers_sigma
            - lb * m_proper(centers_sigma) ** 2 * centers_sigma
        ) / tau
        axes[0][1].plot_wireframe(
            centers_phi, centers_sigma, f_sigma, axlim_clip=True, color="red"
        )
        axes[0][1].set_zlim(np.min(f_sigma) * 1.1, np.max(f_sigma) * 1.1)
        axes[0][1].set_title(r"$m^{(0, 1)}(\phi, \sigma)$")

        # plot first moment in phi, with f_phi
        axes[1][0].plot_wireframe(
            centers_phi, centers_sigma, moments[1].T, axlim_clip=True
        )
        f_phi = (
            -R * centers_phi
            + (m_proper(centers_sigma) - np.abs(centers_phi)) * centers_phi**3
        )
        axes[1][0].plot_wireframe(
            centers_phi, centers_sigma, f_phi, axlim_clip=True, color="red"
        )
        axes[1][0].set_zlim(np.min(f_phi) * 1.1, np.max(f_phi) * 1.1)
        axes[1][0].set_title(r"$m^{(1, 0)}(\phi, \sigma)$")

        # plot second moment in phi, a_phi = g_phi^2 / 2
        axes[1][1].plot_wireframe(
            centers_phi, centers_sigma, moments[2].T, axlim_clip=True
        )
        a_phi_phi = (ep0 + ep1 * centers_phi**2) / 2
        axes[1][1].plot_wireframe(
            centers_phi, centers_sigma, a_phi_phi, axlim_clip=True, color="red"
        )
        axes[1][1].set_zlim(np.min(a_phi_phi) * 0.9, np.max(a_phi_phi) * 1.1)
        axes[1][1].set_title(r"$m^{(1, 1)}(\phi, \sigma)$")

        for pair in axes:
            for ax in pair:
                ax.set_xlabel(r"$\phi$")
                ax.set_ylabel(r"$\sigma$")

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

        fig.savefig(folder_path / "process_pdf_moments_phi_and_sigma_2x2.png")
    else:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=3,
            subplot_kw={"projection": "3d"},
            figsize=(18, 18),
        )
        axes: list[list[Axes3D]]

        axes[0][0].plot_wireframe(centers_phi, centers_sigma, pdf.T, axlim_clip=True)
        axes[0][0].set_title("PDF")

        for moment, power in zip(moments, powers[1:]):
            axes[power[0]][power[1]].plot_wireframe(
                centers_phi, centers_sigma, moment.T, axlim_clip=True
            )
            axes[power[0]][power[1]].set_title(
                r"$m^{(%d, %d)}(\phi, \sigma)$" % (power[0], power[1])
            )
        f_sigma = (U - centers_phi**2 * centers_sigma) / tau
        f_phi = (
            -R * centers_phi
            + (m_proper(centers_sigma) - np.abs(centers_phi)) * centers_phi**3
        )
        a_phi_phi = np.sqrt(ep0 + ep1 * centers_phi**2)
        axes[0][1].plot_wireframe(
            centers_phi,
            centers_sigma,
            (f_sigma),
            axlim_clip=True,
            color="red",
        )
        axes[1][0].plot_wireframe(
            centers_phi,
            centers_sigma,
            (f_phi),
            axlim_clip=True,
            color="red",
        )
        axes[2][0].plot_wireframe(
            centers_phi,
            centers_sigma,
            (a_phi_phi),
            axlim_clip=True,
            color="red",
        )
        for pair in axes:
            for ax in pair:
                ax.set_xlabel(r"$\phi$")
                ax.set_ylabel(r"$\sigma$")

        fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))

        fig.savefig(folder_path / "process_pdf_moments_phi_and_sigma_3x3.png")

f_phi_km = moments[(powers == np.array([1, 0])).all(axis=1)]
f_sigma_km = moments[(powers == np.array([0, 1])).all(axis=1)]
a_phi_km = moments[(powers == np.array([2, 0])).all(axis=1)]

### Build SINDy libraries with sympy
phi_sym = sympy.symbols("p")
sigma_sym = sympy.symbols("s")
m_sym = sympy.Function("m")

#### f phi library:  [1,  φ,  φ^3, φ^5, φ|φ|, φ^3|φ|, φ/sqrt(σ),  φ^3/sqrt(σ), φ*sqrt(σ),  φ^3*sqrt(σ)]
#### expected coeff: [0, -R,    0,   0,    0,      1,         0,            1,         0,            1]
# f phi library:  [1,  φ,  φ^3, φ^5, φ|φ|, φ^3|φ|, φ m(σ), φ^3 m(σ)]
# expected coeff: [0, -R,    0,   0,    0,      1,      0,        1]

f_phi_expr = np.array(
    [
        phi_sym**0,
        phi_sym**1,
        phi_sym**1 * sympy.Abs(phi_sym),
        phi_sym**1 * m_sym(sigma_sym),
        phi_sym**3,
        phi_sym**3 * sympy.Abs(phi_sym),
        phi_sym**3 * m_sym(sigma_sym),
        phi_sym**5,
    ]
)
# + [sympy.Float(1)]
# + [phi_sym ** (2 * i + 1) for i in np.arange(0, 3)]
# + [sympy.Abs(phi_sym) * phi_sym ** (2 * i + 1) for i in np.arange(0, 2)]
# + [phi_sym ** (2 * i + 1) / sympy.Pow(sigma_sym, 0.5) for i in np.arange(0, 2)]
# + [phi_sym ** (2 * i + 1) * sympy.Pow(sigma_sym, 0.5) for i in np.arange(0, 2)]
print("f_phi library:", f_phi_expr)
num_f_phi = len(f_phi_expr)

# f sigma library: [   1,   σ, σ^2, φ^2, φ^2 σ, φ^2 σ^2] ##, φ^4, φ^4 σ, φ^4 σ^2]
# expected coeff:  [U-lb, 2lb, -lb,   0,  lb-1,       0] ##,   0,     0,       0]
f_sigma_expr = np.array(
    [
        phi_sym ** (2 * n) * sigma_sym**m
        for n in np.arange(0, 2)
        for m in np.arange(0, 3)
    ]
)
print("f_sigma library:", f_sigma_expr)
num_f_sigma = len(f_sigma_expr)

#### a phi library:  [  1, σ, σ^2, φ, φ σ, φ σ^2, φ^2, φ^2 σ, φ^2 σ^2]
#### expected coeff: [ep0, 0,   0, 0,   0,     0, ep1,     0,       0]
# a phi library:  [  1, φ, φ^2, φ^3]
# expected coeff: [ep0, 0, ep1,   0]
a_phi_expr = np.array(
    # [phi_sym**n * sigma_sym**m for n in np.arange(0, 3) for m in np.arange(0, 3)]
    [phi_sym**n for n in np.arange(0, 4)]
)
print("a_phi library:", a_phi_expr, flush=True)
num_a_phi = len(a_phi_expr)

# Convert sympy expressions into library matrices
# Initial shape of the library is (n_terms, N, M)
# Once library is constructed from lamdified expressions
# the library is reshaped to (n_terms, N*M)
# because following functions (lstsq) expect the libraries to be only 2D
N, M = centers_phi.shape
lib_f_phi = np.zeros((num_f_phi, N, M))
for k in range(num_f_phi):
    lamb_expr = sympy.lambdify(
        (phi_sym, sigma_sym), f_phi_expr[k], modules=[{"m": m_proper}, "numpy"]
    )
    lib_f_phi[k] = lamb_expr(centers_phi, centers_sigma)
lib_f_phi = lib_f_phi.reshape(-1, N * M)

lib_f_sigma = np.zeros((num_f_sigma, N, M))
for k in range(num_f_sigma):
    lamb_expr = sympy.lambdify(
        (phi_sym, sigma_sym), f_sigma_expr[k], modules=[{"m": m_proper}, "numpy"]
    )
    lib_f_sigma[k] = lamb_expr(centers_phi, centers_sigma)
lib_f_sigma = lib_f_sigma.reshape(-1, N * M)

lib_a_phi = np.zeros((num_a_phi, N, M))
for k in range(num_a_phi):
    lamb_expr = sympy.lambdify(
        (phi_sym, sigma_sym), a_phi_expr[k], modules=[{"m": m_proper}, "numpy"]
    )
    lib_a_phi[k] = lamb_expr(centers_phi, centers_sigma)
lib_a_phi = lib_a_phi.reshape(-1, N * M)

# Initialize Xi with least squares regression (no finite-time corrections)

Xi0 = np.zeros((num_f_phi + num_f_sigma + num_a_phi,))
# Xi0 sections have shape n, libraries have shape (n, NxM), data has shape (N, M)
# thus to be compatible with lstsq reshape lib to (n, NxM).T = (NxM, n), and data to (NxM,)
Xi0[:num_f_phi] = lstsq(lib_f_phi.T, f_phi_km.reshape(N * M), rcond=None)[0]
Xi0[num_f_phi : num_f_phi + num_f_sigma] = lstsq(
    lib_f_sigma.T, f_sigma_km.reshape(N * M), rcond=None
)[0]
Xi0[num_f_phi + num_f_sigma :] = lstsq(
    lib_a_phi.T, a_phi_km.reshape(N * M), rcond=None
)[0]
print("Xi0 =", Xi0, flush=True)

# Initialize adjoint solver
afp = fpsolve.AdjFP(centers, ndim=2)

# Initialize forward steady-state solver
dphi = centers[0][1] - centers[0][0]
dsigma = centers[1][1] - centers[1][0]
fp = fpsolve.SteadyFP([N, M], [dphi, dsigma])


# THESE FUNCTIONS ARE COPIED AND MODIFIED FROM utils.py FOR ADAPTATION TO 2D
def AFP_opt(cost, params):
    ### RUN OPTIMIZATION PROBLEM
    start_time = time()
    Xi0 = params["Xi0"]

    opt_fun = lambda Xi: cost(Xi, params)

    res = minimize(
        opt_fun,
        Xi0,
        method="nelder-mead",
        options={"disp": False, "maxfev": int(1e4), "adaptive": True},
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
    W = params["W"]  # Optimization weights

    N = params["N"]
    M = params["M"]
    # Kramers-Moyal coefficients
    f_x_KM = params["f_x_KM"].flatten()  # (N*M,)
    f_y_KM = params["f_y_KM"].flatten()  # (N*M,)
    a_x_KM = params["a_x_KM"].flatten()  # (N*M,)

    fp: fpsolve.SteadyFP = params["fp"]
    afp: fpsolve.AdjFP = params["afp"]  # Fokker-Planck solvers
    lib_f_x = params["lib_f_x"]  # (n_f_x, N*M)
    lib_f_y = params["lib_f_y"]  # (n_f_y, N*M)
    lib_a_x = params["lib_a_x"]  # (n_a_x, N*M)

    # Construct parameterized drift and diffusion functions from libraries and current coefficients
    f_x_vals = lib_f_x @ Xi[: lib_f_x.shape[1]]  # (n_f_x, N*M).T @ (n_f_x,) = (N*M,)
    f_y_vals = (
        lib_f_y @ Xi[lib_f_x.shape[1] : lib_f_x.shape[1] + lib_f_y.shape[1]]
    )  # (N*M,)
    a_x_vals = lib_a_x @ Xi[lib_f_x.shape[1] + lib_f_y.shape[1] :]  # (N*M,)

    # Solve AFP equation to find finite-time corrected drift/diffusion
    #    corresponding to the current parameters Xi
    afp.precompute_operator(
        [f_x_vals, f_y_vals],
        [a_x_vals, np.zeros((N * M))],
    )
    Q0 = afp.solve(params["tau"], d=0)
    Q1 = afp.solve(params["tau"], d=1)
    if Q0 is None or Q1 is None:
        raise RuntimeError("Failed to solve adjoint focker planck system")
    f_x_tau, a_x_tau = Q0  # (N*M,) (N*M,)
    f_y_tau, a_y_tau = Q1  # (N*M,) (N*M,)

    V = (
        np.sum(W[0] * np.abs(f_x_tau - f_x_KM) ** 2)
        + np.sum(W[1] * np.abs(f_y_tau - f_y_KM) ** 2)
        + np.sum(W[2] * np.abs(a_x_tau - a_x_KM) ** 2)
        + np.sum(W[3] * np.abs(a_y_tau) ** 2)  # should be 0
    )

    # Include PDF constraint via Kullbeck-Leibler divergence regularization
    if params["kl_reg"] > 0:
        p_hist = params["p_hist"]  # Empirical PDF
        p_est = fp.solve(
            [f_x_vals.reshape(N, M), f_y_vals.reshape(N, M)],
            [a_x_vals.reshape(N, M), np.zeros((N, M))],
        )  # Solve Fokker-Planck equation for steady-state PDF
        kl = utils.kl_divergence(p_hist.T, p_est, dx=fp.dx, tol=1e-6)
        kl = max(
            0, kl
        )  # Numerical integration can occasionally produce small negative values
        V += params["kl_reg"] * kl
    return V


def SSR_loop(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """

    # Lists of candidate expressions... coefficients are optimized
    f_x_expr = params["f_x_expr"].copy()
    f_y_expr = params["f_y_expr"].copy()
    a_x_expr = params["a_x_expr"].copy()
    # could be extended to include a_y, however this should be 0 in this case
    lib_f_x = params["lib_f_x"].copy()
    lib_f_y = params["lib_f_y"].copy()
    lib_a_x = params["lib_a_x"].copy()
    # could be extended to include a_y, however this should be 0 in this case
    num_f_x = len(f_x_expr)
    num_f_y = len(f_y_expr)
    num_a_x = len(a_x_expr)
    # could be extended to include a_y, however this should be 0 in this case
    Xi0 = params["Xi0"].copy()

    m = num_f_x + num_f_y + num_a_x

    Xi = np.zeros((m, m - 2), dtype=Xi0.dtype)  # Output results
    V = np.zeros((m - 2))  # Cost at each step

    # Full regression problem as baseline
    Xi[:, 0], V[0] = opt_fun(params)

    # Start with all candidates
    active = np.array([i for i in range(m)])
    active_history = [active]

    # Iterate and threshold
    for k in range(1, m - 2):
        # Loop through remaining terms and find the one that increases the cost function the least
        min_idx = -1
        V[k] = np.inf
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)  # Try deleting this term

            # Break off masks for drift/diffusion
            f_x_active = tmp_active[tmp_active < num_f_x]
            f_y_active = (
                tmp_active[(tmp_active >= num_f_x) * (tmp_active < num_f_x + num_f_y)]
                - num_f_x
            )
            a_x_active = tmp_active[tmp_active >= num_f_x + num_f_y] - (
                num_f_x + num_f_y
            )
            # print(f"{f_x_active=}")
            # print(f"{f_y_active=}")
            # print(f"{a_x_active=}")

            # print(f"{f_x_expr[f_x_active]=}")
            # print(f"{f_y_expr[f_y_active]=}")
            # print(f"{a_x_expr[a_x_active]=}")

            params["f_x_expr"] = f_x_expr[f_x_active]
            params["f_y_expr"] = f_y_expr[f_y_active]
            params["a_x_expr"] = a_x_expr[a_x_active]
            params["lib_f_x"] = lib_f_x[:, f_x_active]
            params["lib_f_y"] = lib_f_y[:, f_y_active]
            params["lib_a_x"] = lib_a_x[:, a_x_active]
            params["Xi0"] = Xi0[tmp_active]  # Z: shouldn't this is found by LSTSQ?

            # Ensure that there is at least one drift x, one drift y, and one diffusion term left
            if len(f_x_active) > 0 and len(f_y_active) > 0 and len(a_x_active) > 0:
                tmp_Xi, tmp_V = opt_fun(params)

                # Keep minimum cost
                if tmp_V < V[k]:
                    min_idx = j
                    V[k] = tmp_V
                    min_Xi = tmp_Xi

        print("Cost: {0}".format(V[k]))
        # Delete least important term
        active = np.delete(active, min_idx)  # Remove inactive index
        Xi0[active] = min_Xi  # type: ignore # Re-initialize with best results from previous
        Xi[active, k] = min_Xi  # type: ignore
        active_history.append(active)
        f_x_active = active[active < num_f_x]
        f_y_active = (
            active[(active >= num_f_x) * (active < num_f_x + num_f_y)] - num_f_x
        )
        a_x_active = active[active >= num_f_x + num_f_y] - (num_f_x + num_f_y)
        print(f"Active f x: {f_x_active}")
        print(f"Active f y: {f_y_active}")
        print(f"Active a x: {a_x_active}", flush=True)

    return Xi, V, active_history


# Optimization parameters
flat_pdf = pdf.flatten()
weight = np.zeros_like(flat_pdf)
weight[flat_pdf > 0] = (
    1 / flat_pdf[flat_pdf > 0]
)  # only non-zero entries of the pdf have weight
weight /= np.nansum(weight)
W = np.array([weight, weight, weight, weight])
params = {
    "W": W,
    "f_x_KM": f_phi_km,
    "f_y_KM": f_sigma_km,
    "a_x_KM": a_phi_km,
    "Xi0": Xi0,
    "f_x_expr": f_phi_expr,
    "f_y_expr": f_sigma_expr,
    "a_x_expr": a_phi_expr,
    "lib_f_x": lib_f_phi.T,
    "lib_f_y": lib_f_sigma.T,
    "lib_a_x": lib_a_phi.T,
    "N": N,
    "M": M,
    "kl_reg": args.kl_reg,
    "fp": fp,
    "afp": afp,
    "p_hist": pdf,
    "tau": dt,
}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: AFP_opt(cost, params)
Xi, V, active_history = SSR_loop(opt_fun, params)

####################
# SSR cost function
####################

labels = [
    r"${0}$".format(sympy.latex(t))
    for t in np.concatenate((f_phi_expr, f_sigma_expr, a_phi_expr))
]

n_terms = len(labels)

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
ax: plt.Axes  # type: ignore
ax2: plt.Axes  # type: ignore
ax.scatter(np.arange(len(V)), V, c="k")
ax.set_xticks(np.arange(n_terms - 2))
ax.set_xticklabels(np.arange(n_terms, 2, -1))
ax.set_xlabel("Sparsity")
ax.set_ylabel(r"Cost")

square = np.zeros_like(Xi)
for i, hist in enumerate(active_history):
    square[hist, i] = 1
square = square.astype(bool)

ax2.pcolor(square, cmap="bone_r", edgecolors="gray")
ax2.axhline(y=num_f_phi, color="red")
ax2.axhline(y=num_f_phi + num_f_sigma, color="red")
ax2.set_yticks(0.5 + np.arange(n_terms))
ax2.set_yticklabels(labels)
ax2.set_xticks(0.5 + np.arange(n_terms - 2))
ax2.set_xticklabels(np.arange(n_terms, 2, -1))
ax2.set_xlabel("Sparsity")
ax2.set_ylabel("Active terms")

fig.savefig(folder_path / "SSR_sparsity_phi_and_sigma.png")

# Select model with the fewest terms before the cost function spikes
threshold = 1e6
n_terms_selected = np.arange(n_terms, 2, -1)[
    np.nonzero(V[1:] - V[:-1] > threshold * V[:-1])[0]
]
while n_terms_selected.size == 0:
    threshold /= 10
    if threshold <= 1:
        print(
            f"WARNING: threshold dropped to {threshold} before finding jump", flush=True
        )
    n_terms_selected = np.arange(n_terms, 2, -1)[
        np.nonzero(V[1:] - V[:-1] > threshold * V[:-1])[0]
    ]
n_terms_selected = n_terms_selected[0]  # take the first spike
print(f"n_terms_selected={n_terms_selected}")
print("With KL regularisation")
print(f"Xi = {Xi[:, 2 - n_terms_selected]}")
print(f"Cost = {V[2 - n_terms_selected]:.1e}")

active = active_history[2 - n_terms_selected]
f_phi_active = active[active < num_f_phi]
f_sigma_active = (
    active[(active >= num_f_phi) * (active < num_f_phi + num_f_sigma)] - num_f_phi
)
a_phi_active = active[active >= num_f_phi + num_f_sigma] - (num_f_phi + num_f_sigma)

params_cpy = params.copy()
params_cpy["f_x_expr"] = f_phi_expr[f_phi_active]
params_cpy["f_y_expr"] = f_sigma_expr[f_sigma_active]
params_cpy["a_x_expr"] = a_phi_expr[a_phi_active]
params_cpy["lib_f_x"] = lib_f_phi.T[:, f_phi_active]
params_cpy["lib_f_y"] = lib_f_sigma.T[:, f_sigma_active]
params_cpy["lib_a_x"] = lib_a_phi.T[:, a_phi_active]
params_cpy["Xi0"] = Xi0[active]
params_cpy["kl_reg"] = 0

unreg_Xi, unreg_cost = opt_fun(params_cpy)
unreg_Xi_expanded = np.zeros_like(Xi0)
unreg_Xi_expanded[active] = unreg_Xi
print("Without KL regularisation")
print(f"Xi = {unreg_Xi_expanded}")
print(f"Cost = {unreg_cost:.1e}", flush=True)

Xi_f_phi = unreg_Xi[:num_f_phi]
Xi_f_sigma = unreg_Xi[num_f_phi : num_f_phi + num_f_sigma]
Xi_a_phi = unreg_Xi[num_f_phi + num_f_sigma :]

# Functions from the expressions
f_phi_sym = utils.sindy_model(Xi_f_phi, f_phi_expr)
f_phi_sindy = sympy.lambdify((phi_sym, sigma_sym), f_phi_sym)
f_sigma_sym = utils.sindy_model(Xi_f_sigma, f_sigma_expr)
f_sigma_sindy = sympy.lambdify((phi_sym, sigma_sym), f_sigma_sym)
a_phi_sym = utils.sindy_model(Xi_a_phi, a_phi_expr)
s_phi_sym = sympy.sqrt(2 * a_phi_expr)
a_phi_sindy = sympy.lambdify((phi_sym, sigma_sym), a_phi_sym)

print(f"dphi = ({f_phi_sym}) dt + ({s_phi_sym}) dbeta")
print(f"dsigma = ({f_sigma_sym}) dt", flush=True)

f_phi_vals = f_phi_sindy(*centers)
f_sigma_vals = f_sigma_sindy(*centers)
a_phi_vals = a_phi_sindy(*centers)

# Check if a scalar (happens when library is a constant)
if np.ndim(f_phi_vals) == 0:
    f_phi_vals = f_phi_vals + 0 * centers
if np.ndim(f_sigma_vals) == 0:
    f_sigma_vals = f_sigma_vals + 0 * centers
if np.ndim(a_phi_vals) == 0:
    a_phi_vals = a_phi_vals + 0 * centers

# Compare PDFs: empirical vs Fokker-Planck solution with model

p_fit = fp.solve([f_phi_vals, f_sigma_vals], [a_phi_vals, np.zeros_like(a_phi_vals)])
kl_div_val = utils.kl_divergence(pdf, p_fit, dx=[dphi, dsigma], tol=1e-6)
print(f"KL divergence: {kl_div_val:.1e}")

afp.precompute_operator(
    [f_phi_vals, f_sigma_vals], [a_phi_vals, np.zeros_like(a_phi_vals)]
)
q0 = afp.solve(dt, 0)
q1 = afp.solve(dt, 1)
if q0 is None or q1 is None:
    raise RuntimeError("Failed to solve adjoint focker planck system")
f_phi_tau, a_phi_tau = q0
f_sigma_tau, a_sigma_tau = q1

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    subplot_kw={"projection": "3d"},
    figsize=(12, 12),
)
axes: list[list[Axes3D]]

# Plot pdf, no 'true' / analytic value
axes[0][0].plot_wireframe(
    centers_phi, centers_sigma, pdf.T, label="Data", axlim_clip=True
)
axes[0][0].plot_wireframe(
    centers_phi, centers_sigma, p_fit.T, label="Fit", axlim_clip=True
)  # transpose?
axes[0][0].legend()
axes[0][0].set_title("PDF")

# plot first moment in sigma/stress, with f_sigma
f_sigma = (
    U
    - (1 - lb) * centers_phi**2 * centers_sigma
    - lb * m_proper(centers_sigma) ** 2 * centers_sigma
) / tau
# TRUE, KM, VALS, TAU
axes[0][1].plot_wireframe(
    centers_phi, centers_sigma, f_sigma, color="red", label="True", axlim_clip=True
)
axes[0][1].plot_wireframe(
    centers_phi, centers_sigma, f_sigma_km, label="KM", axlim_clip=True
)
axes[0][1].plot_wireframe(
    centers_phi, centers_sigma, f_sigma_vals, label="Vals", axlim_clip=True
)
axes[0][1].plot_wireframe(
    centers_phi, centers_sigma, f_sigma_tau, label="Tau", axlim_clip=True
)
axes[0][1].set_zlim(np.min(f_sigma) * 1.1, np.max(f_sigma) * 1.1)
axes[0][1].set_title(r"$m^{(0, 1)}(\phi, \sigma)$")
axes[0][1].legend()

# plot first moment in phi, with f_phi
f_phi = (
    -R * centers_phi + (m_proper(centers_sigma) - np.abs(centers_phi)) * centers_phi**3
)
axes[1][0].plot_wireframe(
    centers_phi, centers_sigma, f_phi, color="red", label="True", axlim_clip=True
)
axes[1][0].plot_wireframe(
    centers_phi, centers_sigma, f_phi_km, label="KM", axlim_clip=True
)
axes[1][0].plot_wireframe(
    centers_phi, centers_sigma, f_phi_vals, label="Vals", axlim_clip=True
)
axes[1][0].plot_wireframe(
    centers_phi, centers_sigma, f_phi_tau, label="Tau", axlim_clip=True
)
axes[1][0].set_zlim(np.min(f_phi) * 1.1, np.max(f_phi) * 1.1)
axes[1][0].set_title(r"$m^{(1, 0)}(\phi, \sigma)$")
axes[1][0].legend()

# plot second moment in phi, a_phi = g_phi^2 / 2
a_phi = (ep0 + ep1 * centers_phi**2) / 2
axes[1][1].plot_wireframe(
    centers_phi, centers_sigma, a_phi, color="red", label="True", axlim_clip=True
)
axes[1][1].plot_wireframe(
    centers_phi, centers_sigma, a_phi_km, label="KM", axlim_clip=True
)
axes[1][1].plot_wireframe(
    centers_phi, centers_sigma, a_phi_vals, label="Vals", axlim_clip=True
)
axes[1][1].plot_wireframe(
    centers_phi, centers_sigma, a_phi_tau, label="Tau", axlim_clip=True
)
axes[1][1].set_zlim(np.min(a_phi) * 0.9, np.max(a_phi) * 1.1)
axes[1][1].set_title(r"$m^{(2, 0)}(\phi, \sigma)$")
axes[1][1].legend()

for pair in axes:
    for ax in pair:
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\sigma$")

fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))

fig.savefig(folder_path / "final_graph_phi_and_sigma.png")
