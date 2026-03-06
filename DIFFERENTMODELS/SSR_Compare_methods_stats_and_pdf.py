# %%
from time import time
from pathlib import Path
from functools import partial
import multiprocessing as mp

import sympy
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from make_and_load_models import get_timeseries_and_KM
from kramersmoyal import km
from utils import sindy_model, jeffreys_divergence, SteadyFP_0D

# %%
SCRATCH_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path(
    f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods_stat_pdf/"
)
FIG_PATH.mkdir(parents=True, exist_ok=True)


# %%
def cost_diffusion(
    xi_C: np.ndarray,
    lib_C_KM: np.ndarray,
    lib_C_timeseries: np.ndarray,
    diffusion_KM: np.ndarray,
    dx_timeseries: np.ndarray,
    normal_dist: np.ndarray,
    normal_dist_edges: np.ndarray,
    alpha: float,
) -> float:
    C_ts = lib_C_timeseries.transpose((1, 2, 0)) @ xi_C
    C_km = (lib_C_KM.T @ xi_C)[None, ...]
    if np.any(C_ts < 0) or np.any(C_km) < 0:
        return np.inf
    if alpha != 1:
        dw_Q = dx_timeseries / np.sqrt(2 * C_ts)
        # pdf for each trajectory separately?
        (dw_Q_hist,), _ = km(dw_Q.flatten(), powers=0, bins=(normal_dist_edges,))  # type: ignore
        norm_dist_dx = normal_dist_edges[1] - normal_dist_edges[0]
        dw_Q_hist /= np.nansum(dw_Q_hist * norm_dist_dx)

        jef_diverg = jeffreys_divergence(dw_Q_hist, normal_dist, dx=norm_dist_dx)
    else:
        jef_diverg = 0

    diffusion_fidelity = np.nansum((C_km - diffusion_KM) ** 2) / np.nansum(
        diffusion_KM**2
    )
    return (1 - alpha) * jef_diverg + alpha / 2 * diffusion_fidelity


def opt_fun_C(
    xi_C_0: np.ndarray,
    lib_C_KM: np.ndarray,
    lib_C_timeseries: np.ndarray,
    diffusion_KM: np.ndarray,
    dx_timeseries: np.ndarray,
    normal_dist: np.ndarray,
    normal_dist_edges: np.ndarray,
    alpha: float,
):
    res = minimize(
        partial(
            cost_diffusion,
            lib_C_KM=lib_C_KM,
            lib_C_timeseries=lib_C_timeseries,
            diffusion_KM=diffusion_KM,
            dx_timeseries=dx_timeseries,
            normal_dist=normal_dist,
            normal_dist_edges=normal_dist_edges,
            alpha=alpha,
        ),
        xi_C_0,
        method="nelder-mead",
        options={"adaptive": True},
    )
    return res.x, res.fun


def cost_drift(
    xi_A: np.ndarray,
    lib_A_KM: np.ndarray,
    lib_A_timeseries: np.ndarray,
    pdf_KM: np.ndarray,
    drift_KM: np.ndarray,
    C_model_KM: np.ndarray,
    dx_KM: float,
    sfp: SteadyFP_0D,
    dx_timeseries: np.ndarray,
    B_model_timeseries: np.ndarray,
    normal_dist: np.ndarray,
    normal_dist_edges: np.ndarray,
    dt: float,
    beta: float = 0,
    gamma: float = 0,
) -> float:
    if xi_A[-1] > 0:
        return np.inf

    if beta != 1:
        dw_P = (
            dx_timeseries - lib_A_timeseries.transpose((1, 2, 0)) @ xi_A * dt
        ) / B_model_timeseries
        (dw_P_hist,), _ = km(dw_P.flatten(), powers=0, bins=(normal_dist_edges,))  # type: ignore
        norm_dist_dx = normal_dist_edges[1] - normal_dist_edges[0]
        dw_P_hist /= np.nansum(dw_P_hist * norm_dist_dx)
        jef_diverg = jeffreys_divergence(dw_P_hist, normal_dist, dx=norm_dist_dx)
        if gamma != 1:
            pdf = sfp.solve(lib_A_KM.T @ xi_A, C_model_KM)
            pdf_diverg = np.sum(jeffreys_divergence(pdf, pdf_KM, dx=dx_KM)) / len(pdf)
        else:
            pdf_diverg = 0
    else:
        pdf_diverg = 0
        jef_diverg = 0

    drift_fidelity = np.nansum(
        ((lib_A_KM.T @ xi_A)[None, ...] - drift_KM) ** 2
    ) / np.nansum(drift_KM**2)
    return (1 - beta) * (
        (1 - gamma) * pdf_diverg + gamma * jef_diverg
    ) + beta / 2 * drift_fidelity


def opt_fun_A(
    xi_A_0: np.ndarray,
    lib_A_KM: np.ndarray,
    lib_A_timeseries: np.ndarray,
    pdf_KM: np.ndarray,
    drift_KM: np.ndarray,
    C_model_KM: np.ndarray,
    dx_KM: float,
    sfp: SteadyFP_0D,
    dx_timeseries: np.ndarray,
    B_model_timeseries: np.ndarray,
    normal_dist: np.ndarray,
    normal_dist_edges: np.ndarray,
    dt: float,
    beta: float,
    gamma: float,
):
    res = minimize(
        partial(
            cost_drift,
            lib_A_KM=lib_A_KM,
            lib_A_timeseries=lib_A_timeseries,
            pdf_KM=pdf_KM,
            drift_KM=drift_KM,
            C_model_KM=C_model_KM,
            dx_KM=dx_KM,
            sfp=sfp,
            dx_timeseries=dx_timeseries,
            B_model_timeseries=B_model_timeseries,
            normal_dist=normal_dist,
            normal_dist_edges=normal_dist_edges,
            dt=dt,
            beta=beta,
            gamma=gamma,
        ),
        xi_A_0,
        method="nelder-mead",
        options={"adaptive": True},
    )
    return res.x, res.fun


# %%
## PLOTTING FUNCTIONS
def timeseries_plots(folder, timeseries, suffix="", slice_=slice(None, None, None)):
    time_stack, x_stack = timeseries
    for i in range(len(time_stack)):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(time_stack[i, slice_], x_stack[i, slice_])
        ax.set_xlabel("Time, $t$")
        ax.set_ylabel("Value, $x$")

        if suffix:
            fig.savefig(folder / f"timeseries_{i}_{suffix}.png")
        else:
            fig.savefig(folder / f"timeseries_{i}.png")
        plt.close(fig)


def KM_plots(folder, KM, suffix=""):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 16), sharex=True)

    ax1.plot(centers, pdf_stack.T, alpha=0.7)
    ax1.set_ylabel("PDF, $P(x)$")
    ax1.set_ylabel("$x$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_ylabel("$x$")

    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    ax3.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax3.set_xlabel("$x$")

    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"kramers_moyal_{suffix}.png")
    else:
        fig.savefig(folder / "kramers_moyal.png")
    plt.close(fig)


def diffusion_plots_one_method(
    folder,
    KM,
    dbeta_pdfs,
    found_diffusion,
    normal_dist_centers,
    normal_dist,
    diffu_expr: str,
    method_name: str,
    suffix="",
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 13))
    ax1.set_title(method_name)
    ax1.plot(normal_dist_centers, normal_dist, linestyle="-", label="Normal")
    ax1.plot(normal_dist_centers, dbeta_pdfs.T, linestyle="--", label="Model")
    ax1.set_ylabel(r"PDF, $P(d\beta)$")
    ax1.set_ylabel(r"$d\beta$")

    ax2.set_title(diffu_expr)
    ax2.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_diffusion, linestyle="-")
    ax2.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax2.set_xlabel("$x$")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    min_ = np.abs(np.min(mean_vals))
    max_ = np.abs(np.max(mean_vals))
    ax2.set_ylim(
        np.min(mean_vals) - 0.01 * (min_ + max_),
        np.max(mean_vals) + 0.01 * (min_ + max_),
    )

    fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"diffusion_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"diffusion_{method_name}.png")

    # ax1.set_yscale("log")

    # if suffix:
    #     fig.savefig(folder / f"diffusion_zoom_{method_name}_{suffix}.png")
    # else:
    #     fig.savefig(folder / f"diffusion_zoom_{method_name}.png")

    plt.close(fig)


def drift_plots_one_method(
    folder,
    KM,
    dbeta_pdfs,
    found_drift,
    normal_dist_centers,
    normal_dist,
    drift_expr: str,
    method_name: str,
    suffix="",
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 13))
    ax1.set_title(method_name)
    ax1.plot(normal_dist_centers, normal_dist, linestyle="-", label="Normal")
    ax1.plot(normal_dist_centers, dbeta_pdfs.T, linestyle="--", label="Model")
    ax1.set_ylabel(r"PDF, $P(d\beta)$")
    ax1.set_xlabel(r"$d\beta$")

    ax2.set_title(drift_expr)
    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_drift, linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_xlabel("$x$")
    mean_vals = np.nanmean(drift_stack, axis=0)
    min_ = np.abs(np.min(mean_vals))
    max_ = np.abs(np.max(mean_vals))
    ax2.set_ylim(
        np.min(mean_vals) - 0.01 * (min_ + max_) / 2,
        np.max(mean_vals) + 0.01 * (min_ + max_) / 2,
    )

    fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"drift_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"drift_{method_name}.png")

    # ax1.set_yscale("log")
    ax2.set_ylim(
        -2,
        2,
    )

    if suffix:
        fig.savefig(folder / f"drift_zoom_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"drift_zoom_{method_name}.png")

    plt.close(fig)


def KM_plots_one_method(
    folder,
    KM,
    found_pdf,
    found_drift,
    found_diffusion,
    drift_expr: str,
    diffu_expr: str,
    method_name: str,
    suffix="",
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 16))
    ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.plot(centers, found_pdf, linestyle="-", label="Model")
    ax1.set_ylabel("PDF, $P(x)$")
    ax1.set_ylabel("$x$")

    ax2.set_title(drift_expr)
    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_drift, linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_ylabel("$x$")
    mean_vals = np.nanmean(drift_stack, axis=0)
    min_ = np.abs(np.min(mean_vals))
    max_ = np.abs(np.max(mean_vals))
    ax2.set_ylim(
        np.min(mean_vals) - 0.01 * (min_ + max_) / 2,
        np.max(mean_vals) + 0.01 * (min_ + max_) / 2,
    )

    ax3.set_title(diffu_expr)
    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.plot(centers, found_diffusion, linestyle="-")
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax3.set_ylabel("$x$")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    min_ = np.abs(np.min(mean_vals))
    max_ = np.abs(np.max(mean_vals))
    ax3.set_ylim(
        np.min(mean_vals) - 0.01 * (min_ + max_),
        np.max(mean_vals) + 0.01 * (min_ + max_),
    )

    fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"kramers_moyal_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"kramers_moyal_{method_name}.png")

    ax1.set_yscale("log")
    ax2.set_ylim(
        -2,
        2,
    )

    if suffix:
        fig.savefig(folder / f"kramers_moyal_zoom_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"kramers_moyal_zoom_{method_name}.png")

    plt.close(fig)


def cost_coeffs_C_plots(folder, costs, xis, diffu_expr, method_name, suffix=""):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax.plot(costs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")

    ax2.imshow(xis.astype(bool).T)
    ylabels = [f"${sympy.latex(expr)}$" for expr in diffu_expr]
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Active Diffusion terms")
    ax2.set_yticks(range(xis.shape[1]), labels=ylabels)
    fig.tight_layout()
    if suffix:
        fig.savefig(folder / f"coeffs_C_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"coeffs_C_{method_name}.png")

    plt.close(fig)


def cost_coeffs_A_plots(folder, costs, xis, drift_expr, method_name, suffix=""):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax.plot(costs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")

    ax2.imshow(xis.astype(bool).T)
    ylabels = [f"${sympy.latex(expr)}$" for expr in drift_expr]
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Active Drift terms")
    ax2.set_yticks(range(xis.shape[1]), labels=ylabels)
    fig.tight_layout()
    if suffix:
        fig.savefig(folder / f"coeffs_A_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"coeffs_A_{method_name}.png")

    plt.close(fig)


def choosing_plots(folder, choosing, method_name, suffix):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(choosing)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Choosing")
    ax.set_title(method_name)

    fig.tight_layout()
    if suffix:
        fig.savefig(folder / f"choosing_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"choosing_{method_name}.png")

    plt.close(fig)


# %%
def poly_lib(
    x_sym,
    order: int,
    centers: np.ndarray,
    xs: np.ndarray,
    EVEN_ABS: bool = False,
    ODD_ABS: bool = False,
):
    arr = []
    for i in range(order):
        if EVEN_ABS and i % 2 == 0 and i != 0:
            arr.append(sympy.Abs(x_sym) * x_sym ** (i - 1))
        elif ODD_ABS and i % 2 == 1:
            arr.append(sympy.Abs(x_sym) * x_sym ** (i - 1))
        else:
            arr.append(x_sym**i)
    lib_expr = np.array(arr)
    lib_KM = np.empty((order, len(centers)))
    lib_timeseries = np.empty((order, *xs.shape))
    for k in range(order):
        lamb = sympy.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)
        lib_timeseries[k] = lamb(xs)

    return lib_expr, lib_KM, lib_timeseries


def SSR_loop_diffusion(
    opt_func_diffusion,
    xi_C_0,
    lib_diffu_KM,
    lib_diffu_timeseries,
    diffusion_KM,
    dx_timeseries,
    normal_dist,
    normal_dist_edges,
    alpha,
):
    n_terms = len(lib_diffu_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)
    min_xis[0], min_Vs[0] = opt_func_diffusion(
        xi_C_0,
        lib_diffu_KM,
        lib_diffu_timeseries,
        diffusion_KM,
        dx_timeseries,
        normal_dist,
        normal_dist_edges,
        alpha,
    )
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        for j in range(1, len(active)):
            tmp_active = np.delete(active.copy(), j)
            if len(tmp_active) == 0:
                continue

            params = [
                xi_C_0[tmp_active],
                lib_diffu_KM[tmp_active],
                lib_diffu_timeseries[tmp_active],
                diffusion_KM,
                dx_timeseries,
                normal_dist,
                normal_dist_edges,
                alpha,
            ]
            params_list.append(params)
            valid_indices.append(j)

        with mp.Pool(NUM_CPUS) as p:
            results = p.starmap(opt_func_diffusion, params_list)
        xis = []
        Vs = []
        for Xi, V in results:
            xis.append(Xi)
            Vs.append(V)
        min_cost = np.nanargmin(Vs)
        min_idx = valid_indices[min_cost]
        min_V = Vs[min_cost]
        min_xi = xis[min_cost]
        active = np.delete(active, min_idx)
        min_Vs[k] = min_V
        min_xis[k, active] = min_xi

    return min_xis, min_Vs


def SSR_loop_drift(
    opt_func_drift,
    xi_A_0,
    lib_drift_KM,
    lib_drift_timeseries,
    pdf_KM,
    drift_KM,
    C_model_KM,
    dx_KM,
    sfp,
    dx_timeseries,
    found_B_timeseries,
    normal_dist,
    normal_dist_edges,
    dt,
    beta,
    gamma,
):
    n_terms = len(lib_drift_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)
    min_xis[0], min_Vs[0] = opt_func_drift(
        xi_A_0,
        lib_drift_KM,
        lib_drift_timeseries,
        pdf_KM,
        drift_KM,
        C_model_KM,
        dx_KM,
        sfp,
        dx_timeseries,
        found_B_timeseries,
        normal_dist,
        normal_dist_edges,
        dt,
        beta,
        gamma,
    )
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        for j in range(len(active)):
            tmp_active = np.delete(active.copy(), j)
            if len(tmp_active) == 0:
                continue

            xi_0_A_reduced = xi_A_0[tmp_active]
            xi_0_A_reduced[-1] = -np.abs(xi_0_A_reduced[-1])
            params = [
                xi_0_A_reduced,
                lib_drift_KM[tmp_active],
                lib_drift_timeseries[tmp_active],
                pdf_KM,
                drift_KM,
                C_model_KM,
                dx_KM,
                sfp,
                dx_timeseries,
                found_B_timeseries,
                normal_dist,
                normal_dist_edges,
                dt,
                beta,
                gamma,
            ]
            params_list.append(params)
            valid_indices.append(j)

        with mp.Pool(NUM_CPUS) as p:
            results = p.starmap(opt_func_drift, params_list)
        xis = []
        Vs = []
        for Xi, V in results:
            xis.append(Xi)
            Vs.append(V)
        min_cost = np.nanargmin(Vs)
        min_idx = valid_indices[min_cost]
        min_V = Vs[min_cost]
        min_xi = xis[min_cost]
        active = np.delete(active, min_idx)
        min_Vs[k] = min_V
        min_xis[k, active] = min_xi

    return min_xis, min_Vs


# %%
equation_names = [
    "Double Well",
    "Triple Well",
    "Softglass (Pitchforking)",
    "Softglass (Normal)",
]
folders = [
    "DoubleWell",
    "TripleWell",
    "SoftglassPitchforking",
    "SoftglassNormal",
]
drift_coefficients = [
    # 1, x, x|x|, x^3, x^3|x|, ...
    [0.0, 1.0, 0.0, -1.0],
    [0.0, -1.0, 0.0, 1.0, 0.0, -0.2],
    [0.0, -0.016, 0.0, 1.1, -1.0],
    [0.0, -0.016, 0.0, 1.1, -1.0],
]
diffusion_coefficients = [
    # [epsilon_0, epsilon_1] -> diffu = sqrt(ep0 + ep1 x^2)
    [0.3, 0.2],  # CHECK IF APPROPRIATE
    [0.3, 0.1],  # CHECK IF APPROPRIATE
    [1e-3, 0.1],
    [1e-5, 0.1],
]

even_abs_simulation = [False, False, True, True]
even_abs_library = [False, False, False, False]
# %%
regression_method_names_diffusion = [
    "SFP_Stats",
    "SFP_Stats",
    "SFP_Stats",
    "KM_SFP_Stats",
    "KM_SFP_Stats",
    "KM_SFP_Stats",
    "KM",
    "KM",
]
alpha_vals = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]
opt_funcs_diffusion = [opt_fun_C] * len(alpha_vals)

regression_method_names_drift = [
    "SFP_Stats",
    "KM_SFP_Stats",
    "KM",
    "SFP_Stats",
    "KM_SFP_Stats",
    "KM",
    "SFP_Stats",
    "KM_SFP_Stats",
]
beta_vals = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5]
opt_funcs_drift = [opt_fun_A] * len(beta_vals)

# %%
NUM_DATASETS = 10
NUM_VALIDATION = 1
NUM_CPUS = 10
dt = 0.001
num_bins = 50
target_metadata = {
    "num_datapoints": 10_000_000,
    "dt": dt,
    "EVEN_ABS": False,
    "coeffs": None,  # fill in later for each equation
    "ep0": None,  # fill in later for each equation
    "ep1": None,  # fill in later for each equation
    "x0": 0.0,
    "num_bins": num_bins,
    "kernel": "gaussian",
}
drift_plus = 1
diffusion_plus = 3

# %%
timeseries_skip = 1000
for equation_number in [0, 1, 2, 3]:  # range(len(equation_names)):
    name = equation_names[equation_number]
    folder = folders[equation_number]
    print("Equation:", name, folder)
    (SCRATCH_PATH / folder).mkdir(parents=True, exist_ok=True)
    (FIG_PATH / folder).mkdir(parents=True, exist_ok=True)
    drift_coef = drift_coefficients[equation_number]
    ep0, ep1 = diffusion_coefficients[equation_number]
    target_metadata["EVEN_ABS"] = even_abs_simulation[equation_number]
    target_metadata["ep0"] = ep0
    target_metadata["ep1"] = ep1
    target_metadata["coeffs"] = drift_coef
    timeseries, KM, val_timeseries, val_KM = get_timeseries_and_KM(
        SCRATCH_PATH / folder, target_metadata, NUM_DATASETS, NUM_VALIDATION, NUM_CPUS
    )

    times, xs = timeseries
    dxs = xs[:, 1:] - xs[:, :-1]

    # time, x, dx: shape (NUM_MODELS, (num_datapoints - 1) // timeseries_skip)
    times = times[:, 1:-1:timeseries_skip]
    xs = xs[:, 1:-1:timeseries_skip]
    dxs = dxs[:, 1::timeseries_skip]

    centers, pdf, drift, diffusion = KM
    # could do this instead as |drift - mean| > N std for example
    # since the value of 0.0 could actually be real
    # instead of being a lack of data
    drift[drift == 0.0] = np.nan
    diffusion[diffusion == 0.0] = np.nan
    KM = (centers, pdf, drift, diffusion)

    timeseries_plots(FIG_PATH / folder, timeseries, folder, slice(0, 100_000, 1))
    KM_plots(FIG_PATH / folder, KM, folder)

    # make libraries for drift and diffusion
    x_sym = sympy.symbols("x")
    num_diffusion = 3 + diffusion_plus
    lib_diffu_expr, lib_diffu_KM, lib_diffu_timeseries = poly_lib(
        x_sym, num_diffusion, centers, xs, False, False
    )
    mask = np.all(np.isfinite(diffusion) + (diffusion > 0), axis=0)
    xi_C_0 = np.abs(
        np.average(lstsq(lib_diffu_KM.T[mask], diffusion.T[mask])[0], axis=1)
    )
    xi_C_0[1::2] /= 10  # odd terms are not great when taking sqrt
    num_drift = len(drift_coef) + drift_plus
    lib_drift_expr, lib_drift_KM, lib_drift_timeseries = poly_lib(
        x_sym,
        num_drift,
        centers,
        xs,
        even_abs_library[equation_number],
        False,
    )
    mask = np.all(np.isfinite(drift), axis=0)
    xi_A_0 = np.average(lstsq(lib_drift_KM.T[mask], drift.T[mask])[0], axis=1)
    xi_A_0[-1] = -np.abs(xi_A_0[-1])

    # regress each method against the same dataset with the same library functions,
    # the same initial conditions, and record the resulting final equation
    # do some plots of the final equation against the datasets (and save the data)
    norm_dist_width = 5
    norm_dist_points = 200
    normal_dist_edges, normal_dist_dx = np.linspace(
        -norm_dist_width * np.sqrt(dt),
        norm_dist_width * np.sqrt(dt),
        norm_dist_points,
        retstep=True,
    )
    normal_dist_centers = (normal_dist_edges[1:] + normal_dist_edges[:-1]) / 2
    normal_dist = np.exp(-(normal_dist_centers**2) / (2 * dt))
    normal_dist /= np.sum(normal_dist * normal_dist_dx)

    sfp = SteadyFP_0D(num_bins, KM[0][1] - KM[0][0])

    method_xi_C_s = []
    method_v_C_s = []

    method_xi_A_s = []
    method_v_A_s = []

    for regression_method_number in range(len(regression_method_names_diffusion)):
        start = time()
        reg_method_name_diffusion = regression_method_names_diffusion[
            regression_method_number
        ]
        alpha_val = alpha_vals[regression_method_number]
        opt_func_diffusion = opt_funcs_diffusion[regression_method_number]

        print("Diffusion", reg_method_name_diffusion, alpha_val, opt_func_diffusion)
        xi_C_s, V_C_s = SSR_loop_diffusion(
            opt_func_diffusion,
            xi_C_0,
            lib_diffu_KM,
            lib_diffu_timeseries,
            diffusion,
            dxs,
            normal_dist,
            normal_dist_edges,
            alpha_val,
        )

        cost_coeffs_C_plots(
            FIG_PATH / folder,
            V_C_s,
            xi_C_s,
            lib_diffu_expr,
            f"{regression_method_number}_{reg_method_name_diffusion}",
            suffix=folder,
        )

        method_xi_C_s.append(xi_C_s)
        method_v_C_s.append(V_C_s)
        end = time()
        print(f"Diffusion SSR time: {end-start}s")
        for xi_C in xi_C_s:
            diffu_expr = (
                "$"
                + sympy.latex(sympy.sqrt(sympy.N(2 * lib_diffu_expr.T @ xi_C, 2)))
                + "$"
            )
            fdiffu_KM = lib_diffu_KM.T @ xi_C
            fdiffu_timeseries = np.sqrt(
                2 * lib_diffu_timeseries.transpose((1, 2, 0)) @ xi_C
            )
            (fpdf,), _ = km(
                (dxs / fdiffu_timeseries).flatten(), powers=0, bins=(normal_dist_edges,)  # type: ignore
            )
            fpdf /= np.nansum(fpdf * (normal_dist_edges[1] - normal_dist_edges[0]))
            diffusion_plots_one_method(
                FIG_PATH / folder,
                KM,
                fpdf,
                fdiffu_KM,
                normal_dist_centers,
                normal_dist,
                diffu_expr,
                f"{np.count_nonzero(xi_C)}_{regression_method_number}_{reg_method_name_diffusion}",
                folder,
            )

        # Choose best xi_C #
        choosing = (V_C_s[1:] - V_C_s[:-1]) / V_C_s[:-1]
        choosing_plots(
            FIG_PATH / folder,
            choosing,
            f"C_{regression_method_number}_{reg_method_name_diffusion}",
            folder,
        )
        chosen = np.nonzero(choosing > np.nanmax(choosing) / 5)[0][0]

        best_C_xi = xi_C_s[chosen]

        found_diffu_expr = lib_diffu_expr.T @ best_C_xi
        print(f"Best Diffusion: B^2 = {2*found_diffu_expr}")
        found_diffu_KM = lib_diffu_KM.T @ best_C_xi
        found_B_timeseries = np.sqrt(
            2 * lib_diffu_timeseries.transpose((1, 2, 0)) @ best_C_xi
        )

        reg_method_name_drift = regression_method_names_drift[regression_method_number]
        beta_val = beta_vals[regression_method_number]
        opt_func_drift = opt_funcs_drift[regression_method_number]

        # With fixed diffusion, determine the best xi_A via SSR_loop_drift #
        print("Drift", reg_method_name_drift, beta_val, opt_func_drift)
        xi_A_s, V_A_s = SSR_loop_drift(
            opt_func_drift,
            xi_A_0,
            lib_drift_KM,
            lib_drift_timeseries,
            pdf,
            drift,
            found_diffu_KM,
            centers[1] - centers[0],
            sfp,
            dxs,
            found_B_timeseries,
            normal_dist,
            normal_dist_edges,
            dt,
            beta_val,
            0.5,  # CHANGE LATER / sweep over?
        )

        cost_coeffs_A_plots(
            FIG_PATH / folder,
            V_A_s,
            xi_A_s,
            lib_drift_expr,
            f"{regression_method_number}_{reg_method_name_diffusion}_{reg_method_name_drift}",
            suffix=folder,
        )

        method_xi_A_s.append(xi_A_s)
        method_v_A_s.append(V_A_s)
        end = time()
        print(f"Drift SSR time: {end-start}s")
        for xi_A in xi_A_s:
            drift_expr = "$" + sympy.latex(sympy.N(lib_drift_expr.T @ xi_A, 2)) + "$"
            fdrift_KM = lib_drift_KM.T @ xi_A
            fdrift_timeseries = lib_drift_timeseries.transpose((1, 2, 0)) @ xi_A
            (fpdf,), _ = km(
                ((dxs - fdrift_timeseries * dt) / found_B_timeseries).flatten(), powers=0, bins=(normal_dist_edges,)  # type: ignore
            )
            fpdf /= np.nansum(fpdf * (normal_dist_edges[1] - normal_dist_edges[0]))
            drift_plots_one_method(
                FIG_PATH / folder,
                KM,
                fpdf,
                fdrift_KM,
                normal_dist_centers,
                normal_dist,
                drift_expr,
                f"{np.count_nonzero(xi_A)}_{regression_method_number}_{reg_method_name_diffusion}_{reg_method_name_drift}",
                folder,
            )
        # Choose best xi_A #
        choosing = (V_A_s[1:] - V_A_s[:-1]) / V_A_s[:-1]
        choosing_plots(
            FIG_PATH / folder,
            choosing,
            f"A_{regression_method_number}_{reg_method_name_diffusion}_{reg_method_name_drift}",
            folder,
        )
        chosen = np.nonzero(choosing > np.nanmax(choosing) / 5)[0][0]
        best_A_xi = xi_A_s[chosen]

        found_drift_expr = lib_drift_expr.T @ best_A_xi
        print(f"Best Drift: A = {found_drift_expr}")
        found_drift_KM = lib_drift_KM.T @ best_A_xi
        found_drift_timeseries = lib_drift_timeseries.transpose((1, 2, 0)) @ best_A_xi

        found_pdf = sfp.solve(found_drift_KM, found_diffu_KM)

        KM_plots_one_method(
            FIG_PATH / folder,
            KM,
            found_pdf,
            found_drift_KM,
            found_diffu_KM,
            "$" + sympy.latex(sympy.N(found_drift_expr, 2)) + "$",
            "$" + sympy.latex(sympy.sqrt(sympy.N(2 * found_diffu_expr, 2))) + "$",
            f"{regression_method_number}_{reg_method_name_diffusion}_{reg_method_name_drift}",
            folder,
        )

        print(
            reg_method_name_diffusion,
            f"dx = ({sympy.N(found_drift_expr, 2)}) dt + {sympy.sqrt(sympy.N(2*found_diffu_expr, 2))} dW\n",
        )
    del timeseries, KM, val_timeseries, val_KM, centers, pdf, drift, diffusion

    print("\n\n")
    np.savez(
        FIG_PATH / folder / "SSR_result.npz",
        method_xi_C_s=method_xi_C_s,
        method_v_C_s=method_v_C_s,
        method_xi_A_s=method_xi_A_s,
        method_v_A_s=method_v_A_s,
    )
