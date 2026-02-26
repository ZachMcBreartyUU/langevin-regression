# %%
from time import time
from pathlib import Path
import os
from functools import partial
import multiprocessing as mp
from typing import Callable, Optional

import sympy
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from make_and_load_models import get_timeseries_and_KM, load_and_stack_KM
from utils import kl_divergence, jeffreys_divergence, SteadyFP

# %%
# SCRATCH_PATH = Path(f"/home/zachuu/scratch/paper/")
SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/compare_methods/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = SCRATCH_PATH

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
def cost(
    xi: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    drift_val = lib_drift.T @ xi[: len(lib_drift)]
    diffu_val = lib_diffu.T @ xi[len(lib_drift) :]
    drift_fid = np.nansum((drifts - drift_val) ** 2) / np.nansum(drifts**2)
    diffusion_fid = np.nansum((diffusions - diffu_val) ** 2) / np.nansum(diffusions**2)
    KM_fid = (drift_fid + diffusion_fid) / 2

    return KM_fid


def cost_scaling(
    scale: float,
    xi: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    scaled_xi = xi * scale

    drift_val = lib_drift.T @ scaled_xi[: len(lib_drift)]
    diffu_val = lib_diffu.T @ scaled_xi[len(lib_drift) :]
    drift_fid = np.nansum((drifts - drift_val) ** 2) / np.nansum(drifts**2)
    diffusion_fid = np.nansum((diffusions - diffu_val) ** 2) / np.nansum(diffusions**2)
    KM_fid = (drift_fid + diffusion_fid) / 2

    return KM_fid


def cost_KL(
    xi: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    drift_val = lib_drift.T @ xi[: len(lib_drift)]
    diffu_val = lib_diffu.T @ xi[len(lib_drift) :]

    drift_fid = np.nansum((drifts - drift_val) ** 2) / np.nansum(drifts**2)
    diffusion_fid = np.nansum((diffusions - diffu_val) ** 2) / np.nansum(diffusions**2)
    KM_fid = (drift_fid + diffusion_fid) / 2
    # if xi[len(lib_drift)] > 0:
    #     return np.inf
    pdf_lib = sfp.solve(drift_val, diffu_val)
    pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
    reg_val = np.sum(
        kl_divergence(pdf_lib[:, None], pdfs.T, centers[1] - centers[0], tol=1e-8)
    ) / len(pdfs)

    return (1 - alpha) * KM_fid + alpha * reg_val


def cost_Jef(
    xi: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    drift_val = lib_drift.T @ xi[: len(lib_drift)]
    diffu_val = lib_diffu.T @ xi[len(lib_drift) :]

    drift_fid = np.nansum((drifts - drift_val) ** 2) / np.nansum(drifts**2)
    diffusion_fid = np.nansum((diffusions - diffu_val) ** 2) / np.nansum(diffusions**2)
    KM_fid = (drift_fid + diffusion_fid) / 2
    # if xi[len(lib_drift)] > 0:
    #     return np.inf
    pdf_lib = sfp.solve(drift_val, diffu_val)
    pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
    reg_val = np.sum(
        jeffreys_divergence(pdf_lib[:, None], pdfs.T, centers[1] - centers[0], tol=1e-8)
    ) / len(pdfs)
    return (1 - alpha) * KM_fid + alpha * reg_val


def cost_Wasserstein(
    xi: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    drift_val = lib_drift.T @ xi[: len(lib_drift)]
    diffu_val = lib_diffu.T @ xi[len(lib_drift) :]

    drift_fid = np.nansum((drifts - drift_val) ** 2) / np.nansum(drifts**2)
    diffusion_fid = np.nansum((diffusions - diffu_val) ** 2) / np.nansum(diffusions**2)
    KM_fid = (drift_fid + diffusion_fid) / 2
    # if xi[len(lib_drift)] > 0:
    #     return np.inf
    pdf_lib = sfp.solve(drift_val, diffu_val)
    pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
    reg_val = 0
    for pdf in pdfs:
        reg_val += np.sum(wasserstein_distance(pdf_lib, pdf))
    reg_val /= len(pdfs)

    return (1 - alpha) * KM_fid + alpha * reg_val


def opt_func(cost, xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    res = minimize(
        partial(
            cost, KM=KM, lib_drift=lib_drift, lib_diffu=lib_diffu, sfp=sfp, alpha=alpha
        ),
        xi0,
        method="nelder-mead",
        options={"adaptive": True},
    )
    return res.x, res.fun


def opt_func_cost(xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    return opt_func(cost, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_cost_KL(xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    return opt_func(cost_KL, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_cost_Jef(xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    return opt_func(cost_Jef, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_cost_Wass(xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    return opt_func(cost_Wasserstein, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_scaling(cost1, xi0, KM, lib_drift, lib_diffu, sfp, alpha):
    res = minimize(
        partial(
            cost1, KM=KM, lib_drift=lib_drift, lib_diffu=lib_diffu, sfp=sfp, alpha=alpha
        ),
        xi0,
        method="nelder-mead",
        options={"adaptive": True},
    )
    xi = res.x
    xi /= np.abs(xi[0])

    res2 = minimize(
        partial(
            cost_scaling,
            xi=xi,
            KM=KM,
            lib_drift=lib_drift,
            lib_diffu=lib_diffu,
            sfp=sfp,
            alpha=alpha,
        ),
        np.ones(1),
    )
    return xi * res2.x[0], res.fun


def opt_func_cost_KL_scaling(xi0, KM, lib_drift, lib_diffu, sfp, alpha=1):
    return opt_func_scaling(cost_KL, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_cost_Jef_scaling(xi0, KM, lib_drift, lib_diffu, sfp, alpha=1):
    return opt_func_scaling(cost_Jef, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


def opt_func_cost_Wass_scaling(xi0, KM, lib_drift, lib_diffu, sfp, alpha=1):
    return opt_func_scaling(cost_Wasserstein, xi0, KM, lib_drift, lib_diffu, sfp, alpha)


# %%
method_names = ["KM", "KM_KL", "KM_Jef", "KM_Wasserstein", "KL", "Jef", "Wasserstein"]
alpha_vals = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
# cost_funcs = [
#     cost_KL,
#     cost_KL,
#     cost_Jef,
#     cost_Wasserstein,
#     cost_KL,
#     cost_Jef,
#     cost_Wasserstein,
# ]
opt_funcs = [
    opt_func_cost,
    opt_func_cost_KL,
    opt_func_cost_Jef,
    opt_func_cost_Wass,
    opt_func_cost_KL_scaling,
    opt_func_cost_Jef_scaling,
    opt_func_cost_Wass_scaling,
]

# %%
NUM_DATASETS = 10
NUM_VALIDATION = 1
NUM_CPUS = 10
dt = 0.001
num_bins = 100
target_metadata = {
    "num_datapoints": 10_000_000,
    "dt": dt,
    "EVEN_ABS": False,
    "coeffs": None,  # fill in later for each equation
    "ep0": None,  # fill in later for each equation
    "ep1": None,  # fill in later for each equation
    "x0": 0.0,
    "num_bins": num_bins,
}


# %%
def timeseries_plots(folder, timeseries, suffix="", slice_=slice(None, None, None)):
    time_stack, x_stack = timeseries
    for i in range(len(time_stack)):
        fig, ax = plt.subplots(figsize=(6, 4))
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    ax1.plot(centers, pdf_stack.T, alpha=0.7)
    ax1.set_ylabel("PDF, $P(x)$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")

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


def KM_plots_one_method(
    folder, KM, found_pdf, found_drift, found_diffusion, method_name: str, suffix=""
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.plot(centers, found_pdf, linestyle="-", label="Model")
    ax1.set_ylabel("PDF, $P(x)$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_drift, linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))

    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.plot(centers, found_diffusion, linestyle="-")
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    ax3.set_ylim(np.min(mean_vals), np.max(mean_vals))

    fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"kramers_moyal_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"kramers_moyal_{method_name}.png")
    plt.close(fig)


def KM_plots_all_methods(
    folder, KM, found_pdfs, found_drifts, found_diffusions, method_names, suffix=""
):
    # I think this plot will look extremely messy
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.set_ylabel("PDF, $P(x)$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    ax3.set_ylim(np.min(mean_vals), np.max(mean_vals))
    for found_pdf, found_drift, found_diffusion, method_name in zip(
        found_pdfs, found_drifts, found_diffusions, method_names
    ):
        ax1.plot(centers, found_pdf, linestyle="-", label=method_name)
        ax2.plot(centers, found_drift, linestyle="-")
        ax3.plot(centers, found_diffusion, linestyle="-")

    fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"kramers_moyal_allmethods_{suffix}.png")
    else:
        fig.savefig(folder / f"kramers_moyal_allmethods.png")
    plt.close(fig)


def cost_coeffs_plots(
    folder, costs, xis, drift_expr, diffu_expr, method_name, suffix=""
):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(13, 6))
    ax.plot(costs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")

    ax2.imshow(xis.astype(bool).T)
    ylabels = [f"${sympy.latex(expr)}$" for expr in drift_expr] + [
        f"${sympy.latex(expr)}$" for expr in diffu_expr
    ]
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Active terms")
    ax2.set_yticks(range(xis.shape[1]), labels=ylabels)
    ax2.axhline(len(drift_expr) - 0.5)
    fig.tight_layout()
    if suffix:
        fig.savefig(folder / f"coeffs_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"coeffs_{method_name}.png")

    plt.close(fig)


def poly_lib(
    x_sym,
    order: int,
    centers: np.ndarray,
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
    for k in range(order):
        lamb = sympy.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)

    return lib_expr, lib_KM


def SSR_loop(opt_func_, KM, xi0, lib_drift_KM, lib_diffu_KM, sfp, alpha):
    n_drift = len(lib_drift_KM)
    n_terms = n_drift + len(lib_diffu_KM)
    min_xis = np.zeros((n_terms - 1, n_terms))
    min_Vs = np.full((n_terms - 1), np.inf)
    min_xis[0], min_Vs[0] = opt_func_(xi0, KM, lib_drift_KM, lib_diffu_KM, sfp, alpha)
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms - 1):
        params_list = []
        valid_indices = []
        for j in range(len(active)):
            tmp_active = np.delete(active.copy(), j)

            drift_active = tmp_active[tmp_active < n_drift]
            diffu_active = tmp_active[tmp_active >= n_drift] - n_drift
            if len(drift_active) == 0 or len(diffu_active) == 0:
                continue

            params = [
                xi0[tmp_active],
                KM,
                lib_drift_KM[drift_active],
                lib_diffu_KM[diffu_active],
                sfp,
                alpha,
            ]
            params_list.append(params)
            valid_indices.append(j)

        with mp.Pool(NUM_CPUS) as p:
            results = p.starmap(opt_func_, params_list)
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


# %%script true
for equation_number in range(len(equation_names)):
    name = equation_names[equation_number]
    folder = folders[equation_number]
    print(name, folder)
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
    centers, pdf, drift, diffusion = KM
    # could do this instead as |drift - mean| > N std for example
    # since the value of 0.0 could actually be real
    # instead of being a lack of data
    drift[drift == 0.0] = np.nan
    diffusion[diffusion == 0.0] = np.nan
    KM = (centers, pdf, drift, diffusion)

    timeseries_plots(FIG_PATH / folder, timeseries, folder, slice(0, 100_000, 1))
    del timeseries, val_timeseries
    KM_plots(FIG_PATH / folder, KM, folder)

    # make libraries for drift and diffusion
    x_sym = sympy.symbols("x")
    num_drift = len(drift_coef) + 1
    lib_drift_expr, lib_drift_KM = poly_lib(
        x_sym, num_drift, KM[0], even_abs_library[equation_number], False
    )
    num_diffusion = 4
    lib_diffu_expr, lib_diffu_KM = poly_lib(x_sym, num_diffusion, KM[0], False, False)
    xi0 = np.zeros(
        num_drift + num_diffusion
    )  # np.random.normal(0, 1, (num_drift + num_diffusion,))
    mask = np.all(np.isfinite(KM[2]), axis=0)
    xi0[:num_drift] = np.average(lstsq(lib_drift_KM.T[mask], KM[2].T[mask])[0], axis=1)
    mask = np.all(np.isfinite(KM[3]), axis=0)
    xi0[num_drift:] = np.abs(
        np.average(lstsq(lib_diffu_KM.T[mask], KM[3].T[mask])[0], axis=1)
    )

    sfp = SteadyFP(num_bins, KM[0][1] - KM[0][0])

    # regress each method against the same dataset with the same library functions,
    # the same initial conditions, and record the resulting final equation
    # do some plots of the final equation against the datasets (and save the data)

    method_xis = []
    method_vs = []

    for method_number in range(len(method_names)):
        start = time()
        method_name = method_names[method_number]
        alpha_val = alpha_vals[method_number]
        opt_func_ = opt_funcs[method_number]
        print(method_name, alpha_val, opt_func_)

        xis, Vs = SSR_loop(
            opt_func_, KM, xi0, lib_drift_KM, lib_diffu_KM, sfp, alpha_val
        )

        cost_coeffs_plots(
            FIG_PATH / folder,
            Vs,
            xis,
            lib_drift_expr,
            lib_diffu_expr,
            f"{method_number}_{method_name}",
            suffix=folder,
        )

        method_xis.append(xis)
        method_vs.append(Vs)
        end = time()
        print(f"{end-start}s\n")
    print("\n\n")
    np.savez(
        SCRATCH_PATH / folder / "SSR_result.npz",
        method_xis=method_xis,
        method_vs=method_vs,
    )

# %%


def eval_jump_difference(series):
    diff = series[1:] - series[:-1]
    max_jump_idx = np.argmax(diff)
    return max_jump_idx


def eval_jump_ratio(series):
    diff = series[1:] / series[:-1]
    max_jump_idx = np.argmax(diff)
    return max_jump_idx


def eval_jump_difference_ratio(series):
    diff = (series[1:] - series[:-1]) / series[:-1]
    max_jump_idx = np.argmax(diff)
    return max_jump_idx


def eval_jump_second_difference_ratio(series):
    diff = (series[2:] - series[:-2]) / series[1:-1]
    max_jump_idx = np.argmax(diff)
    return max_jump_idx


def AIC(series, N):
    """Akaike_information_criterion
    series: cost
    N: number of parameters at each cost level"""
    AIC_ = 2 * N - 2 * np.log(series)
    max_jump_idx = np.argmin(AIC_)
    return max_jump_idx


choose_methods = [
    eval_jump_difference,
    eval_jump_ratio,
    eval_jump_difference_ratio,
    eval_jump_second_difference_ratio,
    AIC,
]
choose_method_names = [
    "eval jump difference",
    "eval jump ratio",
    "eval jump difference ratio",
    "eval jump second difference ratio",
    "AIC",
]

true_drift_exprs = []
true_diffu_exprs = []

table_file = open(SCRATCH_PATH / f"table_file.txt", "w")
for i, (choosing_method, cmn) in enumerate(zip(choose_methods, choose_method_names)):
    print("Method choosing")
    all_drift_exprs = []
    all_diffu_exprs = []
    for equation_number in range(len(equation_names)):
        name = equation_names[equation_number]
        folder = folders[equation_number]
        print(name, folder)
        (SCRATCH_PATH / folder).mkdir(parents=True, exist_ok=True)
        (FIG_PATH / folder).mkdir(parents=True, exist_ok=True)
        drift_coef = drift_coefficients[equation_number]
        ep0, ep1 = diffusion_coefficients[equation_number]
        target_metadata["EVEN_ABS"] = even_abs_simulation[equation_number]
        target_metadata["ep0"] = ep0
        target_metadata["ep1"] = ep1
        target_metadata["coeffs"] = drift_coef
        centers, pdf, drift, diffusion = load_and_stack_KM(
            SCRATCH_PATH / folder, NUM_DATASETS
        )
        # could do this instead as |drift - mean| > N std for example
        # since the value of 0.0 could actually be real
        # instead of being a lack of data
        drift[drift == 0.0] = np.nan
        diffusion[diffusion == 0.0] = np.nan
        KM = (centers, pdf, drift, diffusion)

        # make libraries for drift and diffusion
        x_sym = sympy.symbols("x")
        num_drift = len(drift_coef) + 1
        lib_drift_expr, lib_drift_KM = poly_lib(
            x_sym, num_drift, KM[0], even_abs_library[equation_number], False
        )
        num_diffusion = 4
        lib_diffu_expr, lib_diffu_KM = poly_lib(
            x_sym, num_diffusion, KM[0], False, False
        )
        sfp = SteadyFP(num_bins, KM[0][1] - KM[0][0])

        if i == 0:
            dr = drift_coefficients[equation_number]
            true_drift_exprs.append(lib_drift_expr.T[: len(dr)] @ np.array(dr))
            di = diffusion_coefficients[equation_number]
            true_diffu_exprs.append(lib_diffu_expr.T[: len(di)] @ np.array(di))

        with np.load(SCRATCH_PATH / folder / "SSR_result.npz") as f:
            method_xis = f["method_xis"]
            method_vs = f["method_vs"]
        found_pdfs = []
        found_drifts = []
        found_drift_exprs = []

        found_diffus = []
        found_diffu_exprs = []
        for method_number in range(len(method_names)):
            method_name = method_names[method_number]
            alpha_val = alpha_vals[method_number]
            xis = method_xis[method_number]
            Vs = method_vs[method_number]
            if choosing_method is AIC:
                best_xi = xis[
                    choosing_method(Vs, np.arange(num_drift + num_diffusion + 1, 2, -1))
                ]
            else:
                best_xi = xis[choosing_method(Vs)]

            drift_xi = best_xi[:num_drift]
            diffu_xi = best_xi[num_drift:]

            found_drift = lib_drift_KM.T @ drift_xi
            found_diffu = lib_diffu_KM.T @ diffu_xi
            found_pdf = sfp.solve(found_drift, found_diffu)
            found_pdf /= np.sum(found_pdf * (KM[0][1] - KM[0][0]))

            found_drift_expr = lib_drift_expr.T @ drift_xi
            found_diffu_expr = lib_diffu_expr.T @ diffu_xi
            # print(
            #     method_name,
            #     f"dx = ({sympy.N(found_drift_expr, 2)}) dt + {sympy.sqrt(sympy.N(2*found_diffu_expr, 2))}dW",
            # )
            # print(
            #     f"$dx = ({sympy.latex(sympy.N(found_drift_expr, 2))}) dt + {sympy.latex(sympy.sqrt(sympy.N(2*found_diffu_expr, 2)))} dW$"
            # )

            KM_plots_one_method(
                FIG_PATH / folder,
                KM,
                found_pdf,
                found_drift,
                found_diffu,
                f"{method_number}_{method_name}",
                folder,
            )
            found_pdfs.append(found_pdf)
            found_drifts.append(found_drift)
            found_drift_exprs.append(found_drift_expr)
            found_diffus.append(found_diffu)
            found_diffu_exprs.append(found_diffu_expr)
        KM_plots_all_methods(
            FIG_PATH / folder,
            KM,
            found_pdfs,
            found_drifts,
            found_diffus,
            method_names,
            folder,
        )

        all_drift_exprs.append(found_drift_exprs)
        all_diffu_exprs.append(found_diffu_exprs)

    all_drift_exprs = np.asarray(all_drift_exprs)
    all_diffu_exprs = np.asarray(all_diffu_exprs)

    format_line = "|c" * (len(equation_names) + 1) + "|"
    table_str = rf"""\begin{{table}}[]
    \centering
    \begin{{tabular}}{{{format_line}}}
    \hline
"""
    table_str += (
        "      " + " & ".join(["Method"] + equation_names) + "\\\\ \n \\hline \n"
    )
    table_str_drift = table_str
    table_str_diffu = table_str

    table_str_drift += (
        "    "
        + " & ".join(
            ["Truth"]
            + ["$" + sympy.latex(sympy.N(expr, 2)) + "$" for expr in true_drift_exprs]
        )
        + "\\\\ \n \\hline \n"
    )
    table_str_diffu += (
        "    "
        + " & ".join(
            ["Truth"]
            + ["$" + sympy.latex(sympy.N(expr, 2)) + "$" for expr in true_diffu_exprs]
        )
        + "\\\\ \n \\hline \n"
    )
    for method_number in range(len(method_names)):
        table_str_drift += "    " + " & ".join(
            [method_names[method_number].replace("_", " ")]
            + [
                "$" + sympy.latex(sympy.N(expr, 2)) + "$"
                for expr in all_drift_exprs[:, method_number]
            ]
        )
        table_str_diffu += "      " + " & ".join(
            [method_names[method_number].replace("_", " ")]
            + [
                "$" + sympy.latex(sympy.N(expr, 2)) + "$"
                for expr in 2 * all_diffu_exprs[:, method_number]
            ]
        )
        table_str_drift += "\\\\ \n \\hline \n"
        table_str_diffu += "\\\\ \n \\hline \n"
    table_str_drift += rf"""    \end{{tabular}}
    \caption{{Drift {cmn}}}
\end{{table}}"""
    table_str_diffu += rf"""    \end{{tabular}}
    \caption{{Diffu {cmn}}}
\end{{table}}"""
    print(table_str_drift)
    print(table_str_diffu)
    table_file.write(cmn + "\n")
    table_file.write(table_str_drift + "\n")
    table_file.write(table_str_diffu + "\n")
table_file.close()
