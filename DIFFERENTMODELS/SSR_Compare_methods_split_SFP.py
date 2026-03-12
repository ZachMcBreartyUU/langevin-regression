# %%
from time import time
from pathlib import Path
from functools import partial
import multiprocessing as mp

import sympy as sp
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from make_and_load_models import get_timeseries_and_KM, load_and_stack_KM
from utils import jeffreys_divergence, SteadyFP

# %%
SCRATCH_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path(
    f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods_soft_triple/"
)
FIG_PATH.mkdir(parents=True, exist_ok=True)


# %%
def cost_diffusion(
    xi_C: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    diffu_val = lib_diffu.T @ xi_C
    diffusion_fid = np.nansum(np.abs(diffusions - diffu_val)) / np.nansum(
        np.abs(diffusions)
    )

    if alpha != 1:
        pdf_lib = sfp.solve(np.nanmean(drifts, axis=0), diffu_val)
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        reg_val = np.sum(
            jeffreys_divergence(
                pdf_lib[:, None], pdfs.T, centers[1] - centers[0], tol=1e-8
            )
        ) / len(pdfs)
    else:
        reg_val = 0

    return (1 - alpha) * reg_val + alpha * diffusion_fid


def cost_drift(
    xi_A: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    sfp: SteadyFP,
    beta: float,
) -> float:
    centers, pdfs, drifts, diffusions = KM

    if xi_A[-1] > 0:
        return np.inf

    drift_val = lib_drift.T @ xi_A
    drift_fid = np.nansum(np.abs(drifts - drift_val)) / np.nansum(np.abs(drifts))

    if beta != 1:
        pdf_lib = sfp.solve(drift_val, np.nanmean(diffusions, axis=0))
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        reg_val = np.sum(
            jeffreys_divergence(
                pdf_lib[:, None], pdfs.T, centers[1] - centers[0], tol=1e-8
            )
        ) / len(pdfs)
    else:
        reg_val = 0

    return (1 - beta) * reg_val + beta * drift_fid


def opt_func_diffusion(xi0, KM, lib_diffu, sfp, alpha):
    res = minimize(
        partial(
            cost_diffusion,
            KM=KM,
            lib_diffu=lib_diffu,
            sfp=sfp,
            alpha=alpha,
        ),
        xi0,
        method="nelder-mead",
        options={"adaptive": True},
    )
    return res.x, res.fun


def opt_func_drift(xi0, KM, lib_drift, sfp, beta):
    xi0[-1] = -np.abs(xi0[-1])
    res2 = minimize(
        partial(
            cost_drift,
            KM=KM,
            lib_drift=lib_drift,
            sfp=sfp,
            beta=beta,
        ),
        xi0,
        method="nelder-mead",
        options={"adaptive": True},
    )

    return res2.x, res2.fun


# %%
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

    ax1.set_yscale("log")
    ax2.set_ylim(-2, 2)

    if suffix:
        fig.savefig(folder / f"kramers_moyal_zoom_{suffix}.png")
    else:
        fig.savefig(folder / "kramers_moyal_zoom.png")
    plt.close(fig)


def diffusion_plots_one_method(
    folder,
    KM,
    found_pdf,
    found_diffusion,
    diffu_expr: str,
    method_name: str,
    suffix="",
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 13))
    ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, linestyle="-", label="KM")
    ax1.plot(centers, found_pdf, linestyle="--", label="Model")
    ax1.set_ylabel(r"PDF, $P(x)$")
    ax1.set_ylabel(r"$x$")

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

    ax1.set_yscale("log")

    if suffix:
        fig.savefig(folder / f"diffusion_zoom_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"diffusion_zoom_{method_name}.png")

    plt.close(fig)


def drift_plots_one_method(
    folder,
    KM,
    found_pdf,
    found_drift,
    drift_expr: str,
    method_name: str,
    suffix="",
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 13))
    ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, linestyle="-", label="KM")
    ax1.plot(centers, found_pdf, linestyle="--", label="Model")
    ax1.set_ylabel(r"PDF, $P(x)$")
    ax1.set_xlabel(r"$x$")

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

    ax1.set_yscale("log")
    ax2.set_ylim(-2, 2)

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

    ax2.set_title(drift_expr)
    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_drift, linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
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
    ax2.set_ylim(-2, 2)

    if suffix:
        fig.savefig(folder / f"kramers_moyal_zoom_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"kramers_moyal_zoom_{method_name}.png")

    plt.close(fig)


def KM_plots_all_methods(
    folder, KM, found_pdfs, found_drifts, found_diffusions, method_names, suffix=""
):
    # I think this plot will look extremely messy
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 16))
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


def cost_coeffs_C_plots(folder, costs, xis, diffu_expr, method_name, suffix=""):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax.plot(costs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")

    ax2.imshow(xis.astype(bool).T)
    ylabels = [f"${sp.latex(expr)}$" for expr in diffu_expr]
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
    ylabels = [f"${sp.latex(expr)}$" for expr in drift_expr]
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
    EVEN_ABS: bool = False,
    ODD_ABS: bool = False,
):
    arr = []
    for i in range(order):
        if EVEN_ABS and i % 2 == 0 and i != 0:
            arr.append(sp.Abs(x_sym) * x_sym ** (i - 1))
        elif ODD_ABS and i % 2 == 1:
            arr.append(sp.Abs(x_sym) * x_sym ** (i - 1))
        else:
            arr.append(x_sym**i)
    lib_expr = np.array(arr)
    lib_KM = np.empty((order, len(centers)))
    for k in range(order):
        lamb = sp.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)

    return lib_expr, lib_KM


def SSR_loop_diffusion(opt_func_diffusion, xi_C_0_in, KM, lib_diffu_KM, sfp, alpha):
    n_terms = len(lib_diffu_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)

    mask = np.all(np.isfinite(diffusion), axis=0)

    xi_C_0, _ = opt_func_diffusion(xi_C_0_in, KM, lib_diffu_KM, sfp, 1.0)

    min_xis[0], min_Vs[0] = opt_func_diffusion(xi_C_0, KM, lib_diffu_KM, sfp, alpha)
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        for j in range(len(active)):
            tmp_active = np.delete(active.copy(), j)
            if len(tmp_active) == 0:
                continue
            # xi_C_0_tmp = np.abs(
            #     np.average(
            #         lstsq(lib_diffu_KM[tmp_active].T[mask], diffusion.T[mask])[0],
            #         axis=1,
            #     )
            # )
            xi_C_0_tmp, _ = opt_func_diffusion(
                xi_C_0_in[tmp_active], KM, lib_diffu_KM[tmp_active], sfp, 1.0
            )

            params = [
                xi_C_0_tmp,
                KM,
                lib_diffu_KM[tmp_active],
                sfp,
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


def SSR_loop_drift(opt_func_drift, xi_A_0_in, KM, lib_drift_KM, sfp, beta):
    n_terms = len(lib_drift_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)

    mask = np.all(np.isfinite(drift), axis=0)
    xi_A_0, _ = opt_func_drift(xi_A_0_in, KM, lib_drift_KM, sfp, 1.0)

    min_xis[0], min_Vs[0] = opt_func_drift(xi_A_0, KM, lib_drift_KM, sfp, beta)
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        for j in range(len(active)):
            tmp_active = np.delete(active.copy(), j)

            if len(tmp_active) == 0:
                continue

            # xi_A_0_tmp = np.average(
            #     lstsq(lib_drift_KM[tmp_active].T[mask], drift.T[mask])[0], axis=1
            # )
            xi_A_0_tmp, _ = opt_func_drift(
                xi_A_0_in[tmp_active], KM, lib_drift_KM[tmp_active], sfp, 1.0
            )

            params = [
                xi_A_0_tmp,
                KM,
                lib_drift_KM[tmp_active],
                sfp,
                beta,
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
    "Triple Well Pitch",
    "Triple Well Norm",
]
folders = [
    "DoubleWell",
    "TripleWell",
    "SoftglassPitchforking",
    "SoftglassNormal",
    "TripleWellPitch",
    "TripleWellNorm",
]
drift_coefficients = [
    # 1, x, x|x|, x^3, x^3|x|, ...
    [0.0, 1.0, 0.0, -1.0],
    [0.0, -1.0, 0.0, 1.0, 0.0, -0.2],
    [0.0, -0.016, 0.0, 1.1, -1.0],
    [0.0, -0.016, 0.0, 1.1, -1.0],
    [0.0, -0.016, 0.0, 0.7, 0.0, -0.7],
    [0.0, -0.016, 0.0, 0.7, 0.0, -0.7],
]
diffusion_coefficients = [
    # [epsilon_0, epsilon_1] -> diffu = sqrt(ep0 + ep1 x^2)
    [0.3, 0.2],
    [0.3, 0.1],
    [1e-3, 0.1],
    [1e-5, 0.1],
    [1e-3, 0.1],
    [1e-5, 0.1],
]

even_abs_simulation = [
    False,
    False,
    True,
    True,
    True,
    True,
]
even_abs_library = [
    False,
    False,
    False,
    False,
    False,
    False,
]


# %%
regression_method_names_diffusion = [
    "SFP",
    # "SFP",
    # "SFP_KM",
    "SFP_KM",
    "KM",
]
alpha_vals = [
    0.0,
    # 0.0,
    # 0.5,
    0.5,
    1.0,
]
opt_funcs_diffusion = [opt_func_diffusion] * len(alpha_vals)

regression_method_names_drift = [
    "SFP",
    # "SFP_KM",
    # "SFP",
    "SFP_KM",
    "KM",
]
beta_vals = [
    0.0,
    # 0.5,
    # 0.0,
    0.5,
    1.0,
]
opt_funcs_drift = [opt_func_drift] * len(beta_vals)
# %%
NUM_DATASETS = 10
NUM_VALIDATION = 1
NUM_CPUS = 6
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


# %%script true
for equation_number in [0, 1, 2, 3, 4, 5]:  # range(len(equation_names)):
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
        SCRATCH_PATH / folder,
        target_metadata,
        NUM_DATASETS,
        NUM_VALIDATION,
        NUM_CPUS,
        # (-2.0, 2.0),
    )
    del timeseries, val_timeseries

    # KM = load_and_stack_KM(SCRATCH_PATH / folder, 10)
    centers, pdf, drift, diffusion = KM
    # could do this instead as |drift - mean| > N std for example
    # since the value of 0.0 could actually be real
    # instead of being a lack of data
    drift[drift == 0.0] = np.nan
    diffusion[diffusion == 0.0] = np.nan
    KM = (centers, pdf, drift, diffusion)

    # timeseries_plots(FIG_PATH / folder, timeseries, folder, slice(0, 100_000, 1))
    KM_plots(FIG_PATH / folder, KM, folder)

    # make libraries for drift and diffusion
    x_sym = sp.symbols("x")
    num_drift = len(drift_coef) + drift_plus
    lib_drift_expr, lib_drift_KM = poly_lib(
        x_sym, num_drift, centers, even_abs_library[equation_number], False
    )
    num_diffusion = 3 + diffusion_plus
    lib_diffu_expr, lib_diffu_KM = poly_lib(x_sym, num_diffusion, centers, False, False)
    mask = np.all(np.isfinite(drift), axis=0)
    xi_A_0 = np.average(lstsq(lib_drift_KM.T[mask], drift.T[mask])[0], axis=1)
    mask = np.all(np.isfinite(diffusion), axis=0)
    xi_C_0 = np.abs(
        np.average(lstsq(lib_diffu_KM.T[mask], diffusion.T[mask])[0], axis=1)
    )

    sfp = SteadyFP(num_bins, centers[1] - centers[0])

    # regress each method against the same dataset with the same library functions,
    # the same initial conditions, and record the resulting final equation
    # do some plots of the final equation against the datasets (and save the data)

    method_xi_C_s = []
    method_xi_A_s = []
    method_v_C_s = []
    method_v_A_s = []
    best_xi_C_s = []
    best_xi_A_s = []

    for regression_method_number in range(len(regression_method_names_diffusion)):
        start = time()
        reg_method_name_diffusion = regression_method_names_diffusion[
            regression_method_number
        ]
        alpha_val = alpha_vals[regression_method_number]
        opt_func_diffusion_ = opt_funcs_diffusion[regression_method_number]
        print(reg_method_name_diffusion, alpha_val, opt_func_diffusion_)

        xi_C_s, V_C_s = SSR_loop_diffusion(
            opt_func_diffusion_, xi_C_0, KM, lib_diffu_KM, sfp, alpha_val
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
            diffu_expr = f"${sp.latex(sp.sqrt(sp.N(2 * lib_diffu_expr.T @ xi_C, 2)))}$"
            fdiffu = lib_diffu_KM.T @ xi_C
            fpdf = sfp.solve(np.nanmean(drift, axis=0), fdiffu)
            diffusion_plots_one_method(
                FIG_PATH / folder,
                KM,
                fpdf,
                fdiffu,
                diffu_expr,
                f"{np.count_nonzero(xi_C)}_{regression_method_number}_{reg_method_name_diffusion}",
                folder,
            )
        ######
        reg_method_name_drift = regression_method_names_drift[regression_method_number]
        beta_val = beta_vals[regression_method_number]
        opt_func_drift_ = opt_funcs_drift[regression_method_number]
        print(reg_method_name_drift, beta_val, opt_func_drift_)

        xi_A_s, V_A_s = SSR_loop_drift(
            opt_func_drift_, xi_A_0, KM, lib_drift_KM, sfp, beta_val
        )

        cost_coeffs_A_plots(
            FIG_PATH / folder,
            V_A_s,
            xi_A_s,
            lib_drift_expr,
            f"{regression_method_number}_{reg_method_name_drift}",
            suffix=folder,
        )

        method_xi_A_s.append(xi_A_s)
        method_v_A_s.append(V_A_s)
        end = time()
        print(f"Drift SSR time: {end-start}s")
        for xi_A in xi_A_s:
            drift_expr = f"${sp.latex(sp.N(lib_drift_expr.T @ xi_A, 2))}$"
            fdrift = lib_drift_KM.T @ xi_A
            fpdf = sfp.solve(fdrift, np.nanmean(diffusion, axis=0))
            drift_plots_one_method(
                FIG_PATH / folder,
                KM,
                fpdf,
                fdrift,
                drift_expr,
                f"{np.count_nonzero(xi_A)}_{regression_method_number}_{reg_method_name_drift}",
                folder,
            )

        ## Choose best drift, choose best diffusion ##
        choosing = (V_C_s[1:] - V_C_s[:-1]) / V_C_s[:-1]
        choosing_plots(
            FIG_PATH / folder,
            choosing,
            f"C_{regression_method_number}_{reg_method_name_diffusion}",
            folder,
        )
        chosen = np.nonzero(choosing > np.nanmax(choosing) / 5)[0][0]

        best_xi_C = xi_C_s[chosen]

        choosing = (V_A_s[1:] - V_A_s[:-1]) / V_A_s[:-1]
        choosing_plots(
            FIG_PATH / folder,
            choosing,
            f"A_{regression_method_number}_{reg_method_name_drift}",
            folder,
        )
        chosen = np.nonzero(choosing > np.nanmax(choosing) / 5)[0][0]

        best_xi_A = xi_A_s[chosen]

        found_drift_expr = lib_drift_expr.T @ best_xi_A
        found_diffu_expr = lib_diffu_expr.T @ best_xi_C
        found_drift_KM = lib_drift_KM.T @ best_xi_A
        found_diffu_KM = lib_diffu_KM.T @ best_xi_C

        best_xi_C_s.append(best_xi_C)
        best_xi_A_s.append(best_xi_A)

        print(
            f"dx = ({sp.N(found_drift_expr, 2)}) dt + {sp.sqrt(sp.N(2 * found_diffu_expr, 2))} dW\n",
        )

        found_pdf = sfp.solve(found_drift_KM, found_diffu_KM)
        KM_plots_one_method(
            FIG_PATH / folder,
            KM,
            found_pdf,
            found_drift_KM,
            found_diffu_KM,
            f"${sp.latex(sp.N(found_drift_expr, 2))}$",
            f"${sp.latex(sp.sqrt(sp.N(2 * found_diffu_expr, 2)))}$",
            f"{regression_method_number}_{reg_method_name_diffusion}_{reg_method_name_drift}",
            folder,
        )
    del KM, centers, pdf, drift, diffusion  # , val_KM

    print("\n\n")
    np.savez(
        FIG_PATH / folder / "SSR_result.npz",
        method_xi_C_s=method_xi_C_s,
        method_v_C_s=method_v_C_s,
        method_xi_A_s=method_xi_A_s,
        method_v_A_s=method_v_A_s,
        true_xi_A=drift_coef,
        best_xi_A_s=best_xi_A_s,
        true_xi_C=np.array([ep0, 0.0, ep1]),
        best_xi_C_s=best_xi_C_s,
    )

# with np.load(FIG_PATH / folder / "SSR_result.npz") as f:
#     true_xi_A = f["true_xi_A"]
#     best_xi_A_s = f["best_xi_A_s"]
#     true_xi_C = f["true_xi_C"]
#     best_xi_C_s = f["best_xi_C_s"]

# format_line = "|p{2cm}" + "|p{4cm}" * (len(equation_names)) + "|"
# table_str = rf"""\begin{{table}}[]
# \centering
# \begin{{tabular}}{{{format_line}}}
# \hline
# """
# table_str += "      " + " & ".join(["Method"] + equation_names) + "\\\\ \n  \\hline \n"
# table_str_drift = table_str
# table_str_diffu = table_str
# table_str_drift += (
#     "    "
#     + " & ".join(
#         ["Truth"] + ["$" + sp.latex(sp.N(expr, 2)) + "$" for expr in true_drift_exprs]
#     )
#     + "\\\\ \n  \\hline \n"
# )
# table_str_diffu += (
#     "    "
#     + " & ".join(
#         ["Truth"] + ["$" + sp.latex(sp.N(expr, 2)) + "$" for expr in true_diffu_exprs]
#     )
#     + "\\\\ \n  \\hline \n"
# )
# for equation_number in range(len(equation_names)):
#     for regression_method_number in range(len(regression_method_names)):
#         table_str_drift += "    " + " & ".join(
#             [regression_method_names[regression_method_number].replace("_", " ")]
#             + [
#                 "$" + sp.latex(sp.N(expr, 2)) + "$"
#                 for expr in all_drift_exprs[:, regression_method_number]
#             ]
#         )
#         table_str_diffu += "      " + " & ".join(
#             [regression_method_names[regression_method_number].replace("_", " ")]
#             + [
#                 "$" + sp.latex(sp.N(expr, 2)) + "$"
#                 for expr in 2 * all_diffu_exprs[:, regression_method_number]
#             ]
#         )
#         table_str_drift += "\\\\ \n  \\hline \n"
#         table_str_diffu += "\\\\ \n  \\hline \n"
#     table_str_drift += rf"""    \end{{tabular}}
#     \caption{{Drift }}
#     \end{{table}}"""
#     table_str_diffu += rf"""    \end{{tabular}}
#     \caption{{Diffu }}
#     \end{{table}}"""
#     print(table_str_drift)
#     print(table_str_diffu)
#     with open() as table_file:
#         table_file.write(table_str_drift + "\n")
#         table_file.write(table_str_diffu + "\n")
#         table_file.close()
