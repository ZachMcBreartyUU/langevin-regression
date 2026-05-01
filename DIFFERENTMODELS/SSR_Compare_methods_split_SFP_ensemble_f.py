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

from make_and_load_models import get_timeseries_and_KM, plot_timeseries
from utils import jeffreys_divergence, SteadyFP

# %%
SCRATCH_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path(
    f"/home/zachuu/scratch/seismology/zach/softglass/compare_methods_f_abs/"
)
FIG_PATH.mkdir(parents=True, exist_ok=True)

# %%
DEBUG = True


# %%
p = 2


def cost_diffusion(
    xi_C: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    centers, pdf, drift, diffusion = KM

    diffu_val = lib_diffu.T @ xi_C
    diffusion_fid = np.nansum(np.abs(diffusion - diffu_val) ** p) / np.nansum(
        np.abs(diffusion) ** p
    )

    if alpha != 1:
        pdf_lib = sfp.solve(drift, diffu_val)
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        reg_val = np.sum(
            jeffreys_divergence(pdf_lib, pdf, centers[1] - centers[0], tol=1e-8)
        )
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
    centers, pdf, drift, diffusion = KM

    if xi_A[-1] > 0:
        return np.inf

    drift_val = lib_drift.T @ xi_A
    drift_fid = np.nansum(np.abs(drift - drift_val) ** p) / np.nansum(
        np.abs(drift) ** p
    )

    if beta != 1:
        pdf_lib = sfp.solve(drift_val, diffusion)
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        reg_val = np.sum(
            jeffreys_divergence(pdf_lib, pdf, centers[1] - centers[0], tol=1e-8)
        )
    else:
        reg_val = 0

    return (1 - beta) * reg_val + beta * drift_fid


def opt_func_diffusion(xi0, KMs, lib_diffu, sfp, alpha):
    xs = []
    funs = []

    for KM in KMs:
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
        xs.append(res.x)
        funs.append(res.fun)
    xs = np.asarray(xs)
    funs = np.asarray(funs)
    return xs, funs


def opt_func_drift(xi0, KMs, lib_drift, sfp, beta):
    xi0[-1] = -np.abs(xi0[-1])
    xs = []
    funs = []
    for KM in KMs:
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
        xs.append(res2.x)
        funs.append(res2.fun)
    xs = np.asarray(xs)
    funs = np.asarray(funs)
    return xs, funs


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
        plt.delaxes(ax)
        plt.close(fig)


def KM_plots(folder, KM, suffix=""):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 16))

    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.set_ylabel("PDF, $P(x)$")
    ax1.set_xlabel("$x$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_xlabel("$x$")

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
    ax2.set_ylim(-0.1, 0.1)

    if suffix:
        fig.savefig(folder / f"kramers_moyal_zoom_{suffix}.png")
    else:
        fig.savefig(folder / "kramers_moyal_zoom.png")
    plt.delaxes(ax1)
    plt.delaxes(ax2)
    plt.delaxes(ax3)
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
    # ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, linestyle="", marker="x", label="KM")
    ax1.plot(centers, found_pdf, linestyle="-", label="Model")
    ax1.set_ylabel(r"PDF, $P(x)$")
    ax1.set_ylabel(r"$x$")

    # ax2.set_title(diffu_expr)
    ax2.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_diffusion, linestyle="-")
    ax2.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax2.set_xlabel("$x$")
    # mean_vals = np.nanmean(diffusion_stack, axis=0)
    # min_ = np.abs(np.min(mean_vals))
    # max_ = np.abs(np.max(mean_vals))
    # ax2.set_ylim(
    #     np.min(mean_vals) - 0.01 * (min_ + max_),
    #     np.max(mean_vals) + 0.01 * (min_ + max_),
    # )

    # fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"diffusion_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"diffusion_{method_name}.png")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

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
    # ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, linestyle="", marker="x", label="KM")
    ax1.plot(centers, found_pdf, linestyle="-", label="Model")
    ax1.set_ylabel(r"PDF, $P(x)$")
    ax1.set_xlabel(r"$x$")

    # ax2.set_title(drift_expr)
    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.plot(centers, found_drift, linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_xlabel("$x$")
    # mean_vals = np.nanmean(drift_stack, axis=0)
    # min_ = np.abs(np.min(mean_vals))
    # max_ = np.abs(np.max(mean_vals))
    # ax2.set_ylim(
    #     np.min(mean_vals) - 0.01 * (min_ + max_) / 2,
    #     np.max(mean_vals) + 0.01 * (min_ + max_) / 2,
    # )

    # fig.legend()
    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"drift_{method_name}_{suffix}.png")
    else:
        fig.savefig(folder / f"drift_{method_name}.png")

    ax1.set_yscale("log")
    # ax2.set_ylim(-0.1, 0.1)

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
    # ax1.set_title(method_name)
    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.plot(centers, found_pdf, linestyle="-", label="Model")
    ax1.set_ylabel("PDF, $P(x)$")
    ax1.set_xlabel("$x$")

    # ax2.set_title(drift_expr)
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

    # ax3.set_title(diffu_expr)
    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.plot(centers, found_diffusion, linestyle="-")
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax3.set_xlabel("$x$")
    mean_vals = np.nanmean(diffusion_stack, axis=0)
    min_ = np.abs(np.min(mean_vals))
    max_ = np.abs(np.max(mean_vals))
    ax3.set_ylim(
        np.min(mean_vals) - 0.01 * (min_ + max_),
        np.max(mean_vals) + 0.01 * (min_ + max_),
    )

    # fig.legend()
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
    ax1.set_xlabel("$x$")

    ax2.plot(centers, drift_stack.T, alpha=0.7, linestyle="", marker="x")
    ax2.set_ylabel("Drift, $m^{(1)}(x)$")
    ax2.set_xlabel("$x$")
    mean_vals = np.nanmean(drift_stack, axis=0)
    ax2.set_ylim(np.min(mean_vals), np.max(mean_vals))
    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.set_ylabel("Diffusion, $m^{(2)}(x)$")
    ax3.set_xlabel("$x$")
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
        if EVEN_ABS and i % 2 == 0:  # and i != 0:
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


def poly_lib_sqrt(
    x_sym,
    order: int,
    centers: np.ndarray,
):
    arr = []
    for i in range(2 * order):
        arr.append(x_sym ** (i / 2))
    lib_expr = np.array(arr)
    lib_KM = np.empty((2 * order, len(centers)))
    for k in range(2 * order):
        lamb = sp.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)

    return lib_expr, lib_KM


def SSR_loop_diffusion(
    opt_func_diffusion, xi_C_0_in, KMs, lib_diffu_KM, sfp, alpha, lib_diffu_expr
):
    n_terms = len(lib_diffu_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_xi_vars = np.zeros((n_terms, n_terms))

    min_Vs = np.full((n_terms), np.inf)
    min_V_vars = np.zeros((n_terms))

    xi_C_0s, _ = opt_func_diffusion(xi_C_0_in, KMs, lib_diffu_KM, sfp, 1.0)

    first_xi, first_V = opt_func_diffusion(
        np.nanmean(xi_C_0s, axis=0), KMs, lib_diffu_KM, sfp, alpha
    )
    min_xis[0] = np.nanmean(first_xi, axis=0)
    min_xi_vars[0] = np.nanvar(first_xi, axis=0)
    min_Vs[0] = np.nanmean(first_V)
    min_V_vars[0] = np.nanvar(first_V)
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        active_list = []
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
            xi_C_0s_tmp, _ = opt_func_diffusion(
                xi_C_0_in[tmp_active], KMs, lib_diffu_KM[tmp_active], sfp, 1.0
            )

            params = [
                np.nanmean(xi_C_0s_tmp, axis=0),
                KMs,
                lib_diffu_KM[tmp_active],
                sfp,
                alpha,
            ]
            params_list.append(params)
            valid_indices.append(j)
            active_list.append(tmp_active)

        with mp.Pool(NUM_CPUS) as p:
            results = p.starmap(opt_func_diffusion, params_list)
        xis = []
        xi_vars = []
        Vs = []
        V_vars = []
        for (Xi, V), tmp_active in zip(results, active_list):
            xi_mean = np.nanmean(Xi, axis=0)
            xi_var = np.nanvar(Xi, axis=0)
            V_mean = np.nanmean(V)
            V_var = np.nanvar(V)

            if DEBUG:
                for i in range(len(xi_mean)):
                    expr = str(lib_diffu_expr[tmp_active][i])
                    expr = expr.replace("**", "^")
                    print(
                        f"{expr:>4}: {2*xi_mean[i]: .2e} ± {2*np.sqrt(xi_var[i]):.2e}",
                        end=", ",
                    )
                print(f"{V_mean: .2e} ± {np.sqrt(V_var):.2e}")

            xis.append(xi_mean)
            xi_vars.append(xi_var)
            Vs.append(V_mean)
            V_vars.append(V_var)
        if DEBUG:
            print()
        min_cost = np.nanargmin(Vs)
        min_idx = valid_indices[min_cost]
        min_V = Vs[min_cost]
        min_V_var = V_vars[min_cost]
        min_xi = xis[min_cost]
        min_xi_var = xi_vars[min_cost]
        active = np.delete(active, min_idx)
        min_Vs[k] = min_V
        min_V_vars[k] = min_V_var

        min_xis[k, active] = min_xi
        min_xi_vars[k, active] = min_xi_var

    return min_xis, min_xi_vars, min_Vs, min_V_vars


def SSR_loop_drift_min_variance(
    opt_func_drift, xi_A_0_in, KMs, lib_drift_KM, sfp, beta
):
    n_terms = len(lib_drift_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_xi_vars = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)
    min_V_vars = np.zeros((n_terms))

    xi_A_0s, _ = opt_func_drift(xi_A_0_in, KMs, lib_drift_KM, sfp, 1.0)

    first_xi, first_V = opt_func_drift(
        np.nanmean(xi_A_0s, axis=0), KMs, lib_drift_KM, sfp, beta
    )
    min_xis[0] = np.nanmean(first_xi, axis=0)
    min_xi_vars[0] = np.nanvar(first_xi, axis=0)
    min_Vs[0] = np.nanmean(first_V)
    min_V_vars[0] = np.nanvar(first_V)
    active = np.array(list(range(n_terms)))

    indx_to_remove = np.nanargmax(np.abs(np.sqrt(min_xi_vars[0]) / min_xis[0]))
    np.delete(active, indx_to_remove)

    for k in range(n_terms):
        xi_A_s, V_A_s = opt_func_drift(
            xi_A_0_in[active], KMs, lib_drift_KM[active], sfp, beta
        )
        xi_A_mean = np.nanmean(xi_A_s, axis=0)
        xi_A_var = np.nanvar(xi_A_s, axis=0)

        V_A_mean = np.nanmean(V_A_s)
        V_A_var = np.nanvar(V_A_s)

        min_Vs[k] = V_A_mean
        min_V_vars[k] = V_A_var
        min_xis[k, active] = xi_A_mean
        min_xi_vars[k, active] = xi_A_var

        indx_to_remove = np.nanargmax(np.abs(np.sqrt(xi_A_var) / xi_A_mean))
        active = np.delete(active, indx_to_remove)

    return min_xis, min_xi_vars, min_Vs, min_V_vars


def SSR_loop_drift(
    opt_func_drift, xi_A_0_in, KMs, lib_drift_KM, sfp, beta, lib_drift_expr
):
    n_terms = len(lib_drift_KM)
    min_xis = np.zeros((n_terms, n_terms))
    min_xi_vars = np.zeros((n_terms, n_terms))
    min_Vs = np.full((n_terms), np.inf)
    min_V_vars = np.zeros((n_terms))

    xi_A_0s, _ = opt_func_drift(xi_A_0_in, KMs, lib_drift_KM, sfp, 1.0)

    first_xi, first_V = opt_func_drift(
        np.nanmean(xi_A_0s, axis=0), KMs, lib_drift_KM, sfp, beta
    )
    min_xis[0] = np.nanmean(first_xi, axis=0)
    min_xi_vars[0] = np.nanvar(first_xi, axis=0)
    min_Vs[0] = np.nanmean(first_V)
    min_V_vars[0] = np.nanvar(first_V)
    active = np.array(list(range(n_terms)))

    for k in range(1, n_terms):
        params_list = []
        valid_indices = []
        active_list = []
        for j in range(len(active)):
            tmp_active = np.delete(active.copy(), j)

            if len(tmp_active) == 0:
                continue

            # xi_A_0_tmp = np.average(
            #     lstsq(lib_drift_KM[tmp_active].T[mask], drift.T[mask])[0], axis=1
            # )
            xi_A_0s_tmp, _ = opt_func_drift(
                xi_A_0_in[tmp_active], KMs, lib_drift_KM[tmp_active], sfp, 1.0
            )

            params = [
                np.nanmean(xi_A_0s_tmp, axis=0),
                KMs,
                lib_drift_KM[tmp_active],
                sfp,
                beta,
            ]
            params_list.append(params)
            valid_indices.append(j)
            active_list.append(tmp_active)

        with mp.Pool(NUM_CPUS) as p:
            results = p.starmap(opt_func_drift, params_list)
        xis = []
        xi_vars = []
        Vs = []
        V_vars = []
        for (Xi, V), tmp_active in zip(results, active_list):
            xi_mean = np.nanmean(Xi, axis=0)
            xi_var = np.nanvar(Xi, axis=0)
            V_mean = np.nanmean(V)
            V_var = np.nanvar(V)
            if DEBUG:
                for i in range(len(xi_mean)):
                    expr = str(lib_drift_expr[tmp_active][i])
                    expr = expr.replace("Abs(x)", "|x|").replace("**", "^")
                    print(
                        f"{expr:>7}: {xi_mean[i]: .2e} ± {np.sqrt(xi_var[i]):.2e}",
                        end=", ",
                    )
                print(f"{V_mean: .2e} ± {np.sqrt(V_var):.2e}")
            xis.append(xi_mean)
            xi_vars.append(xi_var)
            Vs.append(V_mean)
            V_vars.append(V_var)
        if DEBUG:
            print()
        min_cost = np.nanargmin(Vs)
        min_idx = valid_indices[min_cost]
        min_V = Vs[min_cost]
        min_V_var = V_vars[min_cost]
        min_xi = xis[min_cost]
        min_xi_var = xi_vars[min_cost]
        active = np.delete(active, min_idx)
        min_Vs[k] = min_V
        min_V_vars[k] = min_V_var
        min_xis[k, active] = min_xi
        min_xi_vars[k, active] = min_xi_var

    return min_xis, min_xi_vars, min_Vs, min_V_vars


# %%
equation_names = [
    "Double Well",
    "Triple Well",
    "Softglass (Pitchforking)",
    "Softglass (Normal)",
    "Triple Well Pitch",
    "Triple Well Norm",
    "Softglass (Small Sigma)",
    "Softglass (Medium Sigma)",
]
folders = [
    "DoubleWell",
    "TripleWell",
    "SoftglassPitchforking",
    "SoftglassNormal",
    "TripleWellPitch",
    "TripleWellNorm",
    "SoftglassSmallSigma",
    "SoftglassMediumSigma",
]
drift_coefficients = [
    # 1, x, x|x|, x^3, x^3|x|, ...
    [0.0, 1.0, 0.0, -1.0],
    [0.0, -1.0, 0.0, 1.0, 0.0, -0.2],
    [0.0, -0.016, 0.0, 1.1, -1.0],
    [0.0, -0.016, 0.0, 1.1, -1.0],
    [0.0, -0.016, 0.0, 0.7, 0.0, -0.7],
    [0.0, -0.016, 0.0, 0.7, 0.0, -0.7],
    [0.0, -0.016, 0.0, 0.4, -1.0],
    [0.0, -0.016, 0.0, 0.7, -1.0],
]
diffusion_coefficients = [
    # [epsilon_0, epsilon_1] -> diffu = sqrt(ep0 + ep1 x^2)
    [0.3, 0.2],
    [0.3, 0.1],
    [1e-3, 0.1],
    [1e-5, 0.1],
    [1e-3, 0.1],
    [1e-5, 0.1],
    [1e-5, 0.1],
    [1e-5, 0.1],
]

even_abs_simulation = [
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
]
even_abs_library = [
    False,
    False,
    True,  # False,
    True,  # False,
    True,  # False,
    True,  # False,
    True,  # False,
    True,  # False,
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
    "bandwidth": 0.1,
}
drift_plus = 1
diffusion_plus = 3


# %%script true
for equation_number in [2, 3, 6, 7]:  # range(len(equation_names)):
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
        (0.0, 2.5),
        f=True,
    )

    # KM = load_and_stack_KM(SCRATCH_PATH / folder, 10)
    centers, pdfs, drifts, diffusions = KM
    # could do this instead as |drift - mean| > N std for example
    # since the value of 0.0 could actually be real
    # instead of being a lack of data
    drifts[drifts == 0.0] = np.nan
    diffusions[diffusions == 0.0] = np.nan
    KM = (centers, pdfs, drifts, diffusions)
    KMs = [(centers, pdfs[i], drifts[i], diffusions[i]) for i in range(NUM_DATASETS)]
    sfp = SteadyFP(num_bins, centers[1] - centers[0])
    TRUE_DRIFT = (
        ep0
        + (ep1 + 2 * drift_coef[1]) * centers
        + 2 * drift_coef[3] * centers**2
        + 2 * drift_coef[4] * centers ** (5 / 2)
    )
    TRUE_DIFFUSION = 2 * ep0 * centers + 2 * ep1 * centers**2
    TRUE_PDF = sfp.solve(TRUE_DRIFT, TRUE_DIFFUSION)
    KMs = [(centers, TRUE_PDF, TRUE_DRIFT, TRUE_DIFFUSION) for i in range(NUM_DATASETS)]
    KM = (
        centers,
        np.asarray([KMsi[1] for KMsi in KMs]),
        np.asarray([KMsi[2] for KMsi in KMs]),
        np.asarray([KMsi[3] for KMsi in KMs]),
    )

    timeseries_plots(FIG_PATH / folder, timeseries, folder, slice(0, 100_000, 1))
    del timeseries, val_timeseries
    KM_plots(FIG_PATH / folder, KM, folder)

    # make libraries for drift and diffusion
    x_sym = sp.symbols("f")
    num_drift = len(drift_coef) // 2 + drift_plus
    lib_drift_expr, lib_drift_KM = poly_lib_sqrt(x_sym, num_drift, centers)
    # lib_drift_expr = lib_drift_expr[[1, 3, 4]]
    # lib_drift_KM = lib_drift_KM[[1, 3, 4]]

    num_diffusion = 3 + diffusion_plus
    lib_diffu_expr, lib_diffu_KM = poly_lib(x_sym, num_diffusion, centers, False, False)
    mask = np.all(np.isfinite(drifts), axis=0)
    xi_A_0 = np.average(lstsq(lib_drift_KM.T[mask], drifts.T[mask])[0], axis=1)
    mask = np.all(np.isfinite(diffusions), axis=0)
    xi_C_0 = np.abs(
        np.average(lstsq(lib_diffu_KM.T[mask], diffusions.T[mask])[0], axis=1)
    )

    # regress each method against the same dataset with the same library functions,
    # the same initial conditions, and record the resulting final equation
    # do some plots of the final equation against the datasets (and save the data)

    method_xi_C_s = []
    method_xi_A_s = []
    method_v_C_s = []
    method_v_A_s = []
    best_xi_C_s = []
    best_xi_A_s = []

    for regression_method_number in [1, 2]:
        # range(len(regression_method_names_diffusion)):
        start = time()
        reg_method_name_diffusion = regression_method_names_diffusion[
            regression_method_number
        ]
        alpha_val = alpha_vals[regression_method_number]
        opt_func_diffusion_ = opt_funcs_diffusion[regression_method_number]
        print(reg_method_name_diffusion, alpha_val, opt_func_diffusion_)

        xi_C_s, xi_C_vars, V_C_s, V_C_vars = SSR_loop_diffusion(
            opt_func_diffusion_,
            xi_C_0,
            KMs,
            lib_diffu_KM,
            sfp,
            alpha_val,
            lib_diffu_expr,
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
        for xi_C, xi_C_var, V_C, V_C_var in zip(xi_C_s, xi_C_vars, V_C_s, V_C_vars):
            diffu_expr = f"${sp.latex(sp.sqrt(sp.N(2 * lib_diffu_expr.T @ xi_C, 2)))}$"
            for i in range(len(xi_C)):
                if xi_C[i] == 0:
                    continue
                print(
                    f"{str(lib_diffu_expr[i]):>4}: {2*xi_C[i]: .2e} ± {2*np.sqrt(xi_C_var[i]):.2e}",
                    end=", ",
                )
            print(f"COST: {V_C_s} ± {np.sqrt(V_C_vars)}\n")
            fdiffu = lib_diffu_KM.T @ xi_C
            fpdf = sfp.solve(np.nanmean(drifts, axis=0), fdiffu)
            diffusion_plots_one_method(
                FIG_PATH / folder,
                KM,
                fpdf,
                fdiffu,
                diffu_expr,
                f"{np.count_nonzero(xi_C)}_{regression_method_number}_{reg_method_name_diffusion}",
                folder,
            )
        print("\n\n")
        ######
        reg_method_name_drift = regression_method_names_drift[regression_method_number]
        beta_val = beta_vals[regression_method_number]
        opt_func_drift_ = opt_funcs_drift[regression_method_number]
        print(reg_method_name_drift, beta_val, opt_func_drift_)

        xi_A_s, xi_A_vars, V_A_s, V_A_vars = SSR_loop_drift(
            opt_func_drift_, xi_A_0, KMs, lib_drift_KM, sfp, beta_val, lib_drift_expr
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
        for xi_A, xi_A_var, V_A, V_A_var in zip(xi_A_s, xi_A_vars, V_A_s, V_A_vars):
            drift_expr = f"${sp.latex(sp.N(lib_drift_expr.T @ xi_A, 2))}$"
            for i in range(len(xi_A)):
                if xi_A[i] == 0:
                    continue
                print(
                    f"{str(lib_drift_expr[i]):>4}: {xi_A[i]: .2e} ± {np.sqrt(xi_A_var[i]):.2e}: {np.abs(np.sqrt(xi_A_var[i]) / xi_A[i]):.2e}",
                    end=", ",
                )
            print(f"COST: {V_A_s} ± {np.sqrt(V_A_vars)}\n")
            fdrift = lib_drift_KM.T @ xi_A
            fpdf = sfp.solve(fdrift, np.nanmean(diffusions, axis=0))
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
    del KM, centers, pdfs, drifts, diffusions  # , val_KM

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
