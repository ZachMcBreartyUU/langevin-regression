from pathlib import Path
from functools import partial
import multiprocessing as mp

import sympy
import symengine
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

from make_and_load_models import get_timeseries_and_KM
from kramersmoyal import km
from utils import sindy_model, jeffreys_divergence

from mpl_toolkits.axisartist import Axes  # typing


def cost_diffusion(
    xi_C,
    lib_C_KM,
    lib_C_timeseries,
    diffusion_KM,
    dx,
    normal_dist,
    normal_dist_edges,
    alpha=0,
) -> float:
    if np.any(xi_C < 0):
        return np.inf
    dw_Q = dx / np.sqrt(2 * lib_C_timeseries.transpose((1, 2, 0)) @ xi_C)
    # pdf for each trajectory separately?
    (dw_Q_hist,), _ = km(dw_Q.flatten(), powers=0, bins=(normal_dist_edges,))  # type: ignore
    norm_dist_dx = normal_dist_edges[1] - normal_dist_edges[0]
    dw_Q_hist /= np.sum(dw_Q_hist * norm_dist_dx)

    jef_diverg = jeffreys_divergence(dw_Q_hist, normal_dist, dx=norm_dist_dx)
    if diffusion_KM.shape[0] != 1:
        diffusion_weight = 1 / np.nanvar(diffusion_KM, axis=0)
    else:
        diffusion_weight = np.ones_like(diffusion_KM[:, 0])
    diffusion_weight /= np.nansum(diffusion_weight)
    diffusion_fidelity = np.nansum(
        diffusion_weight * ((lib_C_KM.T @ xi_C)[None, ...] - diffusion_KM) ** 2
    )
    return (1 - alpha) * jef_diverg + alpha / 2 * diffusion_fidelity


def opt_fun_C(
    xi_C_0,
    lib_C_KM_reduced,
    lib_C_timeseries_reduced,
    alpha,
    diffusion_KM,
    dx,
    normal_dist,
    normal_dist_edges,
):
    res = minimize(
        partial(
            cost_diffusion,
            lib_C_KM=lib_C_KM_reduced,
            lib_C_timeseries=lib_C_timeseries_reduced,
            diffusion_KM=diffusion_KM,
            dx=dx,
            normal_dist=normal_dist,
            normal_dist_edges=normal_dist_edges,
            alpha=alpha,
        ),
        xi_C_0,
        method="nelder-mead",
    )
    return res.x, res.fun


def cost_drift(
    xi_A,
    lib_A_KM,
    lib_A_timeseries,
    drift_KM,
    diffusion_model,
    dx,
    dt,
    normal_dist,
    normal_dist_edges,
    beta=0,
) -> float:
    dw_P = (dx - lib_A_timeseries.transpose((1, 2, 0)) @ xi_A * dt) / diffusion_model
    (dw_P_hist,), _ = km(dw_P.flatten(), powers=0, bins=(normal_dist_edges,))  # type: ignore
    norm_dist_dx = normal_dist_edges[1] - normal_dist_edges[0]
    dw_P_hist /= np.sum(dw_P_hist * norm_dist_dx)
    jef_diverg = jeffreys_divergence(dw_P_hist, normal_dist, dx=norm_dist_dx)

    if drift_KM.shape[0] != 1:
        drift_weight = 1 / np.nanvar(drift_KM, axis=0)
    else:
        drift_weight = np.ones_like(drift_KM[:, 0])
    drift_weight /= np.nansum(drift_weight)
    drift_fidelity = np.nansum(
        drift_weight * ((lib_A_KM.T @ xi_A)[None, ...] - drift_KM) ** 2
    )
    return (1 - beta) * jef_diverg + beta / 2 * drift_fidelity


def opt_fun_A(
    xi_A_0,
    lib_A_KM_reduced,
    lib_A_timeseries_reduced,
    beta,
    drift_KM,
    diffusion_model_timeseries,
    dx,
    dt,
    normal_dist,
    normal_dist_edges,
):
    res = minimize(
        partial(
            cost_drift,
            lib_A_KM=lib_A_KM_reduced,
            lib_A_timeseries=lib_A_timeseries_reduced,
            drift_KM=drift_KM,
            diffusion_model=diffusion_model_timeseries,
            dx=dx,
            dt=dt,
            normal_dist=normal_dist,
            normal_dist_edges=normal_dist_edges,
            beta=beta,
        ),
        xi_A_0,
        method="nelder-mead",
    )
    return res.x, res.fun


def run_model(
    models_dir,
    alpha,
    beta,
    num_datapoints=1_000_000,
    dt=0.001,
    R=0.016,
    ms=0.7,
    ep0=1e-5,
    ep1=0.1,
    num_bins=100,
    NUM_DATASETS=10,
    NUM_VALIDATION=1,
    NUM_CPUS=1,
    timeseries_skip=100,
    norm_dist_width=5,
    norm_dist_points=500,
    num_A_expr=6,
    num_C_expr=5,
):
    target_metadata = {  # Direct or in call?
        "num_datapoints": num_datapoints,
        "dt": dt,
        "EVEN_ABS": True,
        "coeffs": [0, -R, 0, ms, -1.0],
        "ep0": ep0,
        "ep1": ep1,
        "x0": 0.0,
        "num_bins": num_bins,
    }
    # timeseries: (time, x): shape (NUM_MODELS, num_datapoints)
    # models: (centers, pdf, moment_1, moment_2): centers.shape (num_bins,), others.shape (NUM_MODELS, num_bins)
    timeseries, KM, validation_timeseries, validation_KM = get_timeseries_and_KM(
        models_dir,
        target_metadata,
        NUM_DATASETS,
        NUM_VALIDATION,
        NUM_CPUS,
        None,
    )

    time, x = timeseries
    for i in range(NUM_DATASETS):
        fig, ax = plt.subplots()
        ax.plot(time[i], x[i])
        ax.set_xlabel("time, t")
        ax.set_ylabel("value, x")
        fig.savefig(FIG_PATH / f"dataset_{i}.png")
        plt.close(fig)

    dx = x[:, 1:] - x[:, :-1]

    # time, x, dx: shape (NUM_MODELS, (num_datapoints - 1) // timeseries_skip)
    time = time[:, :-1:timeseries_skip]
    x = x[:, :-1:timeseries_skip]
    dx = dx[:, ::timeseries_skip]

    centers_KM, pdf_KM, drift_KM, diffusion_KM = KM
    drift_KM[drift_KM == 0] = np.nan
    diffusion_KM[diffusion_KM == 0] = np.nan

    normal_dist_edges, normal_dist_dx = np.linspace(
        -norm_dist_width * np.sqrt(dt),
        norm_dist_width * np.sqrt(dt),
        norm_dist_points,
        retstep=True,
    )
    normal_dist_centers = (normal_dist_edges[1:] + normal_dist_edges[:-1]) / 2
    normal_dist = np.exp(-(normal_dist_centers**2) / (2 * dt))
    normal_dist /= np.sum(normal_dist * normal_dist_dx)

    # make libraries for drift (A) and diffusion (C) (both timeseries and models)
    x_sym = sympy.symbols("x")

    arr = [x_sym**0]  # 1, x, x|x|, x^3...
    for i in range(1, num_A_expr):
        arr.append(
            sympy.Abs(x_sym) ** (i - 1)  # type: ignore
            * x_sym
            / np.prod(range(1, i + 1), initial=1)
        )
    lib_A_expr = np.array(arr)

    lib_A_KM = np.empty((num_A_expr, num_bins))
    lib_A_timeseries = np.empty((num_A_expr, *x.shape))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, lib_A_expr[k])
        lib_A_KM[k] = lamb_expr(centers_KM)
        lib_A_timeseries[k] = lamb_expr(x)

    lib_C_expr = np.array(
        [
            sympy.Abs(x_sym) ** i / np.prod(range(1, i + 1), initial=1)  # type: ignore
            for i in range(num_C_expr)
        ]
    )

    lib_C_KM = np.empty((num_C_expr, num_bins))
    lib_C_timeseries = np.empty((num_C_expr, *x.shape))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, lib_C_expr[k])
        lib_C_KM[k] = lamb_expr(centers_KM)
        lib_C_timeseries[k] = lamb_expr(x)

    ## ## SPLIT HERE AND DO IN PARALLEL? ## ##
    if isinstance(alpha, (float, int)):
        alpha = [alpha]
    if isinstance(beta, (float, int)):
        beta = [beta]

    cost_C_Jeffs_2d = []
    cost_C_Fids_2d = []
    cost_A_Jeffs_2d = []
    cost_A_Fids_2d = []

    xi_C_model_2d = []
    xi_A_model_2d = []

    for alpha_val in alpha:
        cost_C_Jeffs_1d = []
        cost_C_Fids_1d = []
        cost_A_Jeffs_1d = []
        cost_A_Fids_1d = []
        xi_C_model_1d = []
        xi_A_model_1d = []
        xi_C_0 = np.ones(num_C_expr)
        best_cost_C = np.full((num_C_expr,), np.inf)
        best_xi_C = np.zeros((num_C_expr, num_C_expr))
        active_C = np.array(range(num_C_expr))
        # active_history_C = [active_C]
        for k in range(1, num_C_expr):
            params_list = []
            valid_indices = []
            for j in range(1, len(active_C)):
                tmp_active = active_C.copy()
                tmp_active = np.delete(tmp_active, j)
                if len(tmp_active) == 0:  # not possible?
                    continue

                xi_C_copy = xi_C_0[tmp_active]
                lib_C_KM_reduced = lib_C_KM[tmp_active]
                lib_C_timeseries_reduced = lib_C_timeseries[tmp_active]
                params_list.append(
                    [
                        xi_C_copy,
                        lib_C_KM_reduced,
                        lib_C_timeseries_reduced,
                        alpha_val,
                    ]
                )
                valid_indices.append(j)
            with mp.Pool(min(NUM_CPUS, len(params_list))) as p:
                results = p.starmap(
                    partial(
                        opt_fun_C,
                        diffusion_KM=diffusion_KM,
                        dx=dx,
                        normal_dist=normal_dist,
                        normal_dist_edges=normal_dist_edges,
                    ),
                    params_list,
                )
            Xis = []
            Vs = []
            for Xi, V in results:
                Xis.append(Xi)
                Vs.append(V)
            min_found = np.nanargmin(Vs)
            min_idx = valid_indices[min_found]
            min_V = Vs[min_found]
            min_Xi = Xis[min_found]
            active_C = np.delete(active_C, min_idx)
            best_cost_C[k] = min_V
            best_xi_C[k, active_C] = min_Xi
            # active_history_C.append(active_C)

        # TODO: SELECTION CRITERA -> model before large(st) jump?
        # CURRENT: CHOOSE MODEL WITH 2 PARAMETERS SINCE THATS CORRECT
        xi_C_model = best_xi_C[-2]
        diffusion_model_timeseries = np.sqrt(
            2.0 * lib_C_timeseries.transpose((1, 2, 0)) @ xi_C_model
        )
        C_expr = sindy_model(xi_C_model, lib_C_expr)
        diffusion_expr = sympy.sqrt(sympy.N(2 * C_expr, 2))

        fig, ax = plt.subplots()
        ax.plot(
            centers_KM,
            diffusion_KM.T,
            linestyle="",
            marker="x",
            alpha=0.7,
            label="Data",
        )
        ax.plot(centers_KM, lib_C_KM.T @ xi_C_model, color="r", label="Model")
        ax.set_xlabel("$x$")
        ax.set_ylabel("Diffusion, $C(x)$")
        ax.set_title(f"${sympy.latex(diffusion_expr)}$")
        fig.savefig(FIG_PATH / f"Diffusion_fit_data_alpha_{alpha_val}.png")
        plt.close(fig)

        dw_Q = dx / diffusion_model_timeseries
        (dw_Q_hist,), _ = km(dw_Q.flatten(), bins=(normal_dist_edges,), powers=0)
        dw_Q_hist /= np.sum(dw_Q_hist * normal_dist_dx)

        fig, ax = plt.subplots()
        ax.plot(normal_dist_centers, normal_dist, label="Normal")
        ax.plot(normal_dist_centers, dw_Q_hist, color="r", label="Model")
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"Q measure noise, $dW^\mathbb{Q}$")
        ax.set_title(f"${sympy.latex(diffusion_expr)}$")
        fig.savefig(FIG_PATH / f"Diffusion_fit_dwQ_alpha_{alpha_val}.png")
        plt.close(fig)

        for beta_val in beta:
            xi_A_0 = np.ones(num_A_expr)
            best_cost_A = np.full((num_A_expr,), np.inf)
            best_xi_A = np.zeros((num_A_expr, num_A_expr))
            active_A = np.array(range(num_A_expr))
            # active_history_A = [active_A]
            for k in range(1, num_A_expr):
                params_list = []
                valid_indices = []
                for j in range(len(active_A)):
                    tmp_active = active_A.copy()
                    tmp_active = np.delete(tmp_active, j)
                    if len(tmp_active) == 0:  # not possible?
                        continue

                    xi_A_copy = xi_A_0[tmp_active]
                    lib_A_KM_reduced = lib_A_KM[tmp_active]
                    lib_A_timeseries_reduced = lib_A_timeseries[tmp_active]
                    params_list.append(
                        [
                            xi_A_copy,
                            lib_A_KM_reduced,
                            lib_A_timeseries_reduced,
                            beta_val,
                        ]
                    )
                    valid_indices.append(j)
                with mp.Pool(min(NUM_CPUS, len(params_list))) as p:
                    results = p.starmap(
                        partial(
                            opt_fun_A,
                            drift_KM=drift_KM,
                            diffusion_model_timeseries=diffusion_model_timeseries,
                            dx=dx,
                            dt=dt,
                            normal_dist=normal_dist,
                            normal_dist_edges=normal_dist_edges,
                        ),
                        params_list,
                    )
                Xis = []
                Vs = []
                for Xi, V in results:
                    Xis.append(Xi)
                    Vs.append(V)
                min_found = np.nanargmin(Vs)
                min_idx = valid_indices[min_found]
                min_V = Vs[min_found]
                min_Xi = Xis[min_found]
                active_A = np.delete(active_A, min_idx)
                best_cost_A[k] = min_V
                best_xi_A[k, active_A] = min_Xi
                # active_history_A.append(active_A)

            xi_A_model = best_xi_A[-3]
            drift_expr = sympy.N(sindy_model(xi_A_model, lib_A_expr), 2)
            drift_model_timeseries = lib_A_timeseries.transpose((1, 2, 0)) @ xi_A_model
            fig, ax = plt.subplots()
            ax.plot(
                centers_KM,
                drift_KM.T,
                linestyle="",
                marker="x",
                alpha=0.7,
                label="Data",
            )
            ax.plot(centers_KM, lib_A_KM.T @ xi_A_model, color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel("Drift, $A(x)$")
            ax.set_title(f"${sympy.latex(drift_expr)}$")
            fig.savefig(
                FIG_PATH / f"Drift_fit_data_alpha_{alpha_val}_beta_{beta_val}.png"
            )
            plt.close(fig)

            dw_P = (dx - drift_model_timeseries * dt) / diffusion_model_timeseries
            (dw_P_hist,), _ = km(dw_P.flatten(), bins=(normal_dist_edges,), powers=0)
            dw_P_hist /= np.sum(dw_P_hist * normal_dist_dx)

            fig, ax = plt.subplots()
            ax.plot(normal_dist_centers, normal_dist, label="Normal")
            ax.plot(normal_dist_centers, dw_P_hist, color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel(r"P measure noise, $dW^\mathbb{P}$")
            ax.set_title(f"${sympy.latex(drift_expr)}$")
            fig.savefig(
                FIG_PATH / f"Drift_fit_dwP_alpha_{alpha_val}_beta_{beta_val}.png"
            )
            plt.close(fig)

            print(
                f"{alpha_val=}, {beta_val=}; dx = {drift_expr} dt + {diffusion_expr} dbeta"
            )
            # print(
            #     f"$dx = {sympy.latex(drift_expr)} dt + {sympy.latex(diffusion_expr)} d\beta$"
            # )

            val_time, val_x = validation_timeseries
            val_dx = val_x[:, 1:] - val_x[:, :-1]

            val_time = val_time[:, :-1:timeseries_skip]
            val_x = val_x[:, :-1:timeseries_skip]
            val_dx = val_dx[:, ::timeseries_skip]

            val_centers, val_pdf, val_drift, val_diff = validation_KM
            val_drift[val_drift == 0] = np.nan
            val_diff[val_diff == 0] = np.nan

            drift_lambda = sympy.lambdify(x_sym, drift_expr)
            val_lib_A_KM = drift_lambda(val_centers)[None, ...]
            val_lib_A_timeseries = drift_lambda(val_x)[None, ...]

            diffusion_lambda = sympy.lambdify(x_sym, diffusion_expr)
            val_lib_C_KM = diffusion_lambda(val_centers)[None, ...] ** 2 / 2
            val_lib_C_timeseries = diffusion_lambda(val_x)[None, ...] ** 2 / 2

            fig, ax = plt.subplots()
            ax.plot(
                val_centers,
                val_diff.T,
                linestyle="",
                marker="x",
                alpha=0.7,
                label="Val data",
            )
            ax.plot(val_centers, val_lib_C_KM[0], color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel("Diffusion, $C(x)$")
            ax.set_title(f"${sympy.latex(diffusion_expr)}$")
            fig.savefig(FIG_PATH / f"Diffusion_val_data_alpha_{alpha_val}.png")
            plt.close(fig)

            val_dw_Q = (val_dx) / np.sqrt(2 * val_lib_C_timeseries[0])
            (val_dw_Q_hist,), _ = km(
                val_dw_Q.flatten(), bins=(normal_dist_edges,), powers=0
            )
            val_dw_Q_hist /= np.sum(val_dw_Q_hist * normal_dist_dx)

            fig, ax = plt.subplots()
            ax.plot(normal_dist_centers, normal_dist, label="Normal")
            ax.plot(normal_dist_centers, val_dw_Q_hist, color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel(r"Q measure noise, $dW^\mathbb{Q}$")
            ax.set_title(f"${sympy.latex(diffusion_expr)}$")
            fig.savefig(FIG_PATH / f"Diffusion_val_dwQ_alpha_{alpha_val}.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(
                val_centers,
                val_drift.T,
                linestyle="",
                marker="x",
                alpha=0.7,
                label="Val data",
            )
            ax.plot(val_centers, val_lib_A_KM[0], color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel("Drift, $A(x)$")
            ax.set_title(f"${sympy.latex(drift_expr)}$")
            fig.savefig(
                FIG_PATH / f"Drift_val_data_alpha_{alpha_val}_beta_{beta_val}.png"
            )
            plt.close(fig)

            val_dw_P = (val_dx - val_lib_A_timeseries[0] * dt) / np.sqrt(
                2 * val_lib_C_timeseries[0]
            )
            (val_dw_P_hist,), _ = km(
                val_dw_P.flatten(), bins=(normal_dist_edges,), powers=0
            )
            val_dw_P_hist /= np.sum(val_dw_P_hist * normal_dist_dx)

            fig, ax = plt.subplots()
            ax.plot(normal_dist_centers, normal_dist, label="Normal")
            ax.plot(normal_dist_centers, val_dw_P_hist, color="r", label="Model")
            ax.set_xlabel("$x$")
            ax.set_ylabel(r"P measure noise, $dW^\mathbb{P}$")
            ax.set_title(f"${sympy.latex(drift_expr)}$")
            fig.savefig(
                FIG_PATH / f"Drift_val_dwP_alpha_{alpha_val}_beta_{beta_val}.png"
            )
            plt.close(fig)

            cost_C_Jeff = cost_diffusion(
                np.ones(1),
                val_lib_C_KM,
                val_lib_C_timeseries,
                val_diff,
                val_dx,
                normal_dist,
                normal_dist_edges,
                alpha=0,
            )

            cost_C_Fid = cost_diffusion(
                np.ones(1),
                val_lib_C_KM,
                val_lib_C_timeseries,
                val_diff,
                val_dx,
                normal_dist,
                normal_dist_edges,
                alpha=1,
            )

            cost_A_Jeff = cost_drift(
                np.ones(1),
                val_lib_A_KM,
                val_lib_A_timeseries,
                val_drift,
                val_lib_C_timeseries[0],
                val_dx,
                dt,
                normal_dist,
                normal_dist_edges,
                beta=0,
            )

            cost_A_Fid = cost_drift(
                np.ones(1),
                val_lib_A_KM,
                val_lib_A_timeseries,
                val_drift,
                val_lib_C_timeseries[0],
                val_dx,
                dt,
                normal_dist,
                normal_dist_edges,
                beta=1,
            )
            cost_C_Jeffs_1d.append(cost_C_Jeff)
            cost_C_Fids_1d.append(cost_C_Fid)
            cost_A_Jeffs_1d.append(cost_A_Jeff)
            cost_A_Fids_1d.append(cost_A_Fid)

            xi_C_model_1d.append(xi_C_model)
            xi_A_model_1d.append(xi_A_model)
        cost_C_Jeffs_2d.append(cost_C_Jeffs_1d)
        cost_C_Fids_2d.append(cost_C_Fids_1d)
        cost_A_Jeffs_2d.append(cost_A_Jeffs_1d)
        cost_A_Fids_2d.append(cost_A_Fids_1d)
        xi_C_model_2d.append(xi_C_model_1d)
        xi_A_model_2d.append(xi_A_model_1d)
    return (
        cost_C_Jeffs_2d,
        cost_C_Fids_2d,
        cost_A_Jeffs_2d,
        cost_A_Fids_2d,
        xi_C_model_2d,
        xi_A_model_2d,
    )


SCRATCH_PATH = Path(f"/home/zachuu/scratch/alpha_beta_testing_ms_1.3_zoom")
# SCRATCH_PATH = Path(f"/home/zachuu/scratch/alpha_beta_testing")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = SCRATCH_PATH

alphas = np.linspace(0.5, 0.7, 6)
betas = np.linspace(0, 0.5, 11)

(
    cost_C_Jeffs_2d,
    cost_C_Fids_2d,
    cost_A_Jeffs_2d,
    cost_A_Fids_2d,
    xi_C_model_2d,
    xi_A_model_2d,
) = run_model(
    SCRATCH_PATH,
    alphas,
    betas,
    ms=1.3,
    NUM_VALIDATION=5,
    NUM_CPUS=10,
    num_datapoints=1_000_000,
    timeseries_skip=50,
    num_bins=200,
)

np.savez(
    SCRATCH_PATH / "alpha_beta_sweep_post.npz",
    alphas=alphas,
    betas=betas,
    cost_C_Jeffs_2d=cost_C_Jeffs_2d,
    cost_C_Fids_2d=cost_C_Fids_2d,
    cost_A_Jeffs_2d=cost_A_Jeffs_2d,
    cost_A_Fids_2d=cost_A_Fids_2d,
    xi_C_model_2d=xi_C_model_2d,
    xi_A_model_2d=xi_A_model_2d,
)

with np.load(SCRATCH_PATH / "alpha_beta_sweep_post.npz") as f:
    alphas = f["alphas"]
    betas = f["betas"]
    cost_C_Jeffs_2d = f["cost_C_Jeffs_2d"]
    cost_C_Fids_2d = f["cost_C_Fids_2d"]
    cost_A_Jeffs_2d = f["cost_A_Jeffs_2d"]
    cost_A_Fids_2d = f["cost_A_Fids_2d"]
    xi_C_model_2d = f["xi_C_model_2d"]
    xi_A_model_2d = f["xi_A_model_2d"]

C_mask = np.zeros(len(xi_C_model_2d[0][0]), dtype=bool)
C_mask[0] = True
C_mask[2] = True
xi_C_model_corr_2d = np.count_nonzero(xi_C_model_2d[:, :, C_mask], axis=-1)

A_mask = np.zeros(len(xi_A_model_2d[0][0]), dtype=bool)
A_mask[1] = True
A_mask[3] = True
A_mask[4] = True
xi_A_model_corr_2d = np.count_nonzero(xi_A_model_2d[:, :, A_mask], axis=-1)

# cost_A_Fids_2d[:, 0] = None

beta_grid, alpha_grid = np.meshgrid(betas, alphas)

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, cost_C_Jeffs_2d)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
# ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Jeffrey's divergence C, validation set")
fig.savefig(FIG_PATH / "jeff_C_alpha_beta.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, cost_C_Fids_2d)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
# ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Data fidelity C, validation set")
fig.savefig(FIG_PATH / "fid_C_alpha_beta.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, cost_A_Jeffs_2d)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Jeffrey's divergence A, validation set")
fig.savefig(FIG_PATH / "jeff_A_alpha_beta.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, cost_A_Fids_2d)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Data fidelity A, validation set")
fig.savefig(FIG_PATH / "fid_A_alpha_beta.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(
    alpha_grid,
    beta_grid,
    np.log10(cost_C_Jeffs_2d),
)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
# ax.set_ylabel("beta, diffusion mixing")
ax.set_title("log Jeffrey's divergence C, validation set")
fig.savefig(FIG_PATH / "jeff_C_alpha_beta_log.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(
    alpha_grid,
    beta_grid,
    np.log10(cost_C_Fids_2d),
)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
# ax.set_ylabel("beta, diffusion mixing")
ax.set_title("log Data fidelity C, validation set")
fig.savefig(FIG_PATH / "fid_C_alpha_beta_log.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, np.log10(cost_A_Jeffs_2d))
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
ax.set_ylabel("beta, diffusion mixing")
ax.set_title("log Jeffrey's divergence A, validation set")
fig.savefig(FIG_PATH / "jeff_A_alpha_beta_log.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, np.log10(cost_A_Fids_2d))
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
ax.set_ylabel("beta, diffusion mixing")
ax.set_title("log Data fidelity A, validation set")
fig.savefig(FIG_PATH / "fid_A_alpha_beta_log.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(
    alpha_grid,
    beta_grid,
    xi_C_model_corr_2d,
)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
# ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Number of correct C terms")
fig.savefig(FIG_PATH / "correct_C_alpha_beta.png")

fig, ax = plt.subplots()
cmap = ax.pcolormesh(alpha_grid, beta_grid, xi_A_model_corr_2d)
fig.colorbar(cmap, ax=ax)
ax.set_xlabel("alpha, drift mixing")
ax.set_ylabel("beta, diffusion mixing")
ax.set_title("Number of correct A terms")
fig.savefig(FIG_PATH / "correct_A_alpha_beta.png")

plt.show()
