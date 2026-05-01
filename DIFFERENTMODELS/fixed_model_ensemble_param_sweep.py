# %%
from time import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import json
import symengine
from jitcsde import jitcsde, y
from kramersmoyal import km
from kramersmoyal.kernels import gaussian

import sympy as sp
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from make_and_load_models import (
    get_timeseries_and_KM,
    load_and_stack_KM,
    plot_timeseries,
)
from utils import jeffreys_divergence, SteadyFP
# %%
SCRATCH_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/uncertainty_cube/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path(
    f"/home/zachuu/scratch/seismology/zach/softglass/uncertainty_cube_figures/"
)
FIG_PATH.mkdir(parents=True, exist_ok=True)

# %%
p = 1
maxfev = 10_000
# %%
def cost_diffusion(
    xi_C: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_diffu: np.ndarray,
    sfp: SteadyFP,
    alpha: float,
) -> float:
    """alpha=0: only jeffrey's divergence
    alpha=1: only KM fidelity"""
    centers, pdf, drift, diffusion = KM

    diffu_val = lib_diffu.T @ xi_C
    if np.any(xi_C < 0):
        return np.inf

    diffu_std = 1 / np.std(diffusion, axis=0)
    diffusion_fid = np.nansum(
        diffu_std * np.abs(diffusion - diffu_val) ** p
    ) / np.nansum(diffu_std * np.abs(diffusion) ** p)

    if alpha != 1:
        pdf_lib = sfp.solve(drift, diffu_val)
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        tol = 1e-10
        pdf[(pdf < tol) * (pdf > -tol)] = tol
        pdf_lib[(pdf_lib < tol) * (pdf_lib > -tol)] = tol
        if np.any(pdf_lib < -tol):
            return np.inf
        reg_val = np.sum(
            jeffreys_divergence(pdf_lib, pdf, centers[1] - centers[0], tol=1e-8)
        )
    else:
        reg_val = 0
    # print("JEF:", reg_val)
    # print("DIF:", diffusion_fid)
    return (1 - alpha) * reg_val + alpha * diffusion_fid


def cost_drift(
    xi_A: np.ndarray,
    KM: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lib_drift: np.ndarray,
    sfp: SteadyFP,
    beta: float,
) -> float:
    """beta=0: only jeffrey's divergence
    beta=1: only KM fidelity"""
    centers, pdf, drift, diffusion = KM

    # if xi_A[-1] > 0:
    #     return np.inf

    drift_val = lib_drift.T @ xi_A
    drift_std = 1 / np.std(drift, axis=0)
    drift_fid = np.nansum(drift_std * np.abs(drift - drift_val) ** p) / np.nansum(
        drift_std * np.abs(drift) ** p
    )

    if beta != 1:
        pdf_lib = sfp.solve(drift_val, diffusion)
        pdf_lib /= np.sum(pdf_lib * (centers[1] - centers[0]))
        tol = 1e-10
        pdf[(pdf < tol) * (pdf > -tol)] = tol
        pdf_lib[(pdf_lib < tol) * (pdf_lib > -tol)] = tol
        if np.any(pdf_lib < -tol):
            return np.inf
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
            bounds=[(-10, 10)] * len(lib_diffu),
            options={"adaptive": True, "maxfev": maxfev},
        )
        if not res.success:
            print(res)
        xs.append(res.x)
        funs.append(res.fun)
    xs = np.asarray(xs)
    funs = np.asarray(funs)
    return (
        xs,
        funs
    )


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
            bounds=[(-10, 10)] * len(lib_drift),
            options={"adaptive": True, "maxfev": maxfev},
        )
        if not res2.success:
            print(res2)
        xs.append(res2.x)
        funs.append(res2.fun)

    xs = np.asarray(xs)
    funs = np.asarray(funs)
    return (
        xs,
        funs
    )

# %%
def drift_lib(x_sym, centers):
    lib_expr = np.array(
        [x_sym, x_sym**3, x_sym**3*sp.Abs(x_sym)]
    )
    lib_KM = np.empty((3, len(centers)))
    for k in range(3):
        lamb = sp.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)
    
    return lib_expr, lib_KM

def diffusion_lib(x_sym, centers):
    lib_expr = np.array(
        [x_sym**0, x_sym**2]
    )
    lib_KM = np.empty((2, len(centers)))
    for k in range(2):
        lamb = sp.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)
    
    return lib_expr, lib_KM
# %%

R_range = np.linspace(0, 1, 10)
m_range = np.linspace(0, 2, 10)
ep0_range = np.logspace(-7, -3, 10)
ep1_range = np.linspace(0, 0.5, 10)
np.savez(SCRATCH_PATH / 'RANGES.npz',
         R_range=R_range,
         m_range=m_range,
         ep0_range=ep0_range,
         ep1_range=ep1_range)

R, m, ep0, ep1 = np.meshgrid(R_range, m_range, ep0_range, ep1_range, indexing='ij')


# %%
R_found = np.zeros_like(R)
m_found = np.zeros_like(R)
minus_one_found = np.zeros_like(R)
ep0_found = np.zeros_like(R)
ep1_found = np.zeros_like(R)

dR = np.zeros_like(R)
dm = np.zeros_like(R)
dminus_one = np.zeros_like(R)
dep0 = np.zeros_like(R)
dep1 = np.zeros_like(R)

# %%
def _generate_dataserie(
    N, dt, R, m, ep0, ep1
):
    x0 = 0
    times = np.arange(0, N) * dt
    x = y(0)
    A = [-R *x + m*x**3 - x**3 * symengine.Abs(x)]
    B = [symengine.sqrt(ep0 + ep1 * x**2)]

    ## Integrate model
    SDE = jitcsde(A, B, n=1, additive=False, verbose=False)
    SDE.set_initial_value([x0])
    x_data = np.fromiter(
        (SDE.integrate(t)[0] for t in times),  # type:ignore
        dtype=float,
        count=N,
    )

    return x_data

def _generate_KM(i, edges, models_dir, x_data, dt):
    KM_file = f"KM_{i}.npz"

    x_data = np.append(x_data, -x_data)

    kmc, centers = km(x_data[..., None], bins=(edges,), powers=2, kernel=gaussian)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers = centers[0]
    pdf /= np.nansum(pdf * (edges[1] - edges[0]))
    moment_1 /= dt
    moment_2 /= dt
    # print(f"Saving KM {i+1}", flush=True)
    np.savez(
        models_dir / KM_file,
        centers=centers,
        pdf=pdf,
        moment_1=moment_1,
        moment_2=moment_2,
    )
    return centers, pdf, moment_1, moment_2

def generate(
    models_dir, N_timesteps, N_bins, dt, R, m, ep0, ep1, NUM_DATASETS, NUM_CPUS=1
):
    calls = []
    for j in range(NUM_DATASETS):
        calls.append((N_timesteps, dt, R, m, ep0, ep1))

    with Pool(min(len(calls), NUM_CPUS)) as p:
        x_data_all = p.starmap(_generate_dataserie, calls)
    min_ = np.max(np.min(x_data_all, axis=1))
    max_ = np.min(np.max(x_data_all, axis=1))

    edges = np.linspace(
        min_, max_, N_bins + 1
    )

    calls = []
    for i in range(NUM_DATASETS):
        calls.append((i, edges, models_dir, x_data_all[i], dt))
    with Pool(min(len(calls), NUM_CPUS)) as p:
        KMCs = p.starmap(_generate_KM, calls)
    return KMCs

def load_KM(models_dir, NUM_DATASETS):
    KMs = []
    for i in range(NUM_DATASETS):
        KM_file = f"KM_{i}.npz"
        with np.load(models_dir / KM_file) as f:
            KMs.append([f['centers'],f['pdf'], f['moment_1'], f['moment_2'] ])
    return KMs

#%%
N_timesteps = 10_000_000
N_bins = 20
dt = 0.001
NUM_DATASETS = 5
NUM_CPUS = 5
x_sym = sp.symbols('x')
for Q in range(R.size):
    I = np.unravel_index(Q, R.shape)
    MODEL_PATH = SCRATCH_PATH / f"run_{I[0]}_{I[1]}_{I[2]}_{I[3]}"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if True:
        KMC = generate(MODEL_PATH, N_timesteps, N_bins, dt, R[I],m[I], ep0[I], ep1[I], NUM_DATASETS, NUM_CPUS)
    else:
        KMC = load_KM(MODEL_PATH, NUM_DATASETS)

    # PERFORM MINIMISATION
    centers = KMC[0][0]
    sfp = SteadyFP(N_bins, centers[1] - centers[0])
    drift_lib_expr, drift_lib_KM = drift_lib(x_sym, centers)
    diffu_lib_expr, diffu_lib_KM = diffusion_lib(x_sym, centers)
    xi_drift_0 = np.zeros(3)
    xi_diffu_0 = np.zeros(2)
    xi_diffu_0[0] = 1

    xi_drifts, eval_drifts = opt_func_drift(xi_drift_0, KMC, drift_lib_KM, sfp, 0.5)
    xi_diffus, eval_diffus = opt_func_diffusion(xi_diffu_0, KMC, diffu_lib_KM, sfp, 0.5)

    print(xi_drifts, eval_drifts)
    print(xi_diffus, eval_diffus)
    
    # SAVE RESULTS
    np.savez(SCRATCH_PATH / f'run_{I[0]}_{I[1]}_{I[2]}_{I[3]}.npz', xi_drifts=xi_drifts,xi_diffus=xi_diffus,eval_drifts=eval_drifts,eval_diffus=eval_diffus)

    xi_drift_mean = np.mean(xi_drifts, axis=0)
    xi_drift_std = np.std(xi_drifts, axis=0)
    xi_diffu_mean = np.mean(xi_diffus, axis=0)
    xi_diffu_std = np.std(xi_diffus, axis=0)

    R_found[I] = xi_drift_mean[0]
    m_found[I] = xi_drift_mean[1]
    minus_one_found[I] = xi_drift_mean[2]

    ep0_found[I] = xi_diffu_mean[0]
    ep1_found[I] = xi_diffu_mean[1]

    dR[I] = xi_drift_std[0]
    dm[I] = xi_drift_std[1]
    dminus_one[I] = xi_drift_std[2]

    dep0[I] = xi_diffu_std[0]
    dep1[I] = xi_diffu_std[1]

# Do R plot
for Q_name, Q in zip(['R', 'm', 'ep0', 'ep1'],[R, m, ep0, ep1]):
    for P_name, dP, P in zip(['R', 'm', 'ep0', 'ep1'], [dR, dm, dep0, dep1], [R_found, m_found, ep0_found, ep1_found]):
        plt.plot(Q.flatten(), np.abs(dP / P).flatten())
        plt.savefig(f'd{P_name}{P_name}_{Q_name}.png')


