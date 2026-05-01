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

from make_and_load_models_log import (
    get_timeseries_and_KM_log,
)
from utils import jeffreys_divergence, SteadyFP

# %%
SCRATCH_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/log_x/")
SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path(f"/home/zachuu/scratch/seismology/zach/softglass/log_x_fig/")
FIG_PATH.mkdir(parents=True, exist_ok=True)

# %%
DEBUG = True


def KM_plots(
    folder, KM, lib_drift_KM, drift_coeff, lib_diffu_KM, diffu_coeff, suffix=""
):
    centers, pdf_stack, drift_stack, diffusion_stack = KM
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 16))

    ax1.plot(centers, pdf_stack.T, alpha=0.7, linestyle="", marker="x")
    ax1.set_ylabel("PDF, $P(f)$")
    ax1.set_xlabel("$f$")
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    drift_stack_pos = drift_stack.copy()
    drift_stack_neg = drift_stack.copy()
    drift_stack_pos[drift_stack_pos <= 0] = np.nan
    drift_stack_neg[drift_stack_neg >= 0] = np.nan
    ax2.plot(centers, drift_stack_pos.T, alpha=0.7, c="b", linestyle="", marker="x")
    ax2.plot(centers, -drift_stack_neg.T, alpha=0.7, c="r", linestyle="", marker="x")

    drift = lib_drift_KM.T @ drift_coeff
    drift_pos = drift[drift > 0]
    centers_drift_pos = centers[drift > 0]
    drift_neg = drift[drift < 0]
    centers_drift_neg = centers[drift < 0]
    ax2.plot(centers_drift_pos, drift_pos, c="g", linestyle="-")
    ax2.plot(centers_drift_neg, -drift_neg, c="c", linestyle="-")
    ax2.set_ylabel("Drift, $m^{(1)}(f)$")
    ax2.set_xlabel("$f$")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    # ax2.set_ylim(-0.2, 0.2)
    # ax2.set_xlim(left=1e-2)

    ax3.plot(centers, diffusion_stack.T, alpha=0.7, linestyle="", marker="x")
    ax3.plot(centers, lib_diffu_KM.T @ diffu_coeff, linestyle="-")
    ax3.set_ylabel("Diffusion, $m^{(1)}(f)$")
    ax3.set_xlabel("$f$")
    ax3.set_yscale("log")
    ax3.set_xscale("log")

    fig.tight_layout()

    if suffix:
        fig.savefig(folder / f"kramers_moyal_{suffix}.png")
    else:
        fig.savefig(folder / "kramers_moyal.png")

    plt.delaxes(ax1)
    plt.delaxes(ax2)
    plt.delaxes(ax3)
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
    order: float,
    centers: np.ndarray,
):
    arr = []
    for i in range(round(2 * order)):
        arr.append(x_sym ** (i / 2))
    lib_expr = np.array(arr)
    lib_KM = np.empty((round(2 * order), len(centers)))
    for k in range(round(2 * order)):
        lamb = sp.lambdify(x_sym, lib_expr[k])
        lib_KM[k] = lamb(centers)

    return lib_expr, lib_KM


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
NUM_DATASETS = 10
NUM_VALIDATION = 1
NUM_CPUS = 1
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
    "kernel": "gaussian",
    "bandwidth": "default",
}

bandwidths = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# %%script true
for equation_number in [2, 3, 6, 7]:  # [2, 3, 6, 7]:  # range(len(equation_names)):
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
    for bandwidth in bandwidths:
        target_metadata["bandwidth"] = bandwidth
        timeseries, KM, val_timeseries, val_KM = get_timeseries_and_KM_log(
            SCRATCH_PATH / folder,
            target_metadata,
            NUM_DATASETS,
            NUM_VALIDATION,
            NUM_CPUS,
            # (1e-7, 1.8),
        )
        centers, pdfs, drifts, diffusions = KM

        KMs = [
            (centers, pdfs[i], drifts[i], diffusions[i]) for i in range(NUM_DATASETS)
        ]

        # make libraries for drift and diffusion
        x_sym = sp.symbols("f")
        num_drift = len(drift_coef) / 2 + 0.5
        lib_drift_expr, lib_drift_KM = poly_lib_sqrt(x_sym, num_drift, centers)

        num_diffusion = 3
        lib_diffu_expr, lib_diffu_KM = poly_lib(
            x_sym, num_diffusion, centers, False, False
        )
        KM_plots(
            FIG_PATH / folder,
            KM,
            lib_drift_KM,
            [ep0, 0, ep1 - 2 * drift_coef[1], 0, 2 * drift_coef[3], -2],
            lib_diffu_KM,
            np.array([0, 2 * ep0, 2 * ep1]),
            folder + f"_bw_{bandwidth:.4f}",
        )

        del KM, centers, pdfs, drifts, diffusions, lib_drift_KM, lib_diffu_KM
