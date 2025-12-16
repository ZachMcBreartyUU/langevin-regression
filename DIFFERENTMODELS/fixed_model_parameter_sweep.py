from typing import Optional
from time import time

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# sindy libraries
import sympy

# data generation and final model integration
import symengine
from jitcsde import jitcsde, y

from kramersmoyal import km

from utils import (
    optimise_function,
    optimise_functions,
    sindy_model,
    SteadyFP,
)
from utils_stack import cost_stack, cost_just_jef_stack, cost_alpha_stack
from fixed_model_plotting import do_plot, do_load, SCRATCH_PATH
from make_and_load_models import (
    generate_parameter_space,
    generate_parameter_space_parallel,
    get_parameter_space,
    delete_timeseries,
    plot_models,
)

# SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
import os

NUM_CPUS = int(os.environ.get("SLURM_NTASKS_PER_NODE", default=1))
print(f"{NUM_CPUS=}")

default_metadata = {
    "num_datapoints": 10_000_000,
    "dt": 0.001,
    "EVEN_ABS": True,
    "coeffs": [0.0, -0.016, 0.0, 1.0, -1.0],
    "ep0": 10**-5,
    "ep1": 0.1,
    "x0": [0.0],
    "num_bins": 100,
}


def analyse_model_stack(model_stack, EVEN_ABS, coeffs, ep0, ep1, num_bins):
    centers, pdf_stack, moment_1_stack, moment_2_stack = model_stack

    x_sym = sympy.symbols("x")

    if EVEN_ABS:
        arr = []
        for i in range(len(coeffs)):
            if i % 2 == 0 and i != 0:
                arr.append(x_sym ** (i - 1) * symengine.Abs(x_sym))
            else:
                arr.append(x_sym**i)
        A_lib_expr = np.array(arr)
    else:
        A_lib_expr = np.array([x_sym**i for i in range(len(coeffs))])
    A_lib_expr = A_lib_expr[np.nonzero(coeffs)]
    num_A_expr = len(A_lib_expr)

    lib_A = np.empty((num_A_expr, num_bins))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, A_lib_expr[k])
        lib_A[k] = lamb_expr(centers)

    C_lib_expr = np.array([x_sym**0, x_sym**2])
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers)

    n_terms = num_A_expr + num_C_expr

    # Regress on fixed terms
    Xi0 = np.empty((num_A_expr + num_C_expr))
    A_coeffs = np.average(lstsq(lib_A.T, moment_1_stack.T)[0], axis=1)
    A_coeffs[-1] = -np.abs(A_coeffs[-1])
    Xi0[:num_A_expr] = A_coeffs
    Xi0[num_A_expr:] = np.average(lstsq(lib_C.T, moment_2_stack.T)[0], axis=1)
    # print(f"Least squares: {Xi0=}\n", flush=True)

    first_weights = np.abs(1 / np.std(moment_1_stack, axis=0, keepdims=True))
    second_weights = np.abs(1 / np.std(moment_2_stack, axis=0, keepdims=True))
    first_weights /= np.sum(first_weights, axis=1, keepdims=True)
    second_weights /= np.sum(second_weights, axis=1, keepdims=True)
    weights = np.array([first_weights, second_weights])

    dx = centers[1] - centers[0]
    sfp = SteadyFP(num_bins, dx)

    params = {
        "Ws": weights,
        "Xi0": Xi0,
        "chi0": np.array([1.0]),
        "A_KMs": moment_1_stack,
        "C_KMs": moment_2_stack,
        "A_expr": A_lib_expr,
        "C_expr": C_lib_expr,
        "lib_A": lib_A,
        "lib_C": lib_C,
        "sfp": sfp,
        "pdfs": pdf_stack,
        "dx": dx,
    }
    Xi0, _ = optimise_function(cost_stack, params)
    # print(f"Moment optimise: {Xi0=}\n", flush=True)
    A_coeffs = Xi0[:num_A_expr]
    A_coeffs[-1] = -np.abs(A_coeffs[-1])
    Xi0[:num_A_expr] = A_coeffs
    params["Xi0"] = Xi0

    # opt_func = lambda params: optimise_function(cost_KL, params)
    # xi, cost_val = optimise_function(cost_KL, params)
    xi, cost_val = optimise_functions(cost_just_jef_stack, cost_alpha_stack, params)

    # TODO: Could perform some rounding on the found params, e.g. 0.012324 -> 0.012
    A_sym = sindy_model(xi[:num_A_expr], A_lib_expr)
    A_sindy = sympy.lambdify(x_sym, A_sym)(centers)
    C_sym = sindy_model(xi[num_A_expr:], C_lib_expr)
    C_sindy = sympy.lambdify(x_sym, C_sym)(centers)

    if np.ndim(A_sindy) == 0:
        A_sindy = np.full_like(centers, A_sindy)
    if np.ndim(C_sindy) == 0:
        C_sindy = np.full_like(centers, C_sindy)

    # print(f"dx = ({A_sym}) dt + ({sympy.sqrt(2.0*C_sym)}) dβ")

    ## Directly compare True answer to Found answer
    true_model_xi = np.zeros(n_terms)
    true_model_xi[:num_A_expr] = np.array(coeffs)[np.nonzero(coeffs)]
    true_model_xi[num_A_expr + 0] = ep0 / 2  # B = sqrt(ep0 + ep1 x^2) ->
    true_model_xi[num_A_expr + 1] = ep1 / 2  # C = B^2 / 2 -> ep0 / 2 + ep1 / 2 x^2

    # found_pdf = sfp.solve(A_sindy, C_sindy)

    # differences = np.abs((xi - true_model_xi) / true_model_xi)

    return cost_val, xi, true_model_xi


def analyse_model_stack_multiple(
    model_stack, EVEN_ABS, coeffs, ep0, ep1, num_bins, num_times=10
):
    centers, pdf_stack, moment_1_stack, moment_2_stack = model_stack

    x_sym = sympy.symbols("x")

    if EVEN_ABS:
        arr = []
        for i in range(len(coeffs)):
            if i % 2 == 0 and i != 0:
                arr.append(x_sym ** (i - 1) * symengine.Abs(x_sym))
            else:
                arr.append(x_sym**i)
        A_lib_expr = np.array(arr)
    else:
        A_lib_expr = np.array([x_sym**i for i in range(len(coeffs))])
    A_lib_expr = A_lib_expr[np.nonzero(coeffs)]
    num_A_expr = len(A_lib_expr)

    lib_A = np.empty((num_A_expr, num_bins))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, A_lib_expr[k])
        lib_A[k] = lamb_expr(centers)

    C_lib_expr = np.array([x_sym**0, x_sym**2])
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers)

    n_terms = num_A_expr + num_C_expr

    # print(f"Least squares: {Xi0=}\n", flush=True)

    first_weights = np.abs(1 / np.std(moment_1_stack, axis=0, keepdims=True))
    second_weights = np.abs(1 / np.std(moment_2_stack, axis=0, keepdims=True))
    first_weights /= np.sum(first_weights, axis=1, keepdims=True)
    second_weights /= np.sum(second_weights, axis=1, keepdims=True)
    weights = np.array([first_weights, second_weights])

    dx = centers[1] - centers[0]
    sfp = SteadyFP(num_bins, dx)

    xis = []
    costs = []
    for i in range(num_times):
        print(f"Starting submodel {i}", flush=True)
        # Regress on fixed terms
        Xi0 = 2 * (np.random.random((num_A_expr + num_C_expr)) - 0.5)
        Xi0[num_A_expr - 1] = -np.abs(Xi0[num_A_expr - 1])

        params = {
            "Ws": weights,
            "Xi0": Xi0,
            "chi0": np.array([1.0]),
            "A_KMs": moment_1_stack,
            "C_KMs": moment_2_stack,
            "A_expr": A_lib_expr,
            "C_expr": C_lib_expr,
            "lib_A": lib_A,
            "lib_C": lib_C,
            "sfp": sfp,
            "pdfs": pdf_stack,
            "dx": dx,
        }
        Xi0, _ = optimise_function(cost_stack, params)
        # print(f"Moment optimise: {Xi0=}\n", flush=True)
        A_coeffs = Xi0[:num_A_expr]
        A_coeffs[-1] = -np.abs(A_coeffs[-1])
        Xi0[:num_A_expr] = A_coeffs
        params["Xi0"] = Xi0

        xi, cost_val = optimise_functions(cost_just_jef_stack, cost_alpha_stack, params)
        xis.append(xi)
        costs.append(cost_val)
    ## Directly compare True answer to Found answer
    true_model_xi = np.zeros(n_terms)
    true_model_xi[:num_A_expr] = np.array(coeffs)[np.nonzero(coeffs)]
    true_model_xi[num_A_expr + 0] = ep0 / 2  # B = sqrt(ep0 + ep1 x^2) ->
    true_model_xi[num_A_expr + 1] = ep1 / 2  # C = B^2 / 2 -> ep0 / 2 + ep1 / 2 x^2

    return costs, xis, true_model_xi


def sweep(
    first_range,
    second_range,
    first_name="ep0",
    second_name="ep1",
    generate=True,
    analyse=True,
    NUM_MODELS=10,
    first_log=True,
    second_log=True,
    first_label=r"\epsilon_0",
    second_label=r"\epsilon_1",
):
    first_mesh, second_mesh = np.meshgrid(first_range, second_range)
    first_s = first_mesh.flatten()
    second_s = second_mesh.flatten()
    first_second_models_dirs = [
        SCRATCH_PATH
        / f"MODELS_{first_name}_{second_name}"
        / f"MODELS_{first_name}_{first:.3e}_{second_name}_{second:.3e}"
        for first, second in zip(first_s, second_s)
    ]
    target_metadatas = []
    for first, second in zip(first_s, second_s):
        tm = default_metadata.copy()
        coeffs = tm["coeffs"]
        if first_name in default_metadata.keys():
            tm[first_name] = first
        if second_name in default_metadata.keys():
            tm[second_name] = second

        if first_name == "R":
            coeffs[1] = first
        if second_name == "R":
            coeffs[1] = second

        if second_name == "m_sigma":
            coeffs[3] = first
        if second_name == "m_sigma":
            coeffs[3] = second

        tm["coeffs"] = coeffs

    if generate:
        generate_parameter_space_parallel(
            first_second_models_dirs,
            target_metadatas,
            NUM_MODELS=NUM_MODELS,
            NUM_CPUS=NUM_CPUS,
        )
        for model_dir in first_second_models_dirs:
            # plot the found pdf and moments, not the timeseries
            plot_models(model_dir, NUM_MODELS)
            # clean up the timeseries (since they are huge!)
            delete_timeseries(model_dir)

    if analyse:
        model_stacks = get_parameter_space(first_second_models_dirs, NUM_MODELS=10)

        costs = []
        xis = []
        true_xis = []
        for i, (model_stack, first, second, metadata) in enumerate(
            zip(model_stacks, first_s, second_s, target_metadatas)
        ):
            print(f"model {i}, {first_name}={first:.1e}, {second_name}={second:.1e}")
            cost, xi, true_xi = analyse_model_stack(
                model_stack,
                metadata["EVEN_ABS"],
                metadata["coeffs"],
                metadata["ep0"],
                metadata["ep1"],
                metadata["num_bins"],
            )
            costs.append(cost)
            xis.append(xi)
            true_xis.append(true_xi)

        costs = np.asarray(costs).reshape(first_mesh.shape)
        xis = np.asarray(xis).reshape((*first_mesh.shape, -1))
        true_xis = np.asarray(true_xis).reshape((*first_mesh.shape, -1))

        if first_log:
            first_plot_mesh = np.log10(first_mesh)
        else:
            first_plot_mesh = first_mesh
        if second_log:
            second_plot_mesh = np.log10(second_mesh)
        else:
            second_plot_mesh = second_mesh
        del model_stacks
        np.savez(
            SCRATCH_PATH / f"differences_{first_name}_{second_name}.npz",
            first_plot_mesh=first_plot_mesh,
            second_plot_mesh=second_plot_mesh,
            costs=costs,
            xis=xis,
            true_xis=true_xis,
        )

    with np.load(SCRATCH_PATH / f"differences_{first_name}_{second_name}.npz") as f:
        first_plot_mesh = f["first_plot_mesh"]
        second_plot_mesh = f["second_plot_mesh"]
        costs = f["costs"]
        xis = f["xis"]
        true_xis = f["true_xis"]

    differences = []
    differences_A = []
    differences_C = []
    diffs = np.abs((xis - true_xis) / true_xis)
    differences_A = np.sum(diffs[..., :-2], axis=-1)
    differences_C = np.sum(diffs[..., -2:], axis=-1)
    differences = differences_A + differences_C

    def plot(C, label, filename):
        fig, ax = plt.subplots()
        pcol = ax.pcolor(first_plot_mesh, second_plot_mesh, C)
        cmap = fig.colorbar(pcol, ax=ax, label=label)
        if first_log:
            ax.set_xlabel(rf"$\log {first_label}$")
        else:
            ax.set_xlabel(rf"${first_label}$")
        if second_log:
            ax.set_ylabel(rf"$\log {second_label}$")
        else:
            ax.set_ylabel(rf"${second_label}$")
        fig.savefig(SCRATCH_PATH / filename)
        plt.close(fig)

    coeff_names = ["-Rx", r"m(\sigma)x^3", r"-x^3|x|", r"\epsilon_0", r"\epsilon_1 x^2"]
    # diffs[0, -1] = None
    for i in range(diffs.shape[-1]):
        plot(
            diffs[..., i],
            rf"$\Delta ({coeff_names[i]})$",
            f"differences_{i}_{first_name}_{second_name}.png",
        )
        plot(
            np.log10(diffs[..., i]),
            rf"$\log \Delta ({coeff_names[i]})$",
            f"log_differences_{i}_{first_name}_{second_name}.png",
        )

    plot(
        differences,
        r"$\Delta$ Coefficients",
        f"differences_{first_name}_{second_name}.png",
    )
    plot(
        np.log10(differences),
        r"$\log \Delta$ Coefficients",
        f"log_differences_{first_name}_{second_name}.png",
    )

    plot(
        differences_A,
        r"$\Delta$ Drift",
        f"differences_A_{first_name}_{second_name}.png",
    )
    plot(
        np.log10(differences_A),
        r"$\log \Delta$ Drift",
        f"log_differences_A_{first_name}_{second_name}.png",
    )

    plot(
        differences_C,
        r"$\Delta$ Diffusion",
        f"differences_C_{first_name}_{second_name}.png",
    )
    plot(
        np.log10(differences_C),
        r"$\log \Delta$ Diffusion",
        f"log_differences_C_{first_name}_{second_name}.png",
    )

    plot(costs, r"Cost", f"costs_{first_name}_{second_name}.png")
    plot(np.log10(costs), r"$\log$ Cost", f"log_costs_{first_name}_{second_name}.png")


ep0_range = np.logspace(-5, -1, 11)
ep1_range = np.logspace(-5, -1, 11)
R_range = np.logspace(-5, 1, 11)
m_sigma_range = np.linspace(0.01, 3.01, 11)

ep0_sweep = False
ep1_sweep = False
R_sweep = False
m_sigma_sweep = False

GENERATE = True
ANALYSE = True
NUM_MODELS = 10

# The ordering of the following is arbitrary,
# and actually out of order if you look closely
# This is mostly due to a lack of foresight
if ep0_sweep:
    if ep1_sweep:
        sweep(
            ep0_range,
            ep1_range,
            "ep0",
            "ep1",
            GENERATE,
            ANALYSE,
            NUM_MODELS,
            True,
            True,
            r"\epsilon_0",
            r"\epsilon_1",
        )
    elif R_sweep:
        sweep(
            ep0_range,
            R_range,
            "ep0",
            "R",
            GENERATE,
            ANALYSE,
            NUM_MODELS,
            True,
            True,
            r"\epsilon_0",
            r"R",
        )
    elif m_sigma_sweep:
        sweep(
            m_sigma_range,
            ep0_range,
            "m_sigma",
            "ep0",
            GENERATE,
            ANALYSE,
            NUM_MODELS,
            True,
            True,
            r"m(\sigma)",
            r"\epsilon_0",
        )
    else:
        raise ValueError("only got ep0 :(")
elif ep1_sweep:
    if R_sweep:
        sweep(
            R_range,
            ep1_range,
            "R",
            "ep1",
            GENERATE,
            ANALYSE,
            NUM_MODELS,
            True,
            True,
            r"R",
            r"\epsilon_1",
        )
    elif m_sigma_sweep:
        sweep(
            m_sigma_range,
            ep1_range,
            "m_sigma",
            "ep1",
            GENERATE,
            ANALYSE,
            NUM_MODELS,
            False,
            True,
            r"m(\sigma)",
            r"\epsilon_1",
        )
    else:
        raise ValueError("only got ep1 :(")
elif R_sweep and m_sigma_sweep:
    sweep(
        R_range,
        m_sigma_range,
        "R",
        "m_sigma",
        GENERATE,
        ANALYSE,
        NUM_MODELS,
        True,
        False,
        r"R",
        r"m(\sigma)",
    )
else:
    raise ValueError("Got none :(")

print("DONE")
