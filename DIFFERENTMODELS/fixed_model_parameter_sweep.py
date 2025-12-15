from typing import Optional
from time import time

import numpy as np
from numpy.linalg import lstsq

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
        # Regress on fixed terms
        Xi0 = np.random.random((num_A_expr + num_C_expr))
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


ep0_range = np.logspace(-5, -1, 11)
ep1_range = np.logspace(-5, -1, 11)
R_range = np.logspace(-5, 1, 11)
m_sigma_range = np.linspace(0, 3, 11)

# ep0_mesh, ep1_mesh = np.meshgrid(ep0_range, ep1_range)
# ep0s = ep0_mesh.flatten()
# ep1s = ep1_mesh.flatten()
# ep0_ep1_models_dirs = [
#     SCRATCH_PATH / 'MODELS_ep0_ep1' / f"MODELS_ep0_{ep0:.3e}_ep1_{ep1:.3e}" for ep0, ep1 in zip(ep0s, ep1s)
# ]
# target_metadatas = [
#     {
#         "ep0": ep0,
#         "ep1": ep1,
#         "num_datapoints": default_metadata["num_datapoints"],
#         "dt": default_metadata["dt"],
#         "EVEN_ABS": default_metadata["EVEN_ABS"],
#         "coeffs": default_metadata["coeffs"],
#         "x0": default_metadata["x0"],
#         "num_bins": default_metadata["num_bins"],
#     }
#     for ep0, ep1 in zip(ep0s, ep1s)
# ]

# # generate_parameter_space_parallel(
# #     ep0_ep1_models_dirs, target_metadatas, NUM_MODELS=10, NUM_CPUS=NUM_CPUS
# # )
# # # clean up the timeseries (since they are huge!)
# # for model_dir in ep0_ep1_models_dirs:
# #     delete_timeseries(model_dir)

# model_stacks = get_parameter_space(ep0_ep1_models_dirs, NUM_MODELS=10)

# costs = []
# xis = []
# true_xis = []
# for i, (model_stack, ep0, ep1) in enumerate(zip(model_stacks, ep0s, ep1s)):
#     print(f"model {i}, ep0={ep0:.1e}, ep1={ep1:.1e}")
#     cost, xi, true_xi = analyse_model_stack(
#         model_stack,
#         default_metadata["EVEN_ABS"],
#         default_metadata["coeffs"],
#         ep0,
#         ep1,
#         default_metadata["num_bins"],
#     )
#     costs.append(cost)
#     xis.append(xi)
#     true_xis.append(true_xi)

# costs = np.asarray(costs).reshape(ep0_mesh.shape)
# xis = np.asarray(xis).reshape((*ep0_mesh.shape, -1))
# true_xis = np.asarray(true_xis).reshape((*ep0_mesh.shape, -1))

# log_ep0_mesh = np.log10(ep0_mesh)
# log_ep1_mesh = np.log10(ep1_mesh)
# del model_stacks
# np.savez(
#     SCRATCH_PATH / "differences.npz",
#     log_ep0_mesh=log_ep0_mesh,
#     log_ep1_mesh=log_ep1_mesh,
#     costs=costs,
#     xis=xis,
#     true_xis=true_xis,
# )

# with np.load(SCRATCH_PATH / "differences.npz") as f:
#     log_ep0_mesh = f["log_ep0_mesh"]
#     log_ep1_mesh = f["log_ep1_mesh"]
#     costs = f["costs"]
#     xis = f["xis"]
#     true_xis = f["true_xis"]

# print(log_ep0_mesh.shape)
# print(log_ep1_mesh.shape)
# print(costs.shape)
# print(xis.shape)
# print(true_xis.shape)

# differences = []
# differences_A = []
# differences_C = []
# diffs = np.abs((xis - true_xis) / true_xis)
# differences_A = np.sum(diffs[..., :-2], axis=-1)
# differences_C = np.sum(diffs[..., -2:], axis=-1)
# differences = differences_A + differences_C

# import matplotlib.pyplot as plt


# def plot_ep0_ep1(C, label, filename):
#     fig, ax = plt.subplots()
#     pcol = ax.pcolor(log_ep0_mesh, log_ep1_mesh, C)
#     cmap = fig.colorbar(pcol, ax=ax, label=label)
#     ax.set_xlabel(r"$\log \epsilon_0$")
#     ax.set_ylabel(r"$\log \epsilon_1$")
#     fig.savefig(SCRATCH_PATH / filename)
#     plt.close(fig)


# coeff_names = ["-Rx", r"m(\sigma)x^3", r"-x^3|x|", r"\epsilon_0", r"\epsilon_1 x^2"]
# # diffs[0, -1] = None
# for i in range(diffs.shape[-1]):
#     plot_ep0_ep1(
#         diffs[..., i], rf"$\Delta ({coeff_names[i]})$", f"differences_{i}_ep0_ep1.png"
#     )
#     plot_ep0_ep1(
#         np.log10(diffs[..., i]),
#         rf"$\log \Delta ({coeff_names[i]})$",
#         f"log_differences_{i}_ep0_ep1.png",
#     )

# plot_ep0_ep1(differences, r"$\Delta$ Coefficients", "differences_ep0_ep1.png")
# plot_ep0_ep1(
#     np.log10(differences), r"$\log \Delta$ Coefficients", "log_differences_ep0_ep1.png"
# )

# plot_ep0_ep1(differences_A, r"$\Delta$ Drift", "differences_A_ep0_ep1.png")
# plot_ep0_ep1(
#     np.log10(differences_A), r"$\log \Delta$ Drift", "log_differences_A_ep0_ep1.png"
# )

# plot_ep0_ep1(differences_C, r"$\Delta$ Diffusion", "differences_C_ep0_ep1.png")
# plot_ep0_ep1(
#     np.log10(differences_C), r"$\log \Delta$ Diffusion", "log_differences_C_ep0_ep1.png"
# )

# plot_ep0_ep1(costs, r"Cost", "costs_ep0_ep1.png")
# plot_ep0_ep1(np.log10(costs), r"$\log$ Cost", "log_costs_ep0_ep1.png")
#################
# R_range = np.logspace(-5, 1, 11)
# ep1_range = np.logspace(-5, -1, 11)
# R_mesh, ep1_mesh = np.meshgrid(R_range, ep1_range)
# Rs = R_mesh.flatten()
# ep1s = ep1_mesh.flatten()
# R_ep1_models_dirs = [
#     SCRATCH_PATH / "MODELS_R_ep1" / f"MODELS_R_{R:.3e}_ep1_{ep1:.3e}"
#     for R, ep1 in zip(Rs, ep1s)
# ]
# target_metadatas = [
#     {
#         "coeffs": [0.0, -R, 0.0, 1.0, -1.0],
#         "ep1": ep1,
#         "ep0": default_metadata["ep0"],
#         "num_datapoints": default_metadata["num_datapoints"],
#         "dt": default_metadata["dt"],
#         "EVEN_ABS": default_metadata["EVEN_ABS"],
#         "x0": default_metadata["x0"],
#         "num_bins": default_metadata["num_bins"],
#     }
#     for R, ep1 in zip(Rs, ep1s)
# ]

# # generate_parameter_space_parallel(
# #     R_ep1_models_dirs, target_metadatas, NUM_MODELS=10, NUM_CPUS=NUM_CPUS
# # )

# for model_dir in R_ep1_models_dirs:
#     # plot the found pdf and moments, not the timeseries
#     plot_models(model_dir, 10)
#     # clean up the timeseries (since they are huge!)
#     # delete_timeseries(model_dir)

# # model_stacks = get_parameter_space(R_ep1_models_dirs, NUM_MODELS=10)

# # costs = []
# # xis = []
# # true_xis = []
# # for i, (model_stack, R, ep1) in enumerate(zip(model_stacks, Rs, ep1s)):
# #     print(f"model {i}, R={R:.1e}, ep1={ep1:.1e}")
# #     cost, xi, true_xi = analyse_model_stack(
# #         model_stack,
# #         default_metadata["EVEN_ABS"],
# #         [0.0, -R, 0.0, 1.0, -1.0],
# #         default_metadata["ep0"],
# #         ep1,
# #         default_metadata["num_bins"],
# #     )
# #     costs.append(cost)
# #     xis.append(xi)
# #     true_xis.append(true_xi)

# # costs = np.asarray(costs).reshape(R_mesh.shape)
# # xis = np.asarray(xis).reshape((*R_mesh.shape, -1))
# # true_xis = np.asarray(true_xis).reshape((*R_mesh.shape, -1))

# # log_R_mesh = np.log10(R_mesh)
# # log_ep1_mesh = np.log10(ep1_mesh)
# # del model_stacks
# # np.savez(
# #     SCRATCH_PATH / "differences.npz",
# #     log_R_mesh=log_R_mesh,
# #     log_ep1_mesh=log_ep1_mesh,
# #     costs=costs,
# #     xis=xis,
# #     true_xis=true_xis,
# # )

# # with np.load(SCRATCH_PATH / "differences.npz") as f:
# #     log_R_mesh = f["log_R_mesh"]
# #     log_ep1_mesh = f["log_ep1_mesh"]
# #     costs = f["costs"]
# #     xis = f["xis"]
# #     true_xis = f["true_xis"]

# # print(log_R_mesh.shape)
# # print(log_ep1_mesh.shape)
# # print(costs.shape)
# # print(xis.shape)
# # print(true_xis.shape)

# # differences = []
# # differences_A = []
# # differences_C = []
# # diffs = np.abs((xis - true_xis) / true_xis)
# # differences_A = np.sum(diffs[..., :-2], axis=-1)
# # differences_C = np.sum(diffs[..., -2:], axis=-1)
# # differences = differences_A + differences_C

# # import matplotlib.pyplot as plt


# # def plot_R_ep1(C, label, filename):
# #     fig, ax = plt.subplots()
# #     pcol = ax.pcolor(log_R_mesh, log_ep1_mesh, C)
# #     cmap = fig.colorbar(pcol, ax=ax, label=label)
# #     ax.set_xlabel(r"$\log R$")
# #     ax.set_ylabel(r"$\log \epsilon_1$")
# #     fig.savefig(SCRATCH_PATH / filename)
# #     plt.close(fig)


# # coeff_names = ["-Rx", r"m(\sigma)x^3", r"-x^3|x|", r"\epsilon_0", r"\epsilon_1 x^2"]
# # # diffs[0, -1] = None
# # for i in range(diffs.shape[-1]):
# #     plot_R_ep1(
# #         diffs[..., i], rf"$\Delta ({coeff_names[i]})$", f"differences_{i}_R_ep1.png"
# #     )
# #     plot_R_ep1(
# #         np.log10(diffs[..., i]),
# #         rf"$\log \Delta ({coeff_names[i]})$",
# #         f"log_differences_{i}_R_ep1.png",
# #     )

# # plot_R_ep1(differences, r"$\Delta$ Coefficients", "differences_R_ep1.png")
# # plot_R_ep1(
# #     np.log10(differences), r"$\log \Delta$ Coefficients", "log_differences_R_ep1.png"
# # )

# # plot_R_ep1(differences_A, r"$\Delta$ Drift", "differences_A_R_ep1.png")
# # plot_R_ep1(
# #     np.log10(differences_A), r"$\log \Delta$ Drift", "log_differences_A_R_ep1.png"
# # )

# # plot_R_ep1(differences_C, r"$\Delta$ Diffusion", "differences_C_R_ep1.png")
# # plot_R_ep1(
# #     np.log10(differences_C), r"$\log \Delta$ Diffusion", "log_differences_C_R_ep1.png"
# # )

# # plot_R_ep1(costs, r"Cost", "costs_R_ep1.png")
# # plot_R_ep1(np.log10(costs), r"$\log$ Cost", "log_costs_R_ep1.png")
#####################

m_sigma_mesh, ep1_mesh = np.meshgrid(m_sigma_range, ep1_range)
m_sigma_s = m_sigma_mesh.flatten()
ep1s = ep1_mesh.flatten()
m_sigma_ep1_models_dirs = [
    SCRATCH_PATH / "MODELS_m_sigma_ep1" / f"MODELS_m_sigma_{m_sigma:.3e}_ep1_{ep1:.3e}"
    for m_sigma, ep1 in zip(m_sigma_s, ep1s)
]
target_metadatas = [
    {
        "coeffs": [0.0, -0.016, 0.0, m_sigma, -1.0],
        "ep1": ep1,
        "ep0": default_metadata["ep0"],
        "num_datapoints": default_metadata["num_datapoints"],
        "dt": default_metadata["dt"],
        "EVEN_ABS": default_metadata["EVEN_ABS"],
        "x0": default_metadata["x0"],
        "num_bins": default_metadata["num_bins"],
    }
    for m_sigma, ep1 in zip(m_sigma_s, ep1s)
]

generate_parameter_space_parallel(
    m_sigma_ep1_models_dirs, target_metadatas, NUM_MODELS=10, NUM_CPUS=NUM_CPUS
)

for model_dir in m_sigma_ep1_models_dirs:
    # plot the found pdf and moments, not the timeseries
    plot_models(model_dir, 10)
    # clean up the timeseries (since they are huge!)
    delete_timeseries(model_dir)

model_stacks = get_parameter_space(m_sigma_ep1_models_dirs, NUM_MODELS=10)

costs = []
costs_var = []
xis = []
xis_var = []
true_xis = []
for i, (model_stack, m_sigma, ep1) in enumerate(zip(model_stacks, m_sigma_s, ep1s)):
    print(f"model {i}, R={m_sigma:.1e}, ep1={ep1:.1e}")
    cost, xi, true_xi = analyse_model_stack_multiple(
        model_stack,
        default_metadata["EVEN_ABS"],
        [0.0, -0.016, 0.0, m_sigma, -1.0],
        default_metadata["ep0"],
        ep1,
        default_metadata["num_bins"],
    )
    costs.append(np.mean(cost))
    costs_var.append(np.var(cost))
    xis.append(np.mean(xis, axis=0))
    xis_var.append(np.var(xis, axis=0))
    true_xis.append(true_xi)

costs = np.asarray(costs).reshape(m_sigma_mesh.shape)
costs_var = np.asarray(costs_var).reshape(m_sigma_mesh.shape)
xis = np.asarray(xis).reshape((*m_sigma_mesh.shape, -1))
xis_var = np.asarray(xis_var).reshape((*m_sigma_mesh.shape, -1))
true_xis = np.asarray(true_xis).reshape((*m_sigma_mesh.shape, -1))

log_ep1_mesh = np.log10(ep1_mesh)
del model_stacks
np.savez(
    SCRATCH_PATH / "differences.npz",
    m_sigma_mesh=m_sigma_mesh,
    log_ep1_mesh=log_ep1_mesh,
    costs=costs,
    costs_var=costs_var,
    xis=xis,
    xis_var=xis_var,
    true_xis=true_xis,
)

with np.load(SCRATCH_PATH / "differences.npz") as f:
    m_sigma_mesh = f["m_sigma_mesh"]
    log_ep1_mesh = f["log_ep1_mesh"]
    costs = f["costs"]
    costs_var = f["costs_var"]
    xis = f["xis"]
    xis_var = f["xis_var"]
    true_xis = f["true_xis"]

print(m_sigma_mesh.shape)
print(log_ep1_mesh.shape)
print(costs.shape)
print(costs_var.shape)
print(xis.shape)
print(xis_var.shape)
print(true_xis.shape)

differences = []
differences_A = []
differences_C = []
diffs = np.abs((xis - true_xis) / true_xis)
differences_A = np.sum(diffs[..., :-2], axis=-1)
differences_C = np.sum(diffs[..., -2:], axis=-1)
differences = differences_A + differences_C

import matplotlib.pyplot as plt


def plot_m_sigma_ep1(C, label, filename):
    fig, ax = plt.subplots()
    pcol = ax.pcolor(m_sigma_mesh, log_ep1_mesh, C)
    cmap = fig.colorbar(pcol, ax=ax, label=label)
    ax.set_xlabel(r"$m(\sigma)$")
    ax.set_ylabel(r"$\log \epsilon_1$")
    fig.savefig(SCRATCH_PATH / filename)
    plt.close(fig)


coeff_names = ["-Rx", r"m(\sigma)x^3", r"-x^3|x|", r"\epsilon_0", r"\epsilon_1 x^2"]
for i in range(diffs.shape[-1]):
    plot_m_sigma_ep1(
        diffs[..., i], rf"$\Delta ({coeff_names[i]})$", f"differences_{i}_R_ep1.png"
    )
    plot_m_sigma_ep1(
        np.log10(diffs[..., i]),
        rf"$\log \Delta ({coeff_names[i]})$",
        f"log_differences_{i}_R_ep1.png",
    )

plot_m_sigma_ep1(differences, r"$\Delta$ Coefficients", "differences_R_ep1.png")
plot_m_sigma_ep1(
    np.log10(differences), r"$\log \Delta$ Coefficients", "log_differences_R_ep1.png"
)

plot_m_sigma_ep1(differences_A, r"$\Delta$ Drift", "differences_A_R_ep1.png")
plot_m_sigma_ep1(
    np.log10(differences_A), r"$\log \Delta$ Drift", "log_differences_A_R_ep1.png"
)

plot_m_sigma_ep1(differences_C, r"$\Delta$ Diffusion", "differences_C_R_ep1.png")
plot_m_sigma_ep1(
    np.log10(differences_C), r"$\log \Delta$ Diffusion", "log_differences_C_R_ep1.png"
)

plot_m_sigma_ep1(costs, r"Cost", "costs_R_ep1.png")
plot_m_sigma_ep1(np.log10(costs), r"$\log$ Cost", "log_costs_R_ep1.png")

print("DONE")
