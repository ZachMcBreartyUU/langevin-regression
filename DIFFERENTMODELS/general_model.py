from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist import Axes  # typing
from numpy.linalg import lstsq

# sindy libraries
import sympy

# data generation and final model integration
import symengine
from jitcsde import jitcsde, y

from kramersmoyal import km

from utils import (
    cost_KL,
    optimise_function,
    sindy_model,
    kl_divergence,
    SteadyFP,
    SSR_loop,
)
from utils_parallel import SSR_loop_parallel

SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/")

def run_sindy_model(
    MODEL_NAME,
    LOG_COST=False,
    LARGEST_JUMP=False,
    EVEN_ABS=False,
    PARALLEL=False,
    PDF_WEIGHTS=False,
    dt=0.001,
    num_datapoints=10_000_000,
    num_bins=100,
    kl_reg=0.001,
    ep0=0.2,
    ep1=0.1,
    coeffs=[0.0, -1.0, 0.0, 1.0],
    x0=0.0,
    folderpath: Path = SCRATCH_PATH,
    param="x",
):
    folderpath = folderpath / MODEL_NAME
    folderpath.mkdir(exist_ok=True, parents=True)
    times = np.arange(0, num_datapoints) * dt

    ## Define model: dx = A(x)dt + B(x)dw; C(x) = B^2 / 2
    # Higher order Pitchfork with multiplicative noise
    x = y(0)
    A = 0
    if EVEN_ABS:
        for i in range(len(coeffs)):
            if i % 2 == 0 and i != 0:
                A += x ** (i - 1) * symengine.Abs(x) * coeffs[i]
            else:
                A += x**i * coeffs[i]
    else:
        for i in range(len(coeffs)):
            A += x**i * coeffs[i]
    A = [A]
    B = [symengine.sqrt(ep0 + ep1 * x**2)]

    ## Integrate model
    SDE = jitcsde(A, B, n=1, additive=False)
    SDE.set_initial_value([x0])
    x_data = np.fromiter(
        (SDE.integrate(t)[0] for t in times), dtype=float, count=num_datapoints
    )
    assert x_data.shape == (num_datapoints,)

    fig_data, ax_data = plt.subplots()
    ax_data.plot(times, x_data)
    ax_data.set_ylabel(f"${param}(t)$")
    ax_data.set_xlabel("$t$")

    fig_data.tight_layout()
    fig_data.savefig(folderpath / f"{MODEL_NAME}_data.png")

    ax_data.set_xlim(0, 1000)
    fig_data.savefig(folderpath / f"{MODEL_NAME}_data_zoom.png")
    plt.close(fig_data)

    ## Perform Kramers Moyal
    edges = np.linspace(np.min(x_data), np.max(x_data), num_bins + 1)
    # edges = np.linspace(-0.005, 0.005, num_bins + 1)
    kmc, centers = km(x_data[..., None], bins=(edges,), powers=2)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers_x = centers[0]
    pdf /= np.nansum(pdf)
    moment_1 /= dt
    moment_2 /= dt
    del x_data

    ## Plot pdf and moments
    fig_pdf, ax_pdf = plt.subplots()
    ax_pdf.plot(centers_x, pdf)
    ax_pdf.set_ylabel(f"PDF(${param}$)")
    ax_pdf.set_xlabel(f"${param}$")
    fig_pdf.tight_layout()
    fig_pdf.savefig(folderpath / f"{MODEL_NAME}_pdf.png")
    plt.close(fig_pdf)

    fig_moments, (ax_A, ax_C) = plt.subplots(2)
    ax_A: Axes
    ax_C: Axes
    ax_A.plot(centers_x, moment_1)
    ax_A.set_xlabel(f"${param}$")
    ax_A.set_ylabel(f"First moment, A(${param}$)")
    ax_C.plot(centers_x, moment_2)
    ax_C.set_xlabel(f"${param}$")
    ax_C.set_ylabel(f"Second moment, C(${param}$)")
    fig_moments.savefig(folderpath / f"{MODEL_NAME}_moments.png")
    plt.close(fig_moments)

    ## Make SINDy libraries
    x_sym = sympy.symbols("x")

    # 3 additional terms greater than the largest coefficient is somewhat arbitrary
    if EVEN_ABS:
        arr = []
        for i in range(len(coeffs) + 3):
            if i % 2 == 0 and i != 0:
                arr.append(x_sym ** (i - 1) * symengine.Abs(x_sym))
            else:
                arr.append(x_sym**i)
        A_lib_expr = np.array(arr)
    else:
        A_lib_expr = np.array([x_sym**i for i in range(len(coeffs) + 3)])
    print(f"{A_lib_expr=}")
    num_A_expr = len(A_lib_expr)

    lib_A = np.empty((num_A_expr, num_bins))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, A_lib_expr[k])
        lib_A[k] = lamb_expr(centers_x)

    C_lib_expr = np.array([x_sym**i for i in range(5)])
    print(f"{C_lib_expr=}")
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers_x)

    n_terms = num_A_expr + num_C_expr

    ## Perform SSR
    Xi0 = np.empty((num_A_expr + num_C_expr))
    Xi0[:num_A_expr] = lstsq(lib_A.T, moment_1)[0]
    Xi0[num_A_expr:] = lstsq(lib_C.T, moment_2)[0]
    print(f"{Xi0=}")
    if PDF_WEIGHTS:
        weight = pdf
    else:
        weight = np.ones_like(pdf)
    weights = np.array([weight, weight])

    sfp = SteadyFP(num_bins, centers_x[1] - centers_x[0])

    params = {
        "W": weights,
        "Xi0": Xi0,
        "A_KM": moment_1,
        "C_KM": moment_2,
        "A_expr": A_lib_expr,
        "C_expr": C_lib_expr,
        "lib_A": lib_A,
        "lib_C": lib_C,
        "sfp": sfp,
        "pdf": pdf,
        "kl_reg": kl_reg,
    }

    # opt_func = lambda params: optimise_function(cost_KL, params)
    opt_func = partial(optimise_function, cost_KL)
    if PARALLEL:
        Xi, costs, active_history = SSR_loop_parallel(opt_func, params)
    else:
        Xi, costs, active_history, _ = SSR_loop(opt_func, params)
    print(f"{Xi=}")
    print(f"{costs=}")
    print(f"{active_history=}")

    # choose model
    if LOG_COST:
        costs = np.log(costs)
    dcosts = costs[1:] - costs[:-1]

    if LARGEST_JUMP:
        selected_model = np.argmax(dcosts)
    else:  # First jump
        masked = dcosts
        masked[dcosts < np.max(dcosts) / 10] = 0
        # find the first jump greater than 1/10 of the largest jump
        selected_model = np.nonzero(masked)[0][0]
    print(f"Selected model with sparsity {n_terms - selected_model}")
    chosen_V = costs[selected_model]
    chosen_Xi = Xi[selected_model]

    # TODO: Could perform some rounding on the found params, e.g. 0.012324 -> 0.012
    A_sym = sindy_model(chosen_Xi[:num_A_expr], A_lib_expr)
    A_sindy = sympy.lambdify(x_sym, A_sym)(centers_x)
    C_sym = sindy_model(chosen_Xi[num_A_expr:], C_lib_expr)
    C_sindy = sympy.lambdify(x_sym, C_sym)(centers_x)

    if np.ndim(A_sindy) == 0:
        A_sindy = np.full_like(centers_x, A_sindy)
    if np.ndim(C_sindy) == 0:
        C_sindy = np.full_like(centers_x, C_sindy)

    print(f"dx = ({A_sym}) dt + ({sympy.sqrt(2.0*C_sym)}) dβ")

    pdf_sindy = sfp.solve(A_sindy, C_sindy)
    kl_div = kl_divergence(pdf, pdf_sindy)
    kl_val = kl_div * kl_reg
    print(f"KL div = {kl_div} * {kl_reg} = {kl_val}")
    if LOG_COST:
        init = np.exp(chosen_V)
        print(f"Raw cost = {np.log(init - kl_val)}")
    else:
        print(f"Raw cost = {chosen_V - kl_val}")

    ## Plot results
    # Plot cost and dcost OR log(cost) and dlog(cost)
    fig_cost, (ax_cost, ax_dcost) = plt.subplots(2, figsize=(6, 6))
    ax_cost: Axes
    ax_dcost: Axes
    sparsity = np.arange(n_terms, 1, -1)
    ax_cost.scatter(sparsity, costs)
    ax_dcost.scatter(sparsity[:-1], dcosts)

    min_, max_ = ax_cost.get_xlim()[::-1]
    ax_cost.set_xlim(min_, max_)
    ax_cost.set_xlabel("Sparsity")
    ax_dcost.set_xlim(min_, max_)
    ax_dcost.set_xlabel("Sparsity")

    if LOG_COST:
        # The logging happened when the model was being chosen
        ax_cost.set_ylabel(r"log Cost, $\log V$")
        ax_dcost.set_ylabel(r"dlog Cost, $d\log V$")
        fig_cost.tight_layout()
        fig_cost.savefig(folderpath / f"{MODEL_NAME}_SSR_logcost.png")
    else:
        ax_cost.set_ylabel("Cost, $V$")
        ax_dcost.set_ylabel("dCost, $dV$")
        fig_cost.tight_layout()
        fig_cost.savefig(folderpath / f"{MODEL_NAME}_SSR_cost.png")
    plt.close(fig_cost)

    # Plot history
    fig_history, ax_history = plt.subplots()

    sympy_labels = [
        rf"${sympy.latex(t)}$" for t in np.concatenate((A_lib_expr, C_lib_expr))
    ]

    square = np.zeros_like(Xi.T)
    for i, hist in enumerate(active_history):
        square[hist, i] = 1
    square = square.astype(bool)

    # histories
    ax_history.pcolor(square, cmap="bone_r", edgecolors="gray")
    # drift / diffusion delimiters
    ax_history.axhline(y=num_A_expr, color="red")
    ax_history.set_yticks(0.5 + np.arange(n_terms))
    ax_history.set_yticklabels(sympy_labels)
    ax_history.set_xticks(0.5 + np.arange(n_terms - 1))
    ax_history.set_xticklabels(sparsity)
    ax_history.set_xlabel("Sparsity")
    ax_history.set_ylabel("Active terms")

    fig_history.tight_layout()
    fig_history.savefig(folderpath / f"{MODEL_NAME}_SSR_sparsity.png")
    plt.close(fig_history)

    # Plot found pdf vs KM pdf
    fig_pdf_comp, ax_pdf_comp = plt.subplots()

    # This norming should be redundant
    pdf_normed = pdf / np.nansum(pdf)
    pdf_sindy_normed = pdf_sindy / np.nansum(pdf_sindy)

    ax_pdf_comp.plot(centers_x, pdf_normed, label="Data")
    ax_pdf_comp.plot(centers_x, pdf_sindy_normed, label="SINDy")
    ax_pdf_comp.set_ylabel(f"PDF(${param}$)")
    ax_pdf_comp.set_xlabel(f"${param}$")
    ax_pdf_comp.legend()
    fig_pdf_comp.tight_layout()
    fig_pdf_comp.savefig(folderpath / f"{MODEL_NAME}_pdf_comparison.png")
    plt.close(fig_pdf_comp)

    # Plot found model vs KM moments
    fig_moments_comp, (ax_A_comp, ax_C_comp) = plt.subplots(2)
    ax_A_comp: Axes
    ax_C_comp: Axes
    ax_A_comp.scatter(centers_x, moment_1, marker="x", label="KM")
    ax_A_comp.plot(centers_x, A_sindy, "r", label="Found model")
    ax_A_comp.set_xlabel(r"${param}$")
    ax_A_comp.set_ylabel(r"First moment, A(${param}$)")
    ax_C_comp.scatter(centers_x, moment_2, marker="x", label="KM")
    ax_C_comp.plot(centers_x, C_sindy, "r", label="Found model")
    ax_C_comp.set_xlabel(f"${param}$")
    ax_C_comp.set_ylabel(f"Second moment, C(${param}$)")

    ax_A_comp.legend()
    # ax_C_comp.legend()

    fig_moments_comp.tight_layout()
    fig_moments_comp.savefig(folderpath / f"{MODEL_NAME}_moments_comparison.png")

    ax_A_comp.set_ylim(-0.4, 0.4)
    fig_moments_comp.savefig(folderpath / f"{MODEL_NAME}_moments_comparison_yzoom.png")

    plt.close(fig_moments_comp)

    ## Directly compare True answer to Found answer
    true_model_xi = np.zeros(n_terms)
    # encoding dx = (-x + x^3 - x^5/6)dt + sqrt(ep0 + ep1 x^2)dw
    # this assumes that the models only contain integer powers of x in ascending order
    true_model_xi[: num_A_expr - 3] = np.array(coeffs)
    true_model_xi[num_A_expr + 0] = ep0
    true_model_xi[num_A_expr + 2] = ep1

    differences = np.abs(chosen_Xi - true_model_xi)
    print("Sum of differences in true vs found model:", sum(differences))

    fig_direct, (ax_true, ax_found, ax_difference) = plt.subplots(3)
    ax_true: Axes
    ax_found: Axes
    ax_difference: Axes

    xaxis = np.arange(n_terms)
    ax_true.set_ylabel("True coefficients, a")
    ax_true.scatter(xaxis, true_model_xi)
    ax_found.set_ylabel("Found coefficients, b")
    ax_found.scatter(xaxis, chosen_Xi)
    ax_difference.set_ylabel("$|(a - b)|$")
    ax_difference.scatter(xaxis, differences)
    ax_difference.set_ylim(bottom=0)

    for ax in (ax_true, ax_found, ax_difference):
        ax.set_xticks(xaxis)
        ax.set_xticklabels(sympy_labels)

    fig_direct.tight_layout()
    fig_direct.savefig(folderpath / f"{MODEL_NAME}_direct_comparison.png")
    plt.close(fig_direct)


if __name__ == "__main__":
    import argparse

    df = argparse.ArgumentParser()
    df.add_argument("MODEL_NAME")
    df.add_argument("--LOG_COST", action="store_true")
    df.add_argument("--LARGEST_JUMP", action="store_true")
    df.add_argument("--PARALLEL", action="store_true")
    df.add_argument("--PDF_WEIGHTS", action="store_true")
    df.add_argument("-dt", "--timestep", type=float, default=0.001)
    df.add_argument("-N", "--num-steps", type=int, default=10_000_000)
    df.add_argument("-B", "--num-bins", type=int, default=100)
    df.add_argument("-k", "--kl-reg", type=float, default=1e-3)

    int_param_group = df.add_argument_group("Integration Parameters")
    int_param_group.add_argument("--EVEN_ABS", action="store_true")
    int_param_group.add_argument("-x0", type=float, default=0.0)

    int_param_group.add_argument("-ep0", type=float, default=0.2)
    int_param_group.add_argument("-ep1", type=float, default=0.1)
    # one or more arguments
    int_param_group.add_argument("--coeffs", type=float, nargs="+", default=[0.0])
    args = df.parse_args()
    print(args)

    MODEL_NAME = args.MODEL_NAME  # "pitchfork_higher_order"
    LOG_COST = args.LOG_COST
    LARGEST_JUMP = args.LARGEST_JUMP
    EVEN_ABS = args.EVEN_ABS
    PARALLEL = args.PARALLEL
    PDF_WEIGHTS = args.PDF_WEIGHTS
    dt = args.timestep
    num_datapoints = args.num_steps
    num_bins = args.num_bins
    kl_reg = args.kl_reg
    ep0 = args.ep0
    ep1 = args.ep1

    coeffs = args.coeffs
    x0 = args.x0

    run_sindy_model(
        MODEL_NAME,
        LOG_COST,
        LARGEST_JUMP,
        EVEN_ABS,
        PARALLEL,
        PDF_WEIGHTS,
        dt,
        num_datapoints,
        num_bins,
        kl_reg,
        ep0,
        ep1,
        coeffs,
        x0,
        param=r"\phi",
    )
