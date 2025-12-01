import datetime
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

from utils import (
    optimise_function,
    sindy_model,
    kl_divergence,
    SteadyFP,
    SSR_loop,
    round_array_to_SF,
)
from utils_parallel import SSR_loop_parallel
from utils_stack import cost_stack, cost_KL_stack, cost_jef_stack
from make_and_load_models import get_models


SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/")
print(f"Running on date: {datetime.datetime.now()}")


def run_sindy_model_stacking(
    MODEL_NAME,
    LOG_COST=False,
    LARGEST_JUMP=False,
    EVEN_ABS=False,
    PARALLEL=False,
    ERROR_WEIGHTS=False,
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
    models_dir="MODELS",  # TODO: CL ARGUMENT
    ADD_TERMS=1,  # TODO: CL ARGUMENT
    NUM_STACKS=10,  # TODO: CL ARGUMENT
    EXPLICIT=True,  # TODO: CL ARGUMENT
):
    folderpath = folderpath / MODEL_NAME
    folderpath.mkdir(exist_ok=True, parents=True)

    target_metadata = {
        "num_datapoints": num_datapoints,
        "dt": dt,
        "EVEN_ABS": EVEN_ABS,
        "coeffs": coeffs,
        "ep0": ep0,
        "ep1": ep1,
        "x0": x0,
        "num_bins": num_bins,
    }
    centers, pdfs, moment_1s, moment_2s = get_models(
        folderpath.parent / models_dir, target_metadata, NUM_STACKS
    )
    print(f"{centers.shape=}")
    print(f"{pdfs.shape=}")
    print(f"{moment_1s.shape=}")
    print(f"{moment_2s.shape=}")

    ## Plot pdf and moments
    fig_pdf, ax_pdf = plt.subplots()
    for pdf in pdfs:
        ax_pdf.plot(centers, pdf)
    ax_pdf.set_ylabel(f"PDF(${param}$)")
    ax_pdf.set_xlabel(f"${param}$")
    fig_pdf.tight_layout()
    fig_pdf.savefig(folderpath / f"{MODEL_NAME}_pdf_stack.png")
    plt.close(fig_pdf)

    fig_moments, (ax_A, ax_C) = plt.subplots(2)
    ax_A: Axes
    ax_C: Axes
    for moment_1, moment_2 in zip(moment_1s, moment_2s):
        ax_A.scatter(centers, moment_1, marker="x")
        ax_C.scatter(centers, moment_2, marker="x")
    ax_A.set_xlabel(f"${param}$")
    ax_A.set_ylabel(f"First moment, A(${param}$)")
    ax_C.set_xlabel(f"${param}$")
    ax_C.set_ylabel(f"Second moment, C(${param}$)")
    fig_moments.savefig(folderpath / f"{MODEL_NAME}_moments_stack.png")
    plt.close(fig_moments)

    ## Make SINDy libraries
    x_sym = sympy.symbols("x")

    if EVEN_ABS:
        arr = []
        for i in range(len(coeffs) + ADD_TERMS):
            if i % 2 == 0 and i != 0:
                arr.append(x_sym ** (i - 1) * symengine.Abs(x_sym))
            else:
                arr.append(x_sym**i)
        A_lib_expr = np.array(arr)
    else:
        A_lib_expr = np.array([x_sym**i for i in range(len(coeffs) + ADD_TERMS)])
    print(f"{A_lib_expr=}")
    num_A_expr = len(A_lib_expr)

    lib_A = np.empty((num_A_expr, num_bins))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, A_lib_expr[k])
        lib_A[k] = lamb_expr(centers)

    C_lib_expr = np.array([x_sym**i for i in range(3 + ADD_TERMS)])
    print(f"{C_lib_expr=}")
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers)

    n_terms = num_A_expr + num_C_expr

    ## Perform SSR
    Xi0 = np.empty((num_A_expr + num_C_expr))
    A_coeffs = np.average(lstsq(lib_A.T, moment_1s.T)[0], axis=1)
    A_coeffs[-1] = -np.abs(A_coeffs[-1])
    Xi0[:num_A_expr] = A_coeffs
    Xi0[num_A_expr:] = np.average(lstsq(lib_C.T, moment_2s.T)[0], axis=1)
    print(f"{Xi0=}\n", flush=True)
    if ERROR_WEIGHTS:
        first_weights = pdfs  # (NUM_STACKS, N_BINS)
        second_weights = pdfs / moment_2s  # (NUM_STACKS, N_BINS)
    else:
        # first_weights = np.ones_like(pdfs)
        # second_weights = np.ones_like(pdfs)
        # Calculate weights by the spread of the data
        first_weights = np.abs(
            1  # np.mean(moment_1s, axis=0, keepdims=True)
            / np.std(moment_1s, axis=0, keepdims=True)
        )  # (1, N_BINS)
        second_weights = np.abs(
            1  # np.mean(moment_2s, axis=0, keepdims=True)
            / np.std(moment_2s, axis=0, keepdims=True)
        )  # (1, N_BINS)
    first_weights /= np.sum(first_weights, axis=1, keepdims=True)  # (1, N_BINS)
    second_weights /= np.sum(second_weights, axis=1, keepdims=True)  # (1, N_BINS)

    weights = np.array([first_weights, second_weights])

    fig_weights, (ax_WA, ax_WC) = plt.subplots(2)
    ax_WA: Axes
    ax_WC: Axes
    for first_weight, second_weight in zip(first_weights, second_weights):
        ax_WA.scatter(centers, first_weight)
        ax_WC.scatter(centers, second_weight)
    ax_WA.set_xlabel(f"${param}$")
    ax_WA.set_ylabel(f"First weight, $W_A({param})$")
    ax_WC.set_xlabel(f"${param}$")
    ax_WC.set_ylabel(f"Second weight, $W_C({param})$")
    fig_weights.savefig(folderpath / f"{MODEL_NAME}_weights_stack.png")
    plt.close(fig_weights)

    fig_weights_log, (ax_WA_log, ax_WC_log) = plt.subplots(2)
    ax_WA_log: Axes
    ax_WC_log: Axes
    for first_weight, second_weight in zip(
        np.log10(first_weights), np.log10(second_weights)
    ):
        ax_WA_log.scatter(centers, first_weight)
        ax_WC_log.scatter(centers, second_weight)
    ax_WA_log.set_xlabel(f"${param}$")
    ax_WA_log.set_ylabel(rf"log First weight, $\log W_A({param})$")
    ax_WC_log.set_xlabel(f"${param}$")
    ax_WC_log.set_ylabel(rf"log Second weight, $\log W_C({param})$")
    fig_weights_log.savefig(folderpath / f"{MODEL_NAME}_log_weights_stack.png")
    plt.close(fig_weights_log)

    sfp = SteadyFP(num_bins, centers[1] - centers[0])

    params = {
        "Ws": weights,
        "Xi0": Xi0,
        "A_KMs": moment_1s,
        "C_KMs": moment_2s,
        "A_expr": A_lib_expr,
        "C_expr": C_lib_expr,
        "lib_A": lib_A,
        "lib_C": lib_C,
        "sfp": sfp,
        "pdfs": pdfs,
        "kl_reg": kl_reg,
    }

    if kl_reg > 0:
        opt_func = partial(optimise_function, cost_jef_stack)
        # opt_func = partial(optimise_function, cost_KL_stack)
    else:
        opt_func = partial(optimise_function, cost_stack)

    if PARALLEL:
        Xis, costs, active_history = SSR_loop_parallel(opt_func, params)
    else:
        Xis, costs, active_history, _ = SSR_loop(opt_func, params)
    print(f"{Xis=}")
    print(f"{costs=}")
    print(f"{active_history=}\n")

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
    chosen_Xi = Xis[selected_model]

    # use only the first set of bins for SINDy
    centers_sindy = centers[:num_bins]

    # TODO: Could perform some rounding on the found params, e.g. 0.012324 -> 0.012
    A_sym = sindy_model(chosen_Xi[:num_A_expr], A_lib_expr)
    A_sindy = sympy.lambdify(x_sym, A_sym)(centers_sindy)
    C_sym = sindy_model(chosen_Xi[num_A_expr:], C_lib_expr)
    C_sindy = sympy.lambdify(x_sym, C_sym)(centers_sindy)

    if np.ndim(A_sindy) == 0:
        A_sindy = np.full_like(centers_sindy, A_sindy)
    if np.ndim(C_sindy) == 0:
        C_sindy = np.full_like(centers_sindy, C_sindy)

    print(f"dx = ({A_sym}) dt + ({sympy.sqrt(2.0*C_sym)}) dβ")

    chosen_Xi_round = round_array_to_SF(chosen_Xi, 2)
    A_sym_round = sindy_model(chosen_Xi_round[:num_A_expr], A_lib_expr)
    C_sym_round = sindy_model(chosen_Xi_round[num_A_expr:], C_lib_expr)
    print(
        rf"dx = ({sympy.latex(A_sym_round)}) dt + ({sympy.latex(sympy.sqrt(2.0*C_sym_round))}) d\beta"
    )

    pdf_sindy = sfp.solve(A_sindy, C_sindy)
    kl_divs = kl_divergence(pdfs, pdf_sindy, sfp.dx, tol=1e-6)
    kl_div = np.sum(kl_divs[kl_divs > 0])
    kl_val = kl_div * kl_reg
    print(f"KL div = {kl_div} * {kl_reg} = {kl_val}")
    if LOG_COST:
        init = np.exp(chosen_V)
        print(f"Raw cost = {np.log((init - kl_val) / kl_reg)}")
    else:
        print(f"Raw cost = {(chosen_V - kl_val) / kl_reg}")

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

    square = np.zeros_like(Xis.T)
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

    # Plot found model vs KM moments
    if EXPLICIT:
        for i in range(len(Xis)):
            Xi_rounded = round_array_to_SF(Xis[i], 2)
            A_sym = sindy_model(Xi_rounded[:num_A_expr], A_lib_expr)
            A_sindy = sympy.lambdify(x_sym, A_sym)(centers_sindy)
            C_sym = sindy_model(Xi_rounded[num_A_expr:], C_lib_expr)
            C_sindy = sympy.lambdify(x_sym, C_sym)(centers_sindy)

            if np.ndim(A_sindy) == 0:
                A_sindy = np.full_like(centers_sindy, A_sindy)
            if np.ndim(C_sindy) == 0:
                C_sindy = np.full_like(centers_sindy, C_sindy)

            title = rf"$dx = ({sympy.latex(A_sym)}) dt + ({sympy.latex(sympy.sqrt(2.0*C_sym))}) d\beta$"
            fig_moments_comp, (ax_A_comp, ax_C_comp) = plt.subplots(2)
            ax_A_comp: Axes
            ax_C_comp: Axes

            ax_A_comp.scatter(centers, moment_1s[0], marker="x", label="KM")
            ax_C_comp.scatter(centers, moment_2s[0], marker="x", label="KM")
            for moment_1, moment_2 in zip(moment_1s[1:], moment_2s[1:]):
                ax_A_comp.scatter(centers, moment_1, marker="x")
                ax_C_comp.scatter(centers, moment_2, marker="x")
            ax_A_comp.plot(centers_sindy, A_sindy, "r", label="Found model")
            ax_C_comp.plot(centers_sindy, C_sindy, "r", label="Found model")

            ax_A_comp.set_title(title)
            ax_A_comp.set_xlabel(f"${param}$")
            ax_A_comp.set_ylabel(f"First moment, A(${param}$)")
            ax_C_comp.set_xlabel(f"${param}$")
            ax_C_comp.set_ylabel(f"Second moment, C(${param}$)")

            ax_A_comp.legend()
            # ax_C_comp.legend()

            fig_moments_comp.tight_layout()
            everymodel = folderpath / "everymodel"
            everymodel.mkdir(exist_ok=True)
            fig_moments_comp.savefig(
                everymodel
                / f"{MODEL_NAME}_moments_comparison_sparsity_{sparsity[i]}.png"
            )

            ax_A_comp.set_ylim(-0.4, 0.4)
            fig_moments_comp.savefig(
                everymodel
                / f"{MODEL_NAME}_moments_comparison_yzoom_sparsity_{sparsity[i]}.png"
            )

            plt.close(fig_moments_comp)

            pdf_sindy = sfp.solve(A_sindy, C_sindy)
            # Plot found pdf vs KM pdf
            fig_pdf_comp, ax_pdf_comp = plt.subplots()

            pdfs_normed = pdfs / np.nansum(pdfs, axis=1, keepdims=True)
            pdf_sindy_normed = pdf_sindy / np.nansum(pdf_sindy)
            ax_pdf_comp.plot(centers, pdfs_normed[0], label="Data")
            for pdf_normed in pdfs_normed[1:]:
                ax_pdf_comp.plot(centers, pdf_normed)
            ax_pdf_comp.plot(centers_sindy, pdf_sindy_normed, "--", label="SINDy")

            ax_pdf_comp.set_title(title)
            ax_pdf_comp.set_ylabel(f"PDF(${param}$)")
            ax_pdf_comp.set_xlabel(f"${param}$")
            ax_pdf_comp.legend()
            fig_pdf_comp.tight_layout()
            fig_pdf_comp.savefig(
                folderpath / f"{MODEL_NAME}_pdf_comparison_sparsity_{sparsity[i]}.png"
            )
            plt.close(fig_pdf_comp)
    else:
        fig_moments_comp, (ax_A_comp, ax_C_comp) = plt.subplots(2)
        ax_A_comp: Axes
        ax_C_comp: Axes

        ax_A_comp.scatter(centers, moment_1s[0], marker="x", label="KM")
        ax_C_comp.scatter(centers, moment_2s[0], marker="x", label="KM")
        for moment_1, moment_2 in zip(moment_1s[1:], moment_2s[1:]):
            ax_A_comp.scatter(centers, moment_1, marker="x")
            ax_C_comp.scatter(centers, moment_2, marker="x")
        ax_A_comp.plot(centers_sindy, A_sindy, "r", label="Found model")
        ax_C_comp.plot(centers_sindy, C_sindy, "r", label="Found model")

        ax_A_comp.set_xlabel(f"${param}$")
        ax_A_comp.set_ylabel(f"First moment, A(${param}$)")
        ax_C_comp.set_xlabel(f"${param}$")
        ax_C_comp.set_ylabel(f"Second moment, C(${param}$)")

        ax_A_comp.legend()
        # ax_C_comp.legend()

        fig_moments_comp.tight_layout()
        fig_moments_comp.savefig(folderpath / f"{MODEL_NAME}_moments_comparison.png")

        ax_A_comp.set_ylim(-0.4, 0.4)
        fig_moments_comp.savefig(
            folderpath / f"{MODEL_NAME}_moments_comparison_yzoom.png"
        )

        plt.close(fig_moments_comp)

        # Plot found pdf vs KM pdf
        fig_pdf_comp, ax_pdf_comp = plt.subplots()

        # This norming should be redundant
        pdfs_normed = pdfs / np.nansum(pdfs, axis=1, keepdims=True)
        pdf_sindy_normed = pdf_sindy / np.nansum(pdf_sindy)
        ax_pdf_comp.plot(centers, pdfs_normed[0], label="Data")
        for pdf_normed in pdfs_normed[1:]:
            ax_pdf_comp.plot(centers, pdf_normed)
        ax_pdf_comp.plot(centers_sindy, pdf_sindy_normed, "--", label="SINDy")
        ax_pdf_comp.set_ylabel(f"PDF(${param}$)")
        ax_pdf_comp.set_xlabel(f"${param}$")
        ax_pdf_comp.legend()
        fig_pdf_comp.tight_layout()
        fig_pdf_comp.savefig(folderpath / f"{MODEL_NAME}_pdf_comparison.png")
        plt.close(fig_pdf_comp)

    ## Directly compare True answer to Found answer
    true_model_xi = np.zeros(n_terms)
    true_model_xi[: num_A_expr - ADD_TERMS] = np.array(coeffs)
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
    df.add_argument("--ERROR_WEIGHTS", action="store_true")
    df.add_argument("-dt", "--timestep", type=float, default=0.001)
    df.add_argument("-N", "--num-steps", type=int, default=10_000_000)
    df.add_argument("-B", "--num-bins", type=int, default=100)
    df.add_argument("-k", "--kl-reg", type=float, default=1e-3)
    df.add_argument("-l", "--lasso", type=float, default=0)

    int_param_group = df.add_argument_group("Integration Parameters")
    int_param_group.add_argument("--EVEN_ABS", action="store_true")
    int_param_group.add_argument("-x0", type=float, default=0.0)

    int_param_group.add_argument("-ep0", type=float, default=0.2)
    int_param_group.add_argument("-ep1", type=float, default=0.1)
    # one or more arguments
    int_param_group.add_argument("--coeffs", type=float, nargs="+", default=[0.0])
    args = df.parse_args()
    print(args)

    MODEL_NAME = args.MODEL_NAME
    LOG_COST = args.LOG_COST
    LARGEST_JUMP = args.LARGEST_JUMP
    EVEN_ABS = args.EVEN_ABS
    PARALLEL = args.PARALLEL
    ERROR_WEIGHTS = args.ERROR_WEIGHTS
    dt = args.timestep
    num_datapoints = args.num_steps
    num_bins = args.num_bins
    kl_reg = args.kl_reg
    lasso = args.lasso
    ep0 = args.ep0
    ep1 = args.ep1

    coeffs = args.coeffs
    x0 = args.x0

    run_sindy_model_stacking(
        MODEL_NAME,
        LOG_COST,
        LARGEST_JUMP,
        EVEN_ABS,
        PARALLEL,
        ERROR_WEIGHTS,
        dt,
        num_datapoints,
        num_bins,
        kl_reg,
        ep0,
        ep1,
        coeffs,
        x0,
        param=r"\phi",
        models_dir="MODELS",
    )
