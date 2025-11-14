from typing import Optional
import datetime
from pathlib import Path
from functools import partial
import json

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
    optimise_function,
    sindy_model,
    kl_divergence,
    SteadyFP,
    SSR_loop,
    round_array_to_SF,
)
from utils_parallel import SSR_loop_parallel
from utils_stack import cost_stack, cost_KL_stack, cost_jef_stack

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
    models_dir="MODELS",
):
    ADD_TERMS = 1  # TODO: MAKE INTO FUNCTION ARGUMENT AND CL ARGUMENT
    NUM_STACKS = 10  # TODO: MAKE INTO FUNCTION ARGUMENT AND CL ARGUMENT
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
        first_weights = 1 / np.std(moment_1s, axis=0, keepdims=True)  # (1, N_BINS)
        second_weights = 1 / np.std(moment_2s, axis=0, keepdims=True)  # (1, N_BINS)
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
    ax_WA.set_ylabel(f"First weight, A(${param}$)")
    ax_WC.set_xlabel(f"${param}$")
    ax_WC.set_ylabel(f"Second weight, C(${param}$)")
    fig_weights.savefig(folderpath / f"{MODEL_NAME}_weights_stack.png")
    plt.close(fig_weights)

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
    # compare only the first PDF with the SINDy model (for now?)
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

    # Plot found model vs KM moments
    EXPLICIT = True
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
                everymodel / f"{MODEL_NAME}_moments_comparison_sparsity_{sparsity}.png"
            )

            ax_A_comp.set_ylim(-0.4, 0.4)
            fig_moments_comp.savefig(
                everymodel
                / f"{MODEL_NAME}_moments_comparison_yzoom_sparsity_{sparsity}.png"
            )

            plt.close(fig_moments_comp)
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


def _check_metadata(
    models_dir: Path,
    target_metadata: dict,
):
    if not (models_dir / "metadata.json").exists():
        return False
    with open(models_dir / "metadata.json") as metadata_file:
        metadata = json.load(metadata_file)

    checks = [
        "num_datapoints",
        "dt",
        "EVEN_ABS",
        "coeffs",
        "ep0",
        "ep1",
        "x0",
        "num_bins",
    ]
    for check in checks:
        if metadata[check] != target_metadata[check]:
            return False
    return True


def _check_exists(models_dir: Path, NUM_MODELS: int, filename_template: str):
    for i in range(NUM_MODELS):
        if not (models_dir / (filename_template.format(i))).exists():
            return False
    return True


def load_metadata(models_dir):
    with open(models_dir / "metadata.json") as metadata_file:
        return json.load(metadata_file)


def write_metadata(models_dir, metadata):
    with open(models_dir / "metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file)


def generate_dataseries(models_dir, NUM_MODELS):
    metadata = load_metadata(models_dir)
    for i in range(NUM_MODELS):
        times = np.arange(0, num_datapoints) * dt
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

        self_min = np.min(x_data)
        self_max = np.max(x_data)
        metadata["min_x"] = min(metadata["min_x"], self_min)
        metadata["max_x"] = min(metadata["max_x"], self_max)
        write_metadata(models_dir, metadata)
        print(f"Saving timeseries {i+1} / {NUM_MODELS}", flush=True)
        np.savez(models_dir / f"timeseries_{i}.npz", times=times, x_data=x_data)


def _load_timeseries(timeseries_path):
    with np.load(timeseries_path) as timeseries_file:
        times = timeseries_file["times"]
        x_data = timeseries_file["x_data"]
    return times, x_data


def set_metadata_min_max(models_dir, NUM_MODELS):
    metadata = load_metadata(models_dir)
    new_min = metadata["min_x"]
    new_max = metadata["max_x"]
    for i in range(NUM_MODELS):
        _, x_data = _load_timeseries(models_dir / f"timeseries_{i}.npz")
        self_min = np.min(x_data)
        self_max = np.max(x_data)
        new_min = min(new_min, self_min)
        new_max = max(new_max, self_max)
    metadata["min_x"] = new_min
    metadata["max_x"] = new_max
    write_metadata(models_dir, metadata)
    return new_min, new_max


def generate_models(models_dir, NUM_MODELS):
    # look at the min and max in the metadata
    # file for hint to the bin edges
    metadata = load_metadata(models_dir)
    if metadata["min_x"] == metadata["max_x"]:
        metadata["min_x"], metadata["max_x"] = set_metadata_min_max(
            models_dir, NUM_MODELS
        )

    edges = np.linspace(metadata["min_x"], metadata["max_x"], metadata["num_bins"] + 1)
    centers = edges[1:] - edges[:-1]
    for i in range(NUM_MODELS):
        _, x_data = _load_timeseries(models_dir / f"timeseries_{i}.npz")
        if metadata["EVEN_ABS"]:
            x_data = np.append(x_data, -x_data)

        kmc, centers = km(x_data[..., None], bins=(edges,), powers=2)  # type: ignore
        pdf, moment_1, moment_2 = kmc
        centers = centers[0]
        pdf /= np.nansum(pdf)
        moment_1 /= dt
        moment_2 /= dt
        print(f"Saving model {i+1} / {NUM_MODELS}", flush=True)
        np.savez(
            models_dir / f"model_{i}.npz",
            centers=centers,
            pdf=pdf,
            moment_1=moment_1,
            moment_2=moment_2,
        )


def _load_model(model_path):
    with np.load(model_path) as model_file:
        centers = model_file["centers"]
        pdf = model_file["pdf"]
        moment_1 = model_file["moment_1"]
        moment_2 = model_file["moment_2"]
    return centers, pdf, moment_1, moment_2


def combine_models(models_dir, NUM_MODELS):
    # load all models
    # stack pdfs and moments
    centers = None
    pdfs = []
    moment_1_s = []
    moment_2_s = []
    for i in range(NUM_MODELS):
        centers, pdf, moment_1, moment_2 = _load_model(models_dir / f"model_{i}.npz")
        pdfs.append(pdf)
        moment_1_s.append(moment_1)
        moment_2_s.append(moment_2)
    assert centers is not None
    pdf_stack = np.stack(pdfs, axis=0)
    moment_1_stack = np.stack(moment_1_s, axis=0)
    moment_2_stack = np.stack(moment_2_s, axis=0)
    return centers, pdf_stack, moment_1_stack, moment_2_stack


def get_models(
    models_dir: Path,
    target_metadata: dict,
    NUM_MODELS=10,
):
    # Check if the models directory exists
    models_dir.mkdir(exist_ok=True)
    #   Check if there is metadata and that it is correct
    if not _check_metadata(models_dir, target_metadata):
        print(
            "No or incorrect metadata found: generating data, processing, and stacking",
            flush=True,
        )
        target_metadata["min_x"] = target_metadata["x0"]
        target_metadata["max_x"] = target_metadata["x0"]
        write_metadata(models_dir, target_metadata)
        generate_dataseries(models_dir, NUM_MODELS)
        generate_models(models_dir, NUM_MODELS)
    elif _check_exists(models_dir, NUM_MODELS, "model_{}.npz"):
        print("Found models: loading and stacking", flush=True)
        pass
    elif _check_exists(models_dir, NUM_MODELS, "timeseries_{}.npz"):
        print(
            "Found timeseries but no models: loading data, processing, and stacking",
            flush=True,
        )
        generate_models(models_dir, NUM_MODELS)
    else:
        print(
            "No timeseries or models: generating, processing, and stacking", flush=True
        )
        generate_dataseries(models_dir, NUM_MODELS)
        generate_models(models_dir, NUM_MODELS)
    return combine_models(models_dir, NUM_MODELS)


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
