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
    cost,
    cost_KL,
    optimise_function,
    sindy_model,
    kl_divergence,
    SteadyFP,
    SSR_loop,
)
from utils_parallel import SSR_loop_parallel

from utils_lasso import (
    cost as cost_lasso,
    cost_KL as cost_KL_lasso,
    _square_diff,
    _lasso,
    _kl_reg,
)


SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/")


def run_sindy_model(
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
):
    ADD_TERMS = 0  # TODO: MAKE INTO FUNCTION ARGUMENT AND CL ARGUMENT
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

    plt.close(fig_data)

    fig_data, ax_data = plt.subplots()
    lmt = int(1000 / dt)
    ax_data.plot(times[:lmt], x_data[:lmt])
    ax_data.set_ylabel(f"${param}(t)$")
    ax_data.set_xlabel("$t$")

    fig_data.tight_layout()
    fig_data.savefig(folderpath / f"{MODEL_NAME}_data_zoom.png")
    plt.close(fig_data)

    ## Duplicate the data
    if EVEN_ABS:
        # When we make all even terms odd (by including the abs) then
        # We are assuming symmetry in x, so put this symmetry in the dataset
        x_data = np.append(x_data, -x_data)

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
        lib_A[k] = lamb_expr(centers_x)

    C_lib_expr = np.array([x_sym**i for i in range(3 + ADD_TERMS)])
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
    if ERROR_WEIGHTS:
        first_weight = pdf
        second_weight = pdf / moment_2
        second_weight /= np.sum(second_weight)
    else:
        first_weight = np.ones_like(pdf)
        second_weight = np.ones_like(pdf)
    weights = np.array([first_weight, second_weight])

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

    if kl_reg > 0:
        opt_func = partial(optimise_function, cost_KL)
    else:
        opt_func = partial(optimise_function, cost)

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
    ax_A_comp.set_xlabel(f"${param}$")
    ax_A_comp.set_ylabel(f"First moment, A(${param}$)")
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


def run_sindy_model_LASSO(
    MODEL_NAME,
    EVEN_ABS=False,
    ERROR_WEIGHTS=False,
    dt=0.001,
    num_datapoints=10_000_000,
    num_bins=100,
    kl_reg=0.001,
    lasso=0.01,
    ep0=0.2,
    ep1=0.1,
    coeffs=[0.0, -1.0, 0.0, 1.0],
    x0=0.0,
    folderpath: Path = SCRATCH_PATH,
    param="x",
):
    ADD_TERMS = 0  # TODO: MAKE INTO FUNCTION ARGUMENT AND CL ARGUMENT
    folderpath = folderpath / MODEL_NAME
    folderpath.mkdir(exist_ok=True, parents=True)
    _, centers_x, pdf, moment_1, moment_2 = get_model(
        folderpath, num_datapoints, dt, EVEN_ABS, coeffs, ep0, ep1, x0
    )

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
        lib_A[k] = lamb_expr(centers_x)

    C_lib_expr = np.array([x_sym**i for i in range(3 + ADD_TERMS)])
    print(f"{C_lib_expr=}")
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers_x)

    n_terms = num_A_expr + num_C_expr

    ## Perform LASSO regression
    Xi0 = np.empty((num_A_expr + num_C_expr))
    Xi0[:num_A_expr] = lstsq(lib_A.T, moment_1)[0]
    Xi0[num_A_expr:] = lstsq(lib_C.T, moment_2)[0]
    print(f"{Xi0=}")
    if ERROR_WEIGHTS:
        first_weight = pdf
        second_weight = pdf / moment_2
        second_weight /= np.sum(second_weight)
    else:
        first_weight = np.ones_like(pdf)
        second_weight = np.ones_like(pdf)
    weights = np.array([first_weight, second_weight])

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

    if kl_reg > 0:
        opt_func = partial(optimise_function, cost_KL_lasso)
    else:
        opt_func = partial(optimise_function, cost_lasso)

    Xi, cost_ = opt_func(params)
    Xi[Xi < 1e-8] = 0  # threshold
    print(f"{Xi=}")
    print(f"{cost_=}")
    sparsity = np.count_nonzero(Xi)
    print(f"{sparsity=}")

    # TODO: Could perform some rounding on the found params, e.g. 0.012324 -> 0.012
    A_sym = sindy_model(Xi[:num_A_expr], A_lib_expr)
    A_sindy = sympy.lambdify(x_sym, A_sym)(centers_x)
    C_sym = sindy_model(Xi[num_A_expr:], C_lib_expr)
    C_sindy = sympy.lambdify(x_sym, C_sym)(centers_x)

    if np.ndim(A_sindy) == 0:
        A_sindy = np.full_like(centers_x, A_sindy)
    if np.ndim(C_sindy) == 0:
        C_sindy = np.full_like(centers_x, C_sindy)

    print(f"dx = ({A_sym}) dt + ({sympy.sqrt(2.0*C_sym)}) dβ")

    lasso_val = _lasso(Xi)
    pdf_sindy = sfp.solve(A_sindy, C_sindy)
    kl_val = max(kl_divergence(pdf, pdf_sindy, sfp.dx, tol=1e-6), 0.0)
    sqr_diff = _square_diff(weights, A_sindy, moment_1, C_sindy, moment_2)

    print(f"lasso_val = {lasso} * {lasso_val} = {lasso * lasso_val}")
    print(
        f"kl_val = {(1 - lasso) * (kl_reg)} * {kl_val} = {(1 - lasso) * (kl_reg) * kl_val}"
    )
    print(
        f"sqr_diff = {(1-lasso)*(1-kl_reg)} * {sqr_diff} = {(1-lasso)*(1-kl_reg) * sqr_diff}"
    )

    ## Plot results
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
    ax_A_comp.set_xlabel(f"${param}$")
    ax_A_comp.set_ylabel(f"First moment, A(${param}$)")
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


def _check_metadata(metadata, num_datapoints, dt, EVEN_ABS, coeffs, ep0, ep1, x0):
    return (
        metadata["num_datapoints"] == num_datapoints
        and metadata["dt"] == dt
        and metadata["EVEN_ABS"] == EVEN_ABS
        and metadata["coeffs"] == coeffs
        and metadata["ep0"] == ep0
        and metadata["ep1"] == ep1
        and metadata["x0"] == x0
    )


def _process_dataset(times, x_data):
    ## Perform Kramers Moyal
    edges = np.linspace(np.min(x_data), np.max(x_data), num_bins + 1)
    # edges = np.linspace(-0.005, 0.005, num_bins + 1)
    kmc, centers = km(x_data[..., None], bins=(edges,), powers=2)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers = centers[0]
    pdf /= np.nansum(pdf)
    moment_1 /= dt
    moment_2 /= dt
    return edges, centers, pdf, moment_1, moment_2


def get_model(
    folderpath: Path,
    num_datapoints,
    dt,
    EVEN_ABS,
    coeffs,
    ep0,
    ep1,
    x0,
    SAVE_MODEL=True,
    param="x",
):
    parent_model_path = folderpath.parent / "MODEL"

    if (parent_model_path / "metadata.json").exists():
        # check metadata matches
        with open(parent_model_path / "metadata.json") as metadata_file:
            metadata = json.load(metadata_file)
        if _check_metadata(
            metadata, num_datapoints, dt, EVEN_ABS, coeffs, ep0, ep1, x0
        ):
            if (parent_model_path / "km_model.npz").exists():
                print("Found premade model, skipping data generation", flush=True)
                with np.load(parent_model_path / "km_model.npz") as kramers_file:
                    edges = kramers_file["edges"]
                    centers = kramers_file["centers"]
                    pdf = kramers_file["pdf"]
                    moment_1 = kramers_file["moment_1"]
                    moment_2 = kramers_file["moment_2"]
                return edges, centers, pdf, moment_1, moment_2
            elif (parent_model_path / "timeseries.npz").exists():
                print("Found timeseries dataset, processing", flush=True)
                # Process data set, then save model, then return model
                with np.load(parent_model_path / "timeseries.npz") as datafile:
                    times = datafile["times"]
                    x_data = datafile["x_data"]

                edges, centers, pdf, moment_1, moment_2 = _process_dataset(
                    times, x_data
                )
                if SAVE_MODEL:
                    np.savez(
                        parent_model_path / "km_model.npz",
                        edges=edges,
                        centers=centers,
                        pdf=pdf,
                        moment_1=moment_1,
                        moment_2=moment_2,
                    )
                return edges, centers, pdf, moment_1, moment_2
    print("No premade model or dataset found, generating and processing", flush=True)
    # No or incorrect metadata, or no model or dataset
    # so generate the model
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

    if SAVE_MODEL:
        (parent_model_path).mkdir(exist_ok=True, parents=True)
        metadata = {
            "num_datapoints": num_datapoints,
            "dt": dt,
            "EVEN_ABS": EVEN_ABS,
            "coeffs": coeffs,
            "ep0": ep0,
            "ep1": ep1,
            "x0": x0,
        }
        with open(parent_model_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        np.savez(parent_model_path / "timeseries.npz", times=times, x_data=x_data)

    fig_data, ax_data = plt.subplots()
    ax_data.plot(times, x_data)
    ax_data.set_ylabel(f"${param}(t)$")
    ax_data.set_xlabel("$t$")

    fig_data.tight_layout()
    fig_data.savefig(folderpath / f"{MODEL_NAME}_data.png")

    plt.close(fig_data)

    fig_data, ax_data = plt.subplots()
    lmt = int(1000 / dt)
    ax_data.plot(times[:lmt], x_data[:lmt])
    ax_data.set_ylabel(f"${param}(t)$")
    ax_data.set_xlabel("$t$")

    fig_data.tight_layout()
    fig_data.savefig(folderpath / f"{MODEL_NAME}_data_zoom.png")
    plt.close(fig_data)

    ## Duplicate the data
    if EVEN_ABS:
        # When we make all even terms odd (by including the abs) then
        # We are assuming symmetry in x, so put this symmetry in the dataset
        x_data = np.append(x_data, -x_data)

    edges, centers, pdf, moment_1, moment_2 = _process_dataset(times, x_data)

    if SAVE_MODEL:
        np.savez(
            parent_model_path / "km_model.npz",
            edges=edges,
            centers=centers,
            pdf=pdf,
            moment_1=moment_1,
            moment_2=moment_2,
        )
    return edges, centers, pdf, moment_1, moment_2


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
    if lasso > 0:
        run_sindy_model_LASSO(
            MODEL_NAME,
            EVEN_ABS,
            ERROR_WEIGHTS,
            dt,
            num_datapoints,
            num_bins,
            kl_reg,
            lasso,
            ep0,
            ep1,
            coeffs,
            x0,
            param=r"\phi",
        )
    else:
        run_sindy_model(
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
        )
