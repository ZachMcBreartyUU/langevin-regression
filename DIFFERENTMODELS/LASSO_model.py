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
)

from utils_lasso import (
    cost as cost_lasso,
    cost_KL as cost_KL_lasso,
    _square_diff,
    _lasso,
)


SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/")


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
    ADD_TERMS = 1  # TODO: MAKE INTO FUNCTION ARGUMENT AND CL ARGUMENT
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

    ## Perform LASSO regression
    Xi0 = np.empty((num_A_expr + num_C_expr))
    Xi0[:num_A_expr] = lstsq(lib_A.T, moment_1)[0]
    # Xi0[num_A_expr - 1] = -np.abs(Xi0[num_A_expr - 1])
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

    fig_weights, (ax_WA, ax_WC) = plt.subplots(2)
    ax_WA: Axes
    ax_WC: Axes
    ax_WA.plot(centers_x, first_weight)
    ax_WA.set_xlabel(f"${param}$")
    ax_WA.set_ylabel(f"First weight, A(${param}$)")
    ax_WC.plot(centers_x, second_weight)
    ax_WC.set_xlabel(f"${param}$")
    ax_WC.set_ylabel(f"Second weight, C(${param}$)")
    fig_weights.savefig(folderpath / f"{MODEL_NAME}_weights.png")
    plt.close(fig_weights)

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
        "lasso": lasso,
    }
    if kl_reg > 0:
        opt_func = partial(optimise_function, cost_KL_lasso)
    else:
        opt_func = partial(optimise_function, cost_lasso)

    Xi, cost_ = opt_func(params)
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
        (SDE.integrate(t)[0] for t in times),  # type:ignore
        dtype=float,
        count=num_datapoints,
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
