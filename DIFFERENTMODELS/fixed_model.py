from pathlib import Path
from typing import Optional
from time import time

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
    SteadyFP,
)

SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/fixed_model")
# SCRATCH_PATH.mkdir(parents=True, exist_ok=True)


def run_fixed_model(
    EVEN_ABS=True,
    PDF_WEIGHTS=False,
    dt=0.001,
    num_datapoints=10_000_000,
    num_bins=100,
    kl_reg=0.001,
    ep0=0.2,
    ep1=0.1,
    coeffs=[0.0, -1.0, 0.0, 1.0, -1 / 3],
    x0=0.0,
    seed: Optional[int] = None,
):
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
    SDE = jitcsde(A, B, n=1, additive=False, verbose=False)
    SDE.set_initial_value([x0])
    SDE.set_seed(seed)
    x_data = np.fromiter(
        (SDE.integrate(t)[0] for t in times), dtype=float, count=num_datapoints
    )
    assert x_data.shape == (num_datapoints,)

    ## Perform Kramers Moyal
    edges = np.linspace(np.min(x_data), np.max(x_data), num_bins + 1)
    dedges = edges[1:] - edges[:-1]
    # edges = np.linspace(-0.005, 0.005, num_bins + 1)
    kmc, centers = km(x_data[..., None], bins=(edges,), powers=2)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers_x = centers[0]
    pdf /= np.nansum(pdf * dedges)
    moment_1 /= dt
    moment_2 /= dt
    del x_data

    ## Make SINDy libraries
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
        A_lib_expr = np.array([x_sym**i for i in range(len(coeffs) + 3)])
    # print(f"BEFORE FILTER {A_lib_expr=}")
    # ONLY REGRESS ON THE CORRECT TERMS
    A_lib_expr = A_lib_expr[np.nonzero(coeffs)]
    # print(f"AFTER FILTER {A_lib_expr=}")
    num_A_expr = len(A_lib_expr)

    lib_A = np.empty((num_A_expr, num_bins))
    for k in range(num_A_expr):
        lamb_expr = sympy.lambdify(x_sym, A_lib_expr[k])
        lib_A[k] = lamb_expr(centers_x)

    C_lib_expr = np.array([x_sym**0, x_sym**2])
    # print(f"{C_lib_expr=}")
    num_C_expr = len(C_lib_expr)

    lib_C = np.empty((num_C_expr, num_bins))
    for k in range(num_C_expr):
        lamb_expr = sympy.lambdify(x_sym, C_lib_expr[k])
        lib_C[k] = lamb_expr(centers_x)

    n_terms = num_A_expr + num_C_expr

    # Regress on fixed terms
    Xi0 = np.empty((num_A_expr + num_C_expr))
    Xi0[:num_A_expr] = lstsq(lib_A.T, moment_1)[0]
    Xi0[num_A_expr:] = lstsq(lib_C.T, moment_2)[0]

    if Xi0[-1] > 0:
        Xi0[-1] *= -1
    # print(f"{Xi0=}")
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
    xi, cost_val = optimise_function(cost_KL, params)

    # TODO: Could perform some rounding on the found params, e.g. 0.012324 -> 0.012
    A_sym = sindy_model(xi[:num_A_expr], A_lib_expr)
    A_sindy = sympy.lambdify(x_sym, A_sym)(centers_x)
    C_sym = sindy_model(xi[num_A_expr:], C_lib_expr)
    C_sindy = sympy.lambdify(x_sym, C_sym)(centers_x)

    if np.ndim(A_sindy) == 0:
        A_sindy = np.full_like(centers_x, A_sindy)
    if np.ndim(C_sindy) == 0:
        C_sindy = np.full_like(centers_x, C_sindy)

    # print(f"dx = ({A_sym}) dt + ({sympy.sqrt(2.0*C_sym)}) dβ")

    ## Directly compare True answer to Found answer
    true_model_xi = np.zeros(n_terms)
    true_model_xi[:num_A_expr] = np.array(coeffs)[np.nonzero(coeffs)]
    true_model_xi[num_A_expr + 0] = ep0 / 2  # B = sqrt(ep0 + ep1 x^2) ->
    true_model_xi[num_A_expr + 1] = ep1 / 2  # C = B^2 / 2 -> ep0 / 2 + ep1 / 2 x^2

    found_pdf = sfp.solve(A_sindy, C_sindy)

    differences = np.abs((xi - true_model_xi) / true_model_xi)
    return xi, cost_val, differences, centers_x, pdf, found_pdf


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seed = 987654321
start = time()


# TODO: implement repetition?
def do_test(
    target_var,
    range_var,
    logx=True,
    coeff_labels=["x", "x^3", "x^3|x|", "ep0", "ep1"],
    target_name=None,
):
    if target_name is None:
        target_name = target_var
    start = time()
    print(f"{target_name} targeting {target_var} started", flush=True)

    costs = []
    diffs = []
    centers = []
    pdfs = []
    found_pdfs = []
    for var in range_var:
        print(f"{target_var}={var}, started at {time() - start}", flush=True)
        xi, cost, diff, center, pdf, found_pdf = run_fixed_model(
            **{target_var: var}, seed=seed
        )
        costs.append(cost)
        diffs.append(diff)
        centers.append(center)
        pdfs.append(pdf)
        found_pdfs.append(found_pdf)
    diffs = np.array(diffs).T

    with open(SCRATCH_PATH / f"test_{target_name}.npz", "wb") as f:
        np.savez(
            f,
            range_var=np.asarray(range_var),
            costs=np.asarray(costs),
            diffs=np.asarray(diffs),
            centers=np.asarray(centers),
            pdfs=np.asarray(pdfs),
            found_pdfs=np.asarray(found_pdfs),
        )

    do_plot(
        target_name,
        range_var,
        costs,
        diffs,
        centers,
        pdfs,
        found_pdfs,
        logx,
        coeff_labels,
    )

    print(f"{target_var}, finished at {time() - start}", flush=True)


def do_test_coeffs(
    target_name,
    target_index,
    range_var,
    default_coeffs=[0.0, -1.0, 0.0, 1.0, -1 / 3],
    logx=True,
    coeff_labels=["x", "x^3", "x^3|x|", "ep0", "ep1"],
):

    start = time()
    print(f"{target_name} started", flush=True)
    costs = []
    diffs = []
    centers = []
    pdfs = []
    found_pdfs = []
    for var in range_var:
        print(f"{target_name}={var}, started at {time() - start}", flush=True)
        default_coeffs[target_index] = var
        xi, cost, diff, center, pdf, found_pdf = run_fixed_model(
            coeffs=default_coeffs, seed=seed
        )
        costs.append(cost)
        diffs.append(diff)
        centers.append(center)
        pdfs.append(pdf)
        found_pdfs.append(found_pdf)
    diffs = np.array(diffs).T

    with open(SCRATCH_PATH / f"test_{target_name}.npz", "wb") as f:
        np.savez(
            f,
            range_var=np.asarray(range_var),
            costs=np.asarray(costs),
            diffs=np.asarray(diffs),
            centers=np.asarray(centers),
            pdfs=np.asarray(pdfs),
            found_pdfs=np.asarray(found_pdfs),
        )
    do_plot(
        target_name,
        range_var,
        costs,
        diffs,
        centers,
        pdfs,
        found_pdfs,
        logx,
        coeff_labels,
    )

    print(f"{target_name}, finished at {time() - start}", flush=True)


def do_plot(
    target_var,
    range_var,
    costs,
    diffs,
    centers,
    pdfs,
    found_pdfs,
    logx=True,
    coeff_labels=["x", "x^3", "x^3|x|", "ep0", "ep1"],
    ONLYLABEL=5,
):
    fig_cost, ax_cost = plt.subplots()
    fig_diff, axes_diff = plt.subplots(len(diffs), figsize=(6, len(diffs) * 3))
    fig_pdfs, ax_pdfs = plt.subplots()
    fig_found_pdfs, ax_found_pdfs = plt.subplots()
    if logx:
        ax_cost.semilogx(range_var, costs)
        for diff, ax_diff in zip(diffs, axes_diff):
            ax_diff.semilogx(range_var, diff)
    else:
        ax_cost.plot(range_var, costs)
        for diff, ax_diff in zip(diffs, axes_diff):
            ax_diff.plot(range_var, diff)
    ax_cost.set_xlabel(f"{target_var}")
    ax_cost.set_ylabel("Cost / misfit, V")
    for ax_diff, coeff_lab in zip(axes_diff, coeff_labels):
        ax_diff.set_xlabel(f"{target_var}")
        ax_diff.set_title(
            rf"Absolute difference, $\left|\Delta\xi_{'{'}{coeff_lab}{'}'}\right / \xi_{'{true}'}|$"
        )
    colours = getattr(plt.cm, "jet")(np.linspace(0.1, 0.9, len(pdfs)))
    for center, pdf, colour in zip(centers, pdfs, colours):
        ax_pdfs.plot(center, pdf, color=colour)
    q = plt.cm.ScalarMappable(None, "jet")
    if logx:
        q.set_clim(np.log10(range_var[0]), np.log10(range_var[-1]))
        ticks = np.log10(range_var)
        label = f"log {target_var}"
    else:
        q.set_clim(range_var[0], range_var[-1])
        ticks = range_var
        label = f"{target_var}"

    fig_pdfs.colorbar(q, ax=ax_pdfs, ticks=ticks, label=label)

    ax_pdfs.set_xlabel("x")
    ax_pdfs.set_ylabel("pdf(x)")

    for center, found_pdf, colour in zip(centers, found_pdfs, colours):
        ax_found_pdfs.plot(center, found_pdf, color=colour)
    fig_found_pdfs.colorbar(q, ax=ax_found_pdfs, ticks=ticks, label=label)
    ax_found_pdfs.set_xlabel("x")
    ax_found_pdfs.set_ylabel("pdf(x)")

    fig_cost.tight_layout()
    fig_diff.tight_layout()
    fig_pdfs.tight_layout()
    fig_found_pdfs.tight_layout()
    fig_cost.savefig(SCRATCH_PATH / f"cost_vs_{target_var}.png")
    fig_diff.savefig(SCRATCH_PATH / f"diff_vs_{target_var}.png")
    fig_pdfs.savefig(SCRATCH_PATH / f"pdf_vs_{target_var}.png")
    fig_found_pdfs.savefig(SCRATCH_PATH / f"found_pdf_vs_{target_var}.png")
    plt.close(fig_cost)
    plt.close(fig_diff)
    plt.close(fig_pdfs)
    plt.close(fig_found_pdfs)


def do_load(target_var):
    with np.load(SCRATCH_PATH / f"test_{target_var}.npz") as f:
        # range_var = f["range_var"]
        costs = f["costs"]
        diffs = f["diffs"]
        centers = f["centers"]
        pdfs = f["pdfs"]
        found_pdfs = f["found_pdfs"]
    return (
        # range_var,
        costs,
        diffs,
        centers,
        pdfs,
        found_pdfs,
    )


# ### program splits
# if rank == (0 % size):
#     do_test("num_datapoints", np.logspace(6, 9, 20).astype(int))
# elif rank == (1 % size):
#     do_test("kl_reg", np.logspace(-10, 0, 20))
# elif rank == (2 % size):
#     do_test("dt", np.logspace(-5, -1.5, 10))
# elif rank == (3 % size):
#     do_test("num_bins", np.logspace(1, 4, 20).astype(int))
# elif rank == (4 % size):
#     do_test("ep0", np.logspace(-7, 0, 20))
# elif rank == (5 % size):
#     do_test("ep1", np.logspace(-7, 0, 20))
# elif rank == (6 % size):
#     do_test_coeffs("ax", 1, np.linspace(-3, 3, 20), logx=False)
# elif rank == (7 % size):
#     do_test_coeffs("bx3", 3, np.linspace(-3, 3, 20), logx=False)
# elif rank == (8 % size):
#     do_test_coeffs("cx4", 4, np.linspace(-1, -0.01, 20, endpoint=False), logx=False)
# else:
#     print("Over extended:", rank, size, flush=True)

# range_var = np.logspace(6, 9, 20).astype(int)
# do_plot("num_datapoints", range_var, *do_load("num_datapoints"))
range_var = np.logspace(-10, 0, 20)
do_plot("kl_reg", range_var, *do_load("kl_reg"))
range_var = np.logspace(-5, -1.5, 10)
do_plot("dt", range_var, *do_load("dt"))
# range_var = np.logspace(1, 4, 20).astype(int)
# do_plot("num_bins", range_var, *do_load("num_bins"))
range_var = np.logspace(-7, 0, 20)
do_plot("ep0", range_var, *do_load("ep0"))
range_var = np.logspace(-7, 0, 20)
do_plot("ep1", range_var, *do_load("ep1"))
range_var = np.linspace(-3, 3, 20)
do_plot("ax", range_var, *do_load("ax"), logx=False)
range_var = np.linspace(-3, 3, 20)
do_plot("bx3", range_var, *do_load("bx3"), logx=False)
range_var = np.linspace(-1, -0.01, 20, endpoint=False)
do_plot("cx4", range_var, *do_load("cx4"), logx=False)
