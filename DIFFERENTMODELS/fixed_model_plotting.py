from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist import Axes  # typing

SCRATCH_PATH = Path(f"/scratch/seismology/zach/softglass/fixed_model")


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
):
    plot_cost_and_diff(target_var, range_var, costs, diffs, logx, coeff_labels)
    plot_pdfs(target_var, range_var, centers, pdfs, found_pdfs, logx)


def plot_cost_and_diff(
    target_var,
    range_var,
    costs,
    diffs,
    logx=True,
    coeff_labels=["x", "x^3", "x^3|x|", "ep0", "ep1"],
):
    fig_cost, ax_cost = plt.subplots()
    fig_diff, axes_diff = plt.subplots(len(diffs), figsize=(6, len(diffs) * 3))
    axes_diff: list[Axes]
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

    fig_cost.tight_layout()
    fig_cost.savefig(SCRATCH_PATH / f"cost_vs_{target_var}.png")
    plt.close(fig_cost)

    for ax_diff, coeff_lab in zip(axes_diff, coeff_labels):
        ax_diff.set_xlabel(f"{target_var}")
        ax_diff.set_title(
            rf"Absolute difference, $\left|\Delta\xi_{'{'}{coeff_lab}{'}'}\right / \xi_{'{true}'}|$"
        )

    fig_diff.tight_layout()
    fig_diff.savefig(SCRATCH_PATH / f"diff_vs_{target_var}.png")
    plt.close(fig_diff)


def plot_pdfs(target_var, range_var, centers, pdfs, found_pdfs, logx=True):

    fig_pdfs, ax_pdfs = plt.subplots()
    colours = getattr(plt.cm, "jet")(np.linspace(0.1, 0.9, len(pdfs)))

    for center, pdf, colour in zip(centers, pdfs, colours):
        ax_pdfs.semilogy(center, pdf, color=colour)
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

    fig_pdfs.tight_layout()
    fig_pdfs.savefig(SCRATCH_PATH / f"pdf_vs_{target_var}.png")
    plt.close(fig_pdfs)

    fig_found_pdfs, ax_found_pdfs = plt.subplots()
    for center, found_pdf, colour in zip(centers, found_pdfs, colours):
        ax_found_pdfs.semilogy(center, found_pdf, color=colour)
    fig_found_pdfs.colorbar(q, ax=ax_found_pdfs, ticks=ticks, label=label)
    ax_found_pdfs.set_xlabel("x")
    ax_found_pdfs.set_ylabel("log pdf(x)")

    fig_found_pdfs.tight_layout()
    fig_found_pdfs.savefig(SCRATCH_PATH / f"found_pdf_vs_{target_var}.png")
    plt.close(fig_found_pdfs)

    fig_diff_pdfs, ax_diff_pdfs = plt.subplots()

    for center, pdf, found_pdf, colour in zip(centers, pdfs, found_pdfs, colours):
        ax_diff_pdfs.plot(center, np.abs(pdf - found_pdf), color=colour)
    fig_diff_pdfs.colorbar(q, ax=ax_found_pdfs, ticks=ticks, label=label)
    ax_diff_pdfs.set_xlabel("x")
    ax_diff_pdfs.set_ylabel(r"$\Delta$ pdf(x)")

    fig_diff_pdfs.tight_layout()
    fig_diff_pdfs.savefig(SCRATCH_PATH / f"diff_in_pdf_vs_{target_var}.png")
    plt.close(fig_diff_pdfs)


def do_load(target_var):
    with np.load(SCRATCH_PATH / f"test_{target_var}.npz") as f:
        range_var: np.ndarray = f["range_var"]
        costs: np.ndarray = f["costs"]
        diffs: np.ndarray = f["diffs"]
        centers: np.ndarray = f["centers"]
        pdfs: np.ndarray = f["pdfs"]
        found_pdfs: np.ndarray = f["found_pdfs"]
    return (
        range_var,
        costs,
        diffs,
        centers,
        pdfs,
        found_pdfs,
    )


if __name__ == "__main__":
    # range_var = np.logspace(6, 9, 20).astype(int)
    # do_plot("num_datapoints", range_var, *do_load("num_datapoints"))
    # range_var = np.logspace(-10, 0, 20)
    # do_plot("kl_reg", range_var, *do_load("kl_reg"))
    ld = do_load("dt zoom")
    ld[1][18] = np.inf
    # ld2 = do_load("dt")
    # ld2[1][6] = np.inf
    # ld3 = (
    #     np.append(ld[0], ld2[0]),
    #     np.append(ld[1], ld2[1]),
    #     np.append(ld[2], ld2[2], axis=1),
    #     np.append(ld[3], ld2[3], axis=0),
    #     np.append(ld[4], ld2[4], axis=0),
    #     np.append(ld[5], ld2[5], axis=0),
    # )

    # print(np.log10(ld[0][5:8]))
    do_plot("dt_zoom", *ld)
    # range_var = np.logspace(1, 3, 20).astype(int)
    # do_plot("num_bins", range_var, *do_load("num_bins"))
    # range_var = np.logspace(-7, 0, 20)
    # do_plot("ep0", range_var, *do_load("ep0"))
    # range_var = np.logspace(-7, 0, 20)
    # do_plot("ep1", range_var, *do_load("ep1"))
    # range_var = np.linspace(-3, 3, 20)
    # do_plot("ax", range_var, *do_load("ax"), logx=False)
    # range_var = np.linspace(-3, 3, 20)
    # do_plot("bx3", range_var, *do_load("bx3"), logx=False)
    # range_var = np.linspace(-1, -0.01, 20, endpoint=False)
    # do_plot("cx4", range_var, *do_load("cx4"), logx=False)
