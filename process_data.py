import argparse

import numpy as np
import matplotlib.pyplot as plt
from kramersmoyal import km

from kramers_moyal_log import km_log_bins
import data_loader as dl

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--log-spacing", action="store_true")
ap.add_argument("-c", "--cutoff", type=float, default=250.0)
ap.add_argument("-N", "--num-bins", type=int, default=101)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=-1)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--plot-intermediate", action="store_true")
args = ap.parse_args()
print(args)

folder_path = dl.SCRATCH_PATH / args.folder
print(folder_path)

metadata, times, phi, sigma = dl.get_data(
    folder_path, start=args.start, stop=args.stop, step=args.step
)

if args.plot_intermediate:
    fig, (ax, ax2) = plt.subplots(2)
    # type checking for ax, though matplotlib technically
    # doesn't export Axes, so include the type ignore
    ax: plt.Axes  # type: ignore
    ax2: plt.Axes  # type: ignore
    if args.log_spacing:
        ax.plot(times, np.log(phi**2))
        ax.set_ylabel(r"log Fluidity, $\log f$")
    else:
        ax.plot(times, phi**2)
        ax.set_ylabel(r"Fluidity, $f$")

    ax.set_xlabel("$t$")

    ax2.plot(times, sigma)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"Stress, $\sigma$")

    fig.tight_layout()
    if args.log_spacing:
        fig.savefig(folder_path / "system_graph_log_f.png")
    else:
        fig.savefig(folder_path / "system_graph_f.png")

sigma: np.ndarray
min_sigma = sigma.min()
max_sigma = sigma.max()
avg_sigma = sigma.mean()
std_sigma = sigma.std()
sigma_data = np.array([min_sigma, max_sigma, avg_sigma, std_sigma])

if args.log_spacing:
    fluidity = phi**2
    data = fluidity[fluidity > np.exp(args.cutoff)]
    # num edges = num bins + 1
    log_data = np.log(data)
    edges_log = np.linspace(np.min(log_data), np.max(log_data), args.num_bins + 1)
    centers_log = (edges_log[1:] + edges_log[:-1]) / 2
    widths_log = edges_log[1:] - edges_log[:-1]

    edges = np.exp(edges_log)
    centers = np.exp(centers_log)
    # dlogx = dx / x
    widths = widths_log * centers
    kmc, _ = km_log_bins(timeseries=data, bins=(edges_log,), powers=2)
    pdf, moment1, moment2 = kmc
    # sum(P[x]dx) = sum(P[logx]dlogx) = 1
    pdf /= np.sum(pdf * widths)
    moment1 /= metadata["dt"]
    moment2 /= metadata["dt"]

    np.savez(
        folder_path / "km_log_spacing.npz",
        pdf=pdf,
        moment1=moment1,
        moment2=moment2,
        edges=edges,
        centers=centers,
        widths=widths,
        sigma_data=sigma_data,
    )
else:
    fluidity = phi**2
    data = fluidity[fluidity > args.cutoff]
    # num edges = num bins + 1
    edges = np.linspace(np.min(data), np.max(data), args.num_bins + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    widths = edges[1:] - edges[:-1]

    kmc, _ = km(timeseries=data, bins=(edges,), powers=2)  # type: ignore
    pdf, moment1, moment2 = kmc
    # sum(P[x]dx) = sum(P[logx]dlogx) = 1
    pdf /= np.sum(pdf * widths)
    moment1 /= metadata["dt"]
    moment2 /= metadata["dt"]

    np.savez(
        folder_path / "km_linear_spacing.npz",
        pdf=pdf,
        moment1=moment1,
        moment2=moment2,
        edges=edges,
        centers=centers,
        widths=widths,
        sigma_data=sigma_data,
    )
