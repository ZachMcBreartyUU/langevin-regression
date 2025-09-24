import argparse

import numpy as np
import matplotlib.pyplot as plt
from kramersmoyal import km

from kramers_moyal_log import km_log_bins_2
import data_loader as dl

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--log-spacing-fluidity", action="store_true")
ap.add_argument("--log-spacing-sigma", action="store_true")
ap.add_argument(
    "-c", "--cutoff", type=float, default=-25, help="f > f_cutoff = exp(cutoff)"
)
ap.add_argument("-Nf", "--num-bins-fluidity", type=int, default=101)
ap.add_argument("-Ns", "--num-bins-sigma", type=int, default=101)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--stop", type=int, default=-1)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--plot-intermediate", action="store_true")
args = ap.parse_args()
print(args)
lsf: bool = args.log_spacing_fluidity
lss: bool = args.log_spacing_sigma
cutoff: float = args.cutoff
num_bins_fluidity: int = args.num_bins_fluidity
num_bins_sigma: int = args.num_bins_sigma
start: int = args.start
stop: int = args.stop
step: int = args.step
plot_intermediate: bool = args.plot_intermediate

folder_path = dl.SCRATCH_PATH / args.folder
print(folder_path)

metadata, times, phi, sigma = dl.get_data(
    folder_path, start=start, stop=stop, step=step
)
print(np.shape(times))
print(np.shape(phi))
print(np.shape(sigma))

fluidity = phi**2
mask = (fluidity > np.exp(cutoff)).flatten()
print(np.shape(mask))
times = times[mask]
fluidity = fluidity[mask]
sigma = sigma[mask]

if plot_intermediate:
    fig_f_s, ax_f_s = plt.subplots()
    if lsf and lss:
        ax_f_s.plot(np.log(sigma), np.log(fluidity))
        ax_f_s.set_xlabel(r"log Stress, $\log \sigma$")
        ax_f_s.set_ylabel(r"log Fluidity, $\log f$")
        fig_f_s.tight_layout()
        fig_f_s.savefig(folder_path / "phasespace_log_f_log_s.png")
    elif lsf and not lss:
        ax_f_s.plot(sigma, np.log(fluidity))
        ax_f_s.set_xlabel(r"Stress, $\sigma$")
        ax_f_s.set_ylabel(r"log Fluidity, $\log f$")
        fig_f_s.tight_layout()
        fig_f_s.savefig(folder_path / "phasespace_log_f_s.png")
    elif not lsf and lss:
        ax_f_s.plot(np.log(sigma), fluidity)
        ax_f_s.set_xlabel(r"log Stress, $\log \sigma$")
        ax_f_s.set_ylabel(r"Fluidity, $f$")
        fig_f_s.tight_layout()
        fig_f_s.savefig(folder_path / "phasespace_f_log_s.png")
    else:
        ax_f_s.plot(sigma, fluidity)
        ax_f_s.set_xlabel(r"Stress, $\sigma$")
        ax_f_s.set_ylabel(r"Fluidity, $f$")
        fig_f_s.tight_layout()
        fig_f_s.savefig(folder_path / "phasespace_f_s.png")

    fig_t_fs, (ax_t_f, ax_t_s) = plt.subplots(2, sharex=True)
    ax_t_f: plt.Axes  # type: ignore
    ax_t_s: plt.Axes  # type: ignore

    if lsf:
        ax_t_f.plot(times, np.log(fluidity))
        ax_t_f.set_ylabel(r"log Fluidity, $\log f$")
    else:
        ax_t_f.plot(times, fluidity)
        ax_t_f.set_ylabel(r"Fluidity, $f$")
    ax_t_f.set_xlabel("$t$")

    if lss:
        ax_t_s.plot(times, np.log(sigma))
        ax_t_s.set_ylabel(r"log Stress, $\log \sigma$")
    else:
        ax_t_s.plot(times, sigma)
        ax_t_s.set_ylabel(r"Stress, $\sigma$")
    ax_t_s.set_xlabel("$t$")

    fig_t_fs.tight_layout()
    if lsf and lss:
        fig_t_fs.savefig(folder_path / "dataset_log_f_log_s.png")
    elif lsf and not lss:
        fig_t_fs.savefig(folder_path / "dataset_log_f_s.png")
    elif not lsf and lss:
        fig_t_fs.savefig(folder_path / "dataset_f_log_s.png")
    else:
        fig_t_fs.savefig(folder_path / "dataset_f_s.png")


def get_bins_linear(data, N=100):
    edges = np.linspace(np.min(data), np.max(data), N + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


def get_bins_log(log_data, N=100):
    # num edges = num bins + 1
    edges_log = np.linspace(np.min(log_data), np.max(log_data), N + 1)
    centers_log = (edges_log[1:] + edges_log[:-1]) / 2
    widths_log = edges_log[1:] - edges_log[:-1]

    edges = np.exp(edges_log)
    centers = np.exp(centers_log)
    # dlogx = dx / x
    widths = widths_log * centers

    return edges_log, centers_log, widths_log, edges, centers, widths


if lsf and lss:
    log_fluidity = np.log(fluidity)
    log_sigma = np.log(sigma)
    mask = np.isfinite(log_fluidity) * np.isfinite(log_sigma)
    (
        fluidity_edges_log,
        fluidity_centers_log,
        fluidity_widths_log,
        fluidity_edges,
        fluidity_centers,
        fluidity_widths,
    ) = get_bins_log(log_fluidity[mask], num_bins_fluidity)
    (
        sigma_edges_log,
        sigma_centers_log,
        sigma_widths_log,
        sigma_edges,
        sigma_centers,
        sigma_widths,
    ) = get_bins_log(log_sigma[mask], num_bins_sigma)
    bins = [fluidity_edges_log, sigma_edges_log]
    filename = "km_log_f_log_s.npz"
elif lsf and not lss:
    log_fluidity = np.log(fluidity)
    mask = np.isfinite(log_fluidity) * np.isfinite(sigma)
    (
        fluidity_edges_log,
        fluidity_centers_log,
        fluidity_widths_log,
        fluidity_edges,
        fluidity_centers,
        fluidity_widths,
    ) = get_bins_log(log_fluidity[mask], num_bins_fluidity)
    (
        sigma_edges,
        sigma_centers,
        sigma_widths,
    ) = get_bins_linear(sigma[mask], num_bins_sigma)
    bins = [fluidity_edges_log, sigma_edges]
    filename = "km_log_f_s.npz"
elif not lsf and lss:
    log_sigma = np.log(sigma)
    mask = np.isfinite(fluidity) * np.isfinite(log_sigma)
    (
        fluidity_edges,
        fluidity_centers,
        fluidity_widths,
    ) = get_bins_linear(fluidity[mask], num_bins_fluidity)
    (
        sigma_edges_log,
        sigma_centers_log,
        sigma_widths_log,
        sigma_edges,
        sigma_centers,
        sigma_widths,
    ) = get_bins_log(log_sigma[mask], num_bins_sigma)
    bins = [fluidity_edges, sigma_edges_log]
    filename = "km_f_log_s.npz"
else:
    mask = np.isfinite(fluidity) * np.isfinite(sigma)
    (
        fluidity_edges,
        fluidity_centers,
        fluidity_widths,
    ) = get_bins_linear(fluidity, num_bins_fluidity)
    (
        sigma_edges,
        sigma_centers,
        sigma_widths,
    ) = get_bins_linear(sigma, num_bins_sigma)
    bins = [fluidity_edges, sigma_edges]
    filename = "km_f_s.npz"

timeseries = np.stack([fluidity[mask], sigma[mask]], axis=1)
print(np.shape(timeseries))
kmc, _ = km_log_bins_2(timeseries, [lsf, lss], bins, powers=2, tol=1e-20)

# indices can be found from the powers array returned by km, or by hand
pdf = kmc[0]
moment1_f = kmc[2] / metadata["dt"]
moment1_s = kmc[1] / metadata["dt"]
moment2_f = kmc[6] / metadata["dt"]
moment2_s = kmc[4] / metadata["dt"]

f_grid, s_grid = np.meshgrid(fluidity_widths, sigma_widths)

print(f"{pdf=}")
print(f"{moment1_f=}")
print(f"{moment1_s=}")
print(f"{moment2_f=}")
print(f"{moment2_s=}")

pdf_sum = np.sum(pdf * f_grid * s_grid)
print(f"{pdf_sum=}")  # 0?
pdf /= pdf_sum

assert np.all(np.isfinite(pdf)), "KM pdf is not finite"
assert np.all(np.isfinite(moment1_f)), "KM moment1_f is not finite"
assert np.all(np.isfinite(moment1_s)), "KM moment1_s is not finite"
assert np.all(np.isfinite(moment2_f)), "KM moment2_f is not finite"
assert np.all(np.isfinite(moment2_s)), "KM moment2_s is not finite"

np.savez(
    folder_path / filename,
    pdf=pdf,
    moment1_f=moment1_f,
    moment1_s=moment1_s,
    moment2_f=moment2_f,
    moment2_s=moment2_s,
    fluidity_edges=fluidity_edges,
    fluidity_centers=fluidity_centers,
    fluidity_widths=fluidity_widths,
    sigma_edges=sigma_edges,
    sigma_centers=sigma_centers,
    sigma_widths=sigma_widths,
)
