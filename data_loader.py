from pathlib import Path
import json
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde

SCRATCH_PATH = Path("/scratch/seismology/zach/")


def get_data(folder_path, start=None, stop=None, step=None, memmap=True):
    with open(folder_path / "metadata.json") as f:
        metadata = json.load(f)

    phi_dim: int = metadata["phi_dim"]
    if phi_dim == 0:
        phi_extent = 1
    elif phi_dim == 1:
        y_dim: int = metadata["y_dim"]
        phi_extent = y_dim
    elif phi_dim == 2:
        x_dim: int = metadata["x_dim"]
        y_dim: int = metadata["y_dim"]
        phi_extent = y_dim * x_dim
    else:
        raise ValueError(
            f"Unsupported dimension for phi, got {phi_dim}, expected 0, 1, or 2"
        )
    sigma_dim: int = metadata["sigma_dim"]
    if sigma_dim == 0:
        sigma_extent = 1
    elif sigma_dim == 1:
        y_dim: int = metadata["y_dim"]
        sigma_extent = y_dim
    elif sigma_dim == 2:
        x_dim: int = metadata["x_dim"]
        y_dim: int = metadata["y_dim"]
        sigma_extent = y_dim * x_dim
    else:
        raise ValueError(
            f"Unsupported dimension for sigma, got {sigma_dim}, expected 0, 1, or 2"
        )

    runfile_name = folder_path / f"runfile_{phi_dim}D.npy"
    data = np.load(runfile_name, "r+" if memmap else None)[start:stop:step]

    time = data[:, 0]
    phi = data[:, 1 : 1 + phi_extent]
    sigma = data[:, 1 + phi_extent : 1 + phi_extent + sigma_extent]

    return metadata, time, phi, sigma


def calc_probability_density(
    data: npt.NDArray[np.floating[Any]], N=100, threshold=0.0
) -> Tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    # assuming the data is distributed over many orders of magnitude
    log_data = np.log(data[data > 0])
    min_log_data = np.min(log_data)
    log_threshold = np.log(threshold)
    if min_log_data < log_threshold:
        min_log_data = log_threshold
    max_log_data = np.max(log_data)

    kde_log_xs = np.linspace(min_log_data, max_log_data, N)
    kde_xs = np.exp(kde_log_xs)
    # dlog(x)/dx = 1/x
    d_log_xs = kde_log_xs[1] - kde_log_xs[0]
    d_xs = kde_xs * d_log_xs

    # fit kde to log data, then find value of kde at fixed points
    kde_log_ys = gaussian_kde(log_data)(kde_log_xs)
    # transform the kde from P[log x] to P[x]
    # P[x] = P[log x] * dlog(x) / dx
    kde_ys = kde_log_ys / kde_xs
    # Normalise to 1 in the given range
    kde_ys /= np.sum(kde_ys * d_xs)

    return kde_xs, d_xs, kde_ys
