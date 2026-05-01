from typing import Optional
from functools import partial
from pathlib import Path
import json
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from kramers_moyal_log import km_log_bins
from kramersmoyal.kernels import gaussian, epanechnikov

from make_and_load_models import *
from make_and_load_models import _check_exists, _is_metadata_right


def set_metadata_min_max_log(models_dir, NUM_DATASETS, EVEN_ABS=False):
    metadata = load_metadata(models_dir)
    _, x_data = load_timeseries(models_dir / f"timeseries_{0}.npz")
    x_data = x_data**2
    x_data = x_data[x_data > 0]
    new_min = np.nanmin(x_data)
    new_max = np.nanmax(x_data)
    for i in range(1, NUM_DATASETS):
        _, x_data = load_timeseries(models_dir / f"timeseries_{i}.npz")
        x_data = np.abs(x_data)
        x_data = x_data[x_data > 0]
        self_min = np.nanmin(x_data)
        self_max = np.nanmax(x_data)
        new_min = max(new_min, self_min)
        new_max = min(new_max, self_max)
    metadata["min_x"] = float(new_min)
    metadata["max_x"] = float(new_max)
    write_metadata(models_dir, metadata)
    return new_min, new_max


def _generate_KM_log(
    i, edges, models_dir, metadata, metadata_is_right, validation=False
):
    if validation:
        KM_file_template = "validation_KM_{}.npz"
        KM_file = f"validation_KM_{i}.npz"
        timeseries_file = f"validation_{i}.npz"
    else:
        KM_file_template = "KM_{}.npz"
        KM_file = f"KM_{i}.npz"
        timeseries_file = f"timeseries_{i}.npz"

    if metadata_is_right and _check_exists(models_dir, i, KM_file_template):
        return
    _, x_data = load_timeseries(models_dir / timeseries_file)
    x_data = x_data**2
    x_data = x_data[x_data != 0]

    if metadata["kernel"] == "gaussian":
        kernel = gaussian
    else:
        kernel = epanechnikov
    if metadata["bandwidth"] == "default":
        bw = None
    else:
        bw = metadata["bandwidth"]

    kmc, centers = km_log_bins(x_data[..., None], bins=(edges,), powers=2, kernel=kernel, bw=bw)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers = centers[0]

    pdf /= np.nansum(pdf * (edges[1:] - edges[:-1]))
    moment_1 /= metadata["dt"]
    moment_2 /= metadata["dt"]
    print(f"Saving KM {i+1}", flush=True)
    np.savez(
        models_dir / KM_file,
        centers=centers,
        pdf=pdf,
        moment_1=moment_1,
        moment_2=moment_2,
    )


def generate_KM_log(
    models_dir,
    NUM_DATASETS,
    NUM_VALIDATION,
    NUM_CPUS=1,
    metadata_is_right=False,
    SUGGESTED_MIN_MAX: Optional[tuple[float, float]] = None,
):
    # look at the min and max in the metadata
    # file for hint to the bin edges
    metadata = load_metadata(models_dir)
    metadata["min_x"], metadata["max_x"] = set_metadata_min_max_log(
        models_dir, NUM_DATASETS, metadata["EVEN_ABS"]
    )
    if SUGGESTED_MIN_MAX is None:
        edges = np.linspace(
            np.log(metadata["min_x"]),
            np.log(metadata["max_x"]),
            metadata["num_bins"] + 1,
        )
    else:
        edges = np.linspace(
            np.log(SUGGESTED_MIN_MAX[0]),
            np.log(SUGGESTED_MIN_MAX[1]),
            metadata["num_bins"] + 1,
        )
        metadata_is_right = False
    calls = []
    for i in range(NUM_DATASETS):
        calls.append((i, edges, models_dir, metadata, metadata_is_right, False))
    for i in range(NUM_VALIDATION):
        calls.append((i, edges, models_dir, metadata, metadata_is_right, True))
    with Pool(min(len(calls), NUM_CPUS)) as p:
        p.starmap(_generate_KM_log, calls)


def get_timeseries_and_KM_log(
    models_dir: Path,
    target_metadata: dict,
    NUM_DATASETS=10,
    NUM_VALIDATION_DATASETS=1,
    NUM_CPUS=1,
    SUGGESTED_MIN_MAX: Optional[tuple[float, float]] = None,
):
    # Check if the models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)
    # Check if there is metadata and that it is correct
    timeseries_meta_correct, KM_meta_correct = _is_metadata_right(
        models_dir, target_metadata
    )
    if not timeseries_meta_correct or not KM_meta_correct:
        target_metadata["min_x"] = target_metadata["x0"]
        target_metadata["max_x"] = target_metadata["x0"]
        write_metadata(models_dir, target_metadata)
    generate_dataseries(
        models_dir,
        NUM_DATASETS,
        NUM_VALIDATION_DATASETS,
        metadata_is_right=timeseries_meta_correct,
        NUM_CPUS=NUM_CPUS,
    )
    generate_KM_log(
        models_dir,
        NUM_DATASETS,
        NUM_VALIDATION_DATASETS,
        NUM_CPUS=NUM_CPUS,
        metadata_is_right=KM_meta_correct,
        SUGGESTED_MIN_MAX=SUGGESTED_MIN_MAX,
    )
    return (
        load_and_stack_timeseries(models_dir, NUM_DATASETS),
        load_and_stack_KM(models_dir, NUM_DATASETS),
        load_and_stack_timeseries(models_dir, NUM_VALIDATION_DATASETS, validation=True),
        load_and_stack_KM(models_dir, NUM_VALIDATION_DATASETS, validation=True),
    )
