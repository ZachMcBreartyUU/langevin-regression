from typing import Optional
from functools import partial
from pathlib import Path
import json
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

# data generation and final model integration
import symengine
from jitcsde import jitcsde, y

from kramersmoyal import km
from kramersmoyal.kernels import gaussian, epanechnikov


def _is_metadata_right(
    models_dir: Path,
    target_metadata: dict,
) -> tuple[bool, bool]:
    """timeseries check, KM check"""
    if not (models_dir / "metadata.json").exists():
        return False, False
    with open(models_dir / "metadata.json") as metadata_file:
        metadata: dict = json.load(metadata_file)

    timeseries_checks = [
        "num_datapoints",
        "dt",
        "EVEN_ABS",
        "coeffs",
        "ep0",
        "ep1",
        "x0",
    ]
    KM_checks = ["num_bins", "kernel"]
    for check in timeseries_checks:
        if check not in metadata:
            return False, False
        if metadata[check] != target_metadata[check]:
            return False, False
    for check in KM_checks:
        if check not in metadata:
            return True, False
        if metadata[check] != target_metadata[check]:
            return True, False
    return True, True


def _check_all_exist(models_dir: Path, NUM_DATASETS: int, filename_template: str):
    for i in range(NUM_DATASETS):
        if not _check_exists(models_dir, i, filename_template):
            return False
    return True


def _check_exists(models_dir: Path, i: int, filename_template: str):
    return (models_dir / (filename_template.format(i))).exists()


def load_metadata(models_dir):
    with open(models_dir / "metadata.json") as metadata_file:
        return json.load(metadata_file)


def write_metadata(models_dir, metadata):
    with open(models_dir / "metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file)


def _generate_dataserie(
    j, models_dir, metadata, metadata_is_right=False, validation=False
):
    filename = "timeseries_{}.npz" if not validation else "validation_{}.npz"
    if metadata_is_right and _check_exists(models_dir, j, filename):
        return 0, 0
    num_datapoints = metadata["num_datapoints"]
    dt = metadata["dt"]
    EVEN_ABS = metadata["dt"]
    coeffs = metadata["coeffs"]
    ep0 = metadata["ep0"]
    ep1 = metadata["ep1"]
    x0 = metadata["x0"]

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
    SDE = jitcsde(A, B, n=1, additive=False, verbose=False)
    SDE.set_initial_value([x0])
    x_data = np.fromiter(
        (SDE.integrate(t)[0] for t in times),  # type:ignore
        dtype=float,
        count=num_datapoints,
    )
    assert x_data.shape == (num_datapoints,)

    self_min = np.min(x_data)
    self_max = np.max(x_data)

    print(f"Saving timeseries {j+1} as {filename.format(j)}", flush=True)
    np.savez(models_dir / filename.format(j), times=times, x_data=x_data)

    return self_min, self_max


def generate_dataseries(
    models_dir, NUM_DATASETS, NUM_VALIDATION, metadata_is_right=False, NUM_CPUS=1
):
    metadata = load_metadata(models_dir)
    calls = []
    for j in range(NUM_DATASETS):
        calls.append((j, models_dir, metadata, metadata_is_right, False))
    for j in range(NUM_VALIDATION):
        calls.append((j, models_dir, metadata, metadata_is_right, True))

    with Pool(min(len(calls), NUM_CPUS)) as p:
        mins_maxs = p.starmap(_generate_dataserie, calls)

    mins_maxs = np.array(mins_maxs)[
        :NUM_DATASETS
    ]  # only include models, not validation
    metadata["min_x"] = min(metadata["min_x"], np.min(mins_maxs[:, 0]))
    metadata["max_x"] = max(metadata["max_x"], np.max(mins_maxs[:, 1]))
    write_metadata(models_dir, metadata)


def load_timeseries(timeseries_path):
    with np.load(timeseries_path) as timeseries_file:
        times = timeseries_file["times"]
        x_data = timeseries_file["x_data"]
    return times, x_data


def load_and_stack_timeseries(models_dir, NUM_DATASETS, validation=False):
    times = []
    xs = []
    for i in range(NUM_DATASETS):
        if validation:
            filename = f"validation_{i}.npz"
        else:
            filename = f"timeseries_{i}.npz"
        time, x = load_timeseries(models_dir / filename)
        times.append(time)
        xs.append(x)
    time_stack = np.stack(times, axis=0)
    x_stack = np.stack(xs, axis=0)
    return time_stack, x_stack


def set_metadata_min_max(models_dir, NUM_DATASETS):
    metadata = load_metadata(models_dir)
    new_min = metadata["min_x"]
    new_max = metadata["max_x"]
    for i in range(NUM_DATASETS):
        _, x_data = load_timeseries(models_dir / f"timeseries_{i}.npz")
        self_min = np.min(x_data)
        self_max = np.max(x_data)
        new_min = min(new_min, self_min)
        new_max = max(new_max, self_max)
    metadata["min_x"] = new_min
    metadata["max_x"] = new_max
    write_metadata(models_dir, metadata)
    return new_min, new_max


def _generate_KM(i, edges, models_dir, metadata, metadata_is_right, validation=False):
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
    if metadata["EVEN_ABS"]:
        x_data = np.append(x_data, -x_data)

    if metadata["kernel"] == "gaussian":
        kernel = gaussian
    else:
        kernel = epanechnikov

    kmc, centers = km(x_data[..., None], bins=(edges,), powers=2, kernel=kernel)  # type: ignore
    pdf, moment_1, moment_2 = kmc
    centers = centers[0]
    pdf /= np.nansum(pdf * (edges[1] - edges[0]))
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


def generate_KM(
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
    if metadata["min_x"] == metadata["max_x"]:
        metadata["min_x"], metadata["max_x"] = set_metadata_min_max(
            models_dir, NUM_DATASETS
        )
    if SUGGESTED_MIN_MAX is None:
        edges = np.linspace(
            metadata["min_x"], metadata["max_x"], metadata["num_bins"] + 1
        )
    else:
        edges = np.linspace(
            SUGGESTED_MIN_MAX[0], SUGGESTED_MIN_MAX[1], metadata["num_bins"] + 1
        )
    calls = []
    for i in range(NUM_DATASETS):
        calls.append((i, edges, models_dir, metadata, metadata_is_right, False))
    for i in range(NUM_VALIDATION):
        calls.append((i, edges, models_dir, metadata, metadata_is_right, True))
    with Pool(min(len(calls), NUM_CPUS)) as p:
        p.starmap(_generate_KM, calls)


def load_KM(KM_path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(KM_path) as KM_file:
        centers = KM_file["centers"]
        pdf = KM_file["pdf"]
        moment_1 = KM_file["moment_1"]
        moment_2 = KM_file["moment_2"]
    return centers, pdf, moment_1, moment_2


def load_and_stack_KM(models_dir, NUM_DATASETS, validation=False):
    centers = None
    pdfs = []
    moment_1_s = []
    moment_2_s = []
    for i in range(NUM_DATASETS):
        if validation:
            filename = f"validation_KM_{i}.npz"
        else:
            filename = f"KM_{i}.npz"
        centers, pdf, moment_1, moment_2 = load_KM(models_dir / filename)
        pdfs.append(pdf)
        moment_1_s.append(moment_1)
        moment_2_s.append(moment_2)
    assert centers is not None
    pdf_stack = np.stack(pdfs, axis=0)
    moment_1_stack = np.stack(moment_1_s, axis=0)
    moment_2_stack = np.stack(moment_2_s, axis=0)
    return centers, pdf_stack, moment_1_stack, moment_2_stack


def get_timeseries_and_KM(
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
    write_metadata(models_dir, target_metadata)
    generate_dataseries(
        models_dir,
        NUM_DATASETS,
        NUM_VALIDATION_DATASETS,
        metadata_is_right=timeseries_meta_correct,
        NUM_CPUS=NUM_CPUS,
    )
    generate_KM(
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


def generate_parameter_space(models_dirs, target_metadatas, NUM_DATASETS=1):
    for models_dir, target_metadata in zip(models_dirs, target_metadatas):
        get_timeseries_and_KM(models_dir, target_metadata, NUM_DATASETS)


def generate_parameter_space_parallel(
    models_dirs, target_metadatas, NUM_DATASETS=1, NUM_CPUS=1
):
    with Pool(NUM_CPUS) as pool:
        pool.starmap(
            partial(get_timeseries_and_KM, NUM_DATASETS=NUM_DATASETS),
            zip(models_dirs, target_metadatas),
        )


def get_parameter_space(models_dirs, NUM_DATASETS=1):
    """return a list of (stacked) models from specified directories"""
    stacked_models_list = []
    for models_dir in models_dirs:
        metadata = load_metadata(models_dir)
        stacked_models = get_timeseries_and_KM(models_dir, metadata, NUM_DATASETS)
        stacked_models_list.append(stacked_models)
    return stacked_models_list


def delete_timeseries(models_dir: Path, prefix="timeseries"):
    files = os.listdir(models_dir)
    for file in files:
        if file.startswith(prefix):
            os.remove(models_dir / file)
            print(f"Removed file: {models_dir / file}")


def plot_models(models_dir, NUM_DATASETS):
    centers, pdfs, moment_1s, moment_2s = load_and_stack_KM(models_dir, NUM_DATASETS)

    fig, axes = plt.subplots(3, figsize=(12, 24))
    axes: list[plt.Axes]  # type: ignore
    for pdf, moment_1, moment_2 in zip(pdfs, moment_1s, moment_2s):
        axes[0].scatter(centers, pdf)
        axes[1].scatter(centers, moment_1)
        axes[2].scatter(centers, moment_2)

    axes[0].set_xlabel("$x$")
    axes[1].set_xlabel("$x$")
    axes[2].set_xlabel("$x$")
    axes[0].set_ylabel("PDF, $P(x)$")
    axes[1].set_ylabel("First Moment, $m^{(1)}(x)$")
    axes[2].set_ylabel("Second Moment, $m^{(2)}(x)$")

    fig.savefig(models_dir / f"model_ALL_pdf_moment.png")
    plt.close(fig)
