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


def _check_all_exist(models_dir: Path, NUM_MODELS: int, filename_template: str):
    for i in range(NUM_MODELS):
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


def generate_dataseries(models_dir, NUM_MODELS, wrong_metadata=False):
    metadata = load_metadata(models_dir)
    num_datapoints = metadata["num_datapoints"]
    dt = metadata["dt"]
    EVEN_ABS = metadata["dt"]
    coeffs = metadata["coeffs"]
    ep0 = metadata["ep0"]
    ep1 = metadata["ep1"]
    x0 = metadata["x0"]

    for j in range(NUM_MODELS):
        if not wrong_metadata and _check_exists(models_dir, j, "timeseries_{}.npz"):
            continue
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
        metadata["min_x"] = min(metadata["min_x"], self_min)
        metadata["max_x"] = max(metadata["max_x"], self_max)
        write_metadata(models_dir, metadata)
        print(f"Saving timeseries {j+1} / {NUM_MODELS}", flush=True)
        np.savez(models_dir / f"timeseries_{j}.npz", times=times, x_data=x_data)


def load_timeseries(timeseries_path):
    with np.load(timeseries_path) as timeseries_file:
        times = timeseries_file["times"]
        x_data = timeseries_file["x_data"]
    return times, x_data


def set_metadata_min_max(models_dir, NUM_MODELS):
    metadata = load_metadata(models_dir)
    new_min = metadata["min_x"]
    new_max = metadata["max_x"]
    for i in range(NUM_MODELS):
        _, x_data = load_timeseries(models_dir / f"timeseries_{i}.npz")
        self_min = np.min(x_data)
        self_max = np.max(x_data)
        new_min = min(new_min, self_min)
        new_max = max(new_max, self_max)
    metadata["min_x"] = new_min
    metadata["max_x"] = new_max
    write_metadata(models_dir, metadata)
    return new_min, new_max


def generate_models(models_dir, NUM_MODELS, wrong_metadata=False):
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
        if not wrong_metadata and _check_exists(models_dir, i, "model_{}.npz"):
            continue
        _, x_data = load_timeseries(models_dir / f"timeseries_{i}.npz")
        if metadata["EVEN_ABS"]:
            x_data = np.append(x_data, -x_data)

        kmc, centers = km(x_data[..., None], bins=(edges,), powers=2)  # type: ignore
        pdf, moment_1, moment_2 = kmc
        centers = centers[0]
        pdf /= np.nansum(pdf)
        moment_1 /= metadata["dt"]
        moment_2 /= metadata["dt"]
        print(f"Saving model {i+1} / {NUM_MODELS}", flush=True)
        np.savez(
            models_dir / f"model_{i}.npz",
            centers=centers,
            pdf=pdf,
            moment_1=moment_1,
            moment_2=moment_2,
        )


def load_model(model_path):
    with np.load(model_path) as model_file:
        centers = model_file["centers"]
        pdf = model_file["pdf"]
        moment_1 = model_file["moment_1"]
        moment_2 = model_file["moment_2"]
    return centers, pdf, moment_1, moment_2


def load_and_combine_models(models_dir, NUM_MODELS):
    # load all models
    # stack pdfs and moments
    centers = None
    pdfs = []
    moment_1_s = []
    moment_2_s = []
    for i in range(NUM_MODELS):
        centers, pdf, moment_1, moment_2 = load_model(models_dir / f"model_{i}.npz")
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
    models_dir.mkdir(parents=True, exist_ok=True)
    #   Check if there is metadata and that it is correct
    if not _check_metadata(models_dir, target_metadata):
        print(
            f"No or incorrect metadata found in {models_dir}: generating data, processing, and stacking",
            flush=True,
        )
        target_metadata["min_x"] = target_metadata["x0"]
        target_metadata["max_x"] = target_metadata["x0"]
        write_metadata(models_dir, target_metadata)
        generate_dataseries(models_dir, NUM_MODELS, True)
        generate_models(models_dir, NUM_MODELS, True)
    elif _check_all_exist(models_dir, NUM_MODELS, "model_{}.npz"):
        print("Found models: loading and stacking", flush=True)
        pass
    elif _check_all_exist(models_dir, NUM_MODELS, "timeseries_{}.npz"):
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
    return load_and_combine_models(models_dir, NUM_MODELS)


def generate_parameter_space(models_dirs, target_metadatas, NUM_MODELS=1):
    for models_dir, target_metadata in zip(models_dirs, target_metadatas):
        get_models(models_dir, target_metadata, NUM_MODELS)


def generate_parameter_space_parallel(
    models_dirs, target_metadatas, NUM_MODELS=1, NUM_CPUS=1
):
    with Pool(NUM_CPUS) as pool:
        pool.starmap(
            partial(get_models, NUM_MODELS=NUM_MODELS),
            zip(models_dirs, target_metadatas),
        )


def get_parameter_space(models_dirs, NUM_MODELS=1):
    """return a list of (stacked) models from specified directories"""
    stacked_models_list = []
    for models_dir in models_dirs:
        metadata = load_metadata(models_dir)
        stacked_models = get_models(models_dir, metadata, NUM_MODELS)
        stacked_models_list.append(stacked_models)
    return stacked_models_list


def delete_timeseries(models_dir: Path, prefix="timeseries"):
    files = os.listdir(models_dir)
    for file in files:
        if file.startswith(prefix):
            os.remove(models_dir / file)
            print(f"Removed file: {models_dir / file}")


def plot_models(models_dir, NUM_MODELS):
    centers, pdfs, moment_1s, moment_2s = load_and_combine_models(
        models_dir, NUM_MODELS
    )

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
