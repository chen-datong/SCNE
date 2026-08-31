"""Plot the two-panel XXZ Heisenberg QVC figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_heisenberg_data import data_filename
from heisenberg_model import exact_ground_fidelity, load_spectrum, spectrum_filename


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
matplotlib.rcParams["font.sans-serif"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "cm"


def real_array(values):
    parsed = []
    for value in values:
        if isinstance(value, str):
            parsed.append(complex(value.strip().strip("[]()")))
        else:
            parsed.append(value)
    return np.real(np.asarray(parsed, dtype=np.complex128)).astype(float)


def mean_and_error(values, mode):
    values = real_array(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if mode == "std":
        return mean, standard_deviation
    standard_error = standard_deviation / np.sqrt(values.size) if values.size > 1 else 0.0
    if mode == "sem":
        return mean, standard_error
    if mode == "ci95":
        return mean, 1.96 * standard_error
    raise ValueError(f"Unknown error mode: {mode}")


def load_estimates(data_dir, params, beta, delta):
    path = data_dir / data_filename(
        int(params["n"]),
        int(params["N_U"]),
        int(params["N_M"]),
        float(beta),
        float(delta),
    )
    frame = pd.read_csv(path)
    if len(frame) != int(params["trials"]):
        raise ValueError(
            f"Expected {params['trials']} trials in {path}; found {len(frame)}."
        )
    required = {"pf_rhoO2_fidelity", "pf_rhoO3_fidelity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame


def load_energies(spectrum_dir, params, delta):
    n = int(params["n"])
    j_coupling = float(params["J"])
    path = spectrum_dir / spectrum_filename(n, j_coupling, float(delta))
    return load_spectrum(
        path, expected_n=n, expected_j=j_coupling, expected_delta=delta
    )


def plot(params, data_dir, spectrum_dir, output):
    beta_points = [float(value) for value in params["left"]["beta"]]
    delta_left = float(params["left"]["Delta"])
    delta_points = [float(value) for value in params["right"]["Delta"]]
    beta_right = float(params["right"]["beta"])
    error_mode = params["error_band"]

    beta_mean = {2: [], 3: []}
    beta_error = {2: [], 3: []}
    for beta in beta_points:
        frame = load_estimates(data_dir, params, beta, delta_left)
        for order in (2, 3):
            mean, error = mean_and_error(
                frame[f"pf_rhoO{order}_fidelity"].to_numpy(), error_mode
            )
            beta_mean[order].append(mean)
            beta_error[order].append(error)

    delta_mean = {2: [], 3: []}
    delta_error = {2: [], 3: []}
    for delta in delta_points:
        frame = load_estimates(data_dir, params, beta_right, delta)
        for order in (2, 3):
            mean, error = mean_and_error(
                frame[f"pf_rhoO{order}_fidelity"].to_numpy(), error_mode
            )
            delta_mean[order].append(mean)
            delta_error[order].append(error)

    beta_truth_grid = [float(value) for value in params["truth_grid"]["beta"]]
    energies_left = load_energies(spectrum_dir, params, delta_left)
    beta_truth = {
        order: [
            exact_ground_fidelity(energies_left, beta, order) for beta in beta_truth_grid
        ]
        for order in (1, 2, 3)
    }

    delta_truth_grid = [float(value) for value in params["truth_grid"]["Delta"]]
    delta_truth = {1: [], 2: [], 3: []}
    for delta in delta_truth_grid:
        energies = load_energies(spectrum_dir, params, delta)
        for order in (1, 2, 3):
            delta_truth[order].append(
                exact_ground_fidelity(energies, beta_right, order)
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    truth_styles = {
        1: {"color": "0.8", "lw": 2, "ls": "-.", "label": "No cooling"},
        2: {"color": "0.5", "lw": 2, "ls": "--", "label": r"Cooling, $t=2$"},
        3: {"color": "0.05", "lw": 2, "ls": "-", "label": r"Cooling, $t=3$"},
    }
    for order in (1, 2, 3):
        axes[0].plot(beta_truth_grid, beta_truth[order], **truth_styles[order])
        axes[1].plot(delta_truth_grid, delta_truth[order], **truth_styles[order])

    blue = "#0072B2"
    orange = "#ff7f0e"
    estimate_styles = {
        2: {"fmt": "o", "color": orange, "label": r"QVC, $t=2$"},
        3: {"fmt": "s", "color": blue, "label": r"QVC, $t=3$"},
    }
    for order in (2, 3):
        style = estimate_styles[order]
        axes[0].errorbar(
            beta_points,
            beta_mean[order],
            yerr=beta_error[order],
            fmt=style["fmt"],
            markersize=8,
            mfc=style["color"],
            mec="0.15",
            mew=0.5,
            ecolor=style["color"],
            elinewidth=2,
            capsize=6,
            capthick=2,
            linestyle="none",
            label=style["label"],
        )
        axes[1].errorbar(
            delta_points,
            delta_mean[order],
            yerr=delta_error[order],
            fmt=style["fmt"],
            markersize=8,
            mfc=style["color"],
            mec="0.15",
            mew=0.5,
            ecolor=style["color"],
            elinewidth=2,
            capsize=6,
            capthick=2,
            linestyle="none",
            label=style["label"],
        )

    axes[0].set_xlabel(r"$\beta$", fontsize=23)
    axes[0].set_ylabel(
        r"$\langle \mathrm{gs}|\rho^t|\mathrm{gs}\rangle/\mathrm{tr}(\rho^t)$",
        fontsize=23,
    )
    axes[0].set_xticks(beta_points)
    axes[0].text(0.8, 0.9, "(a)", transform=axes[0].transAxes, fontsize=23)
    axes[1].set_xlabel(r"$\Delta$", fontsize=23)
    axes[1].set_xticks(delta_points)
    axes[1].text(0.08, 0.9, "(b)", transform=axes[1].transAxes, fontsize=23)
    axes[0].set_ylim(-0.02, 1.1)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for axis in axes:
        axis.tick_params(axis="both", which="both", direction="in", labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        ncol=1,
        frameon=True,
        fontsize=15,
        borderpad=0.3,
        labelspacing=0.35,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.25,
        bbox_to_anchor=(0.135, 0.985),
    )
    fig.tight_layout(w_pad=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the Heisenberg QVC figure.")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data" / "heisenberg")
    parser.add_argument(
        "--spectrum-dir", type=Path, default=HERE / "spectra" / "heisenberg"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "figure" / "qvc_heisenberg.pdf"
    )
    args = parser.parse_args()
    params = json.loads(
        (ROOT / "configs" / "qvc_heisenberg_parameters.json").read_text(encoding="utf-8")
    )
    plot(params, args.data_dir, args.spectrum_dir, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
