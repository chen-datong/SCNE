"""Plot CBNE error scaling for estimating tr(rho^3)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import numpy as np
import pandas as pd
from scipy.stats import linregress

from TFIM import build_tfim_pauli_sparse


matplotlib.rcParams["font.sans-serif"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "cm"


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def real_array(values):
    """Parse real or complex-valued CSV cells and return their real parts."""
    parsed = []
    for value in values:
        if isinstance(value, str):
            parsed.append(complex(value.strip().strip("[]()")))
        else:
            parsed.append(value)
    return np.real(np.asarray(parsed, dtype=np.complex128)).astype(float)


def exact_third_moment(n, beta, j_coupling, field):
    """Compute tr(rho^3) for the open-boundary TFIM Gibbs state."""
    hamiltonian = build_tfim_pauli_sparse(
        N=n, J=j_coupling, h=field, pbc=False
    ).toarray()
    energies = np.linalg.eigvalsh(hamiltonian)
    weights = np.exp(-beta * (energies - energies.min()))
    weights /= weights.sum()
    return float(np.sum(weights**3))


def block_mean_mae(values, exact_value, block_size):
    values = np.asarray(values, dtype=float)
    usable = values.size // block_size * block_size
    if usable == 0:
        return np.nan
    block_means = values[:usable].reshape(-1, block_size).mean(axis=1)
    return float(np.mean(np.abs(block_means - exact_value)))


def dedicated_paths(data_dir, n, nu, beta, ancilla_values):
    return [
        data_dir
        / f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}.csv"
        for anc in ancilla_values
    ]


def choose_source(source, data_dir, n, nu, beta, ancilla_values):
    paths = dedicated_paths(data_dir, n, nu, beta, ancilla_values)
    available = [path.exists() for path in paths]
    if source == "rho3":
        missing = [str(path) for path, exists in zip(paths, available) if not exists]
        if missing:
            raise FileNotFoundError("Missing dedicated rho3 data: " + ", ".join(missing))
        return "rho3"
    if source == "paper":
        return "paper"
    if all(available):
        return "rho3"
    if any(available):
        raise FileNotFoundError(
            "Dedicated rho3 data are incomplete. Complete or move them before using --source auto."
        )
    return "paper"


def load_estimates(source, data_dir, n, nu, beta, anc, nm):
    if source == "rho3":
        path = data_dir / f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["NM"].astype(int) == int(nm)]
    else:
        path = data_dir / f"gibbs_pauli_anc_n{n}_Nu{nu}_NM{nm}_beta{beta}_anc{anc}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing paper data: {path}")
        frame = pd.read_csv(path)
    if "third_moment" not in frame.columns:
        raise ValueError(f"Missing third_moment column in {path}")
    return path, real_array(frame["third_moment"].to_numpy())


def collect_errors(params, data_dir, source):
    n = int(params["n"])
    beta = float(params["beta"])
    nu = int(params["left"]["N_U"])
    trials = int(params["trials"])
    ancilla_values = [int(value) for value in params["right"]["n_a"]]
    nm_values = [int(value) for value in params["left"]["N_M"]]
    block_sizes = [int(value) for value in params["right"]["N_U"]]
    source = choose_source(source, data_dir, n, nu, beta, ancilla_values)
    exact_value = exact_third_moment(
        n=n,
        beta=beta,
        j_coupling=float(params["J"]),
        field=float(params["h"]),
    )

    rows = []
    for anc in ancilla_values:
        for nm in nm_values:
            path, estimates = load_estimates(source, data_dir, n, nu, beta, anc, nm)
            if estimates.size != trials:
                raise ValueError(
                    f"Expected {trials} trials in {path}; found {estimates.size}."
                )
            row = {"anc": anc, "NM": nm, "exact_tr_rho3": exact_value}
            for block_size in block_sizes:
                row[f"MAE_NU{block_size}"] = block_mean_mae(
                    estimates, exact_value, block_size
                )
            rows.append(row)
    return pd.DataFrame(rows), source, exact_value


def plot_rho3(errors, params, output_path):
    anc_left = [int(value) for value in params["left"]["n_a"]]
    anc_right = [int(value) for value in params["right"]["n_a"]]
    nu_values = [int(value) for value in params["right"]["N_U"]]
    nm_right = int(params["right"]["N_M"])

    # One vertical column extracted from the 2x3 manuscript layout.
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 8.0))
    blue = cm.Blues
    orange = cm.Oranges
    blue_colors = blue(np.linspace(0.25, 0.9, len(anc_left)))
    orange_colors = orange(np.linspace(0.55, 0.9, len(nu_values)))
    markers = ["o", "s", "^"]
    formatter = ticker.LogFormatterMathtext(base=10)

    for index, anc in enumerate(anc_left):
        subset = errors[errors["anc"] == anc].sort_values("NM")
        axes[0].plot(
            subset["NM"],
            subset["MAE_NU1"],
            marker=markers[index],
            markersize=7,
            linewidth=2,
            color=blue_colors[index],
            label=fr"$n_a={anc}$",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-3, 2e0)
    axes[0].set_xlabel(r"$N_M$", fontsize=23)
    axes[0].set_ylabel("Statistical Error", fontsize=23)
    axes[0].set_title(r"$\mathrm{tr}(\rho^3)$", fontsize=24, pad=8)
    axes[0].legend(
        loc="lower left",
        fontsize=15,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.35,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.2,
    )

    fixed_nm = errors[errors["NM"] == nm_right].sort_values("anc")
    anc_values = fixed_nm["anc"].to_numpy(dtype=float)
    point_handles = []
    point_labels = []
    fit_handles = []
    fit_labels = []
    for index, nu in enumerate(nu_values):
        values = fixed_nm[f"MAE_NU{nu}"].to_numpy(dtype=float)
        mask = np.isfinite(values) & (values > 0)
        slope, intercept, *_ = linregress(
            anc_values[mask], np.log2(values[mask])
        )
        fit_values = np.exp2(intercept + slope * anc_values[mask])
        fit_line, = axes[1].plot(
            anc_values[mask],
            fit_values,
            linestyle="--",
            linewidth=2,
            color=orange_colors[index],
        )
        points = axes[1].scatter(
            anc_values[mask],
            values[mask],
            marker=markers[index],
            s=55,
            linewidth=1.2,
            color=orange_colors[index],
        )
        point_handles.append(points)
        point_labels.append(fr"$N_U={nu}$")
        fit_handles.append(fit_line)
        fit_labels.append(fr"$b={-slope:.2f}$")

    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-3, 1e0)
    axes[1].set_xticks([0, 2, 4, 6, 8])
    axes[1].set_xlabel(r"$n_a$", fontsize=23)
    axes[1].set_ylabel("Statistical Error", fontsize=23)
    point_legend = axes[1].legend(
        point_handles,
        point_labels,
        loc="lower left",
        fontsize=15,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.35,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.2,
        bbox_to_anchor=(0, 0),
    )
    axes[1].add_artist(point_legend)
    axes[1].legend(
        fit_handles,
        fit_labels,
        loc="upper right",
        title=r"Error$\sim 2^{-b n_a}$",
        fontsize=14,
        title_fontsize=14,
        frameon=True,
        handletextpad=0.2,
        handlelength=1.2,
        borderpad=0.35,
        bbox_to_anchor=(1.02, 1.03),
    )

    for axis in axes:
        axis.tick_params(axis="both", which="both", direction="in", labelsize=19, pad=6)
        axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.set_minor_locator(ticker.NullLocator())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the two-panel CBNE error scaling for tr(rho^3)."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "paper", "rho3"),
        default="auto",
        help="use existing paper CSVs, dedicated rho3 rerun CSVs, or auto-detect",
    )
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--output", type=Path, default=HERE / "figure" / "SE_rho3.pdf"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=HERE / "data" / "rho3_statistical_error_n6_Nu1_beta2.0.csv",
    )
    args = parser.parse_args()

    config_path = ROOT / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"][
        "SE_all_observables.pdf"
    ]
    errors, source, exact_value = collect_errors(params, args.data_dir, args.source)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.summary, index=False)
    plot_rho3(errors, params, args.output)
    print(f"Data source: {source}")
    print(f"Exact tr(rho^3): {exact_value:.12g}")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
