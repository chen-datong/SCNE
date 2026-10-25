"""Plot CBNE error scaling for estimating tr(rho^3)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import numpy as np
import pandas as pd
from scipy.stats import linregress

from TFIM import build_tfim_pauli_sparse
from generate_rho3_data import experiment_parameters, state_eigenvalues, rho3_data_path, rho3_summary_path


matplotlib.rcParams["font.sans-serif"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "cm"


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "cbne_observables"


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


def dedicated_paths(data_dir, n, nu, beta, ancilla_values, params=None):
    if params is not None:
        return [rho3_data_path(data_dir, params, anc) for anc in ancilla_values]
    return [
        data_dir
        / f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}.csv"
        for anc in ancilla_values
    ]


def choose_source(source, data_dir, n, nu, beta, ancilla_values, params=None):
    paths = dedicated_paths(data_dir, n, nu, beta, ancilla_values, params=params)
    available = [path.exists() for path in paths]
    if params is not None and params.get("state", "gibbs") == "depolarized":
        if source == "paper":
            raise ValueError("The paper CSVs contain Gibbs states, not depolarized ground states")
        missing = [str(path) for path, exists in zip(paths, available) if not exists]
        if missing:
            raise FileNotFoundError("Missing depolarized-state data. Generate them with "
                                    "generate_rho3_data.py --processes 1. Missing: " + ", ".join(missing))
        return "rho3"
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


def load_estimates(source, data_dir, n, nu, beta, anc, nm, params=None):
    if source == "rho3":
        path = (rho3_data_path(data_dir, params, anc) if params is not None else
                data_dir / f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}.csv")
        frame = pd.read_csv(path)
        frame = frame[frame["NM"].astype(int) == int(nm)]
    else:
        path = data_dir / f"gibbs_pauli_anc_n{n}_Nu{nu}_NM{nm}_beta{beta}_anc{anc}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing paper data: {path}")
        frame = pd.read_csv(path)
    if "third_moment" not in frame.columns:
        raise ValueError(f"Missing third_moment column in {path}")
    if source == "rho3":
        if "idx" not in frame.columns:
            raise ValueError(f"Missing trial index column in {path}")
        frame = frame.sort_values("idx")
        if frame["idx"].tolist() != list(range(len(frame))):
            raise ValueError(f"Incomplete or duplicate trial indices in {path}, N_M={nm}")
    estimates = real_array(frame["third_moment"].to_numpy())
    if not np.isfinite(estimates).all():
        raise ValueError(f"Non-finite estimates in {path}, N_M={nm}")
    return path, estimates


def collect_errors(params, data_dir, source):
    n = int(params["n"])
    beta = float(params.get("beta", 0.0))
    nu = int(params["left"]["N_U"])
    trials = int(params["trials"])
    ancilla_values = [int(value) for value in params["right"]["n_a"]]
    nm_values = [int(value) for value in params["left"]["N_M"]]
    block_sizes = [int(value) for value in params["right"]["N_U"]]
    source = choose_source(source, data_dir, n, nu, beta, ancilla_values, params=params)
    exact_value = float(np.sum(state_eigenvalues(params)**3))

    rows = []
    for anc in ancilla_values:
        for nm in nm_values:
            path, estimates = load_estimates(source, data_dir, n, nu, beta, anc, nm, params=params)
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


def plot_rho3(
    errors,
    params,
    output_path,
    *,
    anc_left=None,
    anc_right=None,
    anc_right_ticks=None,
    panel_labels=("(a)", "(b)"),
    show_nm_fit=False,
    nm_fit_anc=None,
    nm_fit_range=(1000, 10000),
    figure_height=4.2,
    y_limits=(2e-4, 2.0),
    right_y_limits=None,
    frame_layout=None,
    nm_x_limits=None,
):
    if anc_left is None:
        anc_left = [int(value) for value in params["left"]["n_a"]]
    if anc_right is None:
        anc_right = [int(value) for value in params["right"]["n_a"]]
    nu_values = [int(value) for value in params["right"]["N_U"]]
    nm_right = int(params["right"]["N_M"])

    # Match the compact, shared-y layout of SE_vs_NM_and_n.pdf.
    figure_size = (7.2, figure_height) if frame_layout is None else (
        frame_layout["page_width"] / 72, frame_layout["page_height"] / 72
    )
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharey=False)
    blue = cm.Blues
    orange = cm.Oranges
    blue_colors = blue(np.linspace(0.2, 0.9, len(anc_left)))
    orange_colors = orange(np.linspace(0.4, 0.9, len(nu_values)))
    markers = ["o", "s", "^", "d"]
    formatter = ticker.LogFormatterMathtext(base=10)

    for index, anc in enumerate(anc_left):
        subset = errors[errors["anc"] == anc].sort_values("NM")
        if subset.empty:
            raise ValueError(f"No rho3 error data for n_a={anc}")
        axes[0].plot(
            subset["NM"],
            subset["MAE_NU1"],
            marker=markers[index],
            markersize=8,
            linewidth=2,
            color=blue_colors[index],
            label=fr"$n_a = {anc}$",
        )
    if show_nm_fit:
        # Fix the reference exponent at -3/2; fit only its log-space offset.
        fit_anc = max(anc_left) if nm_fit_anc is None else int(nm_fit_anc)
        fit_lo, fit_hi = nm_fit_range
        fit_data = errors[
            (errors["anc"] == fit_anc)
            & (errors["NM"] >= fit_lo)
            & (errors["NM"] <= fit_hi)
        ].sort_values("NM")
        if len(fit_data) < 3 or (fit_data["MAE_NU1"] <= 0).any():
            raise ValueError(f"Need at least three positive n_a={fit_anc} points for the fit")
        fit_slope = -1.5
        fit_intercept = float(np.mean(
            np.log(fit_data["MAE_NU1"]) - fit_slope * np.log(fit_data["NM"])
        ))
        axes[0].set_xscale("log")
        data_x_limits = axes[0].get_xlim() if nm_x_limits is None else nm_x_limits
        fit_nm = np.geomspace(*data_x_limits, 200)
        axes[0].plot(
            fit_nm,
            np.exp(fit_intercept) * fit_nm ** fit_slope,
            "k--",
            linewidth=2,
        )
        axes[0].set_xlim(data_x_limits)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylim(*y_limits)
    nm_max = errors.loc[errors["anc"].isin(anc_left), "NM"].max()
    axes[0].set_xticks([1e3, 1e5, 1e7] if nm_max >= 1e7 else [1e2, 1e4, 1e6])
    if nm_x_limits is not None:
        axes[0].set_xlim(*nm_x_limits)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].text(0.85, 0.90, panel_labels[0], transform=axes[0].transAxes, fontsize=23)
    axes[0].set_xlabel(r"$N_M$", fontsize=23, labelpad=0)
    axes[0].set_ylabel("Statistical Error", fontsize=23)
    axes[0].legend(
        loc="lower left",
        ncol=1,
        fontsize=16,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.4,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.25,
        bbox_to_anchor=(-0.02, -0.02),
    )

    fixed_nm = errors[
        (errors["NM"] == nm_right) & errors["anc"].isin(anc_right)
    ].sort_values("anc")
    if fixed_nm["anc"].tolist() != sorted(set(anc_right)):
        raise ValueError(f"Missing or duplicate ancilla data at N_M={nm_right}")
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
            s=50,
            linewidth=2,
            color=orange_colors[index],
        )
        point_handles.append(points)
        point_labels.append(fr"$N_U = {nu}$")
        fit_handles.append(fit_line)
        fit_labels.append(fr"$b = {-slope:.2f}$")

    axes[1].set_yscale("log")
    axes[1].set_ylim(*(right_y_limits or y_limits))
    if anc_right_ticks is None:
        step = 4 if max(anc_right) > 8 else 2
        anc_right_ticks = sorted(set(range(0, max(anc_right) + 1, step)) | {max(anc_right)})
    axes[1].set_xticks(anc_right_ticks)
    axes[1].set_xlabel(r"$n_a$", fontsize=23, labelpad=0)
    axes[1].text(0.05, 0.90, panel_labels[1], transform=axes[1].transAxes, fontsize=23)
    point_legend = axes[1].legend(
        point_handles,
        point_labels,
        loc="lower left",
        fontsize=16,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.4,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.25,
        bbox_to_anchor=(-0.02, -0.02),
    )
    axes[1].add_artist(point_legend)
    axes[1].legend(
        fit_handles,
        fit_labels,
        loc="upper right",
        title=r"Error$\sim 2^{-b n_a}$",
        fontsize=16,
        title_fontsize=16,
        frameon=True,
        framealpha=0.8,
        labelspacing=0.1,
        handletextpad=0.2,
        handlelength=1.3,
        borderpad=0.2,
        bbox_to_anchor=(1.01, 1.02),
    )

    for axis in axes:
        axis.tick_params(axis="both", which="both", direction="in", labelsize=20, pad=6)
        axis.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=100))
        axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[1].tick_params(labelleft=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_layout is None:
        fig.tight_layout(w_pad=0.5)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    else:
        for axis, (left, right) in zip(axes, frame_layout["spans"]):
            axis.set_position((
                left / frame_layout["page_width"],
                frame_layout["bottom"] / frame_layout["page_height"],
                (right - left) / frame_layout["page_width"],
                frame_layout["height"] / frame_layout["page_height"],
            ))
        fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the two-panel CBNE error scaling for tr(rho^3)."
    )
    parser.add_argument("--state", choices=("depolarized", "gibbs"), default="depolarized")
    parser.add_argument(
        "--source",
        choices=("auto", "paper", "rho3"),
        default="auto",
        help="use existing paper CSVs, dedicated rho3 rerun CSVs, or auto-detect",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=HERE.parent / "figures" / "SE_rho3_anc.pdf"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    params = experiment_parameters(args.state)
    if args.summary is None:
        args.summary = rho3_summary_path(args.data_dir, params)
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
