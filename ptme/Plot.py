import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ptme"
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from scipy.stats import linregress
from utils import D3, D4, ppt3, B2, D5
from Gibbs import load_results_csv, extract_arrays_gibbs
from heisenberg_plot import extract_arrays_depol

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'


def gibbs_ent_detect(
    nA=11,
    nB=1,
    Nu=100,
    NM=1_000_000,
    csv_path=DATA / 'gibbs_dbasis_paper.csv',
    expected_trials=1000,
    outpath=HERE.parent / "figures/Gibbs_ent_detect.pdf",
):
    df = load_results_csv(csv_path)
    betas = np.array([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
    ppt3_mean = np.array([0] * 10, dtype=float)
    ppt3_std = np.array([0] * 10, dtype=float)
    B2_mean = np.array([0] * 10, dtype=float)
    B2_std = np.array([0] * 10, dtype=float)
    for i, beta in enumerate(betas):
        purity_tmp, third_tmp, fourth_tmp, fifth_tmp = extract_arrays_gibbs(df, nA=nA, nB=nB, Nu=Nu, NM=NM, beta=beta)
        if len(purity_tmp) != expected_trials:
            raise ValueError(
                f"Expected {expected_trials} trials for beta={beta}, nA={nA}, "
                f"nB={nB}, N_U={Nu}, N_M={NM}; found {len(purity_tmp)}."
            )
        PPT3 = ppt3(purity_tmp, third_tmp)
        b2 = B2(purity_tmp, third_tmp, fourth_tmp, fifth_tmp)
        ppt3_mean[i] = np.mean(PPT3)
        ppt3_std[i] = np.std(PPT3)
        B2_mean[i] = np.mean(b2)*10
        B2_std[i] = np.std(b2)*10

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(betas, ppt3_mean, marker='s', markersize=7, label=r'$-|B_1|$')
    plt.fill_between(
        betas,
        ppt3_mean - ppt3_std,
        ppt3_mean + ppt3_std,
        alpha=0.3
    )
    plt.plot(betas, B2_mean, marker='o', markersize=7, label=r'$-10|B_2|$')
    plt.fill_between(
        betas,
        B2_mean - B2_std,
        B2_mean + B2_std,
        alpha=0.3
    )
    plt.axhline(y=0, linestyle=(0, (6, 6)), linewidth=1.5, color='black')
    plt.xlabel(r'$\beta$', fontsize=23)
    plt.ylabel('value', fontsize=23)
    plt.xlim(0.59, 1.51)
    plt.legend(fontsize=20, handletextpad=0.3, borderpad=0.3, loc='upper left', bbox_to_anchor=(-0.01, 1.02))
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.show()


def detecting_ent_tfim(csv_filename=DATA / "processed_results2.csv",
                       outpath=HERE.parent / "figures/TFIM_detection_prob.svg"):
    df = pd.read_csv(csv_filename)
    ps = np.array([0.2 + 0.05 * i for i in range(11)]) # use .2f format
    ps = np.round(ps, 2)
    prob_d3s = []
    prob_d4s = []
    prob_d5s = []
    prob_anys = []
    prob_ppt3s = []
    D3_mean = np.array([0] * 11, dtype=float)
    D3_std = np.array([0] * 11, dtype=float)
    D4_mean = np.array([0] * 11, dtype=float)
    D4_std = np.array([0] * 11, dtype=float)
    D5_mean = np.array([0] * 11, dtype=float)
    D5_std = np.array([0] * 11, dtype=float)
    ppt3_mean = np.array([0] * 11, dtype=float)
    ppt3_std = np.array([0] * 11, dtype=float)
    for i, p in enumerate(ps):
        purity, third, fourth, fifth = extract_arrays_depol(df, nA=18, nB=2, NM=100000000, dprob=p)
        d3 = D3(purity, third)
        d4 = D4(purity, third, fourth)
        d5 = D5(purity, third, fourth, fifth)
        ppt3_val = ppt3(purity, third)
        
        # calculate the ratio of positive values
        prob_d3 = d3[d3 > 0].shape[0] / d3.shape[0]
        prob_d4 = d4[d4 > 0].shape[0] / d4.shape[0]
        prob_d5 = d5[d5 > 0].shape[0] / d5.shape[0]
        prob_ppt3 = ppt3_val[ppt3_val > 0].shape[0] / ppt3_val.shape[0]
        # calculate the ratio of one of d3, d4, d5 is positive
        prob_any = (d3 > 0) | (d4 > 0) | (d5 > 0)
        prob_any = prob_any.sum() / prob_any.shape[0]
        prob_d3s.append(prob_d3)
        prob_d4s.append(prob_d4)
        prob_d5s.append(prob_d5)
        prob_anys.append(prob_any)
        prob_ppt3s.append(prob_ppt3)
        D3_mean[i] = np.mean(d3)
        D3_std[i] = np.std(d3)
        D4_mean[i] = np.mean(d4)
        D4_std[i] = np.std(d4)
        D5_mean[i] = np.mean(d5)
        D5_std[i] = np.std(d5)

    fig, axes = plt.subplots(1, 1, figsize=(4, 4.2))
    import matplotlib.cm as cm
    cmap_blue = cm.Blues
    colors = cmap_blue(np.linspace(0.2, 0.9, 3)) 
    orange = cm.Oranges(0.7)
    axes.vlines(x=0.52, ymin=-0.2, ymax=1.2, colors='black', linestyles=(0, (6, 6)), linewidth=1.5)

    axes.plot(ps, prob_d3s, marker='o', markersize=8, label=r'$D_3$', color=colors[0])
    axes.plot(ps, prob_d4s, marker='s', markersize=8, label=r'$D_4$', color=colors[1])
    axes.plot(ps, prob_d5s, marker='^', markersize=8, label=r'$D_5$', color=colors[2])
    axes.plot(ps, prob_anys, marker='d', markersize=8, color=orange)
    axes.set_xlabel(r'$p$', fontsize=23)
    axes.set_ylabel('Detection Probability', fontsize=23)
    axes.set_xticks(ps[::2])
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_ylim(-0.03, 1.03)
    axes.legend(loc='upper right',
        ncol=1,             
        frameon=True,
        fontsize=16,
        borderpad=0.3,       
        labelspacing=0.4,   
        columnspacing=0.5,   
        handletextpad=0.3,   
        handlelength=1.25,
        bbox_to_anchor=(1, 1))
    axes.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)

    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.show()


def plot_mae_t4_nA_convergence(
    data_dir=DATA,
    file_pattern=r"mae_n*_p0.30.csv",
    nB=2,
    tail_points=5,
    right_nm=10_000_000,
    left_nA_to_plot=(10, 14, 18),
    exclude_nA=(6,),
    left_max_nm=5e6,
    outpath=HERE.parent / "figures/mae_t4_vs_NM_and_nA_decay.pdf",
):
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched '{file_pattern}' in {data_dir}")

    n_pattern = re.compile(r"mae_n(\d+)_p0\.30\.csv$")
    traces = []
    for csv_path in csv_files:
        match = n_pattern.match(csv_path.name)
        if not match:
            continue
        n_total = int(match.group(1))
        nA = n_total - nB
        if nA <= 0:
            continue

        df = pd.read_csv(csv_path)
        required = {"sample", "mae_t4"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {missing}")

        df = df.sort_values("sample")
        y4_arr = df["mae_t4"].to_numpy(dtype=float)
        y4_blk10_arr = df["mae_t4_blk10"].to_numpy(dtype=float) if "mae_t4_blk10" in df.columns else None
        y4_blk50_arr = df["mae_t4_blk50"].to_numpy(dtype=float) if "mae_t4_blk50" in df.columns else None
        if right_nm is not None:
            at_nm = df[df["sample"] == int(right_nm)]
            if len(at_nm) != 1:
                raise ValueError(
                    f"Expected one N_M={right_nm} row in {csv_path.name}; found {len(at_nm)}."
                )
            stable_error_t4 = float(at_nm["mae_t4"].iloc[0])
            stable_error_t4_blk10 = float(at_nm["mae_t4_blk10"].iloc[0]) if y4_blk10_arr is not None else np.nan
            stable_error_t4_blk50 = float(at_nm["mae_t4_blk50"].iloc[0]) if y4_blk50_arr is not None else np.nan
        else:
            k = min(tail_points, len(y4_arr))
            stable_error_t4 = float(np.mean(y4_arr[-k:]))
            stable_error_t4_blk10 = float(np.mean(y4_blk10_arr[-k:])) if y4_blk10_arr is not None else np.nan
            stable_error_t4_blk50 = float(np.mean(y4_blk50_arr[-k:])) if y4_blk50_arr is not None else np.nan
        traces.append((nA, df, stable_error_t4, stable_error_t4_blk10, stable_error_t4_blk50))

    traces = [x for x in traces if x[0] not in set(exclude_nA)]

    if len(traces) < 2:
        raise ValueError("Need at least two valid n_A points for exponential-decay fit.")

    traces.sort(key=lambda x: x[0])
    nA_vals = np.array([x[0] for x in traces], dtype=float)
    stable_t4 = np.array([x[2] for x in traces], dtype=float)
    stable_t4_blk10 = np.array([x[3] for x in traces], dtype=float)
    stable_t4_blk50 = np.array([x[4] for x in traces], dtype=float)

    cmap_blue = cm.Blues
    cmap_orange = cm.Oranges
    color_right = cmap_orange(0.7)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    left_nA_set = set(left_nA_to_plot)
    left_traces = [(nA, df, s4, s10, s50) for (nA, df, s4, s10, s50) in traces if nA in left_nA_set]
    left_traces.sort(key=lambda x: x[0])
    if not left_traces:
        raise ValueError(f"No traces matched left_nA_to_plot={left_nA_to_plot}")
    colors_left = cmap_blue(np.linspace(0.2, 0.9, len(left_traces)))

    for i, (nA, df, _, _, _) in enumerate(left_traces):
        df_left = df[df["sample"] <= left_max_nm]
        if df_left.empty:
            continue
        axes[0].plot(
            df_left["sample"],
            df_left["mae_t4"],
            marker="o",
            markersize=7,
            linewidth=2,
            label=rf"$n_A={nA}$",
            color=colors_left[i],
        )

    axes[0].set_xlim(left=4e2, right=left_max_nm * 1.15)
    axes[0].set_ylim(top=2e0, bottom=1e-3)
    axes[0].set_xlabel(r"$N_M$", fontsize=23)
    axes[0].set_ylabel("Statistical Error", fontsize=23)
    axes[0].tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
    axes[0].legend(
        loc="lower left",
        ncol=1,
        frameon=True,
        fontsize=16,
        borderpad=0.3,
        labelspacing=0.4,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.25,
    )
    axes[0].text(0.82, 0.9, "(a)", transform=axes[0].transAxes, fontsize=23)

    right_series = [
        ("N_U=1", stable_t4, "o", cm.Oranges(0.55)),
        ("N_U=10", stable_t4_blk10, "s", cm.Oranges(0.72)),
        ("N_U=50", stable_t4_blk50, "^", cm.Oranges(0.9)),
    ]
    point_handles = []
    point_labels = []
    fit_lines = []
    fit_labels = []
    for name, vals, marker, color in right_series:
        mask = np.isfinite(vals) & (vals > 0)
        if np.sum(mask) < 2:
            continue
        slope, intercept, _, _, _ = linregress(nA_vals[mask], np.log2(vals[mask]))
        fit_vals = np.exp2(intercept + slope * nA_vals[mask])
        b_val = -slope
        line_fit, = axes[1].plot(
            nA_vals[mask],
            fit_vals,
            linestyle="--",
            linewidth=2,
            color=color,
        )
        pt = axes[1].scatter(
            nA_vals[mask],
            vals[mask],
            marker=marker,
            s=55,
            linewidth=1.2,
            color=color,
            label=rf"${name}$",
        )
        point_handles.append(pt)
        point_labels.append(rf"${name}$")
        fit_lines.append(line_fit)
        fit_labels.append(rf"$b = {b_val:.2f}$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$n_A$", fontsize=23)
    axes[1].set_xticks(nA_vals.astype(int))
    axes[1].tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
    leg_points = axes[1].legend(
        point_handles,
        point_labels,
        loc="lower left",
        ncol=1,
        frameon=True,
        fontsize=16,
        borderpad=0.3,
        labelspacing=0.4,
        columnspacing=0.5,
        handletextpad=0.3,
        handlelength=1.25,
        bbox_to_anchor=(0, 0),
    )
    axes[1].add_artist(leg_points)

    axes[1].legend(
        fit_lines,
        fit_labels,
        loc="upper right",
        bbox_to_anchor=(1.02, 1.03),
        title=r"Error$\sim 2^{-bn_A}$",
        fontsize=16,
        title_fontsize=16,
        frameon=True,
        handletextpad=0.2,
        handlelength=1.3,
        borderpad=0.4,
    )
    axes[1].text(0.06, 0.9, "(b)", transform=axes[1].transAxes, fontsize=23)

    math_formatter = ticker.LogFormatterMathtext(base=10)
    axes[0].yaxis.set_major_formatter(math_formatter)
    axes[0].yaxis.set_minor_locator(ticker.NullLocator())
    axes[1].yaxis.set_major_formatter(math_formatter)
    axes[1].yaxis.set_minor_locator(ticker.NullLocator())
    plt.tight_layout(w_pad=0.5)
    plt.savefig(outpath)
    plt.show()
