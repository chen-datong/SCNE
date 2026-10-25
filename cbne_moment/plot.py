import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "cbne_moment"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'


def cal3moment(n, p):
    d = pow(2, n)
    aux = p * p - 2 * p + 1 + 2 / d * p * (1 - p)
    purity = aux + p * p /d
    third = aux * (1 - p) + aux * p / d + p * p * (1 - p) / d / d + p * p * p / d / d
    return purity, third


def average_Ising(n, depth, sample, J, h, p, combine=1):
    J_str = f"{J:.2f}"
    h_str = f"{h:.2f}"
    p_str = f"{p:.2f}"
    with open(DATA_DIR / "TIM_complete" / f"n{n}_depth{depth}_J{J_str}_h{h_str}_p{p_str}_Sample{sample}.jsonl") as f:
        data = [json.loads(line) for line in f if line.strip()]
    cat = 5000
    purity = np.array([d["Purity"] for d in data])[: cat] - 1
    thirdM = (np.array([d["F"] for d in data])[: cat] - 1 - 3 * purity) / 2


    purity = np.mean(np.reshape(purity, (-1, combine)), axis=1)
    thirdM = np.mean(np.reshape(thirdM, (-1, combine)), axis=1)
    exact_p, exact_t = cal3moment(n, p)
    return np.mean(np.abs(thirdM - exact_t)), np.mean(np.abs(purity - exact_p))


def plot_Haar_N(right_nm_by_n=None, outpath=HERE.parent / "figures/SE_vs_NM_and_n.pdf", figure_height=4.2):
    ns = range(5, 21, 3)
    if right_nm_by_n is None:
        import sys
        sys.path.insert(0, str(HERE.parent))
        from check_paper_data import select_cbne_right_counts
        config = json.loads((HERE.parent / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]
        right_nm_by_n = select_cbne_right_counts(config)
    J = h = 1.0
    p = 0.2
    errors1 = []
    errors10 = []
    errors50 = []
    for n in ns:
        N = int(right_nm_by_n[n])
        if n == 17:
            depth=50
        elif n == 20:
            depth=50
        else:
            depth=35
        error = average_Ising(n, depth, N, J, h, p)[0]
        errors1.append(error)
        error = average_Ising(n, depth, N, J, h, p, combine=10)[0]
        errors10.append(error)
        error = average_Ising(n, depth, N, J, h, p, combine=50)[0]
        errors50.append(error)
    import matplotlib.cm as cm
    cmap_blue = cm.Blues
    colors_left = cmap_blue(np.linspace(0.2, 0.9, 4)) 

    fig, axes = plt.subplots(1, 2, figsize=(7.2, figure_height), sharey=True)
    error_NM = {}
    ns1 = [8, 12, 16, 20]
    for n in ns1:
        error = []
        d = 1 << n
        if n == 17:
            depth = 50
        elif n == 16:
            depth = 40
        elif n == 20:
            depth = 50
        else:
            depth = 35
        for NM in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]:
            err = average_Ising(n, depth, NM, J, h, p)[0]
            error.append(err)
        error_NM[n] = error
    NM = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000])
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')


    axes[0].plot(NM, error_NM[8], marker='o', markersize=8, linewidth=2, label=fr"$n = 8$", c=colors_left[0])

    axes[0].plot(NM, error_NM[12], marker='s', markersize=8, linewidth=2, label=fr"$n = 12$", c=colors_left[1])

    axes[0].plot(NM, error_NM[16], marker='^', markersize=8, linewidth=2, label=fr"$n = 16$", c=colors_left[2])


    def line_with_fixed_slope(x, b):
        return -1.5 * x + b

    (b_opt,), cov = curve_fit(line_with_fixed_slope, np.log(NM[5:10]), np.log(error_NM[20][5:10]))
    axes[0].plot(NM, error_NM[20], marker='d', markersize=8, linewidth=2, label=fr"$n = 20$", c=colors_left[3])
    axes[0].plot(NM[:15], np.exp(b_opt - 1.5 * np.log(NM[:15])), linestyle='--', color="0", linewidth=2)

    axes[0].set_xlabel(r'$N_M$', fontsize=23, labelpad=0)
    axes[0].set_ylabel(r'Statistical Error', fontsize=23)
    axes[0].tick_params(axis='both', which='both', direction='in', labelsize=20, pad=6)
    axes[0].legend(loc='lower left',
                    ncol=1,             
                    frameon=True,
                    fontsize=16,
                    borderpad=0.3,       
                    labelspacing=0.4,   
                    columnspacing=0.5,   
                    handletextpad=0.3,   
                    handlelength=1.25,
                    bbox_to_anchor=(-0.02, -0.02))
    axes[0].text(0.85, 0.9, "(a)", transform=axes[0].transAxes, fontsize=23)


    cmap_orange = cm.Oranges  # Orange palette.
    colors_right = cmap_orange(np.linspace(0.4, 0.9, 3))
    axes[1].set_yscale('log')
    slope1, intercept, r_value, p_value, std_err = linregress(ns, np.log2(errors1))
    line1, = axes[1].plot(ns, np.exp2(intercept + slope1 * np.array(ns)), linestyle='--', color=colors_right[0], linewidth=2)
    axes[1].scatter(ns, errors1, marker='o', s=50, linewidth=2, color=colors_right[0], label=fr"$N_U = 1$")
    print(f"slope: {slope1:.2f}")

    slope10, intercept, r_value, p_value, std_err = linregress(ns, np.log2(errors10))
    line10, = axes[1].plot(ns, np.exp2(intercept + slope10 * np.array(ns)), linestyle='--', color=colors_right[1], linewidth=2)
    axes[1].scatter(ns, errors10, marker='s', s=50, linewidth=2, color=colors_right[1], label=fr"$N_U = 10$")
    print(f"slope: {slope10:.2f}")

    slope50, intercept, r_value, p_value, std_err = linregress(ns, np.log2(errors50))
    line50, = axes[1].plot(ns, np.exp2(intercept + slope50 * np.array(ns)), linestyle='--', color=colors_right[2], linewidth=2)
    axes[1].scatter(ns, errors50, marker='^', s=50, linewidth=2, color=colors_right[2], label=fr"$N_U = 50$")
    print(f"slope: {slope50:.2f}")


    axes[1].set_xlabel(r"$n$", fontsize=23, labelpad=0)
    axes[1].tick_params(axis='both', which='both', direction='in', labelsize=20, pad=6)
    
    # Create the first legend manually (for points)
    leg1 = axes[1].legend(loc='lower left',
        ncol=1,             
        frameon=True,
        fontsize=16,
        borderpad=0.3,       
        labelspacing=0.4,   
        columnspacing=0.5,   
        handletextpad=0.3,   
        handlelength=1.25,
        bbox_to_anchor=(-0.02, -0.02)
        )
    # Add the first legend to the axes so it doesn't get removed
    axes[1].add_artist(leg1)

    # Create the second legend (for slopes)
    axes[1].legend([line1, line10, line50], 
                   [fr"$b = {-slope1:.2f}$", fr"$b = {-slope10:.2f}$", fr"$b = {-slope50:.2f}$"],
                   loc='upper right',       # Changed to upper right or adjust bbox
                   bbox_to_anchor=(1.01, 1.02), # Adjust position as needed
                   title=r"Error$\sim 2^{-bn}$",
                   fontsize=16,
                   title_fontsize=16,
                   frameon=True,
                   framealpha=0.8,
                   labelspacing=0.1,
                   handletextpad=0.2, 
                   handlelength=1.3,
                   borderpad=0.2
                   )
    axes[1].text(0.05, 0.9, "(b)", transform=axes[1].transAxes, fontsize=23)
    axes[1].set_xticks([5, 10, 15, 20])
    axes[1].set_ylim(ymin=2e-4, ymax=2)
    plt.tight_layout(w_pad=0.5)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, format='pdf')
    left_x_limits = axes[0].get_xlim()
    plt.close(fig)
    return left_x_limits


SAMPLE_LOWER_CBNE = {12: 380, 13: 520, 14: 810, 15: 1480, 16: 2150, 17: 3500, 18: 5300, 19: 8500}
SAMPLE_LOWER_PTME = {12: 460, 13: 750, 14: 1080, 15: 1840, 16: 2900, 17: 3500, 18: 6100, 19: 13000}

def plot_sample_complexity_lower(nu=200, target_error=0.1, repetitions=50,
                                 outpath=HERE.parent / "figures/sample_lower.pdf"):
    if (nu, target_error, repetitions) != (200, 0.1, 50):
        raise ValueError(
            "The retained lower-bound values only support the manuscript "
            "parameters N_U=200, target_error=0.1, repetitions=50."
        )
    blue = "#1f77b4"
    green = "#2ca02c"
    scale_factor = nu

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)

    x_left = np.array(list(SAMPLE_LOWER_CBNE.keys()), dtype=float)
    y_left = scale_factor * np.array(list(SAMPLE_LOWER_CBNE.values()), dtype=float)
    slope_left, intercept_left, r_left, _, _ = linregress(x_left, np.log2(y_left))
    y_left_fit = np.exp2(intercept_left + slope_left * x_left)
    axes[0].scatter(
        x_left,
        y_left,
        c=blue,
        marker="o",
        s=50,
        edgecolors="0.15",
        linewidth=0.8,
        zorder=3,
    )
    axes[0].plot(
        x_left,
        y_left_fit,
        c=blue,
        linewidth=2,
        linestyle="--",
        zorder=2,
        label=fr"$\log_2(N_U N_M)\propto{slope_left:.3f} n$",
    )
    axes[0].set_xlabel(r"$n$", fontsize=23, labelpad=0)
    axes[0].set_ylabel(r"Sample complexity", fontsize=23)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis='both', which='major', direction='in', labelsize=20)
    axes[0].legend(fontsize=18, loc='lower right')
    axes[0].text(0.05, 0.90, "(a)", transform=axes[0].transAxes, fontsize=23)
    print(f"[CBNE] slope = {slope_left}")
    print(f"[CBNE] intercept = {intercept_left}")
    print(f"[CBNE] r_value = {r_left}")

    x_right = np.array(list(SAMPLE_LOWER_PTME.keys()), dtype=float)
    y_right = scale_factor * np.array(list(SAMPLE_LOWER_PTME.values()), dtype=float)
    slope_right, intercept_right, r_right, _, _ = linregress(x_right, np.log2(y_right))
    y_right_fit = np.exp2(intercept_right + slope_right * x_right)
    axes[1].scatter(
        x_right,
        y_right,
        c=green,
        marker="o",
        s=50,
        edgecolors="0.15",
        linewidth=0.8,
        zorder=3,
    )
    axes[1].plot(
        x_right,
        y_right_fit,
        c=green,
        linewidth=2,
        linestyle="--",
        zorder=2,
        label=fr"$\log_2(N_U N_M)\propto {slope_right:.3f}\, n_A$",
    )
    axes[1].set_xlabel(r"$n_A$", fontsize=23, labelpad=0)
    axes[1].set_yscale("log")
    axes[1].tick_params(axis='both', which='major', direction='in', labelsize=20)
    axes[1].legend(fontsize=18, loc='lower right')
    axes[1].text(0.05, 0.90, "(b)", transform=axes[1].transAxes, fontsize=23)
    print(f"[PTME] slope = {slope_right}")
    print(f"[PTME] intercept = {intercept_right}")
    print(f"[PTME] r_value = {r_right}")

    plt.tight_layout(w_pad=0.5)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, format="pdf")
    plt.show()
