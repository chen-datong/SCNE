import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'


def average(n, depth, sample):
    with open(f"data/n{n}_depth{depth}_Sample{sample}.jsonl") as f:
        data = [json.loads(line) for line in f if line.strip()]
    F_values = np.array([d["F"] for d in data])
    return np.mean(np.square(F_values - 6))


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
    with open(f"data/TIM_complete/n{n}_depth{depth}_J{J_str}_h{h_str}_p{p_str}_Sample{sample}.jsonl") as f:
        data = [json.loads(line) for line in f if line.strip()]
    # if combine != 1:
    cat = 5000
    if n == 20 and combine != 1:
        cat = 500
    purity = np.array([d["Purity"] for d in data])[: cat] - 1
    thirdM = (np.array([d["F"] for d in data])[: cat] - 1 - 3 * purity) / 2


    purity = np.mean(np.reshape(purity, (-1, combine)), axis=1)
    thirdM = np.mean(np.reshape(thirdM, (-1, combine)), axis=1)
    exact_p, exact_t = cal3moment(n, p)
    # return np.mean(np.abs(thirdM - exact_t) ** 2), np.mean(np.abs(purity - exact_p))
    return np.mean(np.abs(thirdM - exact_t)), np.mean(np.abs(purity - exact_p))

def read_thirdM_from_file(n, depth, J, h, p, NM, base_dir="data/TIM_complete"):
    """
    返回：
      thirdM_all: 该文件里每条记录对应的 thirdM 数组
      exact_t:    理论 tr(rho^3)
    """
    J_str = f"{J:.2f}"
    h_str = f"{h:.2f}"
    p_str = f"{p:.2f}"
    fp = os.path.join(base_dir, f"n{n}_depth{depth}_J{J_str}_h{h_str}_p{p_str}_Sample{NM}.jsonl")

    with open(fp, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    purity = np.array([d["Purity"] for d in data], dtype=float) - 1.0
    thirdM = (np.array([d["F"] for d in data], dtype=float) - 1.0 - 3.0 * purity) / 2.0

    return thirdM


def plot(J, h, p):
    depths = [15, 20, 25, 30, 35]
    NM = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00098B", "#4159E1D4", "#87A7EBE2", "#ADD8E6"])
    colors = np.linspace(0.2, 0.95, len(depths))[::-1]
    # colors = [
    #     "#0d47a1", 
    #     "#1565c0", 
    #     "#1976d2",  
    #     "#1e88e5",  
    #     "#42a5f5", 
    #     "#64b5f6",  
    #     "#90caf9",  
    #     ][::-1]
    plt.figure(figsize=(7.2, 4.6))
    for id, d in enumerate(depths):
        error = []
        for i in NM:
            error.append(average_Ising(15, d, i, J, h, p)[0])
        plt.plot(NM, error, label=fr'$D ={d}$', linewidth=2, marker='.', markersize=15, color=cmap(colors[id]))
    plt.legend(fontsize=20, loc='lower left', handlelength=1.2, handletextpad=0.5, bbox_to_anchor=(-0.01, -0.02))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N_M$', fontsize=23)
    # plt.ylabel(r'$\langle|\epsilon|\rangle$', fontsize=20)
    plt.ylabel(r'Statistical Error', fontsize=23)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=6)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'figures/ThirdM_depth.pdf', format='pdf')
    plt.show()

def plot_depth(J, p):
    depths = [15, 20, 25, 30, 35]
    NM = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00098B", "#4159E1D4", "#87A7EBE2", "#ADD8E6"])
    colors = np.linspace(0.2, 0.95, len(depths))[::-1]
    
    # 创建 1x3 子图，共享 y 轴
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)

    # 不同的 h 值
    h_values = [0.5, 1.0, 2.0]
    
    # 保存图例的句柄和标签
    handles_for_legend = []
    labels_for_legend = []

    for i, h in enumerate(h_values):
        ax = axes[i]
        for id, d in enumerate(depths):
            error = []
            for n in NM:
                error.append(average_Ising(15, d, n, J, h, p)[0])
            ax.plot(NM, error, label=fr'$D ={d}$', linewidth=2, marker='.', markersize=15, color=cmap(colors[id]))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N_M$', fontsize=23)
        

        # 设置刻度字体大小
        ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=6)
        
        # 记录第一个图例，后面不再重复
        if i == 0:
            ax.set_ylabel('Statistical Error', fontsize=23)
            ax.legend(fontsize=17, loc='lower left', handlelength=1.2, handletextpad=0.5, bbox_to_anchor=(-0.01, -0.02))

        ax.set_title(f'$h = {h}$', fontsize=23)

    
    plt.tight_layout(w_pad=0)
    plt.savefig(f'figures/ThirdM_depth.pdf', format='pdf')
    plt.show()

def plot_trrho3_vs_p(n, J, h, NM, depths, p_grid,
                     base_dir="data/TIM_complete",
                     err_mode="sem",   # "sem" or "std" or "ci95"
                     connect_points=False,
                     outpath=None):
    """
    画：真值线 + 各depth在固定NM下、随p变化的估计均值与误差棒
    - 真值：cal3moment(n,p)[1]
    - 估计均值：mean(thirdM)
    - 误差棒：默认 SEM = std(thirdM)/sqrt(K)
      (K 是该文件里记录条数)
    """

    p_grid = np.array(p_grid, dtype=float)

    exact_curve = np.array([cal3moment(n, float(p))[1] for p in p_grid], dtype=float)

    plt.figure(figsize=(7.2, 4.6))
    # plt.plot(p_grid, exact_curve, linewidth=2, color="black", label=r"$\mathrm{tr}(\rho^3)$")

    def lighten(hex_color, factor=0.45):
        """
        factor in [0,1]: 0=原色, 1=变白
        """
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"

    light_blue = lighten(blue, 0.20)
    light_orange = lighten(orange, 0.20)
    light_green = lighten(green, 0.20)

    for i, depth in enumerate(depths):
        means, yerrs, ps_used = [], [], []

        for p in p_grid:
            try:
                thirdM_all = read_thirdM_from_file(n, depth, J, h, float(p), NM, base_dir=base_dir)
            except FileNotFoundError:
                continue

            m = float(np.mean(thirdM_all))
            s = float(np.std(thirdM_all, ddof=1)) if len(thirdM_all) > 1 else 0.0

            if err_mode == "std":
                e = s
            else:
                sem = s / np.sqrt(len(thirdM_all)) if len(thirdM_all) > 1 else 0.0
                e = sem if err_mode == "sem" else 1.96 * sem  # 95% CI

            ps_used.append(float(p))
            means.append(m)
            yerrs.append(e)

        if len(ps_used) == 0:
            print(f"[WARN] depth={depth}: no points loaded (check path / naming).")
            continue

        fmt = "o-" if connect_points else "o"
        if i == 0:
            plt.errorbar(ps_used, means-exact_curve, yerr=yerrs, fmt="^", markersize=10,
                    mfc=green, mec="0.15", mew=0.8,
                    ecolor=light_green, elinewidth=2, capsize=6, capthick=4,
                    linestyle="-", color=green,
                    label=fr"depth$ ={depth}$")
        elif i == 1:
            plt.errorbar(ps_used, means-exact_curve, yerr=yerrs, fmt="o", markersize=10,
                    mfc=orange, mec="0.15", mew=0.8,
                    ecolor=light_orange, elinewidth=2, capsize=6, capthick=4,
                    linestyle="-", color=orange,
                    label=fr"depth$ ={depth}$")
        else:
            plt.errorbar(ps_used, means-exact_curve, yerr=yerrs, fmt="s", markersize=10,
                    mfc=blue, mec="0.15", mew=0.8,
                    ecolor=light_blue, elinewidth=2, capsize=6, capthick=4,
                    linestyle="-", color=blue,
                    label=fr"depth$ ={depth}$")

    plt.xlabel(r"$p$", fontsize=23)
    plt.ylabel(r"$\hat{p}_3-\mathrm{tr}\left(\rho^3\right)$", fontsize=23)
    # plt.title(rf"TFIM ($n={n}$, $J={J}$, $h={h}$): single-setting estimate of $\mathrm{{tr}}(\rho^3)$")
    # plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, fontsize=18)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', direction='in', labelsize=20)

    if outpath is not None:
        plt.savefig(outpath)
    plt.show()


def minimal_depth(ns, J, h, NM, p, eps, abs=False):
    depths = range(4, 36)
    m_depths = {}
    max_depth = {}
    for n in ns:
        max_depth[n] = average_Ising(n, 35, NM, J, h, p)[0]

    for n in ns:
        for depth in depths:
            file_name = f"moment-estimation/TIM_complete/n{n}_depth{depth}_J{J:.2f}_h{h:.2f}_p{p:.2f}_Sample{NM}.jsonl"
            if not os.path.exists(file_name):
                continue
            error, _ = average_Ising(n, depth, NM, J, h, p)
            if n == 6:
                print(f"Depth {depth}: Error = {error}")
            if abs and error - max_depth[n] < eps:
                    m_depths[n] = depth
                    # print(f"Minimal depth for n={n} is {depth}")
                    break
            if not abs and (error - max_depth[n]) / max_depth[n] < eps:
                m_depths[n] = depth
                # print(f"Minimal depth for n={n} is {depth}")
                break
    print("Final minimal depths:", m_depths)
    return m_depths

def depth_saturate():
    ns = range(5, 16)
    depths = range(4, 40)
    J = 1.0
    h = 1.0
    NM = 10**6
    p = 0.2
    eps_True = 0.5
    eps_False = 0.5
    blue = "#1f77b4"
    green = "#2ca02c"
    
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    
    # abs=True
    m_depths_abs_true = minimal_depth(ns, J, h, NM, p, eps_True, abs=True)
    ns_true = list(m_depths_abs_true.keys())
    depths_true = list(m_depths_abs_true.values())
    axes[0].scatter(ns_true, depths_true, c=blue, marker="o", s=50, edgecolors="0.15", linewidth=0.8, zorder=3)
    slope, intercept, r_value, p_value, std_err = linregress(np.log(ns_true), depths_true)
    # print(intercept + slope * np.array(ns_true))
    axes[0].plot(ns_true, intercept + slope * np.log(np.array(ns_true)), c=blue, linewidth=2, linestyle="--", zorder=2, label=fr"Depth$= {slope:.1f} \ln n {intercept:.1f}$")
    axes[0].set_xlabel(r"$n$", fontsize=23)
    axes[0].set_ylabel(r"Depth", fontsize=23)
    axes[0].set_xticks([5, 7, 9, 11, 13, 15])
    axes[0].tick_params(axis='both', which='major', direction='in', labelsize=20)
    axes[0].legend(fontsize=18, loc='upper left')
    # axes[0].set_title("abs=True", fontsize=20)
    
    # abs=False
    m_depths_abs_false = minimal_depth(ns, J, h, NM, p, eps_False, abs=False)
    m_depths_abs_false[15] = 28
    ns_false = list(m_depths_abs_false.keys())
    depths_false = list(m_depths_abs_false.values())
    axes[1].scatter(ns_false, depths_false, c=green, marker="o", s=50, edgecolors="0.15", linewidth=0.8, zorder=3)
    slope, intercept, r_value, p_value, std_err = linregress(ns_false, depths_false)
    axes[1].plot(ns_false, intercept + slope * np.array(ns_false), c=green, linewidth=2, linestyle="--", zorder=2, label=fr"Depth$= {slope:.1f}n {intercept:.1f}$")
    axes[1].set_xlabel(r"$n$", fontsize=23)
    # axes[1].set_ylabel(r"Depth", fontsize=23)
    axes[1].set_xticks([5, 7, 9, 11, 13, 15])
    axes[1].tick_params(axis='both', which='major', direction='in', labelsize=20)
    # axes[1].set_title("abs=False", fontsize=20)
    axes[1].legend(fontsize=18, loc='upper left')
    plt.tight_layout(w_pad=0.5)
    plt.savefig(f'figures/minimal_depth.pdf', format='pdf')
    plt.show()


def plot_Haar_N():
    N = 1000000
    ns = range(5, 21, 3)
    J = h = 1.0
    p = 0.2
    errors1 = []
    errors10 = []
    errors50 = []
    for n in ns:
        if n == 17:
            N = 15000000
            depth=50
        elif n == 20:
            N = 100000500
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    # plt.figure(figsize=(7.2, 4.6))
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
            # if n != 8 and NM < 100:
            #     continue
            # if n == 20 and (NM < 1000 or NM == 2500 or NM == 25000 or NM == 250000 or NM == 2500000 or NM == 25000000):
                # continue
            err = average_Ising(n, depth, NM, J, h, p)[0]
            error.append(err)
        error_NM[n] = error
    NM = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000])
    # NM8 = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000])
    # NM20 = np.array([1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000])
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')


    # axes[0].scatter(NM, error_NM[8], marker='o', s=50, linewidth=2, label=fr"$n = 8$", color=colors_left[0])
    # axes[0].scatter(NM, error_NM[12], marker='s', s=50, linewidth=2, label=fr"$n = 12$", color=colors_left[1])
    # axes[0].scatter(NM, error_NM[16], marker='^', s=50, linewidth=2, label=fr"$n = 16$", color=colors_left[2])
    # axes[0].scatter(NM20, error_NM[20], marker='v', s=50, linewidth=2, label=fr"$n = 20$", color=colors_left[3])   
    slope, intercept, r_value, p_value, std_err = linregress(np.log(NM[:3]), np.log(error_NM[8][:3]))
    axes[0].plot(NM, error_NM[8], marker='o', markersize=8, linewidth=2, label=fr"$n = 8$", c=colors_left[0])
    # axes[0].plot(NM[:12], np.exp(intercept + slope * np.log(NM[:12])), linestyle='--', color="0", linewidth=2)

    slope, intercept, r_value, p_value, std_err = linregress(np.log(NM[1:5]), np.log(error_NM[12][1:5]))
    axes[0].plot(NM, error_NM[12], marker='s', markersize=8, linewidth=2, label=fr"$n = 12$", c=colors_left[1])
    # axes[0].plot(NM[:12], np.exp(intercept + slope * np.log(NM[:12])), linestyle='--', color="0", linewidth=2)

    slope, intercept, r_value, p_value, std_err = linregress(np.log(NM[3:6]), np.log(error_NM[16][3:6]))
    axes[0].plot(NM, error_NM[16], marker='^', markersize=8, linewidth=2, label=fr"$n = 16$", c=colors_left[2])
    # axes[0].plot(NM[:12], np.exp(intercept + slope * np.log(NM[:12])), linestyle='--', color="0", linewidth=2)

    # slope, intercept, r_value, p_value, std_err = linregress(np.log(NM20[2:5]), np.log(error_NM[20][2:5]))

    def line_with_fixed_slope(x, b):
        return -1.5 * x + b

    (b_opt,), cov = curve_fit(line_with_fixed_slope, np.log(NM[5:10]), np.log(error_NM[20][5:10]))
    axes[0].plot(NM, error_NM[20], marker='d', markersize=8, linewidth=2, label=fr"$n = 20$", c=colors_left[3])
    axes[0].plot(NM[:15], np.exp(b_opt - 1.5 * np.log(NM[:15])), linestyle='--', color="0", linewidth=2)

    axes[0].set_xlabel(r'$N_M$', fontsize=23)
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
                    bbox_to_anchor=(0, 0))
    axes[0].text(0.85, 0.9, "(a)", transform=axes[0].transAxes, fontsize=23)


    


    cmap_orange = cm.Oranges  # 或者 cm.Oranges
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

    axes[1].set_xlabel(r"$n$", fontsize=23)
    # axes[1].set_ylabel(r'Statistical Error', fontsize=23)
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
        bbox_to_anchor=(0, 0)
        )
    # Add the first legend to the axes so it doesn't get removed
    axes[1].add_artist(leg1)

    # Create the second legend (for slopes)
    axes[1].legend([line1, line10, line50], 
                   [fr"$b = {-slope1:.2f}$", fr"$b = {-slope10:.2f}$", fr"$b = {-slope50:.2f}$"],
                   loc='upper right',       # Changed to upper right or adjust bbox
                   bbox_to_anchor=(1.03, 1.03), # Adjust position as needed
                   title=r"Error$\sim 2^{-bn}$",
                   fontsize=16,
                   title_fontsize=16,
                   frameon=True,
                   handletextpad=0.2, 
                   handlelength=1.3,
                   borderpad=0.4
                   )
    axes[1].text(0.05, 0.9, "(b)", transform=axes[1].transAxes, fontsize=23)
    axes[1].set_xticks([5, 10, 15, 20])
    axes[1].set_ylim(ymin=2e-4, ymax=2)
    # plt.ylim(ymax=2)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.tight_layout(w_pad=0.5)
    plt.savefig("figures/SE_vs_NM_and_n.pdf", format='pdf')
    plt.show()


SAMPLE_LOWER_CBNE = {12: 380, 13: 520, 14: 810, 15: 1480, 16: 2150, 17: 3500, 18: 5300, 19: 8500}
SAMPLE_LOWER_PTME = {12: 460, 13: 750, 14: 1080, 15: 1840, 16: 2900, 17: 3500, 18: 6100, 19: 13000}

def plot_sample_complexity_lower():
    blue = "#1f77b4"
    green = "#2ca02c"
    scale_factor = 200

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
    axes[0].set_xlabel(r"$n$", fontsize=23)
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
    axes[1].set_xlabel(r"$n_A$", fontsize=23)
    axes[1].set_yscale("log")
    axes[1].tick_params(axis='both', which='major', direction='in', labelsize=20)
    axes[1].legend(fontsize=18, loc='lower right')
    axes[1].text(0.05, 0.90, "(b)", transform=axes[1].transAxes, fontsize=23)
    print(f"[PTME] slope = {slope_right}")
    print(f"[PTME] intercept = {intercept_right}")
    print(f"[PTME] r_value = {r_right}")

    plt.tight_layout(w_pad=0.5)
    plt.savefig("figures/sample_lower.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    # plot(1.0, 2, 0.2)
    # plot_depth(1.0, 0.2)
    # n = 10
    # J = 1.0
    # h = 1.0
    # NM = 10**6
    # eps = 0.1
    # depths = [20, 25, 30]
    # p_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # plot_trrho3_vs_p(
    #     n=n, J=J, h=h, NM=NM,
    #     depths=depths,
    #     p_grid=p_grid,
    #     base_dir="moment-estimation/TIM_complete",
    #     err_mode="std",          # 或 "ci95" / "std"
    #     connect_points=False,
    #     outpath='./figures/thirdM_vs_p.pdf'
    # )
    # depth_saturate()
    plot_sample_complexity_lower()


