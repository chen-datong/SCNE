import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

def to_real_array(x):
    """
    Convert an array-like that may contain:
      - floats
      - complex numbers
      - strings like "(0.12-3e-17j)" or "0.12-3e-17j"
    into a float numpy array by parsing as complex then taking real part.
    """
    arr = np.asarray(x)

    # If already numeric
    if np.issubdtype(arr.dtype, np.number):
        return np.real(arr).astype(float)

    # Otherwise parse elementwise (object/string)
    out = np.empty(arr.shape, dtype=float)
    it = np.nditer(arr, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for v in it:
        s = v.item()
        # handle NaN/None
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out[it.multi_index] = np.nan
            continue

        # normalize string
        if isinstance(s, str):
            ss = s.strip()
            # remove wrapping parentheses
            if ss.startswith("(") and ss.endswith(")"):
                ss = ss[1:-1].strip()
            # parse to complex
            out[it.multi_index] = float(np.real(complex(ss)))
        else:
            # fallback: try complex conversion
            out[it.multi_index] = float(np.real(complex(s)))

    return out

def load_df(n, Nu, NM, beta, data_dir="./data", beta_fmt="{:.2f}"):
    """
    读取单个 beta 文件。beta_fmt 控制文件名里 beta 的格式：
      - "{:.2f}" -> beta0.20
      - "{}"     -> beta0.2
    """
    beta_str = beta_fmt.format(beta)
    path = os.path.join(data_dir, f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta_str}.csv")
    return pd.read_csv(path)

def load_df_h(n, Nu, NM, h, data_dir="./data", h_fmt="{:.2f}"):
    """
    读取单个 h 文件。h_fmt 控制文件名里 h 的格式：
      - "{:.2f}" -> h1.00
      - "{}"     -> h1.0
    """
    h_str = h_fmt.format(h)
    path = os.path.join(data_dir, f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_h{h_str}.csv")
    return pd.read_csv(path)

def load_truth_arrays(N, J, h, truth_dir="./data"):
    # 注意：这里的文件名用的是你保存时的格式（J/h 不做格式化）
    f0 = os.path.join(truth_dir, f"tfim_without_cooling_N{N}_J{J}_h{h}_detail.npy")
    f2 = os.path.join(truth_dir, f"tfim_cooling_second_N{N}_J{J}_h{h}_detail.npy")
    f3 = os.path.join(truth_dir, f"tfim_cooling_third_N{N}_J{J}_h{h}_detail.npy")
    return np.load(f0), np.load(f2), np.load(f3)

def load_truth_arrays_h(N, J, beta, truth_dir="./data"):
    # 注意：这里的文件名用的是你保存时的格式（J/h 不做格式化）
    f0 = os.path.join(truth_dir, f"tfim_without_cooling_N{N}_J{J}_beta{beta}_detail.npy")
    f2 = os.path.join(truth_dir, f"tfim_cooling_second_N{N}_J{J}_beta{beta}_detail.npy")
    f3 = os.path.join(truth_dir, f"tfim_cooling_third_N{N}_J{J}_beta{beta}_detail.npy")
    return np.load(f0), np.load(f2), np.load(f3)

def mean_and_err(x, err_mode="sem"):
    """
    x: 1D array
    err_mode: "sem" | "ci95" | "std"
    """
    x = to_real_array(x)
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if err_mode == "std":
        e = s
    else:
        sem = s / np.sqrt(len(x)) if len(x) > 1 else 0.0
        e = sem if err_mode == "sem" else 1.96 * sem
    return m, e

def plot_virtual_cooling_fidelity_scatter(
    # CSV side
    n, Nu, NM, J,
    h_list, h,
    beta_list,
    data_dir="./data",
    h_fmt="{:.2f}",
    beta_fmt="{:.1f}",
    truth_dir="./data",
    # error bars
    err_mode="std",   # "sem" | "ci95" | "std"
    outpath=None,
    strict_inputs=True,
    expected_trials=100,
):
    beta_list = [float(b) for b in beta_list]

    # === load estimates across beta ===
    betas_est2, mean2, err2 = [], [], []
    betas_est3, mean3, err3 = [], [], []

    missing_beta = []
    for beta in beta_list:
        try:
            df = load_df(n, Nu, NM, beta, data_dir=data_dir, beta_fmt=beta_fmt)
        except FileNotFoundError:
            missing_beta.append(beta)
            continue
        if len(df) != expected_trials:
            raise ValueError(
                f"Expected {expected_trials} trials for beta={beta}; found {len(df)}."
            )

        m, e = mean_and_err(df["pf_rhoO2_fidelity"].to_numpy(), err_mode=err_mode)
        betas_est2.append(beta); mean2.append(m); err2.append(e)

        m, e = mean_and_err(df["pf_rhoO3_fidelity"].to_numpy(), err_mode=err_mode)
        betas_est3.append(beta); mean3.append(m); err3.append(e)

    if strict_inputs and missing_beta:
        raise FileNotFoundError(
            f"Missing virtual-cooling data for beta={missing_beta}, "
            f"n={n}, N_U={Nu}, N_M={NM}."
        )

    # === load truth arrays (must align with beta_list order!) ===
    # 你保存的 npy 是单个 beta 一个文件；这里逐个 beta 读出标量真值
    truth_beta, truth0, truth2, truth3 = [], [], [], []

    truth0, truth2, truth3 = load_truth_arrays(N=n, J=J, h=h, truth_dir=truth_dir)
    # truth0 = truth0[:6]
    # truth2 = truth2[:6]
    # truth3 = truth3[:6]

    truth_beta = np.array([0.5 + 0.05 * i for i in range(31)])


    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)

    common_scat_style = {
        "s": 28,               # 相当于 markersize ≈ 5.3
        "edgecolors": "white", # 白色描边，把点和背景线分开
        "linewidths": 0.6,     # 描边稍微细一点，配合小点
        "zorder": 11
    }

    # === Error bar 的帽子宽度也可以微调 ===
    common_err_style = {
        "fmt": "none", 
        "elinewidth": 1.2,     # 线条稍微细一点
        "capsize": 2.5,        # 帽子再小一点
        "capthick": 1.2, 
        "zorder": 10
    }

    # truth as scatter (no lines)
    axes[0].plot(truth_beta, truth0, color="0.8", lw=2, ls='-.', label=r"No cooling")
    # axes[0].plot(truth_beta, truth2, color="0.5", lw=2, ls='--', label=r"Cooling, $t=2$")
    # axes[0].plot(truth_beta, truth3, color="0.05", lw=2, label=r"Cooling, $t=3$")

    axes[0].plot(truth_beta, truth0, color="0.8", lw=2, ls='-.')
    axes[0].plot(truth_beta, truth2, color="0.5", lw=2, ls='--')
    axes[0].plot(truth_beta, truth3, color="0.05", lw=2)

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

    blue = "#0072B2"
    orange = "#ff7f0e"
    light_blue = lighten(blue, 0.20)
    light_orange = lighten(orange, 0.20)

    axes[0].errorbar(betas_est2, mean2, yerr=err2,  fmt="o", markersize=8,
                mfc=orange, mec="0.15", mew=0.5,      
                ecolor=light_orange, elinewidth=2, capsize=6, capthick=10,
                alpha=1.0, 
                label=fr"CBNE, $t=2$")
    axes[0].errorbar(betas_est3, mean3, yerr=err3,
                fmt="s", markersize=8,
                mfc=blue, mec="0.15", mew=0.5,
                ecolor=light_blue, elinewidth=2, capsize=6, capthick=10,
                linestyle="none",
                label=fr"CBNE, $t=3$")

    axes[0].set_xlabel(r"$\beta$", fontsize=23)
    axes[0].set_ylabel(r"$\langle \text{gs}|\rho^t|\text{gs}\rangle/\mathrm{tr}(\rho^t)$", fontsize=23)
    axes[0].set_xticks(beta_list)
    axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylim(-0.02, 1.1)
    # axes[0].set_title(fr"Virtual cooling fidelity (n={n}, J={J}, h={h})")
    # axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    # axes[0].legend(
    #     loc="upper left",   
    #     ncol=2,             
    #     frameon=True,
    #     fontsize=16.5,
    #     borderpad=0.3,       
    #     labelspacing=0.4,   
    #     columnspacing=0.5,   
    #     handletextpad=0.3,   
    #     handlelength=1.4,
    #     bbox_to_anchor=(-0.01, 1.02),   
    # )
    axes[0].tick_params(axis='both', which='both', direction='in', labelsize=20)
    axes[0].text(0.8, 0.9, "(a)", transform=axes[0].transAxes, fontsize=23)

    h_list = [float(b) for b in h_list]

    # === load estimates across beta ===
    hs_est2, mean2, err2 = [], [], []
    hs_est3, mean3, err3 = [], [], []

    missing_h = []
    for ht in h_list:
        try:
            if ht == 0.5 or ht == 1.0 or ht == 1.5:
                h_fmt_local = "{:.1f}"
            else:
                h_fmt_local = h_fmt
            df = load_df_h(n, Nu, NM, ht, data_dir=data_dir, h_fmt=h_fmt_local)
        except FileNotFoundError:
            missing_h.append(ht)
            continue
        if len(df) != expected_trials:
            raise ValueError(
                f"Expected {expected_trials} trials for h={ht}; found {len(df)}."
            )

        m, e = mean_and_err(df["pf_rhoO2_fidelity"].to_numpy(), err_mode=err_mode)
        hs_est2.append(ht); mean2.append(m); err2.append(e)

        m, e = mean_and_err(df["pf_rhoO3_fidelity"].to_numpy(), err_mode=err_mode)
        hs_est3.append(ht); mean3.append(m); err3.append(e)

    if strict_inputs and missing_h:
        raise FileNotFoundError(
            f"Missing virtual-cooling data for h={missing_h}, "
            f"n={n}, N_U={Nu}, N_M={NM}."
        )

    # === load truth arrays (must align with beta_list order!) ===
    truth_h, truth0, truth2, truth3 = [], [], [], []
    truth0, truth2, truth3 = load_truth_arrays_h(N=n, J=J, beta=1.0, truth_dir=truth_dir)

    truth_h = np.array([0.5 + 0.05 * i for i in range(21)])

    axes[1].plot(truth_h, truth0, color="0.8", lw=2, ls='-.', label=r"No cooling")
    axes[1].plot(truth_h, truth2, color="0.5", lw=2, ls='--', label=r"Cooling, $t=2$")
    axes[1].plot(truth_h, truth3, color="0.05", lw=2, label=r"Cooling, $t=3$")

    axes[1].errorbar(hs_est2, mean2, yerr=err2,  fmt="o", markersize=8,
                mfc=orange, mec="0.15", mew=0.5,      
                ecolor=light_orange, elinewidth=2, capsize=6, capthick=10,
                alpha=1.0, 
                label=fr"CBNE ($t=2$)")
    axes[1].errorbar(hs_est3, mean3, yerr=err3,
                fmt="s", markersize=8,
                mfc=blue, mec="0.15", mew=0.5,
                ecolor=light_blue, elinewidth=2, capsize=6, capthick=10,
                linestyle="none",
                label=fr"CBNE ($t=3$)")
    axes[1].set_xlabel(r"$h$", fontsize=23)

    # axes[1].set_ylabel(r"$\langle \text{gs}|\rho^t|\text{gs}\rangle/\mathrm{tr}(\rho^t)$", fontsize=23)
    axes[1].set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    axes[1].tick_params(axis='both', which='both', direction='in', labelsize=20)
    axes[1].text(0.08, 0.9, "(b)", transform=axes[1].transAxes, fontsize=23)


    fig.legend(
        handles, labels, 
        loc="upper left",   
        ncol=1,             
        frameon=True,
        fontsize=16,
        borderpad=0.3,       
        labelspacing=0.4,   
        columnspacing=0.5,   
        handletextpad=0.3,   
        handlelength=1.25,
        bbox_to_anchor=(0.135, 0.985),   
    )
    plt.tight_layout(w_pad=0.05)

    if outpath:
        plt.savefig(outpath)
    plt.show()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Regenerate the virtual-cooling figure using manuscript parameters."
    ).parse_args()
    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]["vcooling.pdf"]
    n = params["n"]
    NM = params["N_M"]
    Nu = params["N_U"]
    beta_list = params["left"]["beta"]

    plot_virtual_cooling_fidelity_scatter(
        n=n,
        Nu=Nu,
        NM=NM,
        J=params["J"],
        h=params["left"]["h"],
        h_list=params["right"]["h"],
        beta_list=beta_list,
        data_dir="./data",
        h_fmt="{:.2f}",
        beta_fmt="{:.1f}",   
        err_mode=params["error_band"],
        expected_trials=params["trials"],
        outpath='./figure/vcooling.pdf'
    )
