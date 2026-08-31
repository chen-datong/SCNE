import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import ticker
from scipy.stats import linregress

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
    f0 = os.path.join(truth_dir, f"tfim_without_cooling_N{N}_J{J}_h{h}.npy")
    f2 = os.path.join(truth_dir, f"tfim_cooling_second_N{N}_J{J}_h{h}.npy")
    f3 = os.path.join(truth_dir, f"tfim_cooling_third_N{N}_J{J}_h{h}.npy")
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
):
    beta_list = [float(b) for b in beta_list]

    # === load estimates across beta ===
    betas_est2, mean2, err2 = [], [], []
    betas_est3, mean3, err3 = [], [], []

    for beta in beta_list:
        try:
            df = load_df(n, Nu, NM, beta, data_dir=data_dir, beta_fmt=beta_fmt)
        except FileNotFoundError:
            continue

        m, e = mean_and_err(df["pf_rhoO2_fidelity"].to_numpy(), err_mode=err_mode)
        betas_est2.append(beta); mean2.append(m); err2.append(e)

        m, e = mean_and_err(df["pf_rhoO3_fidelity"].to_numpy(), err_mode=err_mode)
        betas_est3.append(beta); mean3.append(m); err3.append(e)

    # === load truth arrays (must align with beta_list order!) ===
    # 你保存的 npy 是单个 beta 一个文件；这里逐个 beta 读出标量真值
    truth_beta, truth0, truth2, truth3 = [], [], [], []

    truth0, truth2, truth3 = load_truth_arrays(N=n, J=J, h=h, truth_dir=truth_dir)
    truth0 = truth0[:6]
    truth2 = truth2[:6]
    truth3 = truth3[:6]

    truth_beta = np.array(beta_list)


    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)

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
                mfc=orange, mec=orange, mew=0.5,      
                ecolor=orange, elinewidth=2, capsize=6, capthick=10,
                alpha=1.0, 
                label=fr"CBNE, $t=2$")
    axes[0].errorbar(betas_est3, mean3, yerr=err3,
                fmt="s", markersize=8,
                mfc=blue, mec=blue, mew=0.5,
                ecolor=blue, elinewidth=2, capsize=6, capthick=10,
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

    h_list = [float(b) for b in h_list]

    # === load estimates across beta ===
    hs_est2, mean2, err2 = [], [], []
    hs_est3, mean3, err3 = [], [], []

    for ht in h_list:
        try:
            if ht == 0.5 or ht == 1.0 or ht == 1.5:
                h_fmt_local = "{:.1f}"
            else:
                h_fmt_local = h_fmt
            df = load_df_h(n, Nu, NM, ht, data_dir=data_dir, h_fmt=h_fmt_local)
        except FileNotFoundError:
            continue

        m, e = mean_and_err(df["pf_rhoO2_fidelity"].to_numpy(), err_mode=err_mode)
        hs_est2.append(ht); mean2.append(m); err2.append(e)

        m, e = mean_and_err(df["pf_rhoO3_fidelity"].to_numpy(), err_mode=err_mode)
        hs_est3.append(ht); mean3.append(m); err3.append(e)

    # === load truth arrays (must align with beta_list order!) ===
    truth_h, truth0, truth2, truth3 = [], [], [], []
    truth0, truth2, truth3 = load_truth_arrays_h(N=n, J=J, beta=1.0, truth_dir=truth_dir)

    truth_h = np.array([0.5 + 0.05 * i for i in range(21)])

    axes[1].plot(truth_h, truth0, color="0.8", lw=2, ls='-.', label=r"No cooling")
    axes[1].plot(truth_h, truth2, color="0.5", lw=2, ls='--', label=r"Cooling, $t=2$")
    axes[1].plot(truth_h, truth3, color="0.05", lw=2, label=r"Cooling, $t=3$")

    axes[1].errorbar(hs_est2, mean2, yerr=err2,  fmt="o", markersize=8,
                mfc=orange, mec=orange, mew=0.5,      
                ecolor=orange, elinewidth=2, capsize=6, capthick=10,
                alpha=1.0, 
                label=fr"CBNE ($t=2$)")
    axes[1].errorbar(hs_est3, mean3, yerr=err3,
                fmt="s", markersize=8,
                mfc=blue, mec=blue, mew=0.5,
                ecolor=blue, elinewidth=2, capsize=6, capthick=10,
                linestyle="none",
                label=fr"CBNE ($t=3$)")
    axes[1].set_xlabel(r"$h$", fontsize=23)

    # axes[1].set_ylabel(r"$\langle \text{gs}|\rho^t|\text{gs}\rangle/\mathrm{tr}(\rho^t)$", fontsize=23)
    axes[1].set_xticks(h_list)
    axes[1].tick_params(axis='both', which='both', direction='in', labelsize=20)
    


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
    plt.tight_layout(w_pad=0.2)

    if outpath:
        plt.savefig(outpath)
    plt.show()


def _compute_exact_rho3O(n, beta, observable):
    """
    Compute exact rho^3 O trace for given observable:
      - "X1toXn": tensor product of X on all sites
      - "Z1Z2": Z on first two sites, I elsewhere
      - "psib": |psi0><psi0| - I/d  (psi0 ground state at J=1,h=1)
      - "Hising": TFIM Hamiltonian with J=1,h=1
    """
    # 1) Try cache first
    cache_path = os.path.join("./data", f"exact_rho3O_n{n}_beta{beta}.csv")
    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            if {"observable", "value"} <= set(df_cache.columns):
                sub = df_cache[df_cache["observable"] == observable]
                if not sub.empty:
                    return float(sub.iloc[0]["value"])
        except Exception:
            pass

    d = 1 << n
    rho = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    rho3 = rho @ rho @ rho
    I2 = np.eye(2, dtype=np.complex128)

    if observable == "X1toXn":
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        O = np.eye(1, dtype=np.complex128)
        for _ in range(n):
            O = np.kron(O, X)
    elif observable == "Z1Z2":
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        O = np.eye(1, dtype=np.complex128)
        for i in range(n):
            if i in (0, 1):
                O = np.kron(O, Z)
            else:
                O = np.kron(O, I2)
    elif observable == "psib":
        psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h1.0.npy')
        psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
        O = psib
    elif observable == "Hising":
        # Build dense TFIM Hamiltonian
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        # H = -J sum ZZ - h sum X with J=h=1
        H = np.zeros((d, d), dtype=np.complex128)
        for i in range(n - 1):
            term = np.eye(1, dtype=np.complex128)
            for site in range(n):
                if site == i or site == i + 1:
                    term = np.kron(term, Z)
                else:
                    term = np.kron(term, I2)
            H += -1.0 * term
        for i in range(n):
            term = np.eye(1, dtype=np.complex128)
            for site in range(n):
                if site == i:
                    term = np.kron(term, X)
                else:
                    term = np.kron(term, I2)
            H += -1.0 * term
        O = H
    else:
        raise ValueError(f"Unknown observable: {observable}")

    val = np.trace(rho3 @ O)
    return float(np.real(val))


def compute_and_save_exact_rho3O(n, beta, outpath=None):
    """
    Compute exact values for all supported observables and save to CSV.
    CSV columns: observable, value
    Default outpath: ./data/exact_rho3O_n{n}_beta{beta}.csv
    """
    obs_list = ["X1toXn", "Z1Z2", "psib", "Hising"]
    rows = []
    for obs in obs_list:
        v = _compute_exact_rho3O(n, beta, obs)
        rows.append({"observable": obs, "value": v})
    df = pd.DataFrame(rows)
    if outpath is None:
        outpath = os.path.join("./data", f"exact_rho3O_n{n}_beta{beta}.csv")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    return df

def _collect_mae_for_observable(
    n,
    Nu,
    beta,
    observable,
    anc_list,
    nm_list,
    data_dir="./data",
    expected_trials=1000,
):
    """
    Load per-(anc,NM) CSVs and compute MAE of rho3O estimates vs exact.
    Columns expected:
      - For X1toXn, Z1Z2: gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv
          pf_rhoO3_X1toXn / pf_rhoO3_Z1Z2, third_moment
      - For psib, Hising: gibbs_psibH_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv
          pf_rhoO3_psib / pf_rhoO3_Hising, third_moment
    We reconstruct rho3O_est = pf_rhoO3_* * third_moment, then MAE over repetitions.
    """
    # Map observable to column and filename stem
    if observable in ("X1toXn", "Z1Z2"):
        stem = "gibbs_pauli_anc"
        col_map = {"X1toXn": "pf_rhoO3_X1toXn", "Z1Z2": "pf_rhoO3_Z1Z2"}
    elif observable in ("psib", "Hising"):
        stem = "gibbs_psibH_anc"
        col_map = {"psib": "pf_rhoO3_psib", "Hising": "pf_rhoO3_Hising"}
    else:
        raise ValueError(f"Unknown observable: {observable}")

    exact_val = _compute_exact_rho3O(n, beta, observable)
    rows = []
    beta_str = f"{beta}"
    # Files use beta as raw (e.g., 2.0). Keep both "2.0" and "2" attempts.
    beta_candidates = [f"{beta}", f"{float(beta):.1f}", f"{float(beta):.2f}"]

    def block_mean_mae(values: np.ndarray, exact_val: float, block_size: int) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size < block_size:
            return np.nan
        usable = (arr.size // block_size) * block_size
        arr_block = arr[:usable].reshape(-1, block_size).mean(axis=1)
        return float(np.mean(np.abs(arr_block - exact_val)))

    for anc in anc_list:
        for NM in nm_list:
            df = None
            for bstr in beta_candidates:
                path = os.path.join(
                    data_dir,
                    f"{stem}_n{n}_Nu{Nu}_NM{int(NM)}_beta{bstr}_anc{int(anc)}.csv",
                )
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            if df is None:
                continue
            if expected_trials is not None and len(df) != expected_trials:
                raise ValueError(
                    f"Expected {expected_trials} trials in {path}; found {len(df)}."
                )
            col = col_map[observable]
            if col not in df.columns or "third_moment" not in df.columns:
                continue
            vals = to_real_array(df[col].to_numpy())
            third = to_real_array(df["third_moment"].to_numpy())
            est = vals * third
            mae = float(np.mean(np.abs(est - exact_val)))
            mae_avg10 = block_mean_mae(est, exact_val, 10)
            mae_avg50 = block_mean_mae(est, exact_val, 50)
            rows.append({"anc": int(anc), "NM": int(NM), "MAE": mae, "MAE_avg10": mae_avg10, "MAE_avg50": mae_avg50})
    if not rows:
        return pd.DataFrame(columns=["anc", "NM", "MAE", "MAE_avg10", "MAE_avg50"])
    return pd.DataFrame(rows)


def plot_four_observables_stat_error(
    n,
    Nu,
    beta,
    anc_left=(0, 4, 8),
    anc_right=tuple(range(0, 9)),
    nm_grid=(100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000),
    nm_right=1000000,
    data_dir="./data",
    out_dir="./figure",
):
    """
    Make four two-panel figures for observables: X1toXn, Z1Z2, psib, Hising.
    Left: MAE vs NM at fixed n (multiple n_a).
    Right: MAE vs n_a at fixed (n, NM), with exponential fit.
    """
    os.makedirs(out_dir, exist_ok=True)
    observables = ["X1toXn", "Z1Z2", "psib", "Hising"]
    for obs in observables:
        df = _collect_mae_for_observable(
            n=n, Nu=Nu, beta=beta, observable=obs, anc_list=anc_right, nm_list=nm_grid, data_dir=data_dir
        )
        if df.empty:
            print(f"[WARN] No data found for observable={obs}. Skipping.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
        # Left: MAE vs NM for selected anc
        cmap_blue = cm.Blues
        colors_left = cmap_blue(np.linspace(0.2, 0.9, max(len(anc_left), 1)))
        markers = ["o", "s", "^", "d", "v", "P", "X"]
        for i, anc in enumerate(anc_left):
            sub = df[df["anc"] == int(anc)].sort_values("NM")
            if sub.empty:
                continue
            axes[0].plot(
                sub["NM"],
                sub["MAE"],
                marker=markers[i % len(markers)],
                markersize=7,
                linewidth=2,
                color=colors_left[i % len(colors_left)],
                label=fr"$n_a = {int(anc)}$",
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"$N_M$", fontsize=23)
        axes[0].set_ylabel("Statistical Error", fontsize=23)
        axes[0].set_ylim(bottom=5e-3, top=3e0)
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

        # Right: MAE vs anc at fixed NM
        sub_nm = df[df["NM"] == int(nm_right)].sort_values("anc")
        anc_vals = sub_nm["anc"].to_numpy(dtype=float)
        mae_vals = sub_nm["MAE"].to_numpy(dtype=float)
        mae10_vals = sub_nm["MAE_avg10"].to_numpy(dtype=float)
        mae50_vals = sub_nm["MAE_avg50"].to_numpy(dtype=float)
        cmap_orange = cm.Oranges
        right_series = [
            (r"$N_U=1$", mae_vals, "o", cmap_orange(0.55)),
            (r"$N_U=10$", mae10_vals, "s", cmap_orange(0.72)),
            (r"$N_U=50$", mae50_vals, "^", cmap_orange(0.90)),
        ]
        point_handles, point_labels, fit_lines, fit_labels = [], [], [], []
        for name, vals, marker, color in right_series:
            mask = np.isfinite(vals) & (vals > 0)
            if np.sum(mask) < 2:
                continue
            slope, intercept, *_ = linregress(anc_vals[mask], np.log2(vals[mask]))
            fit_vals = np.exp2(intercept + slope * anc_vals[mask])
            b_val = -slope
            line_fit, = axes[1].plot(
                anc_vals[mask],
                fit_vals,
                linestyle="--",
                linewidth=2,
                color=color,
            )
            pt = axes[1].scatter(
                anc_vals[mask],
                vals[mask],
                marker=marker,
                s=55,
                linewidth=1.2,
                color=color,
                label=name,
            )
            point_handles.append(pt)
            point_labels.append(name)
            fit_lines.append(line_fit)
            fit_labels.append(fr"$b = {b_val:.2f}$")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$n_a$", fontsize=23)
        if anc_vals.size:
            axes[1].set_xticks(anc_vals.astype(int))
        axes[1].tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
        if point_handles:
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
        if fit_lines:
            axes[1].legend(
                fit_lines,
                fit_labels,
                loc="upper right",
                bbox_to_anchor=(1.02, 1.03),
                title=r"Error$\sim 2^{-b n_a}$",
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
        outpath = os.path.join(out_dir, f"SE_vs_NM_and_n_like_{obs}.pdf")
        plt.savefig(outpath, format="pdf")
        plt.show()


def plot_four_observables_stat_error_2x4(
    n,
    Nu,
    beta,
    anc_left=(0, 4, 8),
    anc_right=tuple(range(0, 9)),
    nm_grid=(100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000),
    nm_right=1000000,
    data_dir="./data",
    outpath="./figure/SE_2x3_all_observables.pdf",
    top_ylim=(5e-3, 2e0),
    fig_size=(12.0, 8.0),
):
    """
    Compose a 2x3 figure:
    - Row 1: left panels (MAE vs NM) for [X1...Xn, Z1Z2, psib]
    - Row 2: right panels (MAE vs n_a) for [X1...Xn, Z1Z2, psib]
    Keep plotting style consistent with `plot_four_observables_stat_error`.
    """
    observables = ["X1toXn", "Z1Z2", "psib"]
    title_map = {
        "X1toXn": rf"$X_1X_2\cdots X_{{{n}}}$",
        "Z1Z2": r"$Z_1Z_2$",
        "psib": r"$|\mathrm{gs}\rangle\langle\mathrm{gs}|$",
    }

    # Collect data first so we fail early if everything is missing.
    df_by_obs = {}
    for obs in observables:
        df_obs = _collect_mae_for_observable(
            n=n, Nu=Nu, beta=beta, observable=obs, anc_list=anc_right, nm_list=nm_grid, data_dir=data_dir
        )
        if not df_obs.empty:
            df_by_obs[obs] = df_obs

    if not df_by_obs:
        raise FileNotFoundError("No valid data found for all observables.")

    fig, axes = plt.subplots(2, 3, figsize=fig_size, sharey="row")

    cmap_blue = cm.Blues
    cmap_orange = cm.Oranges
    markers = ["o", "s", "^", "d", "v", "P", "X"]
    colors_left = cmap_blue(np.linspace(0.2, 0.9, max(len(anc_left), 1)))
    right_series_style = [
        (r"$N_U=1$", "MAE", "o", cmap_orange(0.55)),
        (r"$N_U=10$", "MAE_avg10", "s", cmap_orange(0.72)),
        (r"$N_U=50$", "MAE_avg50", "^", cmap_orange(0.90)),
    ]
    math_formatter = ticker.LogFormatterMathtext(base=10)

    for col, obs in enumerate(observables):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        df = df_by_obs.get(obs)

        # If some observable data is missing, keep the slot but mark it.
        if df is None or df.empty:
            ax_top.axis("off")
            ax_bot.axis("off")
            continue

        # ----- top row: MAE vs NM -----
        for i, anc in enumerate(anc_left):
            sub = df[df["anc"] == int(anc)].sort_values("NM")
            if sub.empty:
                continue
            ax_top.plot(
                sub["NM"],
                sub["MAE"],
                marker=markers[i % len(markers)],
                markersize=7,
                linewidth=2,
                color=colors_left[i % len(colors_left)],
                label=fr"$n_a = {int(anc)}$",
            )
        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        ax_top.set_ylim(bottom=float(top_ylim[0]), top=float(top_ylim[1]))
        ax_top.set_xlabel(r"$N_M$", fontsize=23)
        if col == 0:
            ax_top.set_ylabel("Statistical Error", fontsize=23)
        ax_top.tick_params(axis="both", which="both", direction="in", labelsize=19, pad=6)
        ax_top.set_title(title_map.get(obs, obs), fontsize=24, pad=8)
        ax_top.yaxis.set_major_formatter(math_formatter)
        ax_top.yaxis.set_minor_locator(ticker.NullLocator())

        if col == 0:
            ax_top.legend(
                loc="lower left",
                ncol=1,
                frameon=True,
                fontsize=15,
                borderpad=0.3,
                labelspacing=0.35,
                columnspacing=0.5,
                handletextpad=0.3,
                handlelength=1.2,
            )

        # ----- bottom row: MAE vs n_a with fits -----
        sub_nm = df[df["NM"] == int(nm_right)].sort_values("anc")
        anc_vals = sub_nm["anc"].to_numpy(dtype=float)
        ax_bot.set_xticks([0, 2, 4, 6, 8])

        point_handles, point_labels = [], []
        fit_lines, fit_labels = [], []
        for name, col_name, marker, color in right_series_style:
            vals = sub_nm[col_name].to_numpy(dtype=float) if col_name in sub_nm.columns else np.array([])
            mask = np.isfinite(vals) & (vals > 0)
            if np.sum(mask) < 2:
                continue
            slope, intercept, *_ = linregress(anc_vals[mask], np.log2(vals[mask]))
            fit_vals = np.exp2(intercept + slope * anc_vals[mask])
            b_val = -slope
            line_fit, = ax_bot.plot(
                anc_vals[mask],
                fit_vals,
                linestyle="--",
                linewidth=2,
                color=color,
            )
            pt = ax_bot.scatter(
                anc_vals[mask],
                vals[mask],
                marker=marker,
                s=55,
                linewidth=1.2,
                color=color,
                label=name,
            )
            point_handles.append(pt)
            point_labels.append(name)
            fit_lines.append(line_fit)
            fit_labels.append(fr"$b = {b_val:.2f}$")

        ax_bot.set_yscale("log")
        ax_bot.set_xlabel(r"$n_a$", fontsize=23)
        if col == 0:
            ax_bot.set_ylabel("Statistical Error", fontsize=23)
        ax_bot.tick_params(axis="both", which="both", direction="in", labelsize=19, pad=6)
        ax_bot.yaxis.set_major_formatter(math_formatter)
        ax_bot.yaxis.set_minor_locator(ticker.NullLocator())

        if col == 0 and point_handles:
            leg_pts = ax_bot.legend(
                point_handles,
                point_labels,
                loc="lower left",
                ncol=1,
                frameon=True,
                fontsize=15,
                borderpad=0.3,
                labelspacing=0.35,
                columnspacing=0.5,
                handletextpad=0.3,
                handlelength=1.2,
                bbox_to_anchor=(0, 0),
            )
            ax_bot.add_artist(leg_pts)
        if fit_lines:
            ax_bot.legend(
                fit_lines,
                fit_labels,
                loc="upper right",
                bbox_to_anchor=(1.02, 1.03),
                title=r"Error$\sim 2^{-b n_a}$",
                fontsize=14,
                title_fontsize=14,
                frameon=True,
                handletextpad=0.2,
                handlelength=1.2,
                borderpad=0.35,
            )
        ax_bot.set_ylim(4e-3, 1e0)

    plt.tight_layout(w_pad=0.6, h_pad=0.5)
    if outpath:
        outdir = os.path.dirname(outpath)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(outpath, format="pdf")
    plt.show()
def plot_anc(n, NU, beta):
    df = pd.read_csv(f'./data/statistical_error_n{n}_Nu{NU}_beta{beta}.csv')
    required_cols = {"anc", "NM", "MAE_Z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in csv: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    anc_values = sorted(df["anc"].unique())
    plotted_anc = anc_values[::2]
    colors_left = cm.Blues(np.linspace(0.2, 0.9, len(plotted_anc)))

    for i, anc in enumerate(plotted_anc):
        sub = df[df["anc"] == anc].sort_values("NM")
        ax.plot(
            sub["NM"],
            sub["MAE_X"],
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors_left[i],
            label=f"anc={anc}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NM", fontsize=14)
    ax.set_ylabel("MAE_X", fontsize=14)
    # Tighten y-range to highlight the main trend and reduce outlier stretching.
    y_lower = max(float(df["MAE_X"].min()) * 0.9, 1e-4)
    y_upper = float(df["MAE_X"].quantile(0.95)) * 1.1
    ax.set_ylim(6e-3, 2e0)
    ax.set_title(f"MAE_X vs NM (n={n}, Nu={NU}, beta={beta})", fontsize=14)
    ax.legend(fontsize=10, ncol=2)

    # Inset: show exponential-like decay with anc at large NM.
    top_nms = sorted(df["NM"].unique())[-3:]
    df_large = df[df["NM"].isin(top_nms)]
    anc_decay = df_large.groupby("anc", as_index=False)["MAE_X"].mean().sort_values("anc")
    anc_arr = anc_decay["anc"].to_numpy(dtype=float)
    mae_arr = anc_decay["MAE_X"].to_numpy(dtype=float)

    if len(anc_arr) >= 2 and np.all(mae_arr > 0):
        coef = np.polyfit(anc_arr, np.log(mae_arr), 1)
        fit_arr = np.exp(coef[1] + coef[0] * anc_arr)

        axins = ax.inset_axes([0.56, 0.56, 0.4, 0.38])
        axins.plot(anc_arr, mae_arr, "o-", color=cm.Blues(0.75), lw=1.8, ms=4, label="large NM avg")
        axins.plot(anc_arr, fit_arr, "--", color="0.25", lw=1.5, label="exp fit")
        axins.set_yscale("log")
        axins.set_xlabel("anc", fontsize=9)
        axins.set_ylabel("MAE_X", fontsize=9)
        axins.tick_params(axis="both", labelsize=8, direction="in")
        axins.set_title("Large NM decay", fontsize=9)
        axins.legend(fontsize=7, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_statistical_error_two_panel(
    n_list=None,
    Nu=1,
    beta=2.0,
    observable="X",
    observables=None,
    n_left=None,
    anc_left=(0, 4, 8),
    n_right=None,
    anc_right=(0, 1, 2, 3, 4, 5, 6, 7),
    nm_right=1000000,
    outpath="./figure/SE_vs_NM_and_n_like.pdf",
):
    """
    Two-panel plot similar to reference style:
    (a) MAE_{observable} vs NM at fixed n for multiple n_a;
    (b) MAE_{observable} vs n_a at fixed (n, NM).

    If `observables` is provided (e.g. ("X", "Z")), only panel (b) compares
    these observables together; panel (a) keeps a single baseline observable.
    """
    if observables is None:
        obs_list = [str(observable).upper()]
    else:
        obs_list = [str(obs).upper() for obs in observables]
        if len(obs_list) == 0:
            raise ValueError("`observables` cannot be empty.")

    if n_list is None:
        discovered = []
        suffix = f"_Nu{Nu}_beta{beta}.csv"
        for fname in os.listdir("./data"):
            if fname.startswith("statistical_error_n") and fname.endswith(suffix):
                n_str = fname[len("statistical_error_n") : -len(suffix)]
                if n_str.isdigit():
                    discovered.append(int(n_str))
        n_list = sorted(discovered)
    else:
        n_list = sorted(int(x) for x in n_list)

    if len(n_list) == 0:
        raise ValueError("No valid n values found. Please provide n_list or prepare statistical_error files.")
    if n_left is None:
        n_left = n_list[0]
    if n_right is None:
        n_right = n_left

    # ---------- Load all available statistical-error files ----------
    data_by_n = {}
    for n in n_list:
        path = f"./data/statistical_error_n{n}_Nu{Nu}_beta{beta}.csv"
        if os.path.exists(path):
            data_by_n[n] = pd.read_csv(path)

    if n_left not in data_by_n:
        raise FileNotFoundError(
            f"Left panel data not found: ./data/statistical_error_n{n_left}_Nu{Nu}_beta{beta}.csv"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)

    # ---------- (a) MAE_obs vs NM for selected n_a ----------
    cmap_blue = cm.Blues
    colors_left = cmap_blue(np.linspace(0.2, 0.9, max(len(anc_left), 1)))
    markers = ["o", "s", "^", "d", "v", "P", "X"]

    df_left = data_by_n[n_left]
    left_obs = obs_list[0]
    left_col = f"MAE_{left_obs}"
    required_cols_left = {"anc", "NM", left_col}
    missing_left = required_cols_left - set(df_left.columns)
    if missing_left:
        raise KeyError(f"Missing columns in left data: {sorted(missing_left)}")

    for i, anc in enumerate(anc_left):
        sub = df_left[df_left["anc"] == anc].sort_values("NM")
        if sub.empty:
            continue
        axes[0].plot(
            sub["NM"],
            sub[left_col],
            marker=markers[i % len(markers)],
            markersize=7,
            linewidth=2,
            color=colors_left[i % len(colors_left)],
            label=fr"$n_a = {anc}$",
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$N_M$", fontsize=23)
    axes[0].set_ylabel("Statistical Error", fontsize=23)
    axes[0].set_ylim(bottom=5e-3, top=3e0)
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

    # ---------- (b) MAE vs n_a (compare observables only) ----------
    obs_colors = {
        "X": cm.Blues(0.75),
        "Z": cm.Greens(0.75),
    }
    default_obs_colors = cm.tab10(np.linspace(0.1, 0.9, max(len(obs_list), 1)))
    marker_by_obs = ["o", "s", "^", "d", "v", "P", "X"]
    anc_valid = []

    df_right = data_by_n.get(n_right)
    if df_right is None:
        raise FileNotFoundError(
            f"Right panel data not found: ./data/statistical_error_n{n_right}_Nu{Nu}_beta{beta}.csv"
        )
    required_cols_right = {"anc", "NM"}
    for obs in obs_list:
        required_cols_right.add(f"MAE_{obs}")
    missing_right = required_cols_right - set(df_right.columns)
    if missing_right:
        raise KeyError(f"Missing columns in right data: {sorted(missing_right)}")

    series_by_obs = {}
    for obs in obs_list:
        series_by_obs[obs] = []

    for anc in anc_right:
        row = df_right[(df_right["anc"] == anc) & (df_right["NM"] == nm_right)]
        if row.empty:
            continue
        anc_valid.append(int(anc))
        for obs in obs_list:
            k = f"MAE_{obs}"
            series_by_obs[obs].append(float(row.iloc[0][k]))

    lines = []
    line_labels = []
    for j, obs in enumerate(obs_list):
        obs_color = obs_colors.get(obs, default_obs_colors[j % len(default_obs_colors)])
        y = np.asarray(series_by_obs[obs], dtype=float)
        if len(anc_valid) == 0:
            continue
        legend_label = obs if len(obs_list) > 1 else left_obs
        axes[1].scatter(
            anc_valid,
            y,
            marker=marker_by_obs[j % len(marker_by_obs)],
            s=55,
            linewidth=1.2,
            color=obs_color,
            label=legend_label,
        )
        if len(anc_valid) >= 2 and np.all(y > 0):
            slope, intercept, *_ = linregress(anc_valid, np.log2(y))
            y_fit = np.exp2(intercept + slope * np.asarray(anc_valid))
            line, = axes[1].plot(
                anc_valid,
                y_fit,
                linestyle="--",
                color=obs_color,
                linewidth=2,
            )
            lines.append(line)
            if len(obs_list) > 1:
                line_labels.append(fr"${obs}: b = {-slope:.2f}$")
            else:
                line_labels.append(fr"$b = {-slope:.2f}$")

    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$n_a$", fontsize=23)
    axes[1].tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
    leg1 = axes[1].legend(
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
    axes[1].add_artist(leg1)
    if lines:
        axes[1].legend(
            lines,
            line_labels,
            loc="upper right",
            bbox_to_anchor=(1.02, 1.03),
            title=r"Error$\sim 2^{-b n_a}$",
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
    if outpath:
        plt.savefig(outpath, format="pdf")
    plt.show()

def plot_mae_vs_n(
    csv_path="./data/purity_and_third_moment_mae_list.csv",
    outpath="./figure/mae_vs_n.pdf",
    logx=True,
    logy=True,
):
    """
    Plot MAE vs N from csv with columns:
      - N
      - purity_mae
      - third_moment_mae
      - shadow_mae (optional)
    """
    df = pd.read_csv(csv_path)
    required_cols = {"N", "purity_mae", "third_moment_mae"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in csv: {sorted(missing)}")

    df = df.sort_values("N")
    n_vals = df["N"].to_numpy(dtype=float)
    purity_mae = df["purity_mae"].to_numpy(dtype=float)
    third_mae = df["third_moment_mae"].to_numpy(dtype=float)
    has_shadow = "shadow_mae" in df.columns
    shadow_mae = df["shadow_mae"].to_numpy(dtype=float) if has_shadow else None

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(
        n_vals,
        purity_mae,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=cm.Blues(0.72),
        label="Purity MAE",
    )
    ax.plot(
        n_vals,
        third_mae,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color=cm.Oranges(0.75),
        label="Third Moment MAE",
    )
    if has_shadow:
        ax.plot(
            n_vals,
            shadow_mae,
            marker="^",
            markersize=7,
            linewidth=2.2,
            color=cm.Greens(0.72),
            label="Shadow MAE",
        )

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("N", fontsize=20)
    ax.set_ylabel("MAE", fontsize=20)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=16)
    ax.legend(frameon=True, fontsize=14)

    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.show()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]["SE_all_observables.pdf"]
    # n = 14
    # NM = 1000
    # Nu=1  # 或者 [1, 5, 10]
    # beta_list = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]

    # plot_virtual_cooling_fidelity_scatter(
    #     n=n,
    #     Nu=Nu,
    #     NM=NM,
    #     J=1.0,
    #     h=1.0,
    #     h_list=[0.5, 0.75, 1.0, 1.25, 1.5],
    #     beta_list=beta_list,
    #     data_dir="./data",
    #     h_fmt="{:.2f}",
    #     beta_fmt="{:.1f}",   
    #     err_mode="std",
    #     outpath='./figure/vcooling.pdf'
    # )
    # plot_anc(6, 1, 2.0)
    # plot_statistical_error_two_panel(
    #     n_list=range(0, 9),
    #     Nu=1,
    #     beta=2.0,
    #     observables=("X", "Z"),
    #     n_left=6,
    #     anc_left=(0, 4, 8),
    #     n_right=6,
    #     anc_right=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    #     nm_right=1000000,
    #     outpath="./figure/SE_vs_NM_and_n_like_XZ.pdf",
    # )
    # compute_and_save_exact_rho3O(n=6, beta=2.0)
    # plot_four_observables_stat_error(
    # n=6,
    # Nu=1,
    # beta=2.0,
    # anc_left=(0, 4, 8),
    # anc_right=tuple(range(0, 9)),
    # nm_grid=(100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000),
    # nm_right=1000000,
    # data_dir="./data",
    # out_dir="./figure",
    # )
    # plot_mae_vs_n(
    #     csv_path="./data/purity_and_third_moment_mae_list.csv",
    #     outpath="./figure/mae_vs_n.pdf",
    #     logx=True,
    #     logy=True,
    # )
    plot_four_observables_stat_error_2x4(
    n=params["n"],
    Nu=params["left"]["N_U"],
    beta=params["beta"],
    anc_left=tuple(params["left"]["n_a"]),
    anc_right=tuple(params["right"]["n_a"]),
    nm_grid=tuple(params["left"]["N_M"]),
    nm_right=params["right"]["N_M"],
    data_dir="./data",
    outpath="./figure/SE_all_observables.pdf",
    )
