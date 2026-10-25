import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "cbne_observables"
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


def _compute_exact_rho3O(n, beta, observable):
    """
    Compute exact rho^3 O trace for given observable:
      - "X1toXn": tensor product of X on all sites
      - "Z1Z2": Z on first two sites, I elsewhere
      - "psib": |psi0><psi0| - I/d  (psi0 ground state at J=1,h=1)
      - "Hising": TFIM Hamiltonian with J=1,h=1
    """
    # 1) Try cache first
    cache_path = DATA_DIR / f"exact_rho3O_n{n}_beta{beta}.csv"
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
    rho = np.load(DATA_DIR / "TFIM" / f"tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy")
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
        psi0 = np.load(DATA_DIR / "TFIM" / f"tfim_N{n}_gs_J1.0_h1.0.npy")
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
    Default outpath: root/data/cbne_observables/exact_rho3O_n{n}_beta{beta}.csv
    """
    obs_list = ["X1toXn", "Z1Z2", "psib", "Hising"]
    rows = []
    for obs in obs_list:
        v = _compute_exact_rho3O(n, beta, obs)
        rows.append({"observable": obs, "value": v})
    df = pd.DataFrame(rows)
    if outpath is None:
        outpath = DATA_DIR / f"exact_rho3O_n{n}_beta{beta}.csv"
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
    data_dir=DATA_DIR,
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


def plot_gibbs_observables_with_rho3(params, rho3_params, data_dir=DATA_DIR,
                                    outpath=HERE.parent / "figures/SE_all_observables.pdf",
                                    fig_size=(13.0, 7.6)):
    """Draw rho^3 and the three retained Gibbs observables as a 2x4 figure."""
    from plot_rho3 import collect_errors

    for key in ("n", "J", "h", "beta"):
        if float(params[key]) != float(rho3_params[key]):
            raise ValueError(f"Gibbs target-state mismatch for {key}")
    if rho3_params.get("state", "gibbs") != "gibbs":
        raise ValueError("The observable comparison requires Gibbs rho^3 data")

    # Match the retained observable grid without changing the extended Gibbs data.
    rho3_display_params = {
        **rho3_params, "left": dict(params["left"]), "right": dict(params["right"]),
    }
    rho3_errors, _, _ = collect_errors(rho3_display_params, Path(data_dir), "rho3")
    rho3_frame = rho3_errors.rename(columns={
        "MAE_NU1": "MAE", "MAE_NU10": "MAE_avg10", "MAE_NU50": "MAE_avg50",
    })
    specifications = [{
        "name": "rho3", "title": r"$\mathbb{1}$", "frame": rho3_frame,  # \openone (identity)
        "anc_left": rho3_display_params["left"]["n_a"],
        "anc_right": rho3_display_params["right"]["n_a"],
        "nm_right": rho3_display_params["right"]["N_M"],
    }]
    titles = {
        "X1toXn": rf"$X_1X_2\cdots X_{{{params['n']}}}$",
        "Z1Z2": r"$Z_1Z_2$", "psib": r"$|\mathrm{gs}\rangle\langle\mathrm{gs}|$",
    }
    for observable, title in titles.items():
        frame = _collect_mae_for_observable(
            n=params["n"], Nu=params["left"]["N_U"], beta=params["beta"],
            observable=observable, anc_list=params["right"]["n_a"],
            nm_list=params["left"]["N_M"], data_dir=data_dir,
            expected_trials=params["trials"],
        )
        expected = len(params["right"]["n_a"]) * len(params["left"]["N_M"])
        if len(frame) != expected or not np.isfinite(frame[["MAE", "MAE_avg10", "MAE_avg50"]]).all().all():
            raise ValueError(f"Incomplete or invalid observable data for {observable}")
        specifications.append({
            "name": observable, "title": title, "frame": frame,
            "anc_left": params["left"]["n_a"], "anc_right": params["right"]["n_a"],
            "nm_right": params["right"]["N_M"],
        })

    fig, axes = plt.subplots(2, 4, figsize=fig_size, sharex="row", sharey="row")
    markers = ["o", "s", "^", "d"]
    orange_colors = cm.Oranges(np.linspace(0.4, 0.9, 3))
    formatter = ticker.LogFormatterMathtext(base=10)
    legend_options = dict(fontsize=15, frameon=True, borderpad=0.3,
                          labelspacing=0.35, handlelength=1.2, handletextpad=0.3)
    for col, spec in enumerate(specifications):
        top, bottom = axes[:, col]
        frame = spec["frame"]
        colors = cm.Blues(np.linspace(0.2, 0.9, len(spec["anc_left"])))
        for index, anc in enumerate(spec["anc_left"]):
            subset = frame.loc[frame.anc == anc].sort_values("NM")
            top.plot(subset.NM, subset.MAE, marker=markers[index], markersize=7,
                     linewidth=2, color=colors[index], label=fr"$n_a={anc}$")
        top.set_xscale("log")
        top.set_xlabel(r"$N_M$", fontsize=23, labelpad=0)
        top.set_title(spec["title"], fontsize=23, pad=8)
        top.set_xticks([1e3, 1e5, 1e7] if frame.NM.max() >= 1e7 else [1e2, 1e4, 1e6])
        top.xaxis.set_major_formatter(formatter)
        if col in (0, 1):
            top.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02), **legend_options)

        fixed = frame[(frame.NM == spec["nm_right"]) & frame.anc.isin(spec["anc_right"])].sort_values("anc")
        if fixed.anc.tolist() != spec["anc_right"]:
            raise ValueError(f"Missing fixed-NM points for {spec['name']}")
        anc_values = fixed.anc.to_numpy(dtype=float)
        point_handles, fit_handles, fit_labels = [], [], []
        for index, (nu, column) in enumerate([(1, "MAE"), (10, "MAE_avg10"), (50, "MAE_avg50")]):
            values = fixed[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"Invalid MAE for {spec['name']}, N_U={nu}")
            slope, intercept, *_ = linregress(anc_values, np.log2(values))
            line, = bottom.plot(anc_values, np.exp2(intercept + slope * anc_values),
                                "--", color=orange_colors[index], linewidth=2)
            points = bottom.scatter(anc_values, values, marker=markers[index], s=50,
                                    linewidth=1.5, color=orange_colors[index])
            fit_handles.append(line); point_handles.append(points)
            fit_labels.append(fr"$b={-slope:.2f}$")
        bottom.set_xticks(range(0, max(spec["anc_right"]) + 1, 2))
        bottom.set_xlabel(r"$n_a$", fontsize=23, labelpad=0)
        if col == 0:
            points_legend = bottom.legend(point_handles, [r"$N_U=1$", r"$N_U=10$", r"$N_U=50$"],
                                          loc="lower left", bbox_to_anchor=(-0.02, -0.02), **legend_options)
            bottom.add_artist(points_legend)
        bottom.legend(fit_handles, fit_labels, loc="upper right", bbox_to_anchor=(1.01, 1.02),
                      title=r"Error$\sim2^{-bn_a}$", title_fontsize=14, fontsize=14,
                      labelspacing=0.1, borderpad=0.3, handlelength=1.2, handletextpad=0.2)
        for axis in (top, bottom):
            axis.set_yscale("log")
            axis.tick_params(axis="both", which="both", direction="in", labelsize=19, pad=6)
            axis.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=100))
            axis.yaxis.set_major_formatter(formatter)
            axis.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0, 0].set_ylim(2e-4, 2.0)
    axes[1, 0].set_ylim(2e-4, 4.0)
    axes[0, 0].set_ylabel("Statistical Error", fontsize=23)
    axes[1, 0].set_ylabel("Statistical Error", fontsize=23)
    fig.tight_layout(w_pad=-0.5, h_pad=0.4)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, format="pdf")
    plt.close(fig)
