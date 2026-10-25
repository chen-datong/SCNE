from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ptme"
import pandas as pd
import numpy as np
from utils import D3, D4, D5
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'


def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["nA", "nB", "Nu", "NM"]:
        df[c] = df[c].astype(int)
    return df


def extract_arrays_depol(
    df: pd.DataFrame,
    nA: int,
    nB: int,
    NM: int,
    dprob: float,
    Nu: int = None,
):
    """Select matching rows and return purity, thirdM, fourthM, and fifthM."""
    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM) & (df["dprob"] == dprob)]
    if Nu is not None:
        sub = sub[sub["Nu"] == Nu]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}, dprob={dprob}")

    # Sorting only affects row order, not the mean.

    purity  = sub["purity"].to_numpy(float)
    third   = sub["thirdM"].to_numpy(float)
    fourth  = sub["fourthM"].to_numpy(float)
    fifth   = sub["fifthM"].to_numpy(float)

    return purity, third, fourth, fifth

THEORY8 = {
2: 1.0000000000000009,
3: 0.7444526582608291,
4: 0.6794161346582085,
5: 0.6113730102112553,
}


def heisenberg_depol_paper(
    nA=11,
    nB=1,
    Nu=1,
    NM=1_000_000,
    csv_path=DATA / "heisenberg_dbasis_prob_paper.csv",
    outpath=HERE.parent / "figures/Depol_ent_detect.pdf",
    spread="std",
    expected_trials=1000,
):
    """Plot the retained full-range Heisenberg witness data.

    The reference figure uses mean +/- standard deviation across trials.
    Set spread="sem" explicitly for standard errors of the mean instead.
    Measurement counts and trial counts are supplied by paper_parameters.json.
    """
    import os

    df = load_results_csv(csv_path)
    ps = np.round(np.arange(0.0, 1.0, 0.1), 1)
    stats = {"D3": ([], []), "D4": ([], []), "D5": ([], [])}

    for p in ps:
        purity, third, fourth, fifth = extract_arrays_depol(
            df, nA=nA, nB=nB, Nu=Nu, NM=NM, dprob=float(p)
        )
        if len(purity) != expected_trials:
            raise ValueError(
                f"Expected {expected_trials} trials for p={p}, nA={nA}, nB={nB}, "
                f"N_U={Nu}, N_M={NM}; found {len(purity)}."
            )
        values = {
            "D3": D3(purity, third),
            "D4": D4(purity, third, fourth),
            "D5": D5(purity, third, fourth, fifth),
        }
        for name, arr in values.items():
            arr = np.asarray(arr, dtype=float)
            mean = float(np.mean(arr))
            width = float(np.std(arr))
            if spread == "sem":
                width /= np.sqrt(arr.size)
            elif spread != "std":
                raise ValueError("spread must be 'std' or 'sem'")
            stats[name][0].append(mean)
            stats[name][1].append(width)

    colors = {"D3": "#1f77b4", "D4": "#ff7f0e", "D5": "#2ca02c"}
    markers = {"D3": "s", "D4": "o", "D5": "^"}
    labels = {"D3": r"$D_3$", "D4": r"$D_4$", "D5": r"$D_5$"}

    plt.figure(figsize=(7.2, 4.6))
    for name in ("D3", "D4", "D5"):
        mean = np.asarray(stats[name][0])
        width = np.asarray(stats[name][1])
        plt.plot(
            ps,
            mean,
            marker=markers[name],
            markersize=7,
            linewidth=1.8,
            color=colors[name],
            label=labels[name],
        )
        plt.fill_between(
            ps,
            mean - width,
            mean + width,
            color=colors[name],
            alpha=0.3,
            linewidth=0,
        )

    plt.axhline(y=0, linestyle=(0, (6, 6)), linewidth=1.5, color="black")
    plt.xlabel(r"$p$", fontsize=23)
    plt.ylabel("value", fontsize=23, labelpad=0)
    plt.xlim(-0.01, 0.91)
    plt.legend(fontsize=20, handletextpad=0.3, borderpad=0.3, loc="upper right")
    plt.tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.show()
