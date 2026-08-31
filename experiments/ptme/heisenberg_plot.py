import pandas as pd
import numpy as np
from XXZ import ppt_moment, ppt_moment_depolarize
from utils import ppt3, D3, D4, D5
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'




def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["nA", "nB", "Nu", "NM"]:
        df[c] = df[c].astype(int)
    return df

def extract_arrays(df: pd.DataFrame, nA: int, nB: int, NM: int):
    """
    固定 nA, NM，返回四个 numpy 数组：
    purity, thirdM, fourthM, fifthM
    """
    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM)]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}")

    # # 排序只是为了让顺序稳定（不影响平均）
    # sub = sub.sort_values(["nB", "Nu"])

    purity  = sub["purity"].to_numpy(float)
    third   = sub["thirdM"].to_numpy(float)
    fourth  = sub["fourthM"].to_numpy(float)
    fifth   = sub["fifthM"].to_numpy(float)

    return purity, third, fourth, fifth


def extract_arrays_depol(
    df: pd.DataFrame,
    nA: int,
    nB: int,
    NM: int,
    dprob: float,
    Nu: int = None,
):
    """
    固定 nA, NM，返回四个 numpy 数组：
    purity, thirdM, fourthM, fifthM
    """
    sub = df[(df["nA"] == nA) & (df["nB"] == nB) & (df["NM"] == NM) & (df["dprob"] == dprob)]
    if Nu is not None:
        sub = sub[sub["Nu"] == Nu]
    if sub.empty:
        raise ValueError(f"No data for nA={nA}, nB={nB}, NM={NM}, dprob={dprob}")

    # # 排序只是为了让顺序稳定（不影响平均）
    # sub = sub.sort_values(["nB", "Nu"])

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

def mean_abs_errors(purity, third, fourth, fifth, theory=THEORY8):
    """
    返回 dict: {2: err2, 3: err3, 4: err4, 5: err5}
    """

    return {
        2: np.mean(np.abs(purity  - theory[2])),
        3: np.mean(np.abs(third   - theory[3])),
        4: np.mean(np.abs(fourth  - theory[4])),
        5: np.mean(np.abs(fifth   - theory[5])),
    }

def compute_errors_from_csv(csv_path: str, nA: int, nB: int, NM: int, theory):
    df = load_results_csv(csv_path)
    purity, third, fourth, fifth = extract_arrays(df, nA=nA, nB=nB, NM=NM)
    errors = mean_abs_errors(purity, third, fourth, fifth, theory)
    return errors

def exact_depolarize(nA, nB, save_prefix: str = None, load_prefix: str = None):
    if save_prefix:
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ppt3s = []
        D3s = []
        D4s = []
        for p in ps:
            theory = ppt_moment_depolarize(nA, nB, p, 5)
            ppt3s.append(ppt3(theory[2], theory[3]))
            D3s.append(D3(theory[2], theory[3]))
            D4s.append(D4(theory[2], theory[3], theory[4]))
        
        ps_arr = np.array(ps)
        ppt3s = np.array(ppt3s)
        D3s = np.array(D3s)
        D4s = np.array(D4s)

        if save_prefix:
            np.save(save_prefix + "_ps.npy", ps_arr)
            np.save(save_prefix + "_ppt3s.npy", ppt3s)
            np.save(save_prefix + "_D3s.npy", D3s)
            np.save(save_prefix + "_D4s.npy", D4s)
    else:
        ppt3s = np.load(load_prefix + "_ppt3s.npy")
        D3s = np.load(load_prefix + "_D3s.npy")
        D4s = np.load(load_prefix + "_D4s.npy")
    ps_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.plot(ps_arr, ppt3s, label='PPT3')
    plt.plot(ps_arr, D3s, label='D3')
    plt.plot(ps_arr, D4s, label='D4')
    plt.hlines(0, 0.0, 0.9)
    plt.legend()
    plt.show()

    return ps_arr, ppt3s, D3s, D4s


def heisenberg_depol(nA, nB, NM, csv_path='heisenberg_dbasis_prob.csv'):
    df = load_results_csv(csv_path)
    ps = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    ppt3_mean = np.array([0] * 5, dtype=float)
    ppt3_std = np.array([0] * 5, dtype=float)
    D3_mean = np.array([0] * 5, dtype=float)
    D3_std = np.array([0] * 5, dtype=float)
    D3_min = np.array([0] * 5, dtype=float)
    D3_max = np.array([0] * 5, dtype=float)
    D4_mean = np.array([0] * 5, dtype=float)
    D4_std = np.array([0] * 5, dtype=float)
    D5_mean = np.array([0] * 5, dtype=float)
    D5_std = np.array([0] * 5, dtype=float)
    for i, p in enumerate(ps):
        purity_tmp, third_tmp, fourth_tmp, fifth_tmp = extract_arrays_depol(df, nA=nA, nB=nB, NM=NM, dprob=p)
        ppt3 = purity_tmp * purity_tmp - third_tmp
        d3 = D3(purity_tmp, third_tmp)
        d4 = D4(purity_tmp, third_tmp, fourth_tmp)
        d5 = D5(purity_tmp, third_tmp, fourth_tmp, fifth_tmp)
        ppt3_mean[i] = np.mean(ppt3)
        ppt3_std[i] = np.std(ppt3)
        D3_mean[i] = np.mean(d3)
        D3_std[i] = np.std(d3)
        D3_min[i] = np.min(d3)
        D3_max[i] = np.max(d3)
        D4_mean[i] = np.mean(d4)
        D4_std[i] = np.std(d4)
        D5_mean[i] = np.mean(d5)
        D5_std[i] = np.std(d5)

    exact = np.load('heisenberg/depol_nA11_nB1_ppt3s.npy')
    plt.figure(figsize=(7.2, 4.6))
    # plt.plot(ps, ppt3_mean, marker='.', markersize=10, label=r'$p_2^2-p_3$')
    # # plt.plot(ps, exact, label='exact', c='r')
    # plt.fill_between(
    #     ps,
    #     ppt3_mean - ppt3_std,
    #     ppt3_mean + ppt3_std,
    #     alpha=0.3
    # )
    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    plt.errorbar(ps-0.01, D3_mean, yerr=D3_std, marker='s', markersize=8, label=r'$D_3$', color=blue, capsize=5, linewidth=1.0)
    plt.errorbar(ps, D4_mean, yerr=D4_std, marker='o', markersize=8, label=r'$D_4$', color=orange, capsize=5, linewidth=1.0)
    plt.errorbar(ps+0.01, D5_mean, yerr=D5_std, marker='^', markersize=8, label=r'$D_5$', color=green, capsize=5, linewidth=1.0)
    # plt.plot(ps, D3_mean, marker='s', markersize=7, label=r'$D_3$', color=blue)
    # plt.fill_between(
    #     ps,
    #     D3_mean - D3_std,
    #     D3_mean + D3_std,
    #     alpha=0.3
    # )
    # plt.plot(ps, D4_mean, marker='o', markersize=7, label=r'$D_4$', color=orange)
    # plt.fill_between(
    #     ps,
    #     D4_mean - D4_std,
    #     D4_mean + D4_std,
    #     alpha=0.3
    # )
    # plt.plot(ps, D5_mean, marker='^', markersize=7, label=r'$D_5$', color=green)
    # plt.fill_between(
    #     ps,
    #     D5_mean - D5_std,
    #     D5_mean + D5_std,
    #     alpha=0.3
    # )
    plt.axhline(y=0, linestyle=(0, (6, 6)), linewidth=1.5, color='black')
    plt.xlabel(r'$p$', fontsize=23)
    plt.ylabel('value', fontsize=23, labelpad=0)
    plt.xlim(left=0.3-0.01, right=0.71)
    plt.legend(fontsize=20, handletextpad=0.3, borderpad=0.3)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.tight_layout()
    plt.savefig('./figure/Depol_ent_detect.pdf')
    plt.show()


def heisenberg_depol_paper(
    nA=11,
    nB=1,
    Nu=1,
    NM=1_000_000,
    csv_path="heisenberg_dbasis_prob_paper.csv",
    outpath="./figure/Depol_ent_detect.pdf",
    spread="sem",
    expected_trials=1000,
):
    """Reconstruct the full-range plot used in the SCNE3 manuscript.

    The manuscript declares p=0.0,...,0.9, N_U=1, N_M=10^6, 1000 trials,
    and standard-error bands.  Use explicit legacy arguments only when
    auditing the previously retained manuscript PDF.
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
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath)
    plt.show()






# errors = compute_errors_from_csv(
#     "heisenberg_dbasis.csv",
#     nA=8,
#     nB=2,
#     NM=1000,
#     theory=ppt_moment(8, 2, 5)
# )

# print(errors)

# exact_depolarize(11, 1, load_prefix='heisenberg/depol_nA11_nB1')
if __name__ == '__main__':
    heisenberg_depol_paper()
