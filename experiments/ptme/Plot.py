import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from scipy.stats import linregress
from XXZ import ppt_moments_from_rho, depolarize_state
from utils import D3, D4, ppt3, B2, D5, D3ToD5_tfim_dprob
from Gibbs import load_results_csv, extract_arrays_gibbs
from heisenberg_plot import extract_arrays_depol
from PIL import Image, ImageDraw, ImageFont

matplotlib.rcParams["font.sans-serif"] = 'Times New Roman'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

def plot_nA_dependence(csv_filename='experiment_results_dbasis.csv'):
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00098B", "#4159E1D4", "#87A7EBE2", "#ADD8E6"])
    # 获取唯一的 nA 值
    nA_values = df['nA'].unique()
    # 设置画布
    plt.figure(figsize=(7.2, 4.6))

    # 遍历不同的 nA 值，画出每条曲线
    for nA in nA_values[::2]:
        # 筛选出当前 nA 的数据
        nA_data = df[(df['nA'] == nA) & (df['Nu'] == 1)].copy()
        color = cmap(1 - (nA - 5) / (len(nA_values) - 2))
        # 以 NM 为横轴，result 为纵轴画图
        plt.plot(nA_data['NM'], nA_data['thirdM'], label=f'$n_A = {nA}$', marker='.', markersize=10, color = color)

    # 添加图例
    plt.legend(fontsize=20, loc='upper right', ncols=2, columnspacing=1.0, handletextpad=0.3, handlelength=1.4, bbox_to_anchor=(1.01, 1.01))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymax=0.9)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    # plt.ylim(0.4, 9)
    plt.xlabel(r'$N_M/d$', fontsize=23)
    # plt.ylabel(r'$\langle|\epsilon|\rangle$', fontsize=20)
    plt.ylabel(r'Statistical Error', fontsize=23)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # 显示图形
    plt.savefig('./figure/SE_nA.pdf')

    plt.show()


def plot_Nu_dependence(filename='experiment_results_dbasis.csv'):
    df = pd.read_csv(filename)
    cmap = LinearSegmentedColormap.from_list(
        "deep_light_green",
        ["#003300", "#1b5e20", "#388e3c", "#66bb6a", "#c8e6c9"]
    )
    nu_data = df[(df['nA'] == 9)].copy()
    nu = nu_data['Nu'].unique()
    colors = np.linspace(0.2, 0.95, len(nu)-3)[::-1]
    plt.figure(figsize=(7.2, 4.6))
    for i, n in enumerate(nu[:-3]):
        # 以 NM 为横轴，result 为纵轴画图
        plot_data = nu_data[nu_data['Nu'] == n].copy()
        plt.plot(plot_data['NM'], plot_data['thirdM'], label=f'$N_U = {n}$', marker='.', markersize=10, color = cmap(colors[i]))
    plt.legend(fontsize=20, loc='upper right', ncols=2, columnspacing=1.0, handletextpad=0.3, borderpad=0.3, handlelength=1.4, bbox_to_anchor=(1.02, 1.03))
    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # 设置标题和标签
    # plt.xlim(90, 110000)
    plt.ylim(ymax=1.1)
    plt.xlabel(r'$N_M/d$', fontsize=23)
    # plt.ylabel(r'$\langle|\epsilon|\rangle$', fontsize=20)
    plt.ylabel(r'Statistical Error', fontsize=23)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    plt.tight_layout()
    # 显示图形
    plt.savefig('./figure/SE_Nu.pdf')
    plt.show()
    

def plot_converge_val(csv_filename='experiment_results.csv'):
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    cmap = LinearSegmentedColormap.from_list("deep_light_blue", ["#00008B", "#4144E1D5", "#87A7EBE2", "#ADD8E6"])
    # 获取唯一的 nA 值
    nA_values = df['nA'].unique()
    converge = df[(df['NM'] == 100000) & (df['Nu'] == 1)]['result']

    # 设置画布
    plt.figure(figsize=(10, 6.18))

    plt.scatter(nA_values, converge, color = '#00008B')
    coeffs = np.polyfit(nA_values, np.log(converge), 1)
    poly = np.exp(np.polyval(coeffs, nA_values))
    plt.plot(nA_values, poly, linewidth=2, color = '#00008B')


    # 添加图例
    # plt.legend(fontsize=20)
    # plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.ticklabel_format(style='plain', axis='y')
    # 设置标题和标签
    plt.xlabel(r'$n_A$', fontsize=20)
    plt.ylabel(r'$\langle|\epsilon|\rangle$ Convergency value', fontsize=20)
    plt.xticks([5, 6, 7, 8, 9], fontsize=20)
    plt.yticks([0.1, 0.2, 0.4, 0.8], fontsize=20)
    plt.tight_layout()
    # plt.savefig('./figure/convergency_NM.pdf')
    # 显示图形
    plt.show()

def exact_depol_GHZ(nA, nB, sys="B", save_prefix: str = None, load_prefix: str = None):
    if save_prefix:
        N = nA + nB
        d = 2 ** N
        dimA = 2 ** nA
        dimB = 2 ** nB
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1 / np.sqrt(2)
        psi[-1] = 1 / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ppt3s = []
        D3s = []
        D4s = []
        for p in ps:
            rho_depol = depolarize_state(rho, d, p)
            moments = ppt_moments_from_rho(rho_depol, dimA, dimB, t_max=5, sys=sys)
            p2 = moments[2]
            p3 = moments[3]
            p4 = moments[4]
            ppt3s.append(ppt3(p2, p3))
            D3s.append(D3(p2, p3))
            D4s.append(D4(p2, p3, p4))
        ppt3s = np.array(ppt3s)
        D3s = np.array(D3s)
        D4s = np.array(D4s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_ppt3s.npy', ppt3s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_D3s.npy', D3s)
        np.save(f'{save_prefix}_nA{nA}_nB{nB}_D4s.npy', D4s)
    else:
        ppt3s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_ppt3s.npy')
        D3s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_D3s.npy')
        D4s = np.load(f'{load_prefix}_nA{nA}_nB{nB}_D4s.npy')
    ps_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.plot(ps_arr, ppt3s, label='PPT3')
    plt.plot(ps_arr, D3s, label='D3')
    plt.plot(ps_arr, D4s, label='D4')
    plt.hlines(0, 0.0, 0.9)
    plt.legend()
    plt.show()
    
    return ps_arr, ppt3s, D3s, D4s

def gibbs_ent_detect(nA, nB, Nu, NM, csv_path='gibbs_dbasis.csv'):
    df = load_results_csv(csv_path)
    betas = np.array([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
    # betas = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ppt3_mean = np.array([0] * 10, dtype=float)
    ppt3_std = np.array([0] * 10, dtype=float)
    B2_mean = np.array([0] * 10, dtype=float)
    B2_std = np.array([0] * 10, dtype=float)
    for i, beta in enumerate(betas):
        purity_tmp, third_tmp, fourth_tmp, fifth_tmp = extract_arrays_gibbs(df, nA=nA, nB=nB, Nu=Nu, NM=NM, beta=beta)
        # print(np.shape(purity_tmp))
        PPT3 = ppt3(purity_tmp, third_tmp)
        b2 = B2(purity_tmp, third_tmp, fourth_tmp, fifth_tmp)
        ppt3_mean[i] = np.mean(PPT3)
        ppt3_std[i] = np.std(PPT3)
        B2_mean[i] = np.mean(b2)*10
        B2_std[i] = np.std(b2)*10
    # print(B2_mean)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(betas, ppt3_mean, marker='s', markersize=7, label=r'$-|B_1|$')
    # plt.plot(ps, exact, label='exact', c='r')
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
    plt.savefig('./figure/Gibbs_ent_detect.pdf')
    plt.show()

def plot_error_vs_nA_fixed_budget_exact(
    csv_filename="experiment_results_dbasis.csv",
    nB=1,
    Nu_list=(1, 20, 100),
    budget_factor=100,          # N_U * N_M = budget_factor * d
    nm_is_normalized=True,      # True: NM 列是 (N_M/d)；False: NM 列是原始 N_M
    atol=1e-12,
    rtol=1e-6,
    outpath="./figure/SE_vs_nA_fixed_budget.pdf",
):
    df = pd.read_csv(csv_filename)
    required = {"nA", "Nu", "NM", "thirdM"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found columns: {list(df.columns)}")

    nA_values = sorted(df["nA"].unique())

    plt.figure(figsize=(7.2, 4.6))

    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"


    for i, Nu in enumerate(Nu_list):
        xs, ys = [], []

        for nA in nA_values:
            sub = df[(df["nA"] == nA) & (df["Nu"] == Nu)].copy()
            if sub.empty:
                continue

            d = 2 ** (int(nA) + int(nB))
            if nm_is_normalized:
                NM_target = budget_factor / Nu          # target is N_M/d
            else:
                NM_target = budget_factor * d / Nu      # target is N_M

            # 精确匹配目标点（允许浮点误差）
            mask = np.isclose(sub["NM"].to_numpy(dtype=float), NM_target, rtol=rtol, atol=atol)
            hit = sub[mask]
            if hit.empty:
                # 你说肯定有点；如果触发这里，通常是 NM 的存储格式/单位不一致
                raise ValueError(f"Missing exact point for nA={nA}, Nu={Nu}: target NM={NM_target}")

            # 如果同一个点有多次重复（多行），取平均
            y0 = float(np.mean(hit["thirdM"].to_numpy(dtype=float)))
            if y0 <= 0:
                continue

            xs.append(int(nA))
            ys.append(y0)
        if i == 0:
            plt.plot(xs, ys, marker="s", linewidth=2, markersize=7, label=rf"$N_U={Nu}$", color=blue)
        elif i == 1:
            plt.plot(xs, ys, marker="o", linewidth=2, markersize=7, label=rf"$N_U={Nu}$", color=orange)
        else:
            plt.plot(xs, ys, marker="^", linewidth=2, markersize=7, label=rf"$N_U={Nu}$", color=green)

    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    plt.xlabel(r"$n_A$", fontsize=23)
    plt.ylabel("Statistical Error", fontsize=23)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', direction='in', labelsize=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()

def plot_error_vs_nA_fixed_budget_exact_1x2(
    csv_filename="experiment_results_dbasis.csv",
    nB=1,
    Nu_list=(1, 4, 8),
    budget_factors=(1, 100),     # ← 关键：两个 budget
    nm_is_normalized=True,
    atol=1e-12,
    rtol=1e-6,
    outpath="./figure/PTmoment_vs_nA.pdf",
):

    df = pd.read_csv(csv_filename)
    required = {"nA", "Nu", "NM", "thirdM"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found columns: {list(df.columns)}")

    nA_values = sorted(df["nA"].unique())

    fig, axes = plt.subplots(
        1, len(budget_factors),
        figsize=(10.2, 4.2),
        sharey=True
    )

    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    colors = [blue, orange, green]
    markers = ["s", "o", "^"]

    for ax, budget_factor in zip(axes, budget_factors):

        for i, Nu in enumerate(Nu_list):
            xs, ys = [], []

            for nA in nA_values[:-1]:
                sub = df[(df["nA"] == nA) & (df["Nu"] == Nu)].copy()
                if sub.empty:
                    continue

                d = 2 ** (int(nA) + int(nB))
                if nm_is_normalized:
                    NM_target = budget_factor / Nu
                else:
                    NM_target = budget_factor * d / Nu

                mask = np.isclose(
                    sub["NM"].to_numpy(dtype=float),
                    NM_target,
                    rtol=rtol,
                    atol=atol
                )
                hit = sub[mask]
                if hit.empty:
                    raise ValueError(
                        f"Missing exact point for nA={nA}, Nu={Nu}: target NM={NM_target}"
                    )

                y0 = float(np.mean(hit["thirdM"].to_numpy(dtype=float)))
                if y0 <= 0:
                    continue

                xs.append(int(nA))
                ys.append(y0)

            log_ys = np.log(ys)
            slope, intercept, r_value, p_value, std_err = linregress(xs, log_ys)

            # ax.plot(
            #     xs, ys, mec="0.15", mew=0.8,
            #     marker=markers[i],
            #     linewidth=2,
            #     markersize=9,
            #     label=rf"$N_U={Nu}$",
            #     color=colors[i]
            # )
            ax.plot(xs, np.exp(intercept + np.array(xs) * slope), color=colors[i], linewidth=2, linestyle="--", zorder=2)

            ax.scatter(xs, ys, marker=markers[i], s=50, color=colors[i], label=rf"$N_U={Nu}$", edgecolors="0.15", linewidth=0.8, zorder=3)


        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:g}")
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        ax.set_xlabel(r"$n_A$", fontsize=23, labelpad=-2)
        ax.set_xticks([5, 8, 11])
        ax.text(
        0.97, 0.97,
        rf"$N_U N_M = d$" if budget_factor == 1 else rf"$N_U N_M = 100 d$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        bbox=dict(
        pad=0.3,
        facecolor="white",
        edgecolor="none")
        )


        # ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(
            axis="both", which="major",
            direction="in", labelsize=20
        )

    axes[0].set_ylabel("Statistical Error", fontsize=23)
    axes[0].legend(fontsize=18, handletextpad=0.3, borderpad=0.3)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()

def plot_SE_nA_and_Nu(csv_filename='experiment_results_dbasis.csv'):
    df = pd.read_csv(csv_filename)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)

    ax = axes[0]

    cmap_blue = LinearSegmentedColormap.from_list(
        "deep_light_blue",
        ["#00098B", "#4159E1D4", "#87A7EBE2", "#ADD8E6"]
    )

    nA_values = df['nA'].unique()

    for nA in nA_values[1::2]:
        nA_data = df[(df['nA'] == nA) & (df['Nu'] == 1)].copy()
        color = cmap_blue(1 - (nA - 5) / (len(nA_values) - 2))
        ax.plot(
            nA_data['NM'],
            nA_data['thirdM'],
            label=f'$n_A = {nA}$',
            marker='.',
            markersize=12,
            color=color
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymax=0.9)
    ax.set_xlabel(r'$N_M/d$', fontsize=23)
    ax.set_ylabel(r'Statistical Error', fontsize=23)

    ax.legend(
        fontsize=18,
        loc='upper right',
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.3,
        handlelength=1.4,
        bbox_to_anchor=(1.02, 1.03)
    )

    ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # ax.set_title(r'$N_U=1$', fontsize=23)



    ax = axes[1]

    cmap_green = LinearSegmentedColormap.from_list(
        "deep_light_green",
        ["#003300", "#1b5e20", "#388e3c", "#66bb6a", "#c8e6c9"]
    )

    nu_data = df[df['nA'] == 9].copy()
    nu = nu_data['Nu'].unique()
    print(nu)
    colors = np.linspace(0.2, 0.95, 4)[::-1]

    for i, n in enumerate(nu):
        if n not in [1, 4, 16, 64]:
            continue
        plot_data = nu_data[(nu_data['Nu'] == n) & (nu_data['NM'].isin([1, 2, 5, 10, 20, 50, 100]))].copy()
        ax.plot(
            plot_data['NM'],
            plot_data['thirdM'],
            label=f'$N_U = {n}$',
            marker='.',
            markersize=12,
            color=cmap_green(colors[np.where(np.array([1,4,16,64])==n)[0][0]])
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymax=1.2)
    ax.set_xlabel(r'$N_M/d$', fontsize=23)

    ax.legend(
        fontsize=18,
        loc='upper right',
        ncols=2,
        columnspacing=1.0,
        handletextpad=0.3,
        handlelength=1.4,
        bbox_to_anchor=(1.02, 1.03)
    )

    ax.tick_params(axis='both', which='both', direction='in', labelsize=20, pad=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    # ax.set_title(r'$n_A=9$', fontsize=23)


    plt.tight_layout(w_pad=0.1)
    plt.savefig('./figure/SE_nA_Nu.pdf')
    plt.show()

def detecting_ent_tfim(csv_filename='processed_results2.csv'):
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
        # prob_any = (d3 > 0) | (d4 > 0)
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
    # axes.errorbar(ps, D3_mean, yerr=D3_std, marker='s', markersize=4, label=r'$D_3$', color="#1f77b4", capsize=4)
    # axes.errorbar(ps, 10 * D4_mean, yerr=10 * D4_std, marker='o', markersize=4, label=r'$D_4$', color="#ff7f0e", capsize=4)
    # axes.errorbar(ps, 10 * D5_mean, yerr=10 * D5_std, marker='^', markersize=4, label=r'$D_5$', color="#2ca02c", capsize=4)
    # axes.hlines(0, 0.2, 0.7, colors='gray', linestyles=(0, (6, 6)), linewidth=1.5)
    # axes[0].set_xlabel(r'$p$', fontsize=23)
    # axes[0].set_ylabel('Value', fontsize=23)
    # theory3, theory4, theory5 = D3ToD5_tfim_dprob(dprob=ps)
    # prob_d3s_theory = [int(theory3[i] > 0) for i in range(len(ps))]
    # prob_d4s_theory = [int(theory4[i] > 0) for i in range(len(ps))]
    # prob_d5s_theory = [int(theory5[i] > 0) for i in range(len(ps))]
    # prob_anys_theory = [int(theory3[i] > 0 or theory4[i] > 0 or theory5[i] > 0) for i in range(len(ps))]
    import matplotlib.cm as cm
    cmap_blue = cm.Blues
    colors = cmap_blue(np.linspace(0.2, 0.9, 3)) 
    orange = cm.Oranges(0.7)
    axes.vlines(x=0.52, ymin=-0.2, ymax=1.2, colors='black', linestyles=(0, (6, 6)), linewidth=1.5)

    # ax.plot(ps, prob_d3s_theory, label=r'Theory $D_3$', color="#1f77b4", linestyle="--")
    axes.plot(ps, prob_d3s, marker='o', markersize=8, label=r'$D_3$', color=colors[0])
    # ax.plot(ps, prob_d4s_theory, label=r'Theory $D_4$', color="#ff7f0e", linestyle="--")
    axes.plot(ps, prob_d4s, marker='s', markersize=8, label=r'$D_4$', color=colors[1])
    # ax.plot(ps, prob_d5s_theory, label=r'Theory $D_5$', color="#2ca02c", linestyle="--")
    axes.plot(ps, prob_d5s, marker='^', markersize=8, label=r'$D_5$', color=colors[2])
    axes.plot(ps, prob_anys, marker='d', markersize=8, color=orange)
    # axes.plot(ps, prob_ppt3s, marker='x', markersize=8, label=r'PPT3', color='purple')
    # axes.plot(ps, prob_d3s, marker='o', markersize=8, label=r'$D_3$', color=colors[0])
    axes.set_xlabel(r'$p$', fontsize=23)
    axes.set_ylabel('Detection Probability', fontsize=23)
    axes.set_xticks(ps[::2])
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # axes.set_xlim(left=0.2, right=0.8)
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
    # axes.text(-0.25, 0.95, "(b)", transform=axes.transAxes, fontsize=20)

    plt.tight_layout()
    plt.savefig('./figure/TFIM_detection_prob.svg')
    plt.show()

def combine_picture():
    image = Image.open('./PTME.pdf')
    
    # 加载matplotlib图形
    plot_image = Image.open('./figure/TFIM_detection_prob.pdf')

    # 获取图像尺寸
    image_width, image_height = image.size
    plot_width, plot_height = plot_image.size

    # 创建一个新的画布，宽度为两个图像的宽度之和，高度为两者的最大高度
    new_width = image_width + plot_width
    new_height = max(image_height, plot_height)
    combined_image = Image.new("RGB", (new_width, new_height))

    # 将PDF图片和matplotlib图形粘贴到新的画布
    combined_image.paste(image, (0, 0))
    combined_image.paste(plot_image, (image_width, 0))

    # 添加(a)标签和(b)标签
    draw = ImageDraw.Draw(combined_image)

    # 设置字体（这里使用默认字体，您可以选择更合适的字体）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # 在PDF图像上添加 (a) 标签
    draw.text((20, 20), "(a)", font=font, fill="black")
    
    # 在matplotlib图形上添加 (b) 标签
    draw.text((image_width + 20, 20), "(b)", font=font, fill="black")

    # 显示和保存拼接后的图像
    combined_image.show()
    combined_image.save("combined_image_with_labels.jpg")

def plot_mae_t4_by_n(
    data_dir="./data",
    file_pattern=r"mae_n*_p0.30.csv",
    outpath="./figure/mae_t4_by_n_p0.30.pdf",
):
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No csv files matched '{file_pattern}' in {data_dir}")

    n_pattern = re.compile(r"mae_n(\d+)_p0\.30\.csv$")
    plotted_anc = []
    for csv_path in csv_files:
        match = n_pattern.match(csv_path.name)
        if not match:
            continue

        n_val = int(match.group(1))
        df = pd.read_csv(csv_path)
        required = {"sample", "mae_t4"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns: {missing}")

        plotted_anc.append((n_val, df.sort_values("sample")))

    if not plotted_anc:
        raise ValueError("No valid csv files found after filename filtering.")

    plotted_anc.sort(key=lambda x: x[0])
    colors_left = cm.Blues(np.linspace(0.2, 0.9, len(plotted_anc)))
    plt.figure(figsize=(7.2, 4.6))

    for i, (n_val, df) in enumerate(plotted_anc):
        plt.plot(
            df["sample"],
            df["mae_t4"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=rf"$n={n_val}$",
            color=colors_left[i],
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=4e2, right=2e7)
    plt.ylim(top=2e0)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    plt.xlabel(r"$N_M$", fontsize=23)
    plt.ylabel("Statistical Error", fontsize=23)
    plt.legend(fontsize=18, ncols=2, handletextpad=0.3, borderpad=0.3)
    plt.tick_params(axis="both", which="both", direction="in", labelsize=20, pad=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()


def plot_mae_t4_nA_convergence(
    data_dir="./data",
    file_pattern=r"mae_n*_p0.30.csv",
    nB=2,
    tail_points=5,
    left_nA_to_plot=(10, 14, 18),
    exclude_nA=(6,),
    left_max_nm=5e6,
    outpath="./figure/mae_t4_vs_NM_and_nA_decay.pdf",
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


if __name__ == '__main__':
    # 调用函数画图
    # plot_nA_dependence()
    # plot_converge_val()
    # plot_Nu_dependence()
    # gibbs_ent_detect(nA=11, nB=1, Nu=100, NM=1000)
    # plot_error_vs_nA_fixed_budget_exact_1x2()
    # plot_SE_nA_and_Nu()
    # plot_mae_t4_by_n()
    plot_mae_t4_nA_convergence(nB=2)
    # detecting_ent_tfim()
    # combine_picture()