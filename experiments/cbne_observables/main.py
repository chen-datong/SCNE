from token import EXACT_TOKEN_TYPES
import numpy as np
from scipy.stats import unitary_group
import math
import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def estimate_rhot_dprob(rho, n, Nu, NM, t, dprob=0):
    """
    Estimate one or multiple moments of rho:
    - if t is int: return a float (backward compatible)
    - if t is iterable: return dict {t_i: estimate}
    """
    d = pow(2, n)

    if isinstance(t, (list, tuple, set, np.ndarray)):
        t_list = sorted(int(x) for x in t)
        if len(t_list) == 0:
            raise ValueError("t list should not be empty")
        single_t_mode = False
    else:
        t_list = [int(t)]
        single_t_mode = True

    total_counts = {tt: 0.0 for tt in t_list}
    coeff = {}
    for tt in t_list:
        denom = math.comb(NM, tt) * d
        numer = math.comb(d + tt - 1, tt) * math.factorial(tt)
        coeff[tt] = numer / denom

    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        outcomes_idx = np.random.choice(d, size=NM, p=prob)
        counts = np.bincount(outcomes_idx, minlength=d)

        for tt in t_list:
            subtotal = 0.0
            for count in counts:
                if count >= tt:
                    subtotal += math.comb(int(count), tt)
            total_counts[tt] += subtotal * coeff[tt]

    means = {tt: total_counts[tt] / Nu for tt in t_list}
    if single_t_mode:
        return means[t_list[0]]
    return means

def estimate_rhotO_dprob(rho, O, n, Nu, NM, t, dprob=0):
    d = pow(2, n)
    total_counts = 0
    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        O_U = U @ O @ U.conj().T
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        outcomes_idx = np.random.choice(d, size=NM, p=prob)
        counts = np.bincount(outcomes_idx, minlength=d)
        denom = math.comb(NM, t) * d
        numer = math.comb(d + t, t + 1) * math.factorial(t + 1) 
        total = 0
        for count, o_val in zip(counts, np.diag(O_U)):
            if count >= t:
                total += o_val * math.comb(count, t)
        total_counts += total * numer / denom

    return total_counts / Nu

def estimate_rhotAndrhotO_dprob(rho, O_list, n, Nu, NM, t, dprob=0):
    d = pow(2, n)
    total_counts_rhot = 0
    total_counts_rhotO = [0] * len(O_list)
    for _ in tqdm(range(Nu)):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        O_U_list = [U @ O @ U.conj().T for O in O_list]
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        outcomes_idx = np.random.choice(d, size=NM, p=prob)
        counts = np.bincount(outcomes_idx, minlength=d)
        denom_rhot = math.comb(NM, t) * d
        numer_rhot = math.comb(d + t - 1, t) * math.factorial(t) 
        denom_rhotO = math.comb(NM, t) * d
        numer_rhotO = math.comb(d + t, t + 1) * math.factorial(t + 1) 
        for count in counts:
            if count >= t:
                total_counts_rhot += math.comb(count, t) * numer_rhot / denom_rhot
        for i, O_U in enumerate(O_U_list):
            for count, o_val in zip(counts, np.diag(O_U)):
                if count >= t:
                    total_counts_rhotO[i] += o_val * math.comb(count, t) * numer_rhotO / denom_rhotO

    
    return (total_counts_rhot / Nu, [tc / Nu for tc in total_counts_rhotO])

def estimate_rhotAndrhotO_tlist_dprob(rho, O_list, n, Nu, NM, t_list, dprob=0):
    d = pow(2, n)
    total_counts_rhot = {t: 0 for t in t_list}
    total_counts_rhotO = {t: [0] * len(O_list) for t in t_list}
    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        O_U_list = [U @ O @ U.conj().T for O in O_list]
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        outcomes_idx = np.random.choice(d, size=NM, p=prob)
        counts = np.bincount(outcomes_idx, minlength=d)
        for t in t_list:
            denom_rhot = math.comb(NM, t) * d
            numer_rhot = math.comb(d + t - 1, t) * math.factorial(t) 
            denom_rhotO = math.comb(NM, t) * d
            numer_rhotO = math.comb(d + t, t + 1) * math.factorial(t + 1) 
            for count in counts:
                if count >= t:
                    total_counts_rhot[t] += math.comb(count, t) * numer_rhot / denom_rhot
            for i, O_U in enumerate(O_U_list):
                for count, o_val in zip(counts, np.diag(O_U)):
                    if count >= t:
                        total_counts_rhotO[t][i] += o_val * math.comb(count, t) * numer_rhotO / denom_rhotO

    
    return ({t: total_counts_rhot[t] / Nu for t in t_list},
            {t: [tc / Nu for tc in total_counts_rhotO[t]] for t in t_list})

def estimate_rhotAndrhotO_tlist_dprob_ancilla(rho, O_list, n, Nu, NM, t_list, anc, dprob=0):
    d = pow(2, n + anc)
    zero_anc = np.zeros((1 << anc, 1 << anc))
    zero_anc[0, 0] = 1
    rho_ext = np.kron(rho, zero_anc)
    O_ext_list = [np.kron(O, zero_anc) for O in O_list]
    total_counts_rhot = {t: 0 for t in t_list}
    total_counts_rhotO = {t: [0] * len(O_list) for t in t_list}
    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho_ext @ U.conj().T
        O_U_list = [U @ O_ext @ U.conj().T for O_ext in O_ext_list]
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        outcomes_idx = np.random.choice(d, size=NM, p=prob)
        counts = np.bincount(outcomes_idx, minlength=d)
        for t in t_list:
            denom_rhot = math.comb(NM, t) * d
            numer_rhot = math.comb(d + t - 1, t) * math.factorial(t) 
            denom_rhotO = math.comb(NM, t) * d
            numer_rhotO = math.comb(d + t, t + 1) * math.factorial(t + 1) 
            for count in counts:
                if count >= t:
                    total_counts_rhot[t] += math.comb(count, t) * numer_rhot / denom_rhot
            for i, O_U in enumerate(O_U_list):
                for count, o_val in zip(counts, np.diag(O_U)):
                    if count >= t:
                        total_counts_rhotO[t][i] += o_val * math.comb(count, t) * numer_rhotO / denom_rhotO
    
    return ({t: total_counts_rhot[t] / Nu for t in t_list},
            {t: [tc / Nu for tc in total_counts_rhotO[t]] for t in t_list})


def estimate_rhotAndrhotO_tlist_dprob_ancilla_multi_nm(rho, O_list, n, Nu, NM_list, t_list, anc, dprob=0):
    """
    Reuse one maximal sampling trajectory for multiple NM values:
    - sample outcomes with size NM_max once per unitary
    - for each NM in NM_list, use the first NM outcomes
    """
    NM_sorted = sorted(int(x) for x in NM_list)
    if not NM_sorted:
        raise ValueError("NM_list should not be empty")
    if NM_sorted[0] <= 0:
        raise ValueError("All NM values must be positive")

    d = pow(2, n + anc)
    NM_max = NM_sorted[-1]

    zero_anc = np.zeros((1 << anc, 1 << anc))
    zero_anc[0, 0] = 1
    rho_ext = np.kron(rho, zero_anc)
    O_ext_list = [np.kron(O, zero_anc) for O in O_list]

    t_sorted = sorted(int(t) for t in t_list)
    total_counts_rhot = {NM: {t: 0 for t in t_sorted} for NM in NM_sorted}
    total_counts_rhotO = {
        NM: {t: [0] * len(O_list) for t in t_sorted}
        for NM in NM_sorted
    }

    # Precompute normalization factors that depend only on (NM, t, d).
    coeff_rhot = {}
    coeff_rhotO = {}
    for NM in NM_sorted:
        coeff_rhot[NM] = {}
        coeff_rhotO[NM] = {}
        for t in t_sorted:
            denom_rhot = math.comb(NM, t) * d
            numer_rhot = math.comb(d + t - 1, t) * math.factorial(t)
            denom_rhotO = math.comb(NM, t) * d
            numer_rhotO = math.comb(d + t, t + 1) * math.factorial(t + 1)
            coeff_rhot[NM][t] = numer_rhot / denom_rhot
            coeff_rhotO[NM][t] = numer_rhotO / denom_rhotO

    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho_ext @ U.conj().T
        O_U_list = [U @ O_ext @ U.conj().T for O_ext in O_ext_list]
        diag_O_list = [np.diag(O_U) for O_U in O_U_list]
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d

        # One maximal sample; smaller NM reuse prefixes.
        outcomes_idx_max = np.random.choice(d, size=NM_max, p=prob)

        # Incremental prefix counts to avoid repeated full bincount over prefixes.
        counts = np.zeros(d, dtype=np.int64)
        prev_size = 0
        for NM in NM_sorted:
            cur_size = NM
            new_outcomes = outcomes_idx_max[prev_size:cur_size]
            np.add.at(counts, new_outcomes, 1)
            prev_size = cur_size

            counts_f = counts.astype(np.float64, copy=False)
            for t in t_sorted:
                if t == 1:
                    choose_ct = counts_f
                elif t == 2:
                    choose_ct = counts_f * (counts_f - 1.0) * 0.5
                elif t == 3:
                    choose_ct = counts_f * (counts_f - 1.0) * (counts_f - 2.0) / 6.0
                else:
                    # Generic fallback if t_list is extended.
                    choose_ct = np.array(
                        [math.comb(int(c), t) if c >= t else 0.0 for c in counts],
                        dtype=np.float64,
                    )

                total_counts_rhot[NM][t] += float(np.sum(choose_ct) * coeff_rhot[NM][t])
                scale_o = coeff_rhotO[NM][t]
                for i, diag_O in enumerate(diag_O_list):
                    total_counts_rhotO[NM][t][i] += np.sum(diag_O * choose_ct) * scale_o

    rhot_mean = {
        NM: {t: total_counts_rhot[NM][t] / Nu for t in t_sorted}
        for NM in NM_sorted
    }
    rhotO_mean = {
        NM: {t: [tc / Nu for tc in total_counts_rhotO[NM][t]] for t in t_sorted}
        for NM in NM_sorted
    }
    return rhot_mean, rhotO_mean
        

def test(n):
    d = pow(2, n)
    ghz_vec = np.zeros(d, dtype=np.complex128)
    ghz_vec[0] = 1/np.sqrt(2)
    ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 1
    NM = 100000
    t = 2
    dprob = 0.2
    purity = estimate_rhot_dprob(rho_test, n, Nu, NM, 2, dprob)
    print(f"Estimated purity for GHZ state with n={n}: {purity}")
    return purity


def test_t3_with_purity_and_third_moment(n, Nu=1, NM=100000, dprob=0.2, rep=50):
    """
    Estimate moments up to t=3 in one run, then compute:
      purity = rho2 - 1
      third_moment = (rho3 - 3 * purity - 1) / 2
    """
    d = pow(2, n)
    ghz_vec = np.zeros(d, dtype=np.complex128)
    ghz_vec[0] = 1 / np.sqrt(2)
    ghz_vec[-1] = 1 / np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    purity_list = []
    third_moment_list = []
    for _ in range(rep):
        moments = estimate_rhot_dprob(rho_test, n, Nu, NM, [2, 3], dprob)
        purity = moments[2] - 1
        third_moment = (moments[3] - 3 * purity - 1) / 2
        purity_list.append(purity)
        third_moment_list.append(third_moment)
    # Theoretical values for depolarized pure state:
    # rho = (1-p)|psi><psi| + p * I/d, with p=dprob.
    p = float(dprob)
    theory_purity = (1.0 - p) ** 2 + (2.0 * p - p ** 2) / d
    theory_third_moment = (1.0 - p + p / d) ** 3 + (d - 1) * (p / d) ** 3

    purity_mean = float(np.mean(purity_list))
    third_mean = float(np.mean(third_moment_list))
    purity_mae = float(np.mean(np.abs(np.asarray(purity_list, dtype=float) - theory_purity)))
    third_moment_mae = float(np.mean(np.abs(np.asarray(third_moment_list, dtype=float) - theory_third_moment)))
    print(
        f"[n={n}] purity: est={purity_mean:.8f}, theory={theory_purity:.8f}; "
        f"third_moment: est={third_mean:.8f}, theory={theory_third_moment:.8f}; "
        f"MAE(purity)={purity_mae:.8e}, MAE(third)={third_moment_mae:.8e}"
    )

    return {
        "purity": purity_mean,
        "third_moment": third_mean,
        "theory_purity": theory_purity,
        "theory_third_moment": theory_third_moment,
        "purity_mae": purity_mae,
        "third_moment_mae": third_moment_mae,
    }

def experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=1, dprob=0):
    d = pow(2, n)
    # ghz_vec = np.zeros(d, dtype=np.complex128)
    # ghz_vec[0] = 1/np.sqrt(2)
    # ghz_vec[-1] = 1/np.sqrt(2)
    # rho_test = np.outer(ghz_vec, ghz_vec.conj())
    # rho_test = np.load(f'tfim_gibbs_N{n}_beta3.0_J1.0_h1.0.npy')
    # O_list = []
    # ## Define X_1 X_2 ... X_n and Z_1 Z_2
    # # X1toXn = np.eye(1, dtype=np.complex128)
    # # for i in range(n):
    # #     X1toXn = np.kron(X1toXn, X)
    # # O_list.append(X1toXn)
    # # Z1Z2 = np.eye(1, dtype=np.complex128)
    # # for i in range(n):
    # #     if i == 0 or i == 1:
    # #         Z1Z2 = np.kron(Z1Z2, Z)
    # #     else:
    # #         Z1Z2 = np.kron(Z1Z2, np.eye(2, dtype=np.complex128))
    # # O_list.append(Z1Z2)
    # # psib = rho_test - np.eye(d, dtype=np.complex128) / d
    # psi = np.load(f'tfim_N{n}_gs_J1.0_h1.0.npy')
    # psib = np.outer(psi, psi.conj()) - np.eye(d, dtype=np.complex128) / d
    # O_list.append(psib)
    
    t_list = list(range(tmin, t+1))

    rhot_estimate, rhotO_estimates = estimate_rhotAndrhotO_tlist_dprob(rho_test, O_list, n, Nu, NM, t_list, dprob)
    rho1 = rhot_estimate[1]
    rho2 = rhot_estimate[2]
    purity = rho2 - 1
    rho3 = rhot_estimate[3]
    third_moment = (rho3 - 3 * purity - 1) / 2

    rhoO1 = np.array(rhotO_estimates[1])
    # print(rhoO1)
    rhoO2 = np.array(rhotO_estimates[2])
    rhoO3 = np.array(rhotO_estimates[3])
    rho2O = (rhoO2 - 2 * rhoO1) / 2
    rho3O = (rhoO3 - 6 * rho2O - 3 * (1 + purity) * rhoO1) / 6

    pf_rhoO1 = rhoO1
    pf_rhoO2 = rho2O / purity
    pf_rhoO3 = rho3O / third_moment
    
    # print(f'{pf_rhoO1[0]}, {pf_rhoO2[0]}, {pf_rhoO3[0]} for O=X1toXn')
    # print(f'{pf_rhoO1[1]}, {pf_rhoO2[1]}, {pf_rhoO3[1]} for O=Z1Z2')
    
    return (purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3)

def main_experiment_dprob(n, Nu, NM, rep, dprob=0):
    d = pow(2, n)
    data = []
    ghz_vec = np.zeros(d, dtype=np.complex128)
    ghz_vec[0] = 1/np.sqrt(2)
    ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    O_list = []
    psib = rho_test - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)

    for idx in tqdm(range(rep)):
        purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, 3, tmin=1, dprob=dprob)
        # 存储每条数据：idx, purity, third_moment, 以及两个可观测量的三个估计值
        data.append({
            'idx': idx,
            'purity': purity,
            'third_moment': third_moment,
            # 'pf_rhoO1_X1toXn': pf_rhoO1[0],
            # 'pf_rhoO2_X1toXn': pf_rhoO2[0],
            # 'pf_rhoO3_X1toXn': pf_rhoO3[0],
            # 'pf_rhoO1_Z1Z2': pf_rhoO1[1],
            # 'pf_rhoO2_Z1Z2': pf_rhoO2[1],
            # 'pf_rhoO3_Z1Z2': pf_rhoO3[1]
            'pf_rhoO3_fidelity': pf_rhoO3[0]
        })
    
    # 转为 DataFrame
    df = pd.DataFrame(data)
    
    # 保存文件，文件名包含 n, Nu, NM, dprob
    filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{3.0}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def _experiment_dprob_wrapper(args):
    """多进程包装函数，用于 experiment_single 调用"""
    n, Nu, NM, t, tmin, dprob, idx = args
    d = pow(2, n)
    ghz_vec = np.zeros(d, dtype=np.complex128)
    ghz_vec[0] = 1/np.sqrt(2)
    ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    O_list = []
    psib = rho_test - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)
    purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=tmin, dprob=dprob)
    return {
        'idx': idx,
        'purity': purity,
        'third_moment': third_moment,
        'pf_rhoO2_fidelity': pf_rhoO2[0],
        'pf_rhoO3_fidelity': pf_rhoO3[0]
    }

def main_experiment_dprob_parallel(n, Nu, NM, rep, dprob=0, num_processes=None):
    """并行版本的主实验函数，使用 multiprocessing
    
    参数:
        n: 量子比特数
        Nu: unitary数量
        NM: 测量数量
        rep: 重复次数
        dprob: 失真概率
        num_processes: 进程数，如果为None则使用CPU核心数
        
    返回:
        DataFrame: 实验结果
    """
    if num_processes is None:
        num_processes = cpu_count()


    # 准备参数列表
    args_list = [(n, Nu, NM, 3, 1, dprob, idx) for idx in range(rep)]
    
    # 使用多进程池执行实验
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_dprob_wrapper, args_list), total=rep))
    
    # 转为 DataFrame
    df = pd.DataFrame(results)
    
    # 保存文件，文件名包含 n, Nu, NM, dprob
    filename = f"./data/fidelity_n{n}_Nu{Nu}_NM{NM}_dprob{dprob}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def analyze_results(filename):
    """分析实验数据，计算 pf_rhoO3 的平均值和标准差
    
    参数:
        filename: CSV文件路径
        
    返回:
        dict: 包含分析结果的字典
    """
    import pandas as pd
    
    df = pd.read_csv(filename)
    
    # 处理可能是字符串形式的数据（由于数组或复数被保存为字符串）
    def convert_to_float(val):
        if isinstance(val, str):
            # 尝试解析复数格式 "(real+imagj)"
            val = val.strip().strip('[]')
            try:
                # 先尝试作为复数解析
                complex_val = complex(val)
                # 提取实部
                return float(complex_val.real)
            except (ValueError, AttributeError):
                # 如果失败，直接转浮点数
                return float(val)
        return float(val)
    
    # 确保数据为数值类型
    df['pf_rhoO3_X1toXn'] = df['pf_rhoO3_X1toXn'].apply(convert_to_float)
    df['pf_rhoO3_Z1Z2'] = df['pf_rhoO3_Z1Z2'].apply(convert_to_float)
    
    # 计算 pf_rhoO3_X1toXn 和 pf_rhoO3_Z1Z2 的统计量
    results = {
        'pf_rhoO3_X1toXn_mean': df['pf_rhoO3_X1toXn'].mean(),
        'pf_rhoO3_X1toXn_std': df['pf_rhoO3_X1toXn'].std(),
        'pf_rhoO3_Z1Z2_mean': df['pf_rhoO3_Z1Z2'].mean(),
        'pf_rhoO3_Z1Z2_std': df['pf_rhoO3_Z1Z2'].std(),
        'num_samples': len(df)
    }
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Analysis of {filename}")
    print(f"{'='*60}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"\npf_rhoO3_X1toXn:")
    print(f"  Mean: {results['pf_rhoO3_X1toXn_mean']:.6f}")
    print(f"  Std:  {results['pf_rhoO3_X1toXn_std']:.6f}")
    print(f"\npf_rhoO3_Z1Z2:")
    print(f"  Mean: {results['pf_rhoO3_Z1Z2_mean']:.6f}")
    print(f"  Std:  {results['pf_rhoO3_Z1Z2_std']:.6f}")
    print(f"{'='*60}\n")

    
    return results



def analyze_data_fidelity(filename):
    df = pd.read_csv(filename)
    
    # 处理可能是字符串形式的数据（由于数组或复数被保存为字符串）
    def convert_to_float(val):
        if isinstance(val, str):
            # 尝试解析复数格式 "(real+imagj)"
            val = val.strip().strip('[]')
            try:
                # 先尝试作为复数解析
                complex_val = complex(val)
                # 提取实部
                return float(complex_val.real)
            except (ValueError, AttributeError):
                # 如果失败，直接转浮点数
                return float(val)
        return float(val)
    df['pf_rhoO2_fidelity'] = df['pf_rhoO2_fidelity'].apply(convert_to_float)
    df['pf_rhoO3_fidelity'] = df['pf_rhoO3_fidelity'].apply(convert_to_float)
    results = {
        'pf_rhoO2_fidelity_mean': df['pf_rhoO2_fidelity'].mean(),
        'pf_rhoO2_fidelity_std': df['pf_rhoO2_fidelity'].std(),
        'pf_rhoO3_fidelity_mean': df['pf_rhoO3_fidelity'].mean(),
        'pf_rhoO3_fidelity_std': df['pf_rhoO3_fidelity'].std(),
        'num_samples': len(df)
    }
    print(f"\n{'='*60}")
    print(f"Analysis of {filename}")
    print(f"{'='*60}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"\npf_rhoO2_fidelity:")
    print(f"  Mean: {results['pf_rhoO2_fidelity_mean']:.6f}")
    print(f"  Std:  {results['pf_rhoO2_fidelity_std']:.6f}")
    print(f"\npf_rhoO3_fidelity:")
    print(f"  Mean: {results['pf_rhoO3_fidelity_mean']:.6f}")
    print(f"  Std:  {results['pf_rhoO3_fidelity_std']:.6f}")
    print(f"{'='*60}\n")

    return results


def main_experiment_gibbs(n, Nu, NM, rep, beta):
    d = pow(2, n)
    data = []
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    O_list = []
    psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h1.0.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)

    for idx in tqdm(range(rep)):
        purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, 3, tmin=1, dprob=dprob)
        # 存储每条数据：idx, purity, third_moment, 以及两个可观测量的三个估计值
        data.append({
            'idx': idx,
            'purity': purity,
            'third_moment': third_moment,
            'pf_rhoO2_fidelity': pf_rhoO2[0],
            'pf_rhoO3_fidelity': pf_rhoO3[0]
        })
    
    # 转为 DataFrame
    df = pd.DataFrame(data)
    
    # 保存文件，文件名包含 n, Nu, NM, dprob
    filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def _experiment_gibbs_wrapper(args):
    """多进程包装函数，用于 experiment_single 调用"""
    n, Nu, NM, t, tmin, beta, idx = args
    d = pow(2, n)
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    O_list = []
    psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h1.0.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)
    purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=tmin, dprob=dprob)
    return {
        'idx': idx,
        'purity': purity,
        'third_moment': third_moment,
        'pf_rhoO3_fidelity': pf_rhoO3[0]
    }

def main_experiment_gibbs_parallel(n, Nu, NM, rep, beta=3.0, num_processes=None):
    """并行版本的主实验函数，使用 multiprocessing
    
    参数:
        n: 量子比特数
        Nu: unitary数量
        NM: 测量数量
        rep: 重复次数
        beta: 逆温参数
        num_processes: 进程数，如果为None则使用CPU核心数
        
    返回:
        DataFrame: 实验结果
    """
    if num_processes is None:
        num_processes = cpu_count()


    # 准备参数列表
    args_list = [(n, Nu, NM, 3, 1, beta, idx) for idx in range(rep)]
    
    # 使用多进程池执行实验
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper, args_list), total=rep))
    
    # 转为 DataFrame
    df = pd.DataFrame(results)
    
    # 保存文件，文件名包含 n, Nu, NM, beta
    filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def _experiment_gibbs_wrapper_anc(args):
    """多进程包装函数，用于 experiment_single 调用，包含ancilla版本"""
    n, Nu, NM, t, tmin, beta, anc, idx = args
    d = pow(2, n)
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    O_list = []
    # Define X_1 X_2 ... X_n and Z_1 Z_2
    X1toXn = np.eye(1, dtype=np.complex128)
    for i in range(n):
        X1toXn = np.kron(X1toXn, X)
    O_list.append(X1toXn)
    Z1Z2 = np.eye(1, dtype=np.complex128)
    for i in range(n):
        if i == 0 or i == 1:
            Z1Z2 = np.kron(Z1Z2, Z)
        else:
            Z1Z2 = np.kron(Z1Z2, np.eye(2, dtype=np.complex128))
    O_list.append(Z1Z2)
    t_list = list(range(1, t + 1))
    rhot_estimates, rhotO_estimates = estimate_rhotAndrhotO_tlist_dprob_ancilla(rho_test, O_list, n, Nu, NM, t_list, anc=anc, dprob=0)

    rho1 = rhot_estimates[1]
    rho2 = rhot_estimates[2]
    purity = rho2 - 1
    rho3 = rhot_estimates[3]
    third_moment = (rho3 - 3 * purity - 1) / 2

    rhoO1 = np.array(rhotO_estimates[1])
    rhoO2 = np.array(rhotO_estimates[2])
    rhoO3 = np.array(rhotO_estimates[3])
    rho2O = (rhoO2 - 2 * rhoO1) / 2
    rho3O = (rhoO3 - 6 * rho2O - 3 * (1 + purity) * rhoO1) / 6

    pf_rhoO1 = rhoO1
    pf_rhoO2 = rho2O / purity
    pf_rhoO3 = rho3O / third_moment

    return {
        'idx': idx,
        'purity': purity,
        'third_moment': third_moment,
        'pf_rhoO1_X1toXn': pf_rhoO1[0],
        'pf_rhoO2_X1toXn': pf_rhoO2[0],
        'pf_rhoO3_X1toXn': pf_rhoO3[0],
        'pf_rhoO1_Z1Z2': pf_rhoO1[1],
        'pf_rhoO2_Z1Z2': pf_rhoO2[1],
        'pf_rhoO3_Z1Z2': pf_rhoO3[1]
    }


def _experiment_gibbs_wrapper_anc_multi_nm(args):
    """Multi-process wrapper for ancilla experiment with NM reuse."""
    n, Nu, NM_list, t, tmin, beta, anc, idx = args
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    O_list = []
    # Define X_1 X_2 ... X_n and Z_1 Z_2
    X1toXn = np.eye(1, dtype=np.complex128)
    for i in range(n):
        X1toXn = np.kron(X1toXn, X)
    O_list.append(X1toXn)
    Z1Z2 = np.eye(1, dtype=np.complex128)
    for i in range(n):
        if i == 0 or i == 1:
            Z1Z2 = np.kron(Z1Z2, Z)
        else:
            Z1Z2 = np.kron(Z1Z2, np.eye(2, dtype=np.complex128))
    O_list.append(Z1Z2)
    t_list = list(range(tmin, t + 1))

    rhot_by_nm, rhotO_by_nm = estimate_rhotAndrhotO_tlist_dprob_ancilla_multi_nm(
        rho_test, O_list, n, Nu, NM_list, t_list, anc=anc, dprob=0
    )

    rows = []
    for NM in sorted(int(x) for x in NM_list):
        rhot_estimates = rhot_by_nm[NM]
        rhotO_estimates = rhotO_by_nm[NM]
        rho1 = rhot_estimates[1]
        rho2 = rhot_estimates[2]
        purity = rho2 - 1
        rho3 = rhot_estimates[3]
        third_moment = (rho3 - 3 * purity - 1) / 2

        rhoO1 = np.array(rhotO_estimates[1])
        rhoO2 = np.array(rhotO_estimates[2])
        rhoO3 = np.array(rhotO_estimates[3])
        rho2O = (rhoO2 - 2 * rhoO1) / 2
        rho3O = (rhoO3 - 6 * rho2O - 3 * (1 + purity) * rhoO1) / 6

        pf_rhoO1 = rhoO1
        pf_rhoO2 = rho2O / purity
        pf_rhoO3 = rho3O / third_moment

        rows.append({
            'idx': idx,
            'NM': NM,
            'purity': purity,
            'third_moment': third_moment,
            'pf_rhoO1_X1toXn': pf_rhoO1[0],
            'pf_rhoO2_X1toXn': pf_rhoO2[0],
            'pf_rhoO3_X1toXn': pf_rhoO3[0],
            'pf_rhoO1_Z1Z2': pf_rhoO1[1],
            'pf_rhoO2_Z1Z2': pf_rhoO2[1],
            'pf_rhoO3_Z1Z2': pf_rhoO3[1],
        })
    return rows

def main_experiment_gibbs_pauli_anc_parallel(n, Nu, NM, rep, beta, anc, num_processes=None):
    # 这里可以实现基于Pauli测量和ancilla的版本
    if num_processes is None:
        num_processes = cpu_count()
    
    # 准备参数列表
    args_list = [(n, Nu, NM, 3, 1, beta, anc, idx) for idx in range(rep)]

    # 使用多进程池执行实验
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper_anc, args_list), total=rep))
    # 转为 DataFrame
    df = pd.DataFrame(results)
    # 保存文件，文件名包含 n, Nu, NM, beta, anc
    filename = f"./data/gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main_experiment_gibbs_pauli_anc_parallel_nm_sweep(n, Nu, NM_list, rep, beta, anc, num_processes=None):
    """
    Fast NM sweep:
    for each repetition, sample once at max(NM_list), then reuse prefixes for smaller NM.
    """
    if num_processes is None:
        num_processes = cpu_count()
    NM_list = [int(x) for x in NM_list]
    if len(NM_list) == 0:
        raise ValueError("NM_list should not be empty")

    args_list = [(n, Nu, NM_list, 3, 1, beta, anc, idx) for idx in range(rep)]
    with Pool(processes=num_processes) as pool:
        nested_results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper_anc_multi_nm, args_list), total=rep))

    flat_results = [row for rows in nested_results for row in rows]
    df_all = pd.DataFrame(flat_results)
    filename_all = f"./data/gibbs_pauli_anc_n{n}_Nu{Nu}_NMmulti_beta{beta}_anc{anc}.csv"
    df_all.to_csv(filename_all, index=False)
    print(f"Combined NM-sweep results saved to {filename_all}")

    # Keep compatibility with existing downstream code by writing one file per NM.
    for NM in sorted(set(NM_list)):
        df_nm = df_all[df_all["NM"] == NM].drop(columns=["NM"])
        filename_nm = f"./data/gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
        df_nm.to_csv(filename_nm, index=False)
        print(f"Results saved to {filename_nm}")

    return df_all


def _build_tfim_pauli_dense(n, J=1.0, h=1.0):
    """Build dense TFIM Hamiltonian H = -J sum ZZ - h sum X (open boundary)."""
    d = 1 << n
    I2 = np.eye(2, dtype=np.complex128)
    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(n - 1):
        term = np.eye(1, dtype=np.complex128)
        for site in range(n):
            if site == i or site == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I2)
        H += -J * term
    for i in range(n):
        term = np.eye(1, dtype=np.complex128)
        for site in range(n):
            if site == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I2)
        H += -h * term
    return H


def _experiment_gibbs_wrapper_anc_multi_nm_psib_hising(args):
    """Multi-process wrapper using observables: psib and TFIM Hamiltonian."""
    n, Nu, NM_list, t, tmin, beta, anc, idx = args
    d = 1 << n
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h1.0.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    H_ising = _build_tfim_pauli_dense(n=n, J=1.0, h=1.0)
    O_list = [psib, H_ising]
    t_list = list(range(tmin, t + 1))

    rhot_by_nm, rhotO_by_nm = estimate_rhotAndrhotO_tlist_dprob_ancilla_multi_nm(
        rho_test, O_list, n, Nu, NM_list, t_list, anc=anc, dprob=0
    )

    rows = []
    for NM in sorted(int(x) for x in NM_list):
        rhot_estimates = rhot_by_nm[NM]
        rhotO_estimates = rhotO_by_nm[NM]
        rho2 = rhot_estimates[2]
        purity = rho2 - 1
        rho3 = rhot_estimates[3]
        third_moment = (rho3 - 3 * purity - 1) / 2

        rhoO1 = np.array(rhotO_estimates[1])
        rhoO2 = np.array(rhotO_estimates[2])
        rhoO3 = np.array(rhotO_estimates[3])
        rho2O = (rhoO2 - 2 * rhoO1) / 2
        rho3O = (rhoO3 - 6 * rho2O - 3 * (1 + purity) * rhoO1) / 6

        pf_rhoO1 = rhoO1
        pf_rhoO2 = rho2O / purity
        pf_rhoO3 = rho3O / third_moment

        rows.append({
            'idx': idx,
            'NM': NM,
            'purity': purity,
            'third_moment': third_moment,
            'pf_rhoO1_psib': pf_rhoO1[0],
            'pf_rhoO2_psib': pf_rhoO2[0],
            'pf_rhoO3_psib': pf_rhoO3[0],
            'pf_rhoO1_Hising': pf_rhoO1[1],
            'pf_rhoO2_Hising': pf_rhoO2[1],
            'pf_rhoO3_Hising': pf_rhoO3[1],
        })
    return rows


def main_experiment_gibbs_pauli_anc_parallel_nm_sweep_psib_hising(
    n, Nu, NM_list, rep, beta, anc, num_processes=None
):
    """
    NM sweep with ancilla, observables fixed to:
    1) psib = |psi0><psi0| - I/d
    2) Hising (TFIM Hamiltonian with J=1, h=1)
    """
    if num_processes is None:
        num_processes = cpu_count()
    NM_list = [int(x) for x in NM_list]
    if len(NM_list) == 0:
        raise ValueError("NM_list should not be empty")

    args_list = [(n, Nu, NM_list, 3, 1, beta, anc, idx) for idx in range(rep)]
    with Pool(processes=num_processes) as pool:
        nested_results = list(
            tqdm(pool.imap_unordered(_experiment_gibbs_wrapper_anc_multi_nm_psib_hising, args_list), total=rep)
        )

    flat_results = [row for rows in nested_results for row in rows]
    df_all = pd.DataFrame(flat_results)
    filename_all = f"./data/gibbs_psibH_anc_n{n}_Nu{Nu}_NMmulti_beta{beta}_anc{anc}.csv"
    df_all.to_csv(filename_all, index=False)
    print(f"Combined NM-sweep results saved to {filename_all}")

    for NM in sorted(set(NM_list)):
        df_nm = df_all[df_all["NM"] == NM].drop(columns=["NM"])
        filename_nm = f"./data/gibbs_psibH_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
        df_nm.to_csv(filename_nm, index=False)
        print(f"Results saved to {filename_nm}")

    return df_all


def compute_statistical_error(n, Nu):
    rho = np.load(f'./TFIM/tfim_gibbs_N{n}_beta2.0_J1.0_h1.0.npy')
    X1toXn = np.eye(1, dtype=np.complex128)
    for i in range(n):
        X1toXn = np.kron(X1toXn, X)
    Z1Z2 = np.eye(1, dtype=np.complex128)
    for i in range(n):
        if i == 0 or i == 1:
            Z1Z2 = np.kron(Z1Z2, Z)
        else:
            Z1Z2 = np.kron(Z1Z2, np.eye(2, dtype=np.complex128))
    O_list = [X1toXn, Z1Z2]
    rho3 = rho @ rho @ rho
    rho3O_X1toXn = np.trace(rho3 @ X1toXn)
    rho3O_Z1Z2 = np.trace(rho3 @ Z1Z2)
    exact_values = {
        'rho3O_X1toXn': rho3O_X1toXn,
        'rho3O_Z1Z2': rho3O_Z1Z2
    }

    def block_mean_mae(values, exact_val, block_size):
        """Compute MAE after averaging every block_size samples."""
        arr = np.asarray(values, dtype=float)
        if arr.size < block_size:
            return np.nan
        usable = (arr.size // block_size) * block_size
        arr_block = arr[:usable].reshape(-1, block_size).mean(axis=1)
        return float(np.mean(np.abs(arr_block - exact_val)))

    result = []
    for anc in range(9):
        anc_list = []
        for NM in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]:
                df = pd.read_csv(f'./data/gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta2.0_anc{anc}.csv')
                def convert_to_float(val):
                    if isinstance(val, str):
                        # 尝试解析复数格式 "(real+imagj)"
                        val = val.strip().strip('[]')
                        try:
                            # 先尝试作为复数解析
                            complex_val = complex(val)
                            # 提取实部
                            return float(complex_val.real)
                        except (ValueError, AttributeError):
                            # 如果失败，直接转浮点数
                            return float(val)
                    return float(val)
                df['pf_rhoO3_X1toXn'] = df['pf_rhoO3_X1toXn'].apply(convert_to_float) * df['third_moment']
                df['pf_rhoO3_Z1Z2'] = df['pf_rhoO3_Z1Z2'].apply(convert_to_float) * df['third_moment']
                mae_X = np.mean(np.abs(df['pf_rhoO3_X1toXn'] - np.real(exact_values['rho3O_X1toXn'])))
                mae_Z = np.mean(np.abs(df['pf_rhoO3_Z1Z2'] - np.real(exact_values['rho3O_Z1Z2'])))
                mae_X_avg10 = block_mean_mae(df['pf_rhoO3_X1toXn'], np.real(exact_values['rho3O_X1toXn']), 10)
                mae_Z_avg10 = block_mean_mae(df['pf_rhoO3_Z1Z2'], np.real(exact_values['rho3O_Z1Z2']), 10)
                mae_X_avg50 = block_mean_mae(df['pf_rhoO3_X1toXn'], np.real(exact_values['rho3O_X1toXn']), 50)
                mae_Z_avg50 = block_mean_mae(df['pf_rhoO3_Z1Z2'], np.real(exact_values['rho3O_Z1Z2']), 50)

                anc_list.append({
                    'NM': NM,
                    'MAE_X': mae_X,
                    'MAE_Z': mae_Z,
                    'MAE_X_avg10': mae_X_avg10,
                    'MAE_Z_avg10': mae_Z_avg10,
                    'MAE_X_avg50': mae_X_avg50,
                    'MAE_Z_avg50': mae_Z_avg50
                })
        result.append({'anc': anc, 'data': anc_list})

    final_data = []
    for item in result:
        anc_val = item['anc']
        for data_point in item['data']:
            final_data.append({
                'anc': anc_val,
                'NM': data_point['NM'],
                'MAE_X': data_point['MAE_X'],
                'MAE_Z': data_point['MAE_Z'],
                'MAE_X_avg10': data_point['MAE_X_avg10'],
                'MAE_Z_avg10': data_point['MAE_Z_avg10'],
                'MAE_X_avg50': data_point['MAE_X_avg50'],
                'MAE_Z_avg50': data_point['MAE_Z_avg50']
            })
            
    df_res = pd.DataFrame(final_data)
    output_filename = f'./data/statistical_error_n{n}_Nu{Nu}_beta2.0.csv'
    df_res.to_csv(output_filename, index=False)
    print(f"Statistical error results saved to {output_filename}")
    return df_res

def main_test_t3_with_purity_and_third_moment(n, N, eps, dprob, rep):
    d = pow(2, n)
    NM = int(min(N, np.ceil(10 * np.power(d/eps, 2/3))))
    # NM = int(min(N, np.ceil(2 * np.power(d/eps/eps, 1/2))))
    # NM = int(min(N, np.ceil(np.power(d, 2/3))))
    Nu = int(max(1, np.floor(N / NM)))
    result = test_t3_with_purity_and_third_moment(n, Nu, NM, dprob, rep)
    return result


def run_purity_third_moment_mae_sweep(
    n=8,
    N_list=(10, 25, 50, 100, 250, 500, 1000),
    eps=0.1,
    dprob=0.2,
    rep=100,
    out_csv="data/purity_and_third_moment_mae_list.csv",
):
    rows = []
    for N in N_list:
        result = main_test_t3_with_purity_and_third_moment(n, N, eps, dprob, rep)
        rows.append(
            {
                "N": int(N),
                "purity_mae": result["purity_mae"],
                "third_moment_mae": result["third_moment_mae"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved sweep results to {out_csv}")
    return df


def run_purity_third_moment_vs_n_fixed_budget(
    n_list=(2, 3, 4, 5, 6, 7, 8),
    dprob=0.2,
    rep=100,
    prefactor=10.0,
    out_csv="data/purity_and_third_moment_vs_n_fixedN.csv",
):
    """
    Sweep over n with fixed total budget:
      N = ceil(prefactor * d^(2/3)), d = 2^n
    For each n, run estimator and output purity / third_moment.
    """
    rows = []
    for n in n_list:
        d = 2 ** int(n)
        N = int(np.ceil(prefactor * (d/0.1 ** (2.0 / 3.0))))
        # eps=1.0 gives NM=min(N, ceil(10*(d/eps)^(2/3))) ~= N in main_test helper.
        result = main_test_t3_with_purity_and_third_moment(n=int(n), N=N, eps=1.0, dprob=dprob, rep=rep)
        rows.append(
            {
                "n": int(n),
                "d": int(d),
                "N": int(N),
                "purity": float(result["purity"]),
                "third_moment": float(result["third_moment"]),
                "theory_purity": float(result["theory_purity"]),
                "theory_third_moment": float(result["theory_third_moment"]),
                "purity_mae": float(result["purity_mae"]),
                "third_moment_mae": float(result["third_moment_mae"]),
            }
        )
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"Saved n-sweep results to {out_csv}")
    return df


def find_min_N_for_target_mae(
    n=8,
    eps=0.1,
    dprob=0.2,
    rep=100,
    target_mae=0.02,
    metric="both",
    N_start=10,
    N_step=10,
    N_max=5000,
    out_csv=None,
):
    """
    Increase N step-by-step and return the minimal N that reaches target MAE.

    metric:
      - "purity": stop at first purity_mae <= target_mae
      - "third_moment": stop at first third_moment_mae <= target_mae
      - "both": stop only after BOTH metrics have reached target at least once;
                reported purity N / third N are their own first-hit N.
    """
    if metric not in {"purity", "third_moment", "both"}:
        raise ValueError("metric must be one of: 'purity', 'third_moment', 'both'")
    if N_step <= 0:
        raise ValueError("N_step must be positive")
    if N_start <= 0 or N_max < N_start:
        raise ValueError("Require 0 < N_start <= N_max")

    rows = []
    hit_row = None
    first_hit_purity = None
    first_hit_third = None

    for N in range(int(N_start), int(N_max) + 1, int(N_step)):
        result = main_test_t3_with_purity_and_third_moment(n, N, eps, dprob, rep)
        row = {
            "N": int(N),
            "purity_mae": float(result["purity_mae"]),
            "third_moment_mae": float(result["third_moment_mae"]),
        }
        rows.append(row)

        ok_purity = row["purity_mae"] <= target_mae
        ok_third = row["third_moment_mae"] <= target_mae
        if ok_purity and first_hit_purity is None:
            first_hit_purity = row.copy()
        if ok_third and first_hit_third is None:
            first_hit_third = row.copy()

        if (metric == "purity" and ok_purity) or (metric == "third_moment" and ok_third) or (
            metric == "both" and first_hit_purity is not None and first_hit_third is not None
        ):
            hit_row = row
            break

    df_hist = pd.DataFrame(rows)
    if out_csv is not None:
        df_hist.to_csv(out_csv, index=False)
        print(f"Saved N-search history to {out_csv}")

    if hit_row is None:
        print(f"Target not reached up to N={N_max}.")
        return {
            "found": False,
            "min_N": None,
            "purity_first_hit_N": None if first_hit_purity is None else first_hit_purity["N"],
            "third_moment_first_hit_N": None if first_hit_third is None else first_hit_third["N"],
            "history": df_hist,
        }

    print(
        f"Target reached at N={hit_row['N']}: "
        f"purity_mae={hit_row['purity_mae']:.8e}, "
        f"third_moment_mae={hit_row['third_moment_mae']:.8e}"
    )
    return {
        "found": True,
        "min_N": hit_row["N"],
        "purity_first_hit_N": None if first_hit_purity is None else first_hit_purity["N"],
        "third_moment_first_hit_N": None if first_hit_third is None else first_hit_third["N"],
        "purity_first_hit": first_hit_purity,
        "third_moment_first_hit": first_hit_third,
        "hit": hit_row,
        "history": df_hist,
    }


if __name__ == "__main__":
    n = 6
    Nu = 1
    NM = 1000
    rep = 100
    dprob = 0
    beta = 2.0
    anc = 1
    # main_experiment_gibbs(n, Nu, NM, rep, beta)
    # main_experiment(n, Nu=1, NM=10000, rep=100, dprob=0.2)
    # analyze_results(f"./data/experiment_n{n}_Nu{Nu}_NM{NM}_dprob{dprob}.csv")
    # analyze_data_fidelity(f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv")
    # main_experiment_gibbs_pauli_anc_parallel(n, Nu, NM, rep, beta, anc, num_processes=4)
    # compute_statistical_error(n, 1)
    # run_purity_third_moment_mae_sweep(
    #     n=8,
    #     N_list=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
    #     eps=0.1,
    #     dprob=0.2,
    #     rep=100,
    #     out_csv="data/purity_and_third_moment_mae_list.csv",
    # )
    # find_min_N_for_target_mae(
    # n=7,
    # eps=0.1,
    # dprob=0.2,
    # rep=1000,
    # target_mae=0.1,
    # metric="purity",
    # N_start=600,
    # N_step=20,
    # N_max=5000,
    # out_csv=None,
    # )
    # df = run_purity_third_moment_vs_n_fixed_budget(
    # n_list=(3, 4, 5, 6, 7, 8),
    # dprob=0.2,
    # rep=100,
    # prefactor=2.0,  # 对应 N = ceil(10 * d^(2/3))
    # out_csv="data/purity_and_third_moment_vs_n_fixedN.csv",
    # )
    # print(df)
    for n in range(6, 18):
        d = 2 ** n
        N = int(np.ceil(2 * (pow(d/0.1, 2.0 / 3.0))))
        print(f"n: {n}, N: {N}")
