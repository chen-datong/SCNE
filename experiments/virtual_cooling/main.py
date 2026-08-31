import numpy as np
from scipy.stats import unitary_group
import math
import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def estimate_rhot_dprob(rho, n, Nu, NM, t, dprob=0):
    d = pow(2, n)
    total_counts = 0
    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        counts = np.random.multinomial(NM, prob)
        denom = math.comb(NM, t) * d
        numer = math.comb(d + t - 1, t) * math.factorial(t) 
        total = 0
        for count in counts:
            if count >= t:
                total += math.comb(count, t)
        total_counts += total * numer / denom

    return total_counts / Nu

def estimate_rhotO_dprob(rho, O, n, Nu, NM, t, dprob=0):
    d = pow(2, n)
    total_counts = 0
    for _ in range(Nu):
        U = unitary_group.rvs(d)
        rho_U = U @ rho @ U.conj().T
        O_U = U @ O @ U.conj().T
        prob = np.real(np.diag(rho_U))
        prob = prob * (1 - dprob) + dprob / d
        counts = np.random.multinomial(NM, prob)
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
        counts = np.random.multinomial(NM, prob)
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
        counts = np.random.multinomial(NM, prob)
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

def test(n):
    d = pow(2, n)
    ghz_vec = np.zeros(d, dtype=np.complex128)
    ghz_vec[0] = 1/np.sqrt(2)
    ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 1
    NM = 100000
    t = 2
    dprob = 0
    purity = estimate_rhot_dprob(rho_test, n, Nu, NM, t, dprob)
    print(f"Estimated purity for GHZ state with n={n}: {purity}")
    return purity

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


def main_experiment_gibbs(n, Nu, NM, rep, beta, h=1.0, output_axis="beta"):
    d = pow(2, n)
    data = []
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy')
    O_list = []
    psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h{h}.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)

    for idx in tqdm(range(rep)):
        purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, 3, tmin=1, dprob=0)
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
    if output_axis == "beta":
        filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    elif output_axis == "h":
        filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_h{h}.csv"
    else:
        raise ValueError("output_axis must be 'beta' or 'h'")
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def _experiment_gibbs_wrapper(args):
    """多进程包装函数，用于 experiment_single 调用"""
    n, Nu, NM, t, tmin, beta, h, idx = args
    d = pow(2, n)
    rho_test = np.load(f'./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy')
    O_list = []
    psi0 = np.load(f'./TFIM/tfim_N{n}_gs_J1.0_h{h}.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)
    purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=tmin, dprob=0)
    return {
        'idx': idx,
        'purity': purity,
        'third_moment': third_moment,
        'pf_rhoO3_fidelity': pf_rhoO3[0]
    }

def main_experiment_gibbs_parallel(
    n,
    Nu,
    NM,
    rep,
    beta=3.0,
    h=1.0,
    output_axis="beta",
    num_processes=None,
):
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
    args_list = [(n, Nu, NM, 3, 1, beta, h, idx) for idx in range(rep)]
    
    # 使用多进程池执行实验
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper, args_list), total=rep))
    
    # 转为 DataFrame
    df = pd.DataFrame(results)
    
    # 保存文件，文件名包含 n, Nu, NM, beta
    if output_axis == "beta":
        filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    elif output_axis == "h":
        filename = f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_h{h}.csv"
    else:
        raise ValueError("output_axis must be 'beta' or 'h'")
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df



if __name__ == "__main__":
    n = 14
    Nu = 1
    NM = 1_000_000
    rep = 100
    dprob = 0
    beta = 0.5
    main_experiment_gibbs(n, Nu, NM, rep, beta, h=1.0, output_axis="beta")
    # main_experiment(n, Nu=1, NM=10000, rep=100, dprob=0.2)
    # analyze_results(f"./data/experiment_n{n}_Nu{Nu}_NM{NM}_dprob{dprob}.csv")
    analyze_data_fidelity(f"./data/fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv")
