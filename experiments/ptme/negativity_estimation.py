import numpy as np
from scipy.stats import unitary_group
import itertools
from scipy import special
import pandas as pd
import os
import time
from exact_val import calculate_S3
from tqdm import tqdm
import math
# from mpi4py import MPI
import multiprocessing



def get_dims(n_qubits):
    """Calculate the dimension of the Hilbert space for a given number of qubits."""
    return 2**n_qubits

def get_computational_basis_vectors(dim):
    """Generate computational basis vectors for a given dimension."""
    basis = []
    for i in range(dim):
        vec = np.zeros(dim)
        vec[i] = 1
        basis.append(vec)
    return basis

def get_computational_measurement(dim):
    """Generate Computational basis POVM for a given dimension"""
    basis = get_computational_basis_vectors(dim)
    POVM = []
    for b in basis:
        POVM.append(np.outer(b, b.conj()))
    return POVM

def tensor_product(*args):
    """Calculate the tensor product of multiple matrices or vectors."""
    result = args[0]
    for i in range(1, len(args)):
        result = np.kron(result, args[i])
    return result


def get_swap_test_povm_operators(d_A2, d_B):
    """Generate POVM operators for the swap test."""
    if d_A2 != d_B:
        raise ValueError("d_A2 must be equal to d_B for the swap test.")
    d = d_A2
    dim = d * d
    
    I = np.eye(dim)
    
    SWAP = np.zeros((dim, dim), dtype=complex)
    for i in range(d):
        for j in range(d):
            ket_i = np.zeros(d); ket_i[i] = 1
            ket_j = np.zeros(d); ket_j[j] = 1
            
            ket_ij = tensor_product(ket_i, ket_j)
            ket_ji = tensor_product(ket_j, ket_i)
            
            SWAP += np.outer(ket_ji, ket_ij)
            
    Pi_sym = (I + SWAP) / 2
    Pi_asym = (I - SWAP) / 2
    
    return [Pi_sym, Pi_asym] # Corresponds to r = +1 and r = -1


def estimate_negativity_moment(rho, nA, nB, t, Nu, NM):
    """
    Simulates the full protocol from the paper to estimate the t-th negativity moment.
    This version uses an optimized estimator calculation.
    """
    if nA < nB:
        raise ValueError("Protocol requires nA >= nB.")

    nA1 = nA - nB
    nA2 = nB
    
    d_A1 = get_dims(nA1)
    d_A2 = get_dims(nA2)
    d_B = get_dims(nB)
    
    
    computational_basis = get_computational_measurement(d_A1)
    swap_povms = get_swap_test_povm_operators(d_A2, d_B)
    
    M_neg_estimates = []

    for q in range(Nu):
        # if q % 1000 == 0:
        #     print(f"Running for random unitary {q+1}/{Nu}...")
        d_A = d_A1 * d_A2
        Uq = unitary_group.rvs(d_A)
        d_Y = d_A2 * d_B
        # The unitary operator U_op should act as Uq on subsystem A and identity on subsystem B.
        # Since the subsystems are ordered [A1, A2, A3, B], we need to construct the operator for A 
        # and then tensor it with the identity for B.
        # The existing `tensor_product` helper is `kron`, which is what we need.
        U_op = np.kron(Uq, np.eye(d_B))
        rho_Uq = U_op @ rho @ U_op.conj().T
        

        rho_reshaped = rho_Uq.reshape(d_A1, d_Y, d_A1, d_Y)

        probabilities = []

        for Pi_r in swap_povms:
            rho_post_r = np.einsum('ijkl,lj->ik', rho_reshaped, Pi_r)
            for P_b in computational_basis:
                prob = np.real(np.trace(rho_post_r @ P_b))
                probabilities.append(prob)
        probabilities = np.array(probabilities) / np.sum(probabilities)
        num_b_outcomes = len(computational_basis)
        # Sample NM outcomes from the joint distribution
        outcomes = np.random.multinomial(NM, probabilities)
        # Group r outcomes by b, storing only counts of +1 and -1 for efficiency
        b_to_rs = {}  # {b_idx: [n_plus, n_minus]}
        outcome_idx = 0
        for count in outcomes:
            if count > 0:
                b = outcome_idx % num_b_outcomes
                r_idx = outcome_idx // num_b_outcomes
                if b not in b_to_rs:
                    b_to_rs[b] = [0, 0]

                if r_idx == 0:  # r = +1
                    b_to_rs[b][0] += count
                else:  # r = -1
                    b_to_rs[b][1] += count
            outcome_idx += 1

        # Calculate the total sum S_q using the optimized method
        total_sum = 0
        for b in b_to_rs:
            n_plus, n_minus = b_to_rs[b]
            
            sum_for_b = 0
            for k in range(t + 1):
                if k > n_minus or (t - k) > n_plus:
                    continue
                
                term = ((-1)**k) * \
                       special.comb(n_minus, k, exact=True) * \
                       special.comb(n_plus, t - k, exact=True)
                sum_for_b += term
            
            total_sum += sum_for_b

        # Calculate final M_neg_q for this unitary
        M_neg_q = 0
        num_total_combinations = special.comb(NM, t, exact=True)

        if num_total_combinations > 0:
            prefactor = d_A1**(t-1) * d_A2**t
            M_neg_q = prefactor * total_sum / num_total_combinations

        M_neg_estimates.append(M_neg_q)

    final_M_neg = np.mean(M_neg_estimates)
    return final_M_neg

def experiment_GHZ(nA, nB, repeat, csv_filename='experiment_results.csv'):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B
    ghz_vec = np.zeros(dim_total); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 1
    NM = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    experiment = []
    for n in NM:
        print(f'Running for NM = {n}')
        results = []
        for i in tqdm(range(repeat)):
            result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, n)
            results.append(np.abs(result - 4.5))
        experiment.append([nA, nB, Nu, n, np.mean(results)])
    df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'result'])
    file_exists = os.path.exists(csv_filename)
    
    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    return df

# def _worker_GHZ(args):
#     """Helper function for multiprocessing in experiment_GHZ_mp."""
#     rho_test, nA, nB, t, Nu, n = args
#
#     result = estimate_negativity_moment(rho_test, nA, nB, t, Nu, n)
#     return np.abs(result - 4.5)

# def experiment_GHZ_mp(nA, nB, repeat, csv_filename='experiment_results.csv'):
#     d_A = get_dims(nA)
#     d_B = get_dims(nB)
#     dim_total = d_A * d_B
#     ghz_vec = np.zeros(dim_total); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
#     rho_test = np.outer(ghz_vec, ghz_vec.conj())
#     Nu = 1
#     NM = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
#     experiment = []
#
#     for n in NM:
#         print(f'Running for NM = {n}')
#
#         args_list = [(rho_test, nA, nB, 3, Nu, n)] * repeat
#
#         with multiprocessing.Pool(50) as pool:
#             results = list(tqdm(pool.imap_unordered(_worker_GHZ, args_list), total=repeat))
#
#         experiment.append([nA, nB, Nu, n, np.mean(results)])
#
#     df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'result'])
#     file_exists = os.path.exists(csv_filename)
#
#     df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
#     return df
    

def search_converge(nA, nB, step, start, eps, repeat, csv_file='converge.csv'):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B
    ghz_vec = np.zeros(dim_total); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 10
    prev_error = 1
    results = []
    for _ in range(repeat):
        result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, start)
        results.append(np.abs(result - 4.5))
    now_error = np.mean(results)
    print(now_error)
    while prev_error - now_error >= eps:
        print(f'Search for NM = {start + step}')
        prev_error = now_error
        start += step
        results = []
        for _ in range(repeat):
            result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, start)
            results.append(np.abs(result - 4.5))
        now_error = np.mean(results)
        print(now_error)
    df = pd.DataFrame([[nA, nB, Nu, start, eps]], columns=['nA', 'nB', 'Nu', 'NM', 'eps'])
    file_exists = os.path.exists(csv_file)
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    return start
        

def search_converge_MPI(nA, nB, step, start, eps, repeat, csv_file='converge.csv'):
    # MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 当前进程的 ID
    size = comm.Get_size()  # 总进程数

    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B
    ghz_vec = np.zeros(dim_total); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 10
    prev_error = 1
    results = []
    for _ in range(rank, repeat, size):
        result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, start)
        results.append(np.abs(result - 4.5))
    all_results = comm.gather(results, root=0)
    if rank == 0:
        all_results_flat = [item for sublist in all_results for item in sublist]
        now_error = np.mean(all_results_flat)
    while prev_error - now_error >= eps:
        print(f'Search for NM = {start + step}')
        prev_error = now_error
        start += step
        results = []
        for _ in range(rank, repeat, size):
            result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, start)
            results.append(np.abs(result - 4.5))
        all_results = comm.gather(results, root=0)
        if rank == 0:
            all_results_flat = [item for sublist in all_results for item in sublist]
            now_error = np.mean(all_results_flat)
    if rank == 0:
        df = pd.DataFrame([[nA, nB, Nu, start, eps]], columns=['nA', 'nB', 'Nu', 'NM', 'eps'])
        file_exists = os.path.exists(csv_file)
        df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    return start


def experiment_GHZ_MPI(nA, nB, repeat, csv_filename='experiment_results.csv'):
    # MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 当前进程的 ID
    size = comm.Get_size()  # 总进程数

    # 设置实验参数
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B
    ghz_vec = np.zeros(dim_total)
    ghz_vec[0] = 1 / np.sqrt(2)
    ghz_vec[-1] = 1 / np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())
    Nu = 10
    NM = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    experiment = []

    # 并行化 repeat 循环
    for n in NM:
        print(f'Running for NM = {n}')
        results = []

        # 分割 repeat 循环任务
        for i in range(rank, repeat, size):  # 以进程间轮流分配任务
            result = estimate_negativity_moment(rho_test, nA, nB, 3, Nu, n)
            results.append(np.abs(result - 4.5))

        # 每个进程将自己的部分结果收集到 root 进程
        all_results = comm.gather(results, root=0)

        # root 进程汇总所有结果并计算总体平均值
        if rank == 0:
            # 汇总所有进程的结果
            all_results_flat = [item for sublist in all_results for item in sublist]
            overall_mean = np.mean(all_results_flat)
            experiment.append([nA, nB, Nu, n, overall_mean])

    # root 进程处理 CSV 文件保存
    if rank == 0:
        df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'result'])
        file_exists = os.path.exists(csv_filename)
        df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

    return experiment if rank == 0 else None



def _total_sum_from_counts(n_plus, n_minus, t):
    """
    Given counts arrays over b outcomes, compute:
    total_sum = sum_b sum_{k=0..t} (-1)^k C(n_minus[b], k) C(n_plus[b], t-k)
    Using exact integer comb (math.comb).
    """
    total_sum = 0
    for b in range(len(n_plus)):
        npb = int(n_plus[b])
        nmb = int(n_minus[b])
        if npb + nmb < t:
            continue
        sb = 0
        for k in range(t + 1):
            if k <= nmb and (t - k) <= npb:
                sb += ((-1) ** k) * math.comb(nmb, k) * math.comb(npb, t - k)
        total_sum += sb
    return total_sum


def depolarize(dA1, dA2, dB, prob, dprob):
    d = dA1 * dA2 * dB
    dsym = (dB * dB + dB) / 2
    dasym = (dB * dB - dB) / 2
    nprob = len(prob)
    half = int(nprob / 2)
    prob[0 : half] = (1 - dprob) * prob[0 : half] + dprob / d * dsym
    prob[half:] = (1 - dprob) * prob[half: ] + dprob / d * dasym
    prob /= prob.sum()
    return prob
    

def estimate_negativity_moment_multiNM(
    rho, nA, nB, Nu, NM_list, NM_max=None, t_min=2, t_max=5, dprob=0
):
    """
    Run the protocol once with NM_max samples, and compute estimates at each NM in NM_list
    by using the first NM samples (prefix), for all t in [t_min, t_max].

    Returns:
        final_by_t: dict {t: {NM: mean_over_unitaries}}
    """
    if nA < nB:
        raise ValueError("Protocol requires nA >= nB.")
    if t_min < 2 or t_max < t_min:
        raise ValueError("Require 2 <= t_min <= t_max.")

    nA1 = nA - nB
    nA2 = nB

    d_A1 = get_dims(nA1)
    d_A2 = get_dims(nA2)
    d_B  = get_dims(nB)
    d = d_A1 * d_A2 * d_B

    NM_list = sorted(list(NM_list))

    if NM_max is None:
        NM_max = NM_list[-1]
    if NM_max < NM_list[-1]:
        raise ValueError("NM_max must be >= max(NM_list).")

    swap_povms = get_swap_test_povm_operators(d_A2, d_B)

    num_b_outcomes = d_A1
    num_r_outcomes = len(swap_povms)      # should be 2 (+1/-1)
    K = num_b_outcomes * num_r_outcomes   # total joint outcomes

    # prefactors for each t
    t_values = list(range(t_min, t_max + 1))
    prefactor_by_t = {t: (d_A1 ** (t - 1)) * (d_A2 ** t) for t in t_values}

    # per_q_estimates_by_t[t][NM] -> list over unitaries
    per_q_estimates_by_t = {t: {NM: [] for NM in NM_list} for t in t_values}

    for q in range(Nu):
        d_A = d_A1 * d_A2
        Uq = unitary_group.rvs(d_A)
        U_op = np.kron(Uq, np.eye(d_B))
        rho_Uq = U_op @ rho @ U_op.conj().T

        d_Y = d_A2 * d_B
        rho_reshaped = rho_Uq.reshape(d_A1, d_Y, d_A1, d_Y)

        probabilities = []
        for Pi_r in swap_povms:
            rho_post_r = np.einsum('ijkl,lj->ik', rho_reshaped, Pi_r)
            probs_b = np.real(np.diag(rho_post_r))  # length = d_A1
            probabilities.extend(probs_b)

        probabilities = np.array(probabilities, dtype=float)
        probabilities /= probabilities.sum()
        probabilities = depolarize(d_A1, d_A2, d_B, probabilities, dprob)

        # sample NM_max * d individual outcomes
        outcomes_idx = np.random.choice(K, size=NM_max * d, p=probabilities)

        # cumulative counts for each b: plus/minus
        n_plus  = np.zeros(num_b_outcomes, dtype=np.int64)
        n_minus = np.zeros(num_b_outcomes, dtype=np.int64)

        checkpoints = set(np.array(NM_list) * d)

        for i, idx in enumerate(outcomes_idx, start=1):
            b = idx % num_b_outcomes
            r_idx = idx // num_b_outcomes  # 0 -> +1, 1 -> -1

            if r_idx == 0:
                n_plus[b] += 1
            else:
                n_minus[b] += 1

            if i in checkpoints:
                NM_cur = i               # equals NM*d at checkpoint
                NM_key = int(NM_cur / d) # back to NM

                for t in t_values:
                    denom = math.comb(NM_cur, t) if NM_cur >= t else 0
                    if denom == 0:
                        M_neg_q_t = 0.0
                    else:
                        total_sum_t = _total_sum_from_counts(n_plus, n_minus, t)
                        M_neg_q_t = prefactor_by_t[t] * (total_sum_t / denom)

                    per_q_estimates_by_t[t][NM_key].append(M_neg_q_t)

    # average over unitaries
    final_by_t = {
        t: {NM: float(np.mean(per_q_estimates_by_t[t][NM])) for NM in NM_list}
        for t in t_values
    }
    return final_by_t



def _worker_GHZ_multi(args):
    rho_test, nA, nB, Nu, NM_list, t_min, t_max = args

    final_by_t = estimate_negativity_moment_multiNM(
        rho_test, nA, nB, Nu, NM_list, NM_max=max(NM_list), t_min=t_min, t_max=t_max
    )

    t_list = list(range(t_min, t_max + 1))
    # 返回：每个 t 一行，每行按 NM_list 顺序
    # shape: (len(t_list), len(NM_list))
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]

def experiment_GHZ_mp(nA, nB, Nu, repeat, csv_filename='experiment_results_dbasis.csv'):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B

    ghz_vec = np.zeros(dim_total, dtype=np.complex128)
    ghz_vec[0] = 1/np.sqrt(2)
    ghz_vec[-1] = 1/np.sqrt(2)
    rho_test = np.outer(ghz_vec, ghz_vec.conj())

    NM_list = [1, 2, 5, 10, 20, 50, 100]

    # 一次同时算 t=2..5
    t_min, t_max = 2, 3
    args_list = [(rho_test, nA, nB, Nu, NM_list, t_min, t_max)] * repeat

    with multiprocessing.Pool(processes=2) as pool:
        # outs: list of length=repeat
        # each element shape: (len(t_list), len(NM_list))
        outs = list(tqdm(pool.imap_unordered(_worker_GHZ_multi, args_list), total=repeat))

    outs = np.array(outs, dtype=float)


    results = outs[:, 1, :]
    results2 = outs[:, 0, :]
    purity = results2 - 1
    thirdM = (results - 1 - 3 * purity) / 2
    results = thirdM - 0.25  # (repeat, L)
    mean_errors = np.abs(results).mean(axis=0)        # (L,)
    mse = np.square(results).mean(axis=0)

    purity_error = purity - 1
    mean_errors2 = np.abs(purity_error).mean(axis=0)
    mse2 = np.square(purity_error).mean(axis=0)

    experiment = []
    for NM, err, err2 in zip(NM_list, mean_errors, mean_errors2):
        experiment.append([nA, nB, Nu, NM, float(err2), float(err)])

    df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'purity', 'thirdM'])
    file_exists = os.path.exists(csv_filename)
    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

    experiment2 = []
    for NM, err, err2 in zip(NM_list, mse, mse2):
        experiment2.append([nA, nB, Nu, NM, float(err2), float(err)])

    csv_filename2 = 'MSE_dbasis.csv'
    df = pd.DataFrame(experiment2, columns=['nA', 'nB', 'Nu', 'NM', 'purity', 'thirdM'])
    file_exists = os.path.exists(csv_filename2)
    df.to_csv(csv_filename2, mode='a', header=not file_exists, index=False)
    return df


def _worker_heisenberg(args):
    rho_test, nA, nB, Nu, NM_list, t_min, t_max, dprob = args

    final_by_t = estimate_negativity_moment_multiNM(
        rho_test, nA, nB, Nu, NM_list, NM_max=max(NM_list), t_min=t_min, t_max=t_max, dprob=dprob
    )

    t_list = list(range(t_min, t_max + 1))
    # 返回：每个 t 一行，每行按 NM_list 顺序
    # shape: (len(t_list), len(NM_list))
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]


def experiment_heisenberg_mp(nA, nB, Nu, repeat, dprob=0, csv_filename='heisenberg_dbasis_prob.csv'):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B

    psi = np.load(f"./heisenberg/heisenberg_N{nA+nB}_Jz1.0.npy")
    rho_test = np.outer(psi, psi.conj())
    NM_list = [1, 2, 5, 10, 20, 50, 100]

    # 一次同时算 t=2..5
    t_min, t_max = 2, 5
    args_list = [(rho_test, nA, nB, Nu, NM_list, t_min, t_max, dprob)] * repeat
    with multiprocessing.Pool() as pool:
        outs = list(tqdm(pool.imap_unordered(_worker_heisenberg, args_list), total=repeat))
    outs = np.array(outs, dtype=float)
    experiment = []
    for rep in range(outs.shape[0]):
        purity = outs[rep, 0, :] - 1
        thirdM = (outs[rep, 1, :] - 1 - 3 * purity) / 2
        fourthM = (outs[rep, 2, :] - 1 - 6 * purity - 8 * thirdM - 3 * purity ** 2) / 6
        fifthM = (outs[rep, 3, :] - 1 - 10 * purity - 20 * thirdM - 15 * purity ** 2
                  - 30 * fourthM - 20 * purity * thirdM) / 24

        for NM, p2, p3, p4, p5 in zip(NM_list, purity, thirdM, fourthM, fifthM):
            experiment.append([rep, nA, nB, Nu, NM, dprob, float(p2), float(p3), float(p4), float(p5)])
    df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'dprob', 'purity', 'thirdM', 'fourthM', 'fifthM'])
    file_exists = os.path.exists(csv_filename)
    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    return df


def _worker_gibbs(args):
    rho_test, nA, nB, Nu, NM_list, t_min, t_max = args

    final_by_t = estimate_negativity_moment_multiNM(
        rho_test, nA, nB, Nu, NM_list, NM_max=max(NM_list), t_min=t_min, t_max=t_max
    )

    t_list = list(range(t_min, t_max + 1))
    # 返回：每个 t 一行，每行按 NM_list 顺序
    # shape: (len(t_list), len(NM_list))
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]

def experiment_gibbs_mp(nA, nB, Nu, repeat, beta, csv_filename='gibbs_dbasis.csv'):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    rho = np.load(f"./gibbs/gibbs_N{nA+nB}_beta{beta:.2f}.npy")
    NM_list = [1, 2, 5, 10, 20, 50, 100]

    # 一次同时算 t=2..5
    t_min, t_max = 2, 5
    args_list = [(rho, nA, nB, Nu, NM_list, t_min, t_max)] * repeat
    with multiprocessing.Pool() as pool:
        outs = list(tqdm(pool.imap_unordered(_worker_gibbs, args_list), total=repeat))
    outs = np.array(outs, dtype=float)
    experiment = []
    for rep in range(outs.shape[0]):
        purity = outs[rep, 0, :] - 1
        thirdM = (outs[rep, 1, :] - 1 - 3 * purity) / 2
        fourthM = (outs[rep, 2, :] - 1 - 6 * purity - 8 * thirdM - 3 * purity ** 2) / 6
        fifthM = (outs[rep, 3, :] - 1 - 10 * purity - 20 * thirdM - 15 * purity ** 2
                  - 30 * fourthM - 20 * purity * thirdM) / 24

        for NM, p2, p3, p4, p5 in zip(NM_list, purity, thirdM, fourthM, fifthM):
            experiment.append([rep, nA, nB, Nu, NM, beta, float(p2), float(p3), float(p4), float(p5)])
    df = pd.DataFrame(experiment, columns=['nA', 'nB', 'Nu', 'NM', 'beta', 'purity', 'thirdM', 'fourthM', 'fifthM'])
    file_exists = os.path.exists(csv_filename)
    df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
    return df





if __name__ == '__main__':
    # # --- Simulation Parameters ---
    # # As requested, we test with a higher-dimensional GHZ state.
    # # We use a 4-qubit GHZ state with nA=3, nB=1.
    # nA = 11
    # nB = 1
    # t = 3       # Order of the moment to estimate
    # Nu = 1    # Keep sample size high for accuracy
    # NM = 100000    # Keep sample size high for accuracy

    # # System dimensions
    # d_A = get_dims(nA)
    # d_B = get_dims(nB)
    # dim_total = d_A * d_B

    # # Create a 4-qubit GHZ state |psi> = (|0000> + |1111>)/sqrt(2)
    # ghz_vec = np.zeros(dim_total); ghz_vec[0] = 1/np.sqrt(2); ghz_vec[-1] = 1/np.sqrt(2)
    # rho_test = np.outer(ghz_vec, ghz_vec.conj())
    # # vec = get_computational_basis_vectors(dim_total)[0]
    # # rho_test = np.outer(vec, vec.conj())
    # print(f"--- Running Verification with {nA+nB}-Qubit GHZ State ---")
    # print(f"Estimating the {t}-th moment (t={t}).")
    # print(f"System setup: nA={nA}, nB={nB}")
    # print(f"Protocol parameters: Nu={Nu}, NM={NM}")
    
    # # The theoretical value for E[M_neg] for a maximally entangled GHZ state is 2.
    # print(f"Expecting result to be close to {calculate_S3(rho_test, d_A, d_B)}")
    # print(f"----------------------------------------------------")

    # estimated_moment = estimate_negativity_moment(rho_test, nA, nB, t, Nu, NM)

    # print(f"\nFinal Estimated Value: {estimated_moment}")
    for nu in [2, 5, 10]:
        experiment_GHZ_mp(6, 1, nu, 10000)
    # search_converge(7, 1, 50, 300, 0.05, 5000)
    # search_converge(6, 1, 100, 400, 0.01, 5000)