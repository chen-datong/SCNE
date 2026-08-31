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
    
    I = np.eye(dim, dtype=np.complex64)
    
    SWAP = np.zeros((dim, dim), dtype=np.complex64)
    for i in range(d):
        for j in range(d):
            ket_i = np.zeros(d, dtype=np.complex64); ket_i[i] = 1
            ket_j = np.zeros(d, dtype=np.complex64); ket_j[j] = 1
            
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
    



def random_unitary_complex64(N):
    """
    Generate a Haar-random unitary matrix of size N x N using complex64 directly.
    Avoids allocating huge complex128 arrays which cause MemoryError.
    Uses LAPACK cgeqrf/cungqr for strict single precision.
    """
    from scipy.linalg.lapack import cgeqrf, cungqr

    # 1. Generate Z directly in Fortran order for LAPACK in-place optimization
    # Use float32 to save memory during generation
    real_part = np.random.randn(N, N).astype(np.float32)
    imag_part = np.random.randn(N, N).astype(np.float32)
    
    # Create F-contiguous complex64 array
    Z = np.empty((N, N), dtype=np.complex64, order='F')
    Z.real = real_part
    Z.imag = imag_part
    
    del real_part
    del imag_part

    # 2. QR Factorization (cgeqrf)
    # overwrite_a=True works best with F-contiguous
    # Returns: (qr, tau, work, info) or (qr, tau, info)
    
    # Note: scipy.linalg.lapack wrappers returns might vary by version
    # Standard signature for cgeqrf is (qr, tau, info) if lwork is auto-computed
    
    # We let scipy handle lwork allocation (it's small compared to N^2)
    geqrf_result = cgeqrf(Z, overwrite_a=True)
    qr_raw, tau = geqrf_result[:2]
    info = geqrf_result[-1]
    if info < 0:
        raise ValueError(f"cgeqrf illegal value at argument {-info}")
    
    # qr_raw is now sharing memory with Z (if overwrite succeeded) or new array
    # We don't need Z variable pointing to it anymore if separate
    del Z 

    # 3. Extract R diagonal before generating Q
    # The diagonal of R is stored on the diagonal of qr_raw
    r_diag = np.diag(qr_raw).copy()
    
    # 4. Generate Q (cungqr)
    # Overwrites qr_raw with Q
    ungqr_result = cungqr(qr_raw, tau, overwrite_a=True)
    Q = ungqr_result[0]
    info = ungqr_result[-1]
    if info < 0:
        raise ValueError(f"cungqr illegal value at argument {-info}")
    
    # qr_raw/Q is now the unitary Q
    
    # 5. Phase correction
    # r_diag might have zeros if rank deficient, but random matrix is full rank almost surely
    # Use simple loop or broadcasting
    eps = 1e-6
    abs_r = np.abs(r_diag)
    abs_r[abs_r < eps] = 1.0 # Protect against zero div
    phases = r_diag / abs_r
    
    # Broadcast multiply columns
    Q *= phases
    
    return Q


def estimate_negativity_moment_multiNM(
    rho, nA, nB, Nu, NM_list, NM_max=None, t_min=2, t_max=5, dprob=0
):
    """
    Run the protocol once with NM_max samples, and compute estimates at each NM in NM_list
    by using the first NM samples (prefix), for all t in [t_min, t_max].

    Optimized: Supports pure state vector input (rho as 1D array).
    """
    if nA < nB:
        raise ValueError("Protocol requires nA >= nB.")
    if t_min < 2 or t_max < t_min:
        raise ValueError("Require 2 <= t_min <= t_max.")

    # Detect if input is a state vector or density matrix
    is_pure = (rho.ndim == 1)

    # Cast rho to complex64 for memory efficiency
    rho = rho.astype(np.complex64)
    
    nA1 = nA - nB
    nA2 = nB

    d_A1 = get_dims(nA1)
    d_A2 = get_dims(nA2)
    d_B  = get_dims(nB)
    d_A = d_A1 * d_A2
    
    # Check consistency
    if is_pure:
        if rho.shape[0] != d_A * d_B: 
             raise ValueError(f"State vector dim {rho.shape[0]} mismatch expected {d_A*d_B}")
    else:
        if rho.shape[0] != d_A * d_B: 
             raise ValueError(f"Density matrix dim {rho.shape[0]} mismatch expected {d_A*d_B}")

    # For mixed states, we still need old POVMs (slow path)
    if not is_pure:
        swap_povms = get_swap_test_povm_operators(d_A2, d_B)

    NM_list = sorted(list(NM_list))
    if NM_max is None:
        NM_max = NM_list[-1]
    if NM_max < NM_list[-1]:
        raise ValueError("NM_max must be >= max(NM_list).")

    num_b_outcomes = d_A1
    num_r_outcomes = 2
    K = num_b_outcomes * num_r_outcomes

    t_values = list(range(t_min, t_max + 1))
    prefactor_by_t = {t: (d_A1 ** (t - 1)) * (d_A2 ** t) for t in t_values}
    per_q_estimates_by_t = {t: {NM: [] for NM in NM_list} for t in t_values}

    for q in range(Nu):
        # 1. Generate Random Unitary on A
        # Use custom generator for complex64
        Uq = random_unitary_complex64(d_A)
        
        probabilities = []

        if is_pure:
            # === Fast Path for Pure States ===
            # rho is shape (d_total,)
            # Reshape to (dA, dB) to apply U_A
            psi = rho.reshape(d_A, d_B)
            
            # Apply Uq: (dA, dA) @ (dA, dB) -> (dA, dB)
            psi_rot = Uq @ psi 
            
            # Reshape to (dA1, dA2, dB) for measurements
            psi_rot = psi_rot.reshape(d_A1, d_A2, d_B)
            
            # Compute probabilities for each b in A1
            probs_sym = []
            probs_asym = []
            
            for b_idx in range(d_A1):
                phi = psi_rot[b_idx, :, :] # shape (dA2, dB)
                
                # prob_total = <phi|phi>
                # Using einsum or squared norm
                norm_sq = np.sum(np.abs(phi)**2)
                
                swap_overlap = np.vdot(phi.T, phi).real # Must be real for Hermitian obs
                
                p_sym = 0.5 * (norm_sq + swap_overlap)
                p_asym = 0.5 * (norm_sq - swap_overlap)
                
                # Check for numerical noise
                if p_sym < 0: p_sym = 0.0
                if p_asym < 0: p_asym = 0.0
                
                probs_sym.append(p_sym)
                probs_asym.append(p_asym)
                
            # Combine to match legacy order: First all Syms (r=0), then all Asyms (r=1)
            probabilities.extend(probs_sym)
            probabilities.extend(probs_asym)
            
        else:
            # === Slow Path for Mixed States ===
            U_op = np.kron(Uq, np.eye(d_B, dtype=np.complex64))
            rho_Uq = U_op @ rho @ U_op.conj().T

            d_Y = d_A2 * d_B
            rho_reshaped = rho_Uq.reshape(d_A1, d_Y, d_A1, d_Y)

            for Pi_r in swap_povms:
                rho_post_r = np.einsum('ijkl,lj->ik', rho_reshaped, Pi_r)
                probs_b = np.real(np.diag(rho_post_r))  # length = d_A1
                probabilities.extend(probs_b)

        probabilities = np.array(probabilities, dtype=np.float64) # Keep probabilities high precision for sampling
        sum_p = probabilities.sum()
        if sum_p > 0:
            probabilities /= sum_p
            
        if dprob > 0:
            probabilities = depolarize(d_A1, d_A2, d_B, probabilities, dprob)

        # Draw exactly N_M outcomes.  Multinomial increments preserve the
        # prefix distribution while avoiding an O(N_M) Python loop.
        n_plus  = np.zeros(num_b_outcomes, dtype=np.int64)
        n_minus = np.zeros(num_b_outcomes, dtype=np.int64)
        previous_nm = 0

        for NM_cur in NM_list:
            increment = np.random.multinomial(NM_cur - previous_nm, probabilities)
            n_plus += increment[:num_b_outcomes]
            n_minus += increment[num_b_outcomes:]
            previous_nm = NM_cur

            for t in t_values:
                denom = math.comb(NM_cur, t) if NM_cur >= t else 0
                if denom == 0:
                    M_neg_q_t = 0.0
                else:
                    total_sum_t = _total_sum_from_counts(n_plus, n_minus, t)
                    M_neg_q_t = prefactor_by_t[t] * (total_sum_t / denom)

                per_q_estimates_by_t[t][NM_cur].append(M_neg_q_t)

    # average over unitaries
    final_by_t = {
        t: {NM: float(np.mean(per_q_estimates_by_t[t][NM])) for NM in NM_list}
        for t in t_values
    }
    return final_by_t




def _worker_heisenberg(args):
    rho_test, nA, nB, Nu, NM_list, t_min, t_max, dprob = args

    final_by_t = estimate_negativity_moment_multiNM(
        rho_test, nA, nB, Nu, NM_list, NM_max=max(NM_list), t_min=t_min, t_max=t_max, dprob=dprob
    )

    t_list = list(range(t_min, t_max + 1))
    # 返回：每个 t 一行，每行按 NM_list 顺序
    # shape: (len(t_list), len(NM_list))
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]


def experiment_heisenberg_mp(
    nA,
    nB,
    Nu,
    repeat,
    dprob=0,
    csv_filename='heisenberg_dbasis_prob_paper.csv',
    NM_list=None,
):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B

    psi = np.load(f"./heisenberg/heisenberg_N{nA+nB}_Jz1.0.npy")
    # Optimize: Pass pure state vector psi directly instead of constructing density matrix
    # rho_test = np.outer(psi, psi.conj()) 
    rho_test = psi
    
    NM_list = [1_000_000] if NM_list is None else sorted(int(x) for x in NM_list)

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







if __name__ == '__main__':

    for nu in [2, 5, 10]:
        experiment_GHZ_mp(6, 1, nu, 10000)
    # search_converge(7, 1, 50, 300, 0.05, 5000)
    # search_converge(6, 1, 100, 400, 0.01, 5000)
