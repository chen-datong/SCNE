import numpy as np
from scipy.stats import unitary_group
import pandas as pd
import os
from tqdm import tqdm
import math
import multiprocessing
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "ptme"


def get_dims(n_qubits):
    """Calculate the dimension of the Hilbert space for a given number of qubits."""
    return 2**n_qubits


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


#     """Helper function for multiprocessing in experiment_GHZ_mp."""
#

#
#
#
#
#
#
    

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

    # Current API uses total shots N_M; legacy CSV NM represented N_M / d.
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
    # One row per t, with columns in NM_list order.
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]


def experiment_heisenberg_mp(
    nA,
    nB,
    Nu,
    repeat,
    dprob=0,
    csv_filename=DATA / "heisenberg_dbasis_prob_paper.csv",
    NM_list=None,
    processes=1,
):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    dim_total = d_A * d_B

    psi = np.load(DATA / "heisenberg" / f"heisenberg_N{nA+nB}_Jz1.0.npy")
    rho_test = np.outer(psi, psi.conj())
    NM_list = [1_000_000] if NM_list is None else sorted(int(x) for x in NM_list)

    # Compute moments t=2 through 5 together.
    t_min, t_max = 2, 5
    args_list = [(rho_test, nA, nB, Nu, NM_list, t_min, t_max, dprob)] * repeat
    with multiprocessing.Pool(processes=processes) as pool:
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
    # One row per t, with columns in NM_list order.
    return [[final_by_t[t][NM] for NM in NM_list] for t in t_list]

def experiment_gibbs_mp(
    nA,
    nB,
    Nu,
    repeat,
    beta,
    csv_filename=DATA / "gibbs_dbasis_paper.csv",
    NM_list=None,
    processes=1,
):
    d_A = get_dims(nA)
    d_B = get_dims(nB)
    rho = np.load(DATA / "gibbs" / f"gibbs_N{nA+nB}_beta{beta:.2f}.npy")
    NM_list = [1_000_000] if NM_list is None else sorted(int(x) for x in NM_list)

    # Compute moments t=2 through 5 together.
    t_min, t_max = 2, 5
    args_list = [(rho, nA, nB, Nu, NM_list, t_min, t_max)] * repeat
    with multiprocessing.Pool(processes=processes) as pool:
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
