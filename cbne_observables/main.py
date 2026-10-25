from pathlib import Path
import numpy as np
from scipy.stats import unitary_group
import math
import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cbne_observables"
STATE_DIR = DATA_DIR / "TFIM"

X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


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
        

def _experiment_gibbs_wrapper_anc(args):
    """Worker wrapper for experiment_single, including ancillas."""
    n, Nu, NM, t, tmin, beta, anc, idx = args
    d = pow(2, n)
    rho_test = np.load(STATE_DIR / f'tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
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
    rho_test = np.load(STATE_DIR / f'tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
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
    # Pauli measurements with ancillas can be added here.
    if num_processes is None:
        num_processes = cpu_count()
    
    # Build worker arguments.
    args_list = [(n, Nu, NM, 3, 1, beta, anc, idx) for idx in range(rep)]

    # Run the worker pool.
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper_anc, args_list), total=rep))
    # Collect results.
    df = pd.DataFrame(results)
    # Save results with parameters in the filename.
    filename = DATA_DIR / f"gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
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
    filename_all = DATA_DIR / f"gibbs_pauli_anc_n{n}_Nu{Nu}_NMmulti_beta{beta}_anc{anc}.csv"
    df_all.to_csv(filename_all, index=False)
    print(f"Combined NM-sweep results saved to {filename_all}")

    # Keep compatibility with existing downstream code by writing one file per NM.
    for NM in sorted(set(NM_list)):
        df_nm = df_all[df_all["NM"] == NM].drop(columns=["NM"])
        filename_nm = DATA_DIR / f"gibbs_pauli_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
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
    rho_test = np.load(STATE_DIR / f'tfim_gibbs_N{n}_beta{beta}_J1.0_h1.0.npy')
    psi0 = np.load(STATE_DIR / f'tfim_N{n}_gs_J1.0_h1.0.npy')
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
    filename_all = DATA_DIR / f"gibbs_psibH_anc_n{n}_Nu{Nu}_NMmulti_beta{beta}_anc{anc}.csv"
    df_all.to_csv(filename_all, index=False)
    print(f"Combined NM-sweep results saved to {filename_all}")

    for NM in sorted(set(NM_list)):
        df_nm = df_all[df_all["NM"] == NM].drop(columns=["NM"])
        filename_nm = DATA_DIR / f"gibbs_psibH_anc_n{n}_Nu{Nu}_NM{NM}_beta{beta}_anc{anc}.csv"
        df_nm.to_csv(filename_nm, index=False)
        print(f"Results saved to {filename_nm}")

    return df_all
