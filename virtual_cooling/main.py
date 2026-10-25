import numpy as np
from scipy.stats import unitary_group
import math
import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "virtual_cooling"


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


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


def experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=1, dprob=0):
    d = pow(2, n)
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
    
    t_list = list(range(tmin, t+1))

    rhot_estimate, rhotO_estimates = estimate_rhotAndrhotO_tlist_dprob(rho_test, O_list, n, Nu, NM, t_list, dprob)
    rho1 = rhot_estimate[1]
    rho2 = rhot_estimate[2]
    purity = rho2 - 1
    rho3 = rhot_estimate[3]
    third_moment = (rho3 - 3 * purity - 1) / 2

    rhoO1 = np.array(rhotO_estimates[1])
    rhoO2 = np.array(rhotO_estimates[2])
    rhoO3 = np.array(rhotO_estimates[3])
    rho2O = (rhoO2 - 2 * rhoO1) / 2
    rho3O = (rhoO3 - 6 * rho2O - 3 * (1 + purity) * rhoO1) / 6

    pf_rhoO1 = rhoO1
    pf_rhoO2 = rho2O / purity
    pf_rhoO3 = rho3O / third_moment
    
    
    return (purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3)


def main_experiment_gibbs(n, Nu, NM, rep, beta, h=1.0, output_axis="beta"):
    d = pow(2, n)
    data = []
    rho_test = np.load(DATA / "TFIM" / f'tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy')
    O_list = []
    psi0 = np.load(DATA / "TFIM" / f'tfim_N{n}_gs_J1.0_h{h}.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)

    for idx in tqdm(range(rep)):
        purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, 3, tmin=1, dprob=0)
        # Store the trial index, moments, and observable estimates.
        data.append({
            'idx': idx,
            'purity': purity,
            'third_moment': third_moment,
            'pf_rhoO2_fidelity': pf_rhoO2[0],
            'pf_rhoO3_fidelity': pf_rhoO3[0]
        })
    
    # Collect results.
    df = pd.DataFrame(data)
    
    # Save results with parameters in the filename.
    if output_axis == "beta":
        filename = DATA / f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    elif output_axis == "h":
        filename = DATA / f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_h{h}.csv"
    else:
        raise ValueError("output_axis must be 'beta' or 'h'")
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df

def _experiment_gibbs_wrapper(args):
    """Worker wrapper for experiment_single."""
    n, Nu, NM, t, tmin, beta, h, idx = args
    d = pow(2, n)
    rho_test = np.load(DATA / "TFIM" / f'tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy')
    O_list = []
    psi0 = np.load(DATA / "TFIM" / f'tfim_N{n}_gs_J1.0_h{h}.npy')
    psib = np.outer(psi0, psi0.conj()) - np.eye(d, dtype=np.complex128) / d
    O_list.append(psib)
    purity, third_moment, pf_rhoO1, pf_rhoO2, pf_rhoO3 = experiment_single(n, rho_test, O_list, Nu, NM, t, tmin=tmin, dprob=0)
    return {
        'idx': idx,
        'purity': purity,
        'third_moment': third_moment,
        'pf_rhoO2_fidelity': pf_rhoO2[0],
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
    """Run trials with multiprocessing and return a DataFrame.

Use n qubits, Nu settings, NM shots, and rep trials at inverse temperature beta.
num_processes=None uses the CPU count."""
    if num_processes is None:
        num_processes = cpu_count()


    # Build worker arguments.
    args_list = [(n, Nu, NM, 3, 1, beta, h, idx) for idx in range(rep)]
    
    # Run the worker pool.
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_experiment_gibbs_wrapper, args_list), total=rep))
    
    # Collect results.
    df = pd.DataFrame(results)
    
    # Save results with parameters in the filename.
    if output_axis == "beta":
        filename = DATA / f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_beta{beta}.csv"
    elif output_axis == "h":
        filename = DATA / f"fidelity_gibbs_n{n}_Nu{Nu}_NM{NM}_h{h}.csv"
    else:
        raise ValueError("output_axis must be 'beta' or 'h'")
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df
