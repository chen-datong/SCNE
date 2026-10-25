"""Generate dedicated CBNE data for estimating tr(rho^3)."""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from TFIM import build_tfim_pauli_sparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "cbne_observables"


def gibbs_eigenvalues(n, beta, j_coupling, field):
    hamiltonian = build_tfim_pauli_sparse(
        N=n, J=j_coupling, h=field, pbc=False
    ).toarray()
    energies = np.linalg.eigvalsh(hamiltonian)
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum()


def experiment_parameters(state="depolarized"):
    key = "SE_rho3_anc.pdf" if state == "depolarized" else "SE_rho3_gibbs.pdf"
    return json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"][key]


def state_eigenvalues(params):
    """Only the spectrum matters for Haar-random measurement statistics."""
    n = int(params["n"])
    if params.get("state", "gibbs") == "gibbs":
        return gibbs_eigenvalues(n, float(params["beta"]), float(params["J"]), float(params["h"]))
    if params["state"] != "depolarized":
        raise ValueError(f"Unknown state: {params['state']}")
    probability = float(params["depolarizing_probability"])
    if not 0 <= probability <= 1:
        raise ValueError("Depolarizing probability must lie in [0, 1]")
    dimension = 1 << n
    weights = np.full(dimension, probability / dimension)
    weights[0] += 1.0 - probability
    return weights


def rho3_data_path(data_dir, params, anc, trials=None):
    n, nu = int(params["n"]), int(params["left"]["N_U"])
    suffix = "" if trials is None or trials == int(params["trials"]) else f"_trials{trials}"
    if params.get("state", "gibbs") == "depolarized":
        probability = float(params["depolarizing_probability"])
        name = f"rho3_depol_anc_n{n}_Nu{nu}_NMmulti_p{probability}_anc{anc}{suffix}.csv"
    else:
        beta = float(params["beta"])
        name = f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}{suffix}.csv"
    return Path(data_dir) / name


def rho3_summary_path(data_dir, params):
    n, nu = int(params["n"]), int(params["left"]["N_U"])
    if params.get("state", "gibbs") == "depolarized":
        tag = f"depol_n{n}_Nu{nu}_p{float(params['depolarizing_probability'])}"
    else:
        tag = f"n{n}_Nu{nu}_beta{float(params['beta'])}"
    return Path(data_dir) / f"rho3_statistical_error_{tag}.csv"


def one_trial(args):
    eigenvalues, total_dimension, nm_values, seed, anc, index = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, anc, index]))
    rank = eigenvalues.size
    gaussian = (
        rng.standard_normal((total_dimension, rank))
        + 1j * rng.standard_normal((total_dimension, rank))
    ) / np.sqrt(2.0)
    isometry, _ = np.linalg.qr(gaussian, mode="reduced")
    probabilities = np.abs(isometry) ** 2 @ eigenvalues
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    counts = np.zeros(total_dimension, dtype=np.int64)
    previous_nm = 0
    rows = []
    for nm in nm_values:
        counts += rng.multinomial(nm - previous_nm, probabilities)
        previous_nm = nm
        counts_float = counts.astype(float, copy=False)
        choose_two = counts_float * (counts_float - 1.0) / 2.0
        choose_three = choose_two * (counts_float - 2.0) / 3.0
        raw_two = (total_dimension + 1.0) * np.sum(choose_two) / math.comb(nm, 2)
        raw_three = (
            (total_dimension + 1.0)
            * (total_dimension + 2.0)
            * np.sum(choose_three)
            / math.comb(nm, 3)
        )
        purity = raw_two - 1.0
        third_moment = (raw_three - 3.0 * purity - 1.0) / 2.0
        rows.append(
            {
                "idx": index,
                "NM": nm,
                "purity": purity,
                "third_moment": third_moment,
            }
        )
    return rows


def parse_ancilla_values(text, allowed):
    if text == "all":
        return allowed
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Specify at least one ancilla size")
    if any(value < 0 for value in values):
        raise ValueError("Ancilla sizes must be nonnegative")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Generate the paper-parameter CBNE tr(rho^3) experiment."
    )
    parser.add_argument("--state", choices=("depolarized", "gibbs"), default="depolarized")
    parser.add_argument("--dry-run", action="store_true", help="Print settings and paths without simulating or writing data")
    parser.add_argument(
        "--ancilla",
        default="all",
        help="all configured ancilla sizes, or explicit nonnegative n_a values",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="independent repetitions per n_a (default: paper value of 1000)",
    )
    parser.add_argument(
        "--processes", type=int, default=1, help="worker process count"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if args.processes < 1:
        raise ValueError("--processes must be positive")

    params = experiment_parameters(args.state)
    n = int(params["n"])
    nu = int(params["left"]["N_U"])
    paper_trials = int(params["trials"])
    trials = paper_trials if args.trials is None else args.trials
    if trials < 1:
        raise ValueError("--trials must be positive")
    nm_values = sorted(int(value) for value in params["left"]["N_M"])
    allowed_ancilla = [int(value) for value in params["right"]["n_a"]]
    ancilla_values = parse_ancilla_values(args.ancilla, allowed_ancilla)
    outputs = {anc: rho3_data_path(args.output_dir, params, anc, trials) for anc in ancilla_values}
    if args.dry_run:
        print(f"State: {params['model']}; n={n}; processes={args.processes}; trials={trials}")
        print(f"N_M: {nm_values}; n_a: {ancilla_values}")
        print(f"Exact tr(rho^3): {np.sum(state_eigenvalues(params)**3):.12g}")
        for anc, path in outputs.items():
            matrix_mib = (1 << (n + anc)) * (1 << n) * 16 / 2**20
            print(f"n_a={anc}: one complex isometry is {matrix_mib:g} MiB (QR and temporaries require additional memory); {path}")
        return
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing rho3 data: " + ", ".join(existing)
        )

    eigenvalues = state_eigenvalues(params)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for anc in ancilla_values:
        total_dimension = 1 << (n + anc)
        jobs = (
            (eigenvalues, total_dimension, nm_values, args.seed, anc, index)
            for index in range(trials)
        )
        with Pool(processes=args.processes) as pool:
            nested = list(
                tqdm(
                    pool.imap_unordered(one_trial, jobs),
                    total=trials,
                    desc=f"n_a={anc}",
                )
            )
        frame = pd.DataFrame(row for trial_rows in nested for row in trial_rows)
        frame.sort_values(["NM", "idx"]).to_csv(outputs[anc], index=False)
        print(f"Saved {outputs[anc]}")


if __name__ == "__main__":
    main()
