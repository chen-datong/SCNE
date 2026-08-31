"""Generate dedicated CBNE data for estimating tr(rho^3)."""

import argparse
import json
import math
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from TFIM import build_tfim_pauli_sparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def gibbs_eigenvalues(n, beta, j_coupling, field):
    hamiltonian = build_tfim_pauli_sparse(
        N=n, J=j_coupling, h=field, pbc=False
    ).toarray()
    energies = np.linalg.eigvalsh(hamiltonian)
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum()


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
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Ancilla values outside the paper grid: {invalid}")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Generate the paper-parameter CBNE tr(rho^3) experiment."
    )
    parser.add_argument("--ancilla", default="all", help="all or comma-separated n_a values")
    parser.add_argument(
        "--processes", type=int, default=min(4, cpu_count()), help="worker process count"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    if args.processes < 1:
        raise ValueError("--processes must be positive")

    config_path = ROOT / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"][
        "SE_all_observables.pdf"
    ]
    n = int(params["n"])
    beta = float(params["beta"])
    nu = int(params["left"]["N_U"])
    trials = int(params["trials"])
    nm_values = sorted(int(value) for value in params["left"]["N_M"])
    allowed_ancilla = [int(value) for value in params["right"]["n_a"]]
    ancilla_values = parse_ancilla_values(args.ancilla, allowed_ancilla)
    eigenvalues = gibbs_eigenvalues(
        n, beta, float(params["J"]), float(params["h"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        anc: args.output_dir
        / f"rho3_anc_n{n}_Nu{nu}_NMmulti_beta{beta}_anc{anc}.csv"
        for anc in ancilla_values
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing rho3 data: " + ", ".join(existing)
        )

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
