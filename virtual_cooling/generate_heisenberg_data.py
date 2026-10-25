"""Generate QVC estimates for the XXZ Heisenberg Gibbs state."""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from heisenberg_model import gibbs_eigenvalues, load_spectrum, spectrum_filename


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "virtual_cooling"
ROOT = HERE.parent
_WORKER = {}


def data_filename(n, nu, nm, beta, delta):
    return (
        f"qvc_heisenberg_n{n}_Nu{nu}_NM{nm}_"
        f"beta{beta:.2f}_Delta{delta:.2f}.csv"
    )


def initialize_worker(energies, beta, nu, nm, seed, condition_id, column_block):
    dimension = int(energies.size)
    _WORKER.clear()
    _WORKER.update(
        {
            "weights": gibbs_eigenvalues(energies, beta),
            "dimension": dimension,
            "nu": int(nu),
            "nm": int(nm),
            "seed": int(seed),
            "condition_id": int(condition_id),
            "column_block": int(column_block),
            "moment_coefficients": {
                order: (
                    math.comb(dimension + order - 1, order)
                    * math.factorial(order)
                    / (math.comb(nm, order) * dimension)
                )
                for order in (1, 2, 3)
            },
            "observable_coefficients": {
                order: (
                    math.comb(dimension + order, order + 1)
                    * math.factorial(order + 1)
                    / (math.comb(nm, order) * dimension)
                )
                for order in (1, 2, 3)
            },
        }
    )


def diagonal_probabilities(unitary, weights, column_block):
    """Compute diag(U rho U^dagger) without materializing |U|^2 in full."""
    dimension = weights.size
    probabilities = np.zeros(dimension, dtype=float)
    for start in range(0, dimension, column_block):
        stop = min(start + column_block, dimension)
        block = unitary[:, start:stop]
        probabilities += (np.abs(block) ** 2) @ weights[start:stop]
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()
    return probabilities


def choose_counts(counts):
    counts = counts.astype(float, copy=False)
    choose_one = counts
    choose_two = counts * (counts - 1.0) / 2.0
    choose_three = choose_two * (counts - 2.0) / 3.0
    return {1: choose_one, 2: choose_two, 3: choose_three}


def simulate_trial(index):
    from scipy.stats import unitary_group

    dimension = _WORKER["dimension"]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [_WORKER["seed"], _WORKER["condition_id"], int(index)]
        )
    )
    raw_moments = {order: 0.0 for order in (1, 2, 3)}
    raw_observables = {order: 0.0 for order in (1, 2, 3)}

    for _ in range(_WORKER["nu"]):
        unitary = unitary_group.rvs(dimension, random_state=rng)
        probabilities = diagonal_probabilities(
            unitary, _WORKER["weights"], _WORKER["column_block"]
        )
        observable_diagonal = np.abs(unitary[:, 0]) ** 2 - 1.0 / dimension
        counts = rng.multinomial(_WORKER["nm"], probabilities)
        combinations = choose_counts(counts)
        for order in (1, 2, 3):
            raw_moments[order] += (
                np.sum(combinations[order])
                * _WORKER["moment_coefficients"][order]
            )
            raw_observables[order] += (
                np.dot(observable_diagonal, combinations[order])
                * _WORKER["observable_coefficients"][order]
            )

    for order in (1, 2, 3):
        raw_moments[order] /= _WORKER["nu"]
        raw_observables[order] /= _WORKER["nu"]

    purity = raw_moments[2] - 1.0
    third_moment = (raw_moments[3] - 3.0 * purity - 1.0) / 2.0
    rho2_observable = (raw_observables[2] - 2.0 * raw_observables[1]) / 2.0
    rho3_observable = (
        raw_observables[3]
        - 6.0 * rho2_observable
        - 3.0 * (1.0 + purity) * raw_observables[1]
    ) / 6.0
    if purity == 0.0 or third_moment == 0.0:
        raise ZeroDivisionError("Estimated normalization moment is zero")
    identity_offset = 1.0 / dimension
    return {
        "idx": int(index),
        "purity": float(np.real(purity)),
        "third_moment": float(np.real(third_moment)),
        "pf_rhoO2_fidelity": float(np.real(rho2_observable / purity + identity_offset)),
        "pf_rhoO3_fidelity": float(
            np.real(rho3_observable / third_moment + identity_offset)
        ),
    }


def conditions(params, axis):
    result = []
    if axis in ("beta", "all"):
        delta = float(params["left"]["Delta"])
        result.extend((float(beta), delta, "beta") for beta in params["left"]["beta"])
    if axis in ("delta", "all"):
        beta = float(params["right"]["beta"])
        result.extend((beta, float(delta), "Delta") for delta in params["right"]["Delta"])
    return result


def condition_seed_id(beta, delta):
    """Stable seed component independent of whether axes are run separately."""
    return int(round(float(beta) * 100)) * 10_000 + int(round(float(delta) * 100))


def main():
    parser = argparse.ArgumentParser(
        description="Run the paper-grid QVC experiment for the XXZ Heisenberg model."
    )
    parser.add_argument("--axis", choices=("beta", "delta", "all"), default="all")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--column-block", type=int, default=64)
    parser.add_argument("--spectrum-dir", type=Path, default=DATA / "spectra" / "heisenberg")
    parser.add_argument("--output-dir", type=Path, default=DATA / "heisenberg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.processes < 1 or args.column_block < 1:
        raise ValueError("--processes and --column-block must be positive")

    params = json.loads(
        (ROOT / "qvc_heisenberg_parameters.json").read_text(encoding="utf-8")
    )
    n = int(params["n"])
    j_coupling = float(params["J"])
    nu = int(params["N_U"])
    nm = int(params["N_M"])
    trials = int(params["trials"])
    jobs = conditions(params, args.axis)
    spectra = {
        delta: args.spectrum_dir / spectrum_filename(n, j_coupling, delta)
        for _, delta, _ in jobs
    }
    missing_spectra = [str(path) for path in spectra.values() if not path.exists()]
    outputs = [
        args.output_dir / data_filename(n, nu, nm, beta, delta)
        for beta, delta, _ in jobs
    ]
    dimension = 1 << n
    unitary_gib = dimension * dimension * 16 / 1024**3
    print(f"Conditions: {len(jobs)}, trials per condition: {trials}")
    print(f"Each Haar unitary alone occupies about {unitary_gib:.2f} GiB (complex128).")
    print(f"Worker processes: {args.processes}; use more than one only with sufficient RAM.")
    for (beta, delta, axis_name), output in zip(jobs, outputs):
        status = "exists" if output.exists() else "missing"
        print(
            f"  [{status}] {axis_name}: beta={beta:.2f}, "
            f"Delta={delta:.2f} -> {output}"
        )
    if args.dry_run:
        if missing_spectra:
            print("Missing prerequisite spectra:")
            for path in missing_spectra:
                print(f"  {path}")
        return
    if missing_spectra:
        raise FileNotFoundError("Missing Heisenberg spectra: " + ", ".join(missing_spectra))
    from tqdm import tqdm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (beta, delta, _), output in zip(jobs, outputs):
        if output.exists() and not args.overwrite:
            print(f"Skipping existing data {output}")
            continue
        energies = load_spectrum(
            spectra[delta], expected_n=n, expected_j=j_coupling, expected_delta=delta
        )
        initializer_args = (
            energies,
            beta,
            nu,
            nm,
            args.seed,
            condition_seed_id(beta, delta),
            args.column_block,
        )
        if args.processes == 1:
            initialize_worker(*initializer_args)
            results = [
                simulate_trial(index)
                for index in tqdm(range(trials), desc=f"beta={beta:.2f}, Delta={delta:.2f}")
            ]
        else:
            with Pool(
                processes=args.processes,
                initializer=initialize_worker,
                initargs=initializer_args,
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(simulate_trial, range(trials)),
                        total=trials,
                        desc=f"beta={beta:.2f}, Delta={delta:.2f}",
                    )
                )
        pd.DataFrame(results).sort_values("idx").to_csv(output, index=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
