"""Audit Heisenberg QVC spectra and simulation data."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "virtual_cooling"
sys.path.insert(0, str(EXPERIMENT))

from generate_heisenberg_data import data_filename  # noqa: E402
from heisenberg_model import load_spectrum, spectrum_filename  # noqa: E402


def report(label, missing):
    if not missing:
        print(f"OK       {label}")
        return True
    print(f"MISSING  {label}")
    for item in missing:
        print(f"         {item}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--data-only-spectra",
        action="store_true",
        help="do not require the 0.05-spaced Delta spectra used for smooth truth lines",
    )
    args = parser.parse_args()
    params = json.loads(
        (ROOT / "qvc_heisenberg_parameters.json").read_text(encoding="utf-8")
    )
    n = int(params["n"])
    j_coupling = float(params["J"])
    nu = int(params["N_U"])
    nm = int(params["N_M"])
    trials = int(params["trials"])
    spectrum_dir = ROOT / "data" / "virtual_cooling" / "spectra" / "heisenberg"
    data_dir = ROOT / "data" / "virtual_cooling" / "heisenberg"

    deltas = {float(params["left"]["Delta"])}
    key = "right" if args.data_only_spectra else "truth_grid"
    deltas.update(float(value) for value in params[key]["Delta"])
    spectrum_missing = []
    for delta in sorted(deltas):
        path = spectrum_dir / spectrum_filename(n, j_coupling, delta)
        if not path.exists():
            spectrum_missing.append(path.name)
            continue
        try:
            load_spectrum(path, expected_n=n, expected_j=j_coupling, expected_delta=delta)
        except ValueError as error:
            spectrum_missing.append(f"{path.name}: {error}")

    conditions = [
        (float(beta), float(params["left"]["Delta"]))
        for beta in params["left"]["beta"]
    ]
    conditions.extend(
        (float(params["right"]["beta"]), float(delta))
        for delta in params["right"]["Delta"]
    )
    data_missing = []
    required_columns = {
        "idx",
        "purity",
        "third_moment",
        "pf_rhoO2_fidelity",
        "pf_rhoO3_fidelity",
    }
    for beta, delta in conditions:
        path = data_dir / data_filename(n, nu, nm, beta, delta)
        if not path.exists():
            data_missing.append(path.name)
            continue
        frame = pd.read_csv(path)
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            data_missing.append(f"{path.name}: missing {sorted(missing_columns)}")
        elif len(frame) != trials:
            data_missing.append(f"{path.name}: expected {trials} rows, found {len(frame)}")

    checks = [
        report("Heisenberg spectra", spectrum_missing),
        report("Heisenberg QVC data", data_missing),
    ]
    if not all(checks) and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
