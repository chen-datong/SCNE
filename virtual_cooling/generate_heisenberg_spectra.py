"""Generate XXZ spectra required by the Heisenberg QVC experiment."""

import argparse
import json
from pathlib import Path

from heisenberg_model import full_spectrum, save_spectrum, spectrum_filename


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "virtual_cooling"
ROOT = HERE.parent


def unique(values):
    return sorted(set(float(value) for value in values))


def requested_deltas(params, axis, data_only):
    deltas = []
    if axis in ("beta", "all"):
        deltas.append(params["left"]["Delta"])
    if axis in ("delta", "all"):
        key = "right" if data_only else "truth_grid"
        deltas.extend(params[key]["Delta"])
    return unique(deltas)


def main():
    parser = argparse.ArgumentParser(
        description="Generate full XXZ spectra for the Heisenberg QVC parameter grid."
    )
    parser.add_argument("--axis", choices=("beta", "delta", "all"), default="all")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="for the Delta axis, generate only five sampled points, not the smooth truth grid",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA / "spectra" / "heisenberg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    params = json.loads(
        (ROOT / "qvc_heisenberg_parameters.json").read_text(encoding="utf-8")
    )
    n = int(params["n"])
    j_coupling = float(params["J"])
    deltas = requested_deltas(params, args.axis, args.data_only)
    outputs = [
        args.output_dir / spectrum_filename(n, j_coupling, delta) for delta in deltas
    ]
    print(f"Model: {params['hamiltonian']}")
    print(f"n={n}, J={j_coupling}, Delta values={deltas}")
    for path in outputs:
        status = "exists" if path.exists() else "missing"
        print(f"  [{status}] {path}")
    if args.dry_run:
        return
    for delta, output in zip(deltas, outputs):
        if output.exists() and not args.overwrite:
            print(f"Skipping existing spectrum {output}")
            continue
        print(f"Diagonalizing n={n}, J={j_coupling}, Delta={delta} ...")
        energies = full_spectrum(n, j_coupling, delta)
        save_spectrum(output, energies, n, j_coupling, delta)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
