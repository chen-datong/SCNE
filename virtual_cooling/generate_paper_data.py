"""Generate TFIM cooling data using total measurement counts N_M."""
import argparse
import json
from pathlib import Path
from main import main_experiment_gibbs_parallel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "virtual_cooling"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("beta", "h", "all"), default="all")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="show plan only; do not construct states or simulate")
    parser.add_argument("--prepare-states", action="store_true", help="prepare missing TFIM states only, then exit (dense, high memory)")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    p = json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]["vcooling.pdf"]
    n, nu, nm, trials = p["n"], p["N_U"], p["N_M"], p["trials"]
    jobs = []
    if args.axis in ("beta", "all"):
        jobs.extend(("beta", beta, p["left"]["h"]) for beta in p["left"]["beta"])
    if args.axis in ("h", "all"):
        jobs.extend(("h", p["right"]["beta"], h) for h in p["right"]["h"])
    states = {}
    outputs = []
    for axis, beta, h in jobs:
        states[DATA / "TFIM" / f"tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy"] = ("gibbs", beta, h)
        states[DATA / "TFIM" / f"tfim_N{n}_gs_J1.0_h{h}.npy"] = ("gs", beta, h)
        outputs.append(DATA / f"fidelity_gibbs_n{n}_Nu{nu}_NM{nm}_{axis}{beta if axis == 'beta' else h}.csv")
    missing = [path for path in states if not path.exists()]
    print(f"TFIM: n={n}, N_U={nu}, total N_M={nm}, trials={trials}, conditions={len(jobs)}, processes={args.processes}")
    print(f"Legacy NM1000 used total N_M=1000*2^{n}={1000*(1 << n)}; current NM is total shots.")
    print(f"Missing state inputs: {len(missing)}")
    for path in missing:
        print(path)
    if args.dry_run:
        return
    if args.prepare_states:
        from TFIM import gibbs_full_rho_tfim_to_npy, ground_state_full
        for path in missing:
            kind, beta, h = states[path]
            path.parent.mkdir(parents=True, exist_ok=True)
            function = gibbs_full_rho_tfim_to_npy if kind == "gibbs" else ground_state_full
            function(str(path), N=n, beta=beta, J=p["J"], h=h, pbc=False)
        return
    if missing:
        raise FileNotFoundError("TFIM state inputs are missing. --prepare-states creates them using dense diagonalization.")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing[0]}. Preserve existing outputs before starting a fresh run.")
    for axis, beta, h in jobs:
        main_experiment_gibbs_parallel(n, nu, nm, trials, beta=beta, h=h,
                                      output_axis=axis, num_processes=args.processes)


if __name__ == "__main__":
    main()
