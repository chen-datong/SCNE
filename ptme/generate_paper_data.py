"""Generate PTME paper data using total measurement counts N_M."""
import argparse
import json
from pathlib import Path
from negativity_estimation import experiment_gibbs_mp, experiment_heisenberg_mp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ptme"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("gibbs", "depol"), required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="show parameters and missing states without simulating")
    parser.add_argument("--prepare-states", action="store_true", help="prepare missing Gibbs state inputs only, then exit")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    if args.prepare_states and args.figure != "gibbs":
        parser.error("--prepare-states is for gibbs; the depol ground state is already retained")
    figures = json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]
    params = figures["Gibbs_ent_detect.pdf" if args.figure == "gibbs" else "Depol_ent_detect.pdf"]
    n = params["n_A"] + params["n_B"]
    if args.figure == "gibbs":
        values = params["beta"]
        output = DATA / "gibbs_dbasis_paper.csv"
        states = [DATA / "gibbs" / f"gibbs_N{n}_beta{beta:.2f}.npy" for beta in values]
    else:
        values = params["depolarizing_probability"]
        output = DATA / "heisenberg_dbasis_prob_paper.csv"
        states = [DATA / "heisenberg" / f"heisenberg_N{n}_Jz1.0.npy"]
    missing = [p for p in states if not p.exists()]
    print(f"{args.figure}: n_A={params['n_A']}, n_B={params['n_B']}, N_U={params['N_U']}, total N_M={params['N_M']}, trials={params['trials']}, conditions={len(values)}, processes={args.processes}")
    print(f"Current NM is the total shot count; legacy NM=N_M/d, d={1 << n}.")
    print(f"Output: {output}")
    print(f"Missing state inputs: {len(missing)}")
    for p in missing:
        print(p)
    if args.dry_run:
        return
    if args.prepare_states:
        from Gibbs import gibbs_full_rho_tfim_to_npy
        for beta, path in zip(values, states):
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            gibbs_full_rho_tfim_to_npy(str(path), N=n, beta=beta,
                                       J=params["J"], h=params["h"], pbc=False)
        return
    if missing:
        raise FileNotFoundError("Missing state inputs; use --figure gibbs --prepare-states first.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Preserve it before starting a fresh run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    for value in values:
        kwargs = dict(Nu=params["N_U"], repeat=params["trials"],
                      NM_list=[params["N_M"]], csv_filename=str(output), processes=args.processes)
        if args.figure == "gibbs":
            experiment_gibbs_mp(params["n_A"], params["n_B"], beta=float(value), **kwargs)
        else:
            experiment_heisenberg_mp(params["n_A"], params["n_B"], dprob=float(value), **kwargs)


if __name__ == "__main__":
    main()
