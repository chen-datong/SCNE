"""Generate virtual-cooling data with the manuscript-declared parameters."""

import argparse
import json
from pathlib import Path

from main import main_experiment_gibbs_parallel


def require_states(n, beta, h):
    paths = [
        Path(f"./TFIM/tfim_gibbs_N{n}_beta{beta}_J1.0_h{h}.npy"),
        Path(f"./TFIM/tfim_N{n}_gs_J1.0_h{h}.npy"),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Generate the required TFIM states with TFIM.py first: " + ", ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("beta", "h", "all"), default="all")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]["vcooling.pdf"]
    n, nu, nm, trials = params["n"], params["N_U"], params["N_M"], params["trials"]
    if args.axis in ("beta", "all"):
        for beta in params["left"]["beta"]:
            h = params["left"]["h"]
            require_states(n, beta, h)
            main_experiment_gibbs_parallel(
                n, nu, nm, trials, beta=beta, h=h,
                output_axis="beta", num_processes=args.processes,
            )
    if args.axis in ("h", "all"):
        for h in params["right"]["h"]:
            beta = params["right"]["beta"]
            require_states(n, beta, h)
            main_experiment_gibbs_parallel(
                n, nu, nm, trials, beta=beta, h=h,
                output_axis="h", num_processes=args.processes,
            )


if __name__ == "__main__":
    main()
