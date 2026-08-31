"""Generate PTME entanglement-detection data at manuscript parameters."""

import argparse
import json
from pathlib import Path

from negativity_estimation import experiment_gibbs_mp, experiment_heisenberg_mp


def require_absent(path):
    if path.exists():
        raise FileExistsError(
            f"Refusing to append duplicate paper data to {path}. "
            "Move the existing file aside before starting a fresh run."
        )


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing state inputs: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("gibbs", "depol"), required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]
    if args.figure == "gibbs":
        figure = params["Gibbs_ent_detect.pdf"]
        output = Path("gibbs_dbasis_paper.csv")
        require_absent(output)
        require_files(
            Path(f"gibbs/gibbs_N{figure['n_A'] + figure['n_B']}_beta{beta:.2f}.npy")
            for beta in figure["beta"]
        )
        for beta in figure["beta"]:
            experiment_gibbs_mp(
                figure["n_A"], figure["n_B"], Nu=figure["N_U"],
                repeat=figure["trials"], beta=float(beta),
                csv_filename=str(output), NM_list=[figure["N_M"]],
            )
    else:
        figure = params["Depol_ent_detect.pdf"]
        output = Path("heisenberg_dbasis_prob_paper.csv")
        require_absent(output)
        require_files([
            Path(f"heisenberg/heisenberg_N{figure['n_A'] + figure['n_B']}_Jz1.0.npy")
        ])
        for probability in figure["depolarizing_probability"]:
            experiment_heisenberg_mp(
                figure["n_A"], figure["n_B"], Nu=figure["N_U"],
                repeat=figure["trials"], dprob=float(probability),
                csv_filename=str(output), NM_list=[figure["N_M"]],
            )


if __name__ == "__main__":
    main()
