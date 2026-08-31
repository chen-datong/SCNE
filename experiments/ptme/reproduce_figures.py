"""Regenerate the PTME numerical figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from Plot import (
    detecting_ent_tfim,
    gibbs_ent_detect,
    plot_mae_t4_nA_convergence,
)
from heisenberg_plot import heisenberg_depol_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retained-data",
        action="store_true",
        help="redraw historical PDFs with their retained non-caption parameter subsets",
    )
    parser.add_argument(
        "--figure",
        choices=("all", "tfim", "scaling", "gibbs", "depol"),
        default="all",
        help="reproduce one figure or the complete PTME numerical set",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]
    scaling = params["SE_vs_NM_and_nA.pdf"]
    gibbs = params["Gibbs_ent_detect.pdf"]
    depol = params["Depol_ent_detect.pdf"]

    selected = {"tfim", "scaling", "gibbs", "depol"} if args.figure == "all" else {args.figure}
    gibbs_csv = "gibbs_dbasis.csv" if args.retained_data else "gibbs_dbasis_paper.csv"
    depol_csv = (
        "heisenberg_dbasis_prob.csv"
        if args.retained_data
        else "heisenberg_dbasis_prob_paper.csv"
    )
    required = []
    if "gibbs" in selected:
        required.append(gibbs_csv)
    if "depol" in selected:
        required.append(depol_csv)
    missing = [name for name in required if not Path(name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data for the selected parameter mode: " + ", ".join(missing)
        )

    if "tfim" in selected:
        detecting_ent_tfim(csv_filename="processed_results2.csv")
    if "scaling" in selected:
        plot_mae_t4_nA_convergence(
            data_dir="./data",
            nB=scaling["n_B"],
            right_nm=None if args.retained_data else scaling["right"]["N_M"],
            outpath="./figure/SE_vs_NM_and_nA.pdf",
        )
    if "gibbs" in selected:
        gibbs_ent_detect(
            nA=gibbs["n_A"],
            nB=gibbs["n_B"],
            Nu=gibbs["N_U"],
            NM=1000 if args.retained_data else gibbs["N_M"],
            csv_path=gibbs_csv,
            expected_trials=100 if args.retained_data else gibbs["trials"],
        )
    if "depol" in selected:
        heisenberg_depol_paper(
            nA=depol["n_A"],
            nB=depol["n_B"],
            Nu=depol["N_U"],
            NM=100 if args.retained_data else depol["N_M"],
            csv_path=depol_csv,
            outpath="./figure/Depol_ent_detect.pdf",
            spread="std" if args.retained_data else depol["error_band"],
            expected_trials=depol["trials"],
        )


if __name__ == "__main__":
    main()
