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
    gibbs_csv = "gibbs_dbasis_paper.csv"
    depol_csv = "heisenberg_dbasis_prob_paper.csv"
    required = []
    if "gibbs" in selected:
        required.append(gibbs_csv)
    if "depol" in selected:
        required.append(depol_csv)
    missing = [name for name in required if not Path(name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing manuscript-parameter data for the selected figures: "
            + ", ".join(missing)
        )

    if "tfim" in selected:
        detecting_ent_tfim(csv_filename="processed_results2.csv")
    if "scaling" in selected:
        plot_mae_t4_nA_convergence(
            data_dir="./data",
            nB=scaling["n_B"],
            right_nm=scaling["right"]["N_M"],
            outpath="./figure/SE_vs_NM_and_nA.pdf",
        )
    if "gibbs" in selected:
        gibbs_ent_detect(
            nA=gibbs["n_A"],
            nB=gibbs["n_B"],
            Nu=gibbs["N_U"],
            NM=gibbs["N_M"],
            csv_path=gibbs_csv,
            expected_trials=gibbs["trials"],
        )
    if "depol" in selected:
        heisenberg_depol_paper(
            nA=depol["n_A"],
            nB=depol["n_B"],
            Nu=depol["N_U"],
            NM=depol["N_M"],
            csv_path=depol_csv,
            outpath="./figure/Depol_ent_detect.pdf",
            spread=depol["error_band"],
            expected_trials=depol["trials"],
        )


if __name__ == "__main__":
    main()
