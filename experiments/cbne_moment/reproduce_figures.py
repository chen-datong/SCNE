"""Regenerate the two active CBNE moment figures."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import plot


RETAINED_RIGHT_NM = {
    5: 1_000_000,
    8: 1_000_000,
    11: 1_000_000,
    14: 1_000_000,
    17: 15_000_000,
    20: 100_000_500,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retained-data",
        action="store_true",
        help="redraw the historical manuscript PDF from the retained nonuniform N_M data",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]
    scaling = params["SE_vs_NM_and_n.pdf"]
    lower = params["sample_lower.pdf"]

    os.makedirs("figures", exist_ok=True)
    right_nm_by_n = (
        RETAINED_RIGHT_NM
        if args.retained_data
        else {n: scaling["right"]["N_M"] for n in scaling["right"]["n"]}
    )
    plot.plot_Haar_N(right_nm_by_n=right_nm_by_n)
    plot.plot_sample_complexity_lower(
        nu=lower["N_U"],
        target_error=lower["target_error"],
        repetitions=lower["trials"],
    )


if __name__ == "__main__":
    main()
