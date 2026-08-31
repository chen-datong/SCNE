"""Regenerate the two active CBNE moment figures."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import plot


def main():
    argparse.ArgumentParser(
        description="Regenerate CBNE figures using manuscript parameters."
    ).parse_args()

    config_path = Path(__file__).resolve().parents[2] / "configs" / "paper_parameters.json"
    params = json.loads(config_path.read_text(encoding="utf-8"))["figures"]
    scaling = params["SE_vs_NM_and_n.pdf"]
    lower = params["sample_lower.pdf"]

    os.makedirs("figures", exist_ok=True)
    right_nm_by_n = {
        n: scaling["right"]["N_M"] for n in scaling["right"]["n"]
    }
    plot.plot_Haar_N(right_nm_by_n=right_nm_by_n)
    plot.plot_sample_complexity_lower(
        nu=lower["N_U"],
        target_error=lower["target_error"],
        repetitions=lower["trials"],
    )


if __name__ == "__main__":
    main()
