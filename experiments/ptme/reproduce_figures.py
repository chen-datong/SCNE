"""Regenerate the PTME figures for which retained processed data are available."""

import matplotlib

matplotlib.use("Agg")

from Plot import (
    detecting_ent_tfim,
    gibbs_ent_detect,
    plot_mae_t4_nA_convergence,
)
from heisenberg_plot import heisenberg_depol_paper


def main():
    detecting_ent_tfim(csv_filename="processed_results2.csv")
    plot_mae_t4_nA_convergence(
        data_dir="./data",
        nB=2,
        outpath="./figure/SE_vs_NM_and_nA.pdf",
    )
    gibbs_ent_detect(
        nA=11,
        nB=1,
        Nu=100,
        NM=1000,
        csv_path="gibbs_dbasis.csv",
    )
    heisenberg_depol_paper(
        nA=11,
        nB=1,
        NM=100,
        csv_path="heisenberg_dbasis_prob.csv",
        outpath="./figure/Depol_ent_detect.pdf",
        spread="std",
    )


if __name__ == "__main__":
    main()
