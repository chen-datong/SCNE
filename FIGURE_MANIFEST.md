# Active numerical figures in SCNE3

| Figure | Experiment directory | Main entry point | Reproduction status |
|---|---|---|---|
| `SE_vs_NM_and_n.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Data included. |
| `sample_lower.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Values are hard-coded in `plot.py`. |
| `vcooling.pdf` | `experiments/virtual_cooling` | `python plot.py` | Data included; retained code uses `NM=1000`, unlike the manuscript caption. |
| `SE_all_observables.pdf` | `experiments/cbne_observables` | `python plot.py` | Required processed CSV files included. |
| `SE_vs_NM_and_nA.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Data included. |
| `Gibbs_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Data included. |
| `Depol_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Reconstructed paper-style code included; old remote code only made the legacy plot. |

The exact PDFs copied from the manuscript directory are under
`reference/paper_figures/`.  `reference/local_legacy/` contains the older local
`Depol_ent_detect` for comparison.

The non-numerical `PTME.pdf` protocol/composition figure is excluded. The
standalone numerical panel that was embedded in it can still be regenerated as
`experiments/ptme/figure/TFIM_detection_prob.svg`.
