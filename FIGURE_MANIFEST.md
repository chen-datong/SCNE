# Active numerical figures in SCNE3

| Figure | Experiment directory | Main entry point | Reproduction status |
|---|---|---|---|
| `SE_vs_NM_and_n.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Left-panel data included; fixed `N_M=10^8` right-panel data require regeneration. |
| `sample_lower.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Values are hard-coded in `plot.py`. |
| `vcooling.pdf` | `experiments/virtual_cooling` | `python plot.py` | Caption-compliant `N_M=10^6` data require regeneration. |
| `SE_all_observables.pdf` | `experiments/cbne_observables` | `python plot.py` | Required processed CSV files included. |
| `SE_vs_NM_and_nA.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Data included; default code selects exactly `N_M=10^7`. |
| `Gibbs_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Caption-compliant `N_M=10^6`, 1000-trial data require regeneration. |
| `Depol_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Caption-compliant `N_M=10^6` data require regeneration; default bands are standard errors. |

The exact PDFs copied from the manuscript directory are under
`reference/paper_figures/`. `reference/local_legacy/` contains the older local
`Depol_ent_detect` plus archived outputs generated from retained non-caption
parameter subsets.

The non-numerical `PTME.pdf` protocol/composition figure is excluded. The
standalone numerical panel that was embedded in it can still be regenerated as
`experiments/ptme/figure/TFIM_detection_prob.svg`.
