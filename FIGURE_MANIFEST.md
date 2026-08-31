# Active numerical figures in SCNE3

| Figure | Experiment directory | Main entry point | Reproduction status |
|---|---|---|---|
| `SE_vs_NM_and_n.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Data included. |
| `sample_lower.pdf` | `experiments/cbne_moment` | `python reproduce_figures.py` | Values are hard-coded in `plot.py`. |
| `vcooling.pdf` | `experiments/virtual_cooling` | `python plot.py` | Data included; retained code uses `NM=1000`, unlike the manuscript caption. |
| `SE_all_observables.pdf` | `experiments/cbne_observables` | `python plot.py` | Required processed CSV files included. |
| `PTME.pdf` | `experiments/ptme` | `python reproduce_figures.py`, then edit/export `PTME.drawio` | Numerical panel can be regenerated; final composition is a Draw.io export. |
| `SE_vs_NM_and_nA.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Data included. |
| `Gibbs_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Data included. |
| `Depol_ent_detect.pdf` | `experiments/ptme` | `python reproduce_figures.py` | Reconstructed paper-style code included; old remote code only made the legacy plot. |

The exact PDFs copied from the manuscript directory are under
`reference/paper_figures/`.  `reference/local_legacy/` contains the older local
`Depol_ent_detect` for comparison.
