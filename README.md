# Quantum Nonlinear Properties from a Single Measurement Setting — Numerical Experiments

Numerical code for the manuscript and supplementary figures.

## Layout

- `cbne_moment/`: CBNE moment scaling and threshold search.
- `cbne_observables/`: ancilla scaling of moments and observables.
- `virtual_cooling/`: TFIM and XXZ Heisenberg virtual cooling.
- `ptme/`: partial-transpose moments and entanglement detection.
- `data/`: numerical inputs, grouped by the same topics.
- `figures/`: the shared output folder for all PDF and SVG figures.
- `tests/`: local checks, excluded from Git.

## Run

```powershell
python -m pip install -r requirements.txt
python reproduce_paper_figures.py --list
python reproduce_paper_figures.py --all --check-only
python reproduce_paper_figures.py --figure cbne-scaling
python reproduce_paper_figures.py --figure virtual-cooling
```

Run from the repository root. Absolute script paths also work. `--check-only`
reads inputs without drawing or simulating; `--output-dir PATH` changes the
shared figure destination. The plotting commands never start simulations.

| Figure | Selection | Output | Plot implementation |
|---|---|---|---|
| Fig. 2 | `cbne-scaling` | `SE_scaling.pdf` | `cbne_observables/plot_combined_scaling.py` |
| Fig. 3 | `virtual-cooling` | `vcooling_models.pdf` | `virtual_cooling/plot_combined_cooling.py` |
| Fig. 4(b) | `tfim-detection` | `TFIM_detection_prob.svg` | `ptme/Plot.py` |
| Fig S1 | `cbne-sample-lower` | `sample_lower.pdf` | `cbne_moment/plot.py` |
| Fig S2 | `cbne-observables` | `SE_all_observables.pdf` | `cbne_observables/plot.py` |
| Fig S3 | `ptme-scaling` | `SE_vs_NM_and_nA.pdf` | `ptme/Plot.py` |
| Fig S4 | `gibbs-detection` | `Gibbs_ent_detect.pdf` | `ptme/Plot.py` |
| Fig S5 | `depol-detection` | `Depol_ent_detect.pdf` | `ptme/heisenberg_plot.py` |

## Supporting files

`reproduce_paper_figures.py` is the single plotting entry point.
`check_paper_data.py` and `check_qvc_heisenberg.py` validate its inputs.
The C++/ITensor backends and current data-processing tools remain alongside
their experiment code. See each topic README for its model and data-generation functions.

## AI assistance

This code repository was organized in collaboration with AI. Please contact the authors if you have any questions or encounter any issues.
