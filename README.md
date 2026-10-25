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

## Parameters and retained data

`paper_parameters.json` stores figure parameters and historical file mappings;
`qvc_heisenberg_parameters.json` stores Heisenberg cooling parameters.
All current generators use total N_M. Historical PTME and TFIM cooling labels
used N_M/d: old NM1000 means 4096000 shots for 12 qubits and 16384000 for
14 qubits. Describe retained results using their actual budgets and trial counts.

Fig. 2(b) uses N_M=100000000 for all six sizes n=5,8,11,14,17,20,
with 5000 independent random-unitary records per size. Averaging blocks of
N_U=1,10,50 gives 5000,500,100 estimates for every size. The n=20 input
was updated on 2026-10-24 from two servers (2600 + 2400 records with disjoint
seed ranges). Its original eight-field JSONL format and numerical values are
preserved. Parameters are recorded in `paper_parameters.json`.
The cooling plot prefers a complete new TFIM series when supplied.
Fig S4/Fig S5 use the restored legacy data at total N_M=4096000, stored as
`data/ptme/gibbs_dbasis_paper.csv` and `heisenberg_dbasis_prob_paper.csv`.
Only the historical NM1000 label was converted to total shots; moment values
are unchanged. Fig S4 uses N_U=100 and 100 trials per beta; Fig S5 uses
N_U=1 and 1000 trials per depolarizing probability. Their error bands are
standard deviations for both figures. The Fig S5 caption should say
"standard deviation" rather than "standard error". Paper captions must
report these actual budgets and counts.

Fig S1 uses stored search minima; Fig S3 and Fig. 4(b) have processed inputs. Original
search/simulation records are not all bundled. Fig. 4's schematic is assembled
outside this repository.

## Supporting files

`reproduce_paper_figures.py` is the single plotting entry point.
`check_paper_data.py` and `check_qvc_heisenberg.py` validate its inputs.
The C++/ITensor backends and current data-processing tools remain alongside
their experiment code. See each topic README for its model and data-generation functions.

## AI assistance

This code repository was organized in collaboration with AI. Please contact the authors if you have any questions or encounter any issues.
