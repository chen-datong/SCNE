# SCNE3 numerical experiments

This repository consolidates the code and retained data used by the eight
active numerical figures in `paper/SCNE3.tex`.  Source directories outside
this repository were not modified.

## Layout

- `experiments/cbne_moment`: third-moment error scaling and sample-complexity plot.
- `experiments/virtual_cooling`: virtual-cooling fidelity plot.
- `experiments/cbne_observables`: ancilla-assisted observable estimates.
- `experiments/ptme`: PT-moment, entanglement-detection, and PTME composition code.
- `reference/paper_figures`: exact PDFs used by the manuscript.
- `paper/SCNE3.tex`: the audited TeX source.
- `FIGURE_MANIFEST.md`: figure-to-code mapping and reproduction status.
- `REMOTE_AUDIT.md`: GitHub comparison and local-only updates.

## Python setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run each command from the listed experiment directory because the historical
scripts use relative paths.

```powershell
cd experiments\cbne_moment
python reproduce_figures.py

cd ..\virtual_cooling
python plot.py

cd ..\cbne_observables
python plot.py

cd ..\ptme
python reproduce_figures.py
```

The C++ simulators use ITensor and retain their original build files/source.
They are located under `experiments/cbne_moment/simulator` and
`experiments/ptme/tensor_simulator`.  The processed inputs required for the
included plotting commands are already present, so ITensor is not required to
redraw the figures.

## Known manuscript/code differences

1. The exact `vcooling.pdf` was produced from files named with `NM=1000`, while
   the manuscript caption says `N_M=10^6`.
2. The right panel of `SE_vs_NM_and_n.pdf` does not use a single fixed
   `N_M=10^8` for every system size.
3. The right panel of `SE_vs_NM_and_nA.pdf` averages the last five sample-size
   points rather than selecting only `N_M=10^7`.
4. `Gibbs_ent_detect.pdf` uses 100 retained rows for the `N_U=100, NM=1000`
   subset, although the caption says 1000 independent trials.
5. The manuscript `Depol_ent_detect.pdf` uses ten depolarizing probabilities
   and shaded standard-deviation bands.  The remote code only has an older
   five-point error-bar version.  The reconstructed function documents this;
   `spread="sem"` is available if the caption's “standard error” wording is to
   be followed literally.

## Data policy

This snapshot includes only data needed by the active plotting paths, rather
than every raw simulation file from the three historical directories.  No file
exceeds GitHub's 100 MB per-file limit.
