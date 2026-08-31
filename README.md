# SCNE3 numerical experiments

This repository consolidates the code and retained data used by the seven
active numerical figures in `paper/SCNE3.tex`. Source directories outside
this repository were not modified.

## Layout

- `experiments/cbne_moment`: third-moment error scaling and sample-complexity plot.
- `experiments/virtual_cooling`: virtual-cooling fidelity plot.
- `experiments/cbne_observables`: ancilla-assisted observable estimates.
  Its derived `tr(rho^3)` run and two-panel plot are documented in
  `experiments/cbne_observables/RHO3_EXPERIMENT.md`.
- `experiments/ptme`: PT-moment and entanglement-detection simulations and plots.
- `reference/paper_figures`: exact PDFs used by the manuscript.
- `reference/local_legacy`: archived outputs made from retained non-caption data.
- `paper/SCNE3.tex`: the audited TeX source.
- `FIGURE_MANIFEST.md`: figure-to-code mapping and reproduction status.
- `PARAMETERS.md`: manuscript parameter contract and data-coverage boundary.
- `RUN_REQUIRED.md`: simulations still required, commands, and prerequisites.
- `configs/paper_parameters.json`: machine-readable manuscript parameters.
- `REMOTE_AUDIT.md`: GitHub comparison and local-only updates.
- `TMP_AUDIT.md`: comparison with the additional candidates under `../tmp`.

## Python setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run each command from the listed experiment directory because the historical
scripts use relative paths. The default entry points enforce the parameters
declared in the manuscript captions and stop when matching data are absent.

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

The retained snapshot does not yet contain every strict parameter combination.
Run `python scripts/check_paper_data.py --allow-missing` from the repository root
to see the exact coverage, and see `RUN_REQUIRED.md` for the commands that still
need to be run. Maintained entry points accept only the manuscript parameter
contract. Historical non-caption outputs are archived under
`reference/local_legacy` for provenance and are not consumed by those entry
points.

The C++ simulators use ITensor and retain their original build files/source.
They are located under `experiments/cbne_moment/simulator` and
`experiments/ptme/tensor_simulator`. Historical processed inputs remain for
provenance but are not used by maintained entry points. New simulations are
required for the caption-parameter combinations listed in `PARAMETERS.md`.

## Archived manuscript/data differences

The maintained default code now follows the manuscript parameter contract. The
following statements describe only the older data retained for provenance:

1. The exact `vcooling.pdf` was produced from files named with `NM=1000`, while
   the manuscript caption says `N_M=10^6`. New simulations use `N_M` as the
   total number of outcomes rather than multiplying it by the Hilbert dimension.
2. The right panel of `SE_vs_NM_and_n.pdf` does not use a single fixed
   `N_M=10^8` for every system size.
3. The historical right panel of `SE_vs_NM_and_nA.pdf` averages the last five
   sample-size points. The default code now selects exactly `N_M=10^7`.
4. `Gibbs_ent_detect.pdf` uses 100 retained rows for the `N_U=100, NM=1000`
   subset, although the caption says 1000 independent trials.
5. The manuscript `Depol_ent_detect.pdf` visibly uses standard-deviation bands,
   although its caption says standard error. The default code now uses standard
   error. The historical standard-deviation rendering is archived only.

## Data policy

This snapshot includes only data needed by the active plotting paths, rather
than every raw simulation file from the three historical directories. No file
exceeds GitHub's 100 MB per-file limit.

The non-numerical `PTME.pdf` protocol/composition figure is intentionally not
included. Its standalone numerical entanglement-detection panel remains at
`experiments/ptme/figure/TFIM_detection_prob.svg`.
