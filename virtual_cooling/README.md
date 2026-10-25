# Virtual cooling

```powershell
python reproduce_paper_figures.py --figure virtual-cooling
```

This writes the combined TFIM/XXZ figure to `../figures/vcooling_models.pdf`.
`plot_combined_cooling.py` assembles it; `plot.py` and `plot_heisenberg.py`
render the two model rows.

TFIM model/preparation and simulation are in `TFIM.py` and `main.py`.
`generate_paper_data.py` supports `--dry-run`, `--prepare-states` (state inputs
only), and `--processes 1`. The dense 14-qubit implementation requires several
large matrices; one complex128 square matrix alone is 4 GiB.

`heisenberg_model.py`, `generate_heisenberg_spectra.py`, and
`generate_heisenberg_data.py` provide the XXZ model, spectra and trials.
The convention is H=J sum(XX+YY+Delta ZZ), open boundaries, Pauli operators,
J=1. Parameters are in `../qvc_heisenberg_parameters.json`.

Inputs are in `../data/virtual_cooling/`. Old TFIM NM1000 file labels mean
total N_M=16384000; Heisenberg trial data use total N_M=10^6.
Existing TFIM results can be reused with their actual measurement budget.
