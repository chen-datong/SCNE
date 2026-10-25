# CBNE ancilla and observable experiments

From the repository root:

```powershell
python reproduce_paper_figures.py --figure cbne-scaling
python reproduce_paper_figures.py --figure cbne-observables
```

- `plot_combined_scaling.py`: Fig. 2; upper-row rendering is in `cbne_moment/plot.py`.
- `plot.py`: Fig S2, the 2x4 Gibbs moment/observable figure.
- `plot_rho3.py`: shared rho^3 statistics and panel rendering.
- `generate_rho3_data.py`: the single rho^3 generator; `--state gibbs` selects
  Gibbs data and the default selects depolarized data. `--dry-run` prints the
  plan, `--ancilla` selects values, and `--processes` defaults to one.
- `main.py`, `TFIM.py`: observable simulation and TFIM state preparation.

Parameters are in `../paper_parameters.json`; inputs in `../data/cbne_observables/`;
outputs in `../figures/`. Fig. 2 uses the depolarized six-qubit state at p=0.2.
Fig S2 uses the six-qubit Gibbs state at beta=2, J=h=1, with shared axes within
each row. Its displayed grid stops at n_a=8 and N_M=10^6.
