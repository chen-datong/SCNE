# CBNE experiment for `tr(rho^3)`

This derived experiment uses exactly the state and sampling grid declared for
`SE_all_observables.pdf`:

- six-qubit open-boundary TFIM Gibbs state;
- `J=h=1`, `beta=2`;
- 1000 independent repetitions;
- upper panel: `N_U=1`, `n_a=0,4,8`, and the same 13-point `N_M` grid;
- lower panel: fixed `N_M=1000000`, `n_a=0,...,8`, and `N_U=1,10,50`.

The retained paper CSVs already contain a `third_moment` column, so the new
two-panel figure can be produced immediately without rerunning measurements:

```powershell
cd experiments\cbne_observables
python plot_rho3.py --source paper
```

This writes `figure/SE_rho3.pdf` and the plotted MAE values to
`data/rho3_statistical_error_n6_Nu1_beta2.0.csv`.

To generate an independent raw data set from scratch, run:

```powershell
python generate_rho3_data.py --processes 4
python plot_rho3.py --source rho3
```

The fresh simulator samples Haar-random measurement bases through a Haar
isometry, avoiding construction of a dense `2^(n+n_a)` square unitary. It still
performs 1000 repetitions for each of nine ancilla sizes and is therefore a
large run. Output files are protected against accidental overwrite; use
`--overwrite` only when intentionally replacing a complete earlier rerun.

For the lower panel, each group of 10 or 50 independent `N_U=1` trials is
averaged before computing the error, reproducing the `N_U=10` and `N_U=50`
curves used by `SE_all_observables.pdf`.
