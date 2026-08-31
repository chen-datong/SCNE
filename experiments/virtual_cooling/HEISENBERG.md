# QVC with the XXZ Heisenberg model

This variant keeps the numerical layout of `vcooling.pdf` and replaces the
TFIM by the open-boundary antiferromagnetic XXZ Hamiltonian

```text
H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}).
```

The convention uses Pauli matrices without spin-1/2 factors, `J=1`, and even
system size `n=14`. The left panel fixes `Delta=1` and uses the original six
beta values. The right panel fixes `beta=1` and replaces the TFIM field scan by
`Delta=0.5,0.75,1.0,1.25,1.5`. Both panels use `N_U=1`, `N_M=1000000`, 100
trials, virtual-cooling orders 2 and 3, and standard-deviation error bars.

All parameters are stored in `../../configs/qvc_heisenberg_parameters.json`.

## Run order

Run from `experiments/virtual_cooling`.

Inspect the spectrum plan without starting diagonalization:

```powershell
python generate_heisenberg_spectra.py --dry-run
```

Generate the spectra. The smooth right-panel truth curves require 21 values of
`Delta`; use `--data-only` if only the five simulation points are needed first.

```powershell
python generate_heisenberg_spectra.py
```

Inspect the measurement plan without starting the experiment:

```powershell
python generate_heisenberg_data.py --dry-run
```

After the spectra exist, run all 11 sampled conditions and plot:

```powershell
python generate_heisenberg_data.py --processes 1
python ..\..\scripts\check_qvc_heisenberg.py
python plot_heisenberg.py
```

The output figure is `figure/qvc_heisenberg.pdf`. Heisenberg outputs use their
own `spectra/heisenberg` and `data/heisenberg` directories and never overwrite
the TFIM files.

## Resource warning

The spectrum code exploits fixed-magnetization blocks and stores only 16384
eigenvalues per `Delta`; it does not construct a dense 14-qubit Gibbs matrix.
The exact Haar-measurement step remains expensive: one `2^14` square
complex128 unitary alone occupies 4 GiB, before QR workspace. The runner
therefore defaults to one process. Increasing `--processes` multiplies the
memory requirement.

The estimator works in the common energy eigenbasis. This is exactly equivalent
to rotating the Gibbs density matrix and its ground-state projector together,
because the Haar ensemble is invariant under a fixed change of basis.
