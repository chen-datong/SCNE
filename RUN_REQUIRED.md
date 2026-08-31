# Simulations still required

Four caption-compliant data sets are still missing. The commands below generate
raw/processed simulation data; plotting should be done only after the matching
data check passes. All commands assume the Python environment in `README.md`.

## 1. CBNE size scaling: `SE_vs_NM_and_n.pdf`, right panel

Run from `experiments/cbne_moment/simulator` in a Bash environment:

```bash
bash run_experiments.sh
```

This runs `n = 5, 8, 11, 14, 17, 20`, with `N_M=100000000` and 5000
repetitions per size. It must create the six matching JSONL files under
`experiments/cbne_moment/data/TIM_complete`.

Prerequisite currently absent: a compiled ITensor `main` executable and the MPS
inputs it expects. Build/restore those before launching this run.

## 2. Virtual cooling: `vcooling.pdf`

Run from `experiments/virtual_cooling`:

```powershell
python generate_paper_data.py --axis all
```

This generates 11 CSV files at `n=14`, `N_U=1`, `N_M=1000000`, and 100
trials: six beta points and five field-strength points.

Prerequisite currently absent: the corresponding 14-qubit TFIM Gibbs-state and
ground-state `.npy` files under `experiments/virtual_cooling/TFIM`. `TFIM.py`
contains the state-generation functions, but its example `__main__` block is
not configured for the complete paper grid and should not be run unchanged.

## 3. Gibbs entanglement detection: `Gibbs_ent_detect.pdf`

Run from `experiments/ptme`:

```powershell
python generate_paper_data.py --figure gibbs
```

This generates `gibbs_dbasis_paper.csv` for ten beta values, using `n_A=11`,
`n_B=1`, `N_U=100`, `N_M=1000000`, and 1000 trials per beta.

Prerequisite currently absent: the ten 12-qubit Gibbs-state files
`gibbs/gibbs_N12_beta0.60.npy` through `gibbs/gibbs_N12_beta1.50.npy`.

## 4. Depolarizing-noise entanglement detection: `Depol_ent_detect.pdf`

Run from `experiments/ptme`:

```powershell
python generate_paper_data.py --figure depol
```

This generates `heisenberg_dbasis_prob_paper.csv` for `p=0.0,0.1,...,0.9`,
using `n_A=11`, `n_B=1`, `N_U=1`, `N_M=1000000`, and 1000 trials per noise
value. The required Heisenberg ground-state file is already present, so this is
the only outstanding simulation whose state prerequisite is currently ready.

## Data that does not need to be rerun

- `SE_vs_NM_and_n.pdf`, left panel;
- `sample_lower.pdf`;
- `SE_all_observables.pdf`;
- `SE_vs_NM_and_nA.pdf`.

## Check and plot after the runs

From the repository root, confirm that all paper data are complete:

```powershell
python scripts/check_paper_data.py
```

Then regenerate the affected figures from their experiment directories:

```powershell
cd experiments\cbne_moment
python reproduce_figures.py

cd ..\virtual_cooling
python plot.py

cd ..\ptme
python reproduce_figures.py --figure gibbs
python reproduce_figures.py --figure depol
```

These simulations are computationally large and are not launched by the data
checker or plotting commands.
