# Manuscript parameter contract

The machine-readable parameter source is `configs/paper_parameters.json`.
Throughout the maintained code, `N_M` now means the total number of measurement
outcomes per random-unitary setting. It is no longer a multiplier that is
silently multiplied by the Hilbert-space dimension.

The default reproduction entry points use the parameter values stated in the
manuscript captions. They validate that matching data exist instead of silently
substituting another parameter value. There is no fallback or compatibility
mode for historical non-caption data.

## Data that must be regenerated

The retained snapshot predates this parameter contract. New simulations are
required for the following caption-declared combinations:

- `SE_vs_NM_and_n.pdf`, right panel: fixed `N_M=100000000` for every system size;
- `vcooling.pdf`: `N_M=1000000` for all beta and field values;
- `Gibbs_ent_detect.pdf`: `N_U=100`, `N_M=1000000`, and 1000 trials per beta;
- `Depol_ent_detect.pdf`: `N_U=1`, `N_M=1000000`, and 1000 trials per noise value.

The retained data are not relabelled and must not be cited as results from the
caption-declared parameter combinations. Existing non-caption output PDFs are
archived in `reference/local_legacy`; maintained entry points do not consume
them.

## Existing data already consistent with the captions

- the left panel of `SE_vs_NM_and_n.pdf`;
- `sample_lower.pdf` (the plotted minima remain hard-coded source values);
- `SE_all_observables.pdf`;
- `SE_vs_NM_and_nA.pdf` after selecting the exact `N_M=10000000` row for its
  right panel rather than averaging the last five sample-size points.

The active `experiments/ptme/figure/SE_vs_NM_and_nA.pdf` has been regenerated
with that exact selection. The original manuscript PDF remains unchanged under
`reference/paper_figures` and is expected to differ from the corrected output.

Run `python scripts/check_paper_data.py` from the repository root to audit the
current data coverage against the parameter contract. See `RUN_REQUIRED.md` for
the exact outstanding runs and their prerequisites.

## Caption-compliant generation entry points

Run these from their own experiment directories after preparing the required
ITensor build or exact TFIM/Heisenberg state files:

```text
experiments/cbne_moment/simulator/run_experiments.sh
experiments/virtual_cooling/generate_paper_data.py
experiments/ptme/generate_paper_data.py --figure gibbs
experiments/ptme/generate_paper_data.py --figure depol
```

These are large simulations. The generation scripts encode the caption values
but are not executed as part of ordinary plotting or repository verification.
The current repository does not bundle the compiled ITensor executable/MPS
inputs, the 14-qubit dense TFIM state files, or the 12-qubit Gibbs state files.
Those prerequisites must be prepared before the corresponding new simulations
can be launched. The retained 12-qubit Heisenberg ground-state vector is present.
