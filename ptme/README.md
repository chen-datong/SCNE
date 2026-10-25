# Partial-transpose moments and entanglement detection

From the repository root:

```powershell
python reproduce_paper_figures.py --figure ptme-scaling
python reproduce_paper_figures.py --figure tfim-detection
python reproduce_paper_figures.py --figure gibbs-detection
python reproduce_paper_figures.py --figure depol-detection
```

- `Plot.py`, `heisenberg_plot.py`: Fig S3, Fig. 4(b), Fig S4 and Fig S5 plotting.
- `negativity_estimation.py`: PT-moment estimation and trial generation.
- `Gibbs.py`, `XXZ.py`, `utils.py`: model construction and witness formulas.
- `generate_paper_data.py --figure gibbs|depol`: new total-shot data;
  supports `--dry-run`, `--processes 1`, and Gibbs-only `--prepare-states`.
- `tensor_simulator/`: ITensor backend; `process_jsonl_results.py` prepares
  detection inputs; `calculate_stat.py` and `process_exact_results.py` prepare
  MAEs; `min_nm_third_mae.py` performs the threshold search.

Inputs are in `../data/ptme/`; outputs in `../figures/`.
Fig S4/Fig S5 use `gibbs_dbasis_paper.csv` and
`heisenberg_dbasis_prob_paper.csv` in `data/ptme/`. Both use total N_M=4096000. Their old NM1000 labels represented
N_M/d with d=4096; only that label was converted, preserving all moment values.
Fig S4 has 100 trials per beta at N_U=100 (standard-deviation bands).
Fig S5 has 1000 trials per depolarizing probability at N_U=1
(standard-deviation bands, matching the reference figure). The Fig S5 caption
should say "standard deviation" rather than "standard error". `paper_parameters.json` records these actual values;
the generator also follows that configuration.
