# CBNE moments

- `plot.py`: upper row of Fig. 2 and sample-complexity Fig S1.
- `simulator/main.cpp`: ITensor simulation and TFIM ground-state preparation.
- `simulator/run_experiments.sh`: `--dry-run` checks the plan;
  `--prepare-states` generates missing MPS inputs only; no flag runs measurements.
- `min_nm_grouped_third_mae.py`: recompute threshold minima from original records.

Draw Fig. 2 using `python reproduce_paper_figures.py --figure cbne-scaling`
from the repository root. All default outputs go to `../figures/`.
Fig S1 is reproduced from the stored search minima in `plot.py`.

## Completed Fig. 2(b) data

`data/cbne_moment/TIM_complete/` contains 5000 records per size for
n=5,8,11,14,17,20 at total N_M=100000000. The first five sizes were completed
on 2026-10-22; n=20 was updated on 2026-10-24 using 2600 records from server A
and 2400 from server B, with disjoint seed ranges. The unified plotting entry
point selects these files through `check_paper_data.py`.

The original eight-field JSONL schema and numerical values are unchanged.
MaxDim is 500 for n<=14, 600 for n=17, and 900 for n=20.

Draw Fig S1 with `python reproduce_paper_figures.py --figure cbne-sample-lower`.
The C++ runner uses MaxDim=500 for n<=14, 600 for n=17 and 900 for n=20.

## Run the C++ simulator

Install ITensor C++ and a C++ compiler. From `cbne_moment/simulator/`, run:

```bash
make LIBRARY_DIR=/path/to/itensor
bash run_experiments.sh --dry-run
bash run_experiments.sh --prepare-states
bash run_experiments.sh
```

Replace `/path/to/itensor` with your ITensor installation directory. State
inputs are generated in `MPS/`. Existing output files are protected from overwrite.
