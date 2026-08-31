# Audit of the additional `tmp` candidates

Checked on 2026-08-31. The directory `../tmp` contains 1,618 files, including
15 Python files, 23 PDFs, and two nested Git repositories. The review covered
all Python files and every candidate PDF corresponding to an active manuscript
figure.

## Results

| Candidate | Finding | Decision |
|---|---|---|
| `tmp/Moment` | Its nested `moment-estimation` checkout is at commit `ada9130`, an older remote snapshot than the `b7848a6` checkout already audited. Its `SE_vs_NM_and_n.pdf` differs visibly from the manuscript (rendered pixel MAE 0.06745). | Do not replace the consolidated code. The repository version renders pixel-identically to the manuscript at the same resolution. |
| `tmp/virtual distillation` | Its plot is an earlier layout without panel labels and uses a coarser truth grid (pixel MAE 0.06228). `main.py` also contains an experimental ancilla sweep, but that path is not used by any active SCNE3 figure and its required input data are absent. | Keep the consolidated manuscript version; exclude the unrelated incomplete ancilla sweep. |
| `tmp/Negativity2` | This is the clean remote commit `2cff36d`. Its `Gibbs_ent_detect.pdf` is byte-identical to the manuscript, so the same code/data are already retained. Its depolarizing-noise plot is the old five-point version over `p=0.3,...,0.7` (pixel MAE 0.05881). | No new code is needed. Keep the reconstructed ten-point manuscript routine, whose rendered pixel MAE is 0.00072. |
| `tmp/Negativity` | Standalone early GHZ/PT-moment prototypes for small systems, plus one convergence PNG and two notes. They do not read the retained manuscript datasets and do not generate any of the eight active figure names. | Exclude from the figure-reproduction repository. |

## Conclusion

The `tmp` directory contains useful historical confirmation, especially the
byte-identical Gibbs figure, but no newer or more accurate source for a missing
SCNE3 manuscript figure. Copying its older plotting scripts would reduce
fidelity. Therefore no candidate code from `tmp` replaces the versions under
`experiments/`.
