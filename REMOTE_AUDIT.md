# GitHub remote audit

Checked on 2026-08-31.

## Known remotes

| Local source | Remote | Remote `main` | Result |
|---|---|---|---|
| `Moment(1)/moment-estimation` | `https://github.com/chen-datong/moment-estimation.git` | `b7848a610b53dda587d05b5b0df458c569d7eddc` (2026-02-09) | Local checkout is at the same remote commit. |
| `moment/Negativity2` | `https://github.com/chen-datong/Negativity2.git` | `2cff36d4e2e8ce107f52999b2fa876afd51a1abc` (2026-02-08) | Local checkout is at the same remote commit, but important manuscript files are newer uncommitted local changes. |

Only the `main` branch exists on either remote.  The public repositories under
the same GitHub account do not contain another SCNE/CBNE/PTME codebase.

## Important conclusion

The remote repositories are not newer than the local files.  In particular,
the following useful manuscript-era artifacts are local-only:

- modified `Negativity2/Plot.py`;
- `processed_results2.csv`;
- the final PTME protocol/composition assets (intentionally excluded from this numerical repository);
- `SE_vs_NM_and_nA.pdf` and its `mae_n*_p0.30.csv` inputs;
- the full-range manuscript version of `Depol_ent_detect.pdf`.

The tracked remote `heisenberg_plot.py` only produces the older five-point
error-bar plot. This repository adds `heisenberg_depol_paper()` with the
caption-declared ten-point range, `N_M=10^6`, 1000 trials, and standard-error
bands. Explicit compatibility mode can still audit the retained PDF's
standard-deviation rendering.
