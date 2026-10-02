# Nonlinear likelihood and score comparison

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| sir_d18 / 2 / 1008 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 1 / 1 | -81.8123245 | -143.582336, 52.0719452, 38.652504 |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-sir_d18 | 0 | 77.11 |
