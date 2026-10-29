# Nonlinear likelihood and score comparison

Heuristic comparison: not_established_missing_reference_or_comparators. See summary.json for conditional paired errors.

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / covariance_only | 0 / 1 | unavailable | unavailable |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / guarded_pairwise | 0 / 1 | unavailable | unavailable |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-sir_d18 | 2 | 92.28 |
