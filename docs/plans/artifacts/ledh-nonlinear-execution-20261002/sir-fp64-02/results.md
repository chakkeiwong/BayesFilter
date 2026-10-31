# Nonlinear likelihood and score comparison

Heuristic comparison: not_established_missing_reference_or_comparators. See summary.json for conditional paired errors.

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / covariance_only | 2 / 2 | -975.573545 | 884.172978, -456.477896, 571.352608 |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / guarded_pairwise | 1 / 2 | -1040.87806 | 3230.85495, -1581.97484, 936.41325 |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-sir_d18 | 2 | 80.21 |
