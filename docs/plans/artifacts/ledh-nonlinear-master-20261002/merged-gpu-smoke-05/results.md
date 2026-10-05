# Nonlinear likelihood and score comparison

Heuristic comparison: not_established_missing_reference_or_comparators. See summary.json for conditional paired errors.

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| predator_prey / 2 / 1008 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 1 / 1 | -15.6159315 | -38.1113663, -0.746935725, -0.0373195671, 7.76226664, 3.10609818, -4.25494814 |
| sir_d18 / 2 / 1008 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 1 / 1 | -81.8123932 | -143.584335, 52.0722885, 38.6530685 |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-predator_prey | 0 | 39.30 |
| worker-0002-sir_d18 | 0 | 58.60 |
