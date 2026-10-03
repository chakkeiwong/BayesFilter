# Nonlinear likelihood and score comparison

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| predator_prey / 1 / 36 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 1 / 1 | -4.40527153 | 6.47290993, 0.0477669649, -0.00520191016, -0.807607651, 1.44238019, -1.99135852 |
| sir_d18 / 1 / 36 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 0 / 1 | unavailable | unavailable |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-predator_prey | 0 | 39.61 |
| worker-0002-sir_d18 | 2 | 36.10 |
