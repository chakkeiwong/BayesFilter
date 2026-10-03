# Nonlinear likelihood and score comparison

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| predator_prey / 2 / 72 / 260201 / 0 | iid_dual_cap / original | 1 / 1 | -15.5330391 | -39.2024765, -0.763751805, -0.0374134891, 7.89234066, 3.00860214, -4.14407349 |
| predator_prey / 2 / 72 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 1 / 1 | -15.6170959 | -39.9731979, -0.771629989, -0.0390072986, 7.54793453, 3.52906108, -4.82089949 |
| sir_d18 / 2 / 72 / 260201 / 0 | iid_dual_cap / original | 0 / 1 | unavailable | unavailable |
| sir_d18 / 2 / 72 / 260201 / 0 | iid_dual_cap / guarded_pairwise | 0 / 1 | unavailable | unavailable |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-predator_prey | 0 | 133.85 |
| worker-0002-sir_d18 | 2 | 110.66 |
