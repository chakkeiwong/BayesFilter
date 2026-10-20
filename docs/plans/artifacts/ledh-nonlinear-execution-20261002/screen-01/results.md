# Nonlinear likelihood and score comparison

Heuristic comparison: not_established_missing_reference_or_comparators. See summary.json for conditional paired errors.

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| predator_prey / 20 / 1008 / 260401 / 0 | iid_dual_cap / covariance_only | 2 / 2 | -97.4387016 | -33.3663082, -0.521889836, 0.0223338883, 4.4075315, -9.18847752, 11.424664 |
| predator_prey / 20 / 1008 / 260401 / 0 | iid_dual_cap / original | 2 / 2 | -97.4388123 | -33.5629845, -0.527369857, 0.022287325, 4.41377139, -9.20778894, 11.4398022 |
| predator_prey / 20 / 1008 / 260401 / 0 | iid_dual_cap / richer_marginal | 2 / 2 | -97.3689995 | -33.4925518, -0.545005798, 0.0199153805, 4.56110454, -8.6144557, 10.7226715 |
| predator_prey / 20 / 1008 / 260401 / 0 | iid_dual_cap / richer_pairwise | 2 / 2 | -97.3483238 | -33.5500336, -0.549612075, 0.0197138768, 4.57080007, -8.56857681, 10.665524 |
| predator_prey / 20 / 1008 / 260401 / 0 | iid_dual_cap / guarded_pairwise | 2 / 2 | -97.3503494 | -33.5277176, -0.546957374, 0.0196376313, 4.56825876, -8.54556799, 10.6375685 |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / covariance_only | 0 / 2 | unavailable | unavailable |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / original | 0 / 2 | unavailable | unavailable |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / richer_marginal | 0 / 2 | unavailable | unavailable |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / richer_pairwise | 0 / 2 | unavailable | unavailable |
| sir_d18 / 20 / 1008 / 260401 / 0 | iid_dual_cap / guarded_pairwise | 0 / 2 | unavailable | unavailable |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-predator_prey | 0 | 135.46 |
| worker-0002-sir_d18 | 2 | 188.05 |
