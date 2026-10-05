# Nonlinear likelihood and score comparison

Heuristic comparison: not_established_missing_reference_or_comparators. See summary.json for conditional paired errors.

Means and standard errors are conditional on each dataset and parameter point. Missing references are unavailable, not zero. Invalid/unfinished runs are counted; means over valid runs alone must not support a ranking.

| Model / T / N / dataset / point | Route / arm | Valid / returned | Mean log likelihood | Score means |
|---|---|---:|---:|---|
| predator_prey / 20 / 1008 / 260402 / 0 | iid_dual_cap / covariance_only | 4 / 4 | -99.2975941 | -21.1051588, 0.0374907791, 0.019272699, 1.53950936, -6.71427178, 8.29984879 |
| predator_prey / 20 / 1008 / 260402 / 0 | iid_dual_cap / original | 3 / 4 | -99.2926127 | -21.1711636, 0.0382668978, 0.0198137456, 1.48714868, -6.81959852, 8.43001874 |
| predator_prey / 20 / 1008 / 260402 / 0 | iid_dual_cap / richer_marginal | 4 / 4 | -99.2878494 | -21.2183242, 0.0345742498, 0.018828928, 1.52920124, -6.58846974, 8.14454401 |
| predator_prey / 20 / 1008 / 260402 / 0 | iid_dual_cap / richer_pairwise | 4 / 4 | -99.2878799 | -21.221417, 0.034575454, 0.0189681044, 1.50847352, -6.61170053, 8.17371929 |
| predator_prey / 20 / 1008 / 260402 / 0 | iid_dual_cap / guarded_pairwise | 4 / 4 | -99.2871284 | -21.2151017, 0.0351280868, 0.0189033826, 1.52016082, -6.60116327, 8.16064453 |
| predator_prey / 20 / 1008 / 260402 / 0 | previous_inverse_cdf / covariance_only | 4 / 4 | -99.3263531 | -20.9205403, 0.0414377879, 0.0195999728, 1.48792246, -6.76876557, 8.36708522 |
| predator_prey / 20 / 1008 / 260402 / 0 | previous_inverse_cdf / original | 4 / 4 | -99.31991 | -20.9540119, 0.0419956278, 0.0198131748, 1.48274654, -6.82083595, 8.43249416 |
| predator_prey / 20 / 1008 / 260402 / 0 | previous_inverse_cdf / richer_marginal | 4 / 4 | -99.3142891 | -21.1015801, 0.0372979585, 0.0188303185, 1.53539953, -6.58897674, 8.14619923 |
| predator_prey / 20 / 1008 / 260402 / 0 | previous_inverse_cdf / richer_pairwise | 4 / 4 | -99.3115425 | -21.1044083, 0.0369013697, 0.0189568158, 1.53212965, -6.62277424, 8.18731952 |
| predator_prey / 20 / 1008 / 260402 / 0 | previous_inverse_cdf / guarded_pairwise | 4 / 4 | -99.3126259 | -21.1131444, 0.0362626705, 0.0189237618, 1.53027913, -6.61175573, 8.17402053 |
| predator_prey / 20 / 1008 / 260402 / 0 | repaired_permutation / covariance_only | 4 / 4 | -99.3247433 | -20.9272776, 0.0412008381, 0.0194803872, 1.4931168, -6.74011135, 8.33178067 |
| predator_prey / 20 / 1008 / 260402 / 0 | repaired_permutation / original | 4 / 4 | -99.320013 | -20.9613914, 0.0411265949, 0.019875335, 1.47876391, -6.8338213, 8.44730473 |
| predator_prey / 20 / 1008 / 260402 / 0 | repaired_permutation / richer_marginal | 4 / 4 | -99.3153229 | -21.0874424, 0.0376900453, 0.0187943862, 1.53661445, -6.57927811, 8.13447428 |
| predator_prey / 20 / 1008 / 260402 / 0 | repaired_permutation / richer_pairwise | 4 / 4 | -99.3120441 | -21.1321969, 0.0353361005, 0.0188162252, 1.53182751, -6.58705044, 8.14263439 |
| predator_prey / 20 / 1008 / 260402 / 0 | repaired_permutation / guarded_pairwise | 4 / 4 | -99.3129501 | -21.104486, 0.0365236038, 0.0188795459, 1.53291717, -6.60218596, 8.16207528 |

Full per-coordinate SE, reference values and kinds are in `values-and-scores.csv`. Paired changes are in `summary.json`; all realized values are in `rows.json`.

| Worker | Exit | Wall seconds |
|---|---:|---:|
| worker-0001-predator_prey | 2 | 153.79 |
| worker-0002-predator_prey | 0 | 166.47 |
| worker-0003-predator_prey | 0 | 174.99 |
