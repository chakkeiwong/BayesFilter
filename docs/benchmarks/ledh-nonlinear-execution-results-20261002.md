# Nonlinear LEDH execution — 2 October 2026

Predator–prey returned finite likelihoods and full scores in all 70 T=20/N=1008 runs; 69 passed the separate trace checks. SIR d=18 failed all ten FP32 runs. Four same-data FP64 SIR evaluations were finite, but their likelihoods were far below the independent particle reference. Double precision repairs this observed numerical failure; it does not repair the downstream approximation.

These are diagnostic results at the declared truth, with untuned inherited controls. They do not establish a ranking between repair variants, scope admission, or HMC readiness. Every realized score coordinate and its reference are in [actual-values-and-scores.csv](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/actual-values-and-scores.csv).

## Experiment and independent reference

The shared canonical analytical LEDH executor used T=20, N=1008, GPU/XLA/FP32/TF32, and memory growth. Dataset 260401 used IID designs 260501–260502 for all five arms and both models. Predator–prey dataset 260402 additionally used designs 260501–260504 under IID, previous inverse-CDF ancestry and repaired permutation ancestry. Precision diagnostics replayed the exact FP32 observations and random inputs in FP64. A fresh FP32 SIR run preserved already-computed stage checks; no validity threshold or numerical protection changed.

The independent bootstrap particle filter uses the same initial law, transition-before-observation timing, saved observations and parameter values. Its score is the posterior average of accumulated analytical complete-data scores (Fisher identity), rather than a derivative through discrete resampling. Initial-state derivatives vanish. Gaussian mean and covariance derivatives are both retained. This targets the exact observed-data score as particle count increases; the finite-particle estimate is approximate. LEDH instead differentiates its own finite numerical likelihood. Agreement between those different finite computations is an accuracy question, not an identity.

The old predator–prey bootstrap fixture observed x0 before transitioning, so it was unsuitable for this comparison. The new reference passed fixed-state derivative finite differences in both models and a separate exact linear-Gaussian Kalman fixture. Four independent replications were run at each of N=8192, 32768, 131072, 524288. Estimates below use log of the mean likelihood and likelihood-weighted Fisher scores. Delete-one-replication jackknife MCSE measures replication noise; it does not measure residual particle bias. The larger rungs were added to resolve the observed reference noise, not to select for agreement with LEDH.

| Model / data seed | Reference particles | Log likelihood | Jackknife MCSE | Score vector | Score MCSE |
|---|---:|---:|---:|---|---|
| predator_prey / 260401 | 8192 | -97.320561 | 0.039003 | -32.916, -0.555093, 0.020703, 4.38198, -8.70589, 10.8404 | 1.077, 0.06311, 0.00143, 0.07203, 0.2671, 0.3463 |
| predator_prey / 260401 | 32768 | -97.373981 | 0.025789 | -33.705, -0.544774, 0.0199385, 4.55883, -8.68971, 10.7987 | 0.5896, 0.0168, 0.0006263, 0.06893, 0.1503, 0.1803 |
| predator_prey / 260402 | 8192 | -99.270178 | 0.050164 | -19.9033, 0.0488468, 0.0188408, 1.53256, -6.55371, 8.10838 | 1.282, 0.05447, 0.001015, 0.2305, 0.07725, 0.1033 |
| predator_prey / 260402 | 32768 | -99.321704 | 0.026594 | -21.5471, 0.0345818, 0.0181612, 1.55184, -6.46557, 7.98394 | 0.3264, 0.007855, 0.0001875, 0.01971, 0.06378, 0.07668 |
| sir_d18 / 260401 | 8192 | -678.119000 | 0.215663 | 6.52719, -31.2449, 5.75406 | 66.31, 26.52, 0.3564 |
| sir_d18 / 260401 | 32768 | -677.999379 | 0.021480 | 114.965, -71.0636, 5.15601 | 25.47, 7.205, 0.2765 |
| predator_prey / 260401 | 131072 | -97.348757 | 0.008520 | -32.8565, -0.554664, 0.0203593, 4.46922, -8.66844, 10.7852 | 0.1722, 0.009844, 0.0001347, 0.02384, 0.04891, 0.06101 |
| predator_prey / 260401 | 524288 | -97.342974 | 0.006575 | -32.9052, -0.543943, 0.0205295, 4.48734, -8.7324, 10.8655 | 0.0755, 0.00339, 9.419e-05, 0.008272, 0.022, 0.0277 |
| predator_prey / 260402 | 131072 | -99.321912 | 0.011908 | -21.2665, 0.0286087, 0.0188092, 1.53141, -6.57019, 8.12535 | 0.2779, 0.006872, 0.0002137, 0.04035, 0.03807, 0.04645 |
| predator_prey / 260402 | 524288 | -99.311525 | 0.003656 | -21.1402, 0.039307, 0.018684, 1.50221, -6.5311, 8.0744 | 0.09711, 0.001551, 6.656e-06, 0.005059, 0.002713, 0.002705 |
| sir_d18 / 260401 | 131072 | -678.046435 | 0.008898 | 92.9613, -62.8857, 5.60647 | 5.642, 3.132, 0.1033 |
| sir_d18 / 260401 | 524288 | -678.077466 | 0.014864 | 106.313, -65.9603, 5.58635 | 5.536, 2.42, 0.03538 |

Predator–prey score order is (r, K, a, s, u, v). SIR score order is (log infection-rate scale, log removal-rate scale, log observation-noise scale). At N524288 the minimum particle ESS was 11240 and 34022 on the two PP datasets, and 33055 on SIR; final distinct initial ancestors remained at least 12942, 14116 and 7701 respectively. These diagnostics do not certify reference convergence.

## Actual LEDH likelihoods and full scores

The following means include every finite returned evaluation, including the two explicitly flagged trace-parity failures. Counts separate finiteness from validity. The reference above is approximate, not an oracle. Replication standard errors and individual values are retained in the JSON/CSV; small-seed means are descriptive.

| Model / data / precision | Ancestry / arm | Valid / finite / returned | Mean log likelihood | Mean score vector |
|---|---|---:|---:|---|
| predator_prey / 260401 / float32 | iid_dual_cap / covariance_only | 2/2/2 | -97.438702 | -33.3663, -0.52189, 0.0223339, 4.40753, -9.18848, 11.4247 |
| predator_prey / 260401 / float32 | iid_dual_cap / original | 2/2/2 | -97.438812 | -33.563, -0.52737, 0.0222873, 4.41377, -9.20779, 11.4398 |
| predator_prey / 260401 / float32 | iid_dual_cap / richer_marginal | 2/2/2 | -97.368999 | -33.4926, -0.545006, 0.0199154, 4.5611, -8.61446, 10.7227 |
| predator_prey / 260401 / float32 | iid_dual_cap / richer_pairwise | 2/2/2 | -97.348324 | -33.55, -0.549612, 0.0197139, 4.5708, -8.56858, 10.6655 |
| predator_prey / 260401 / float32 | iid_dual_cap / guarded_pairwise | 2/2/2 | -97.350349 | -33.5277, -0.546957, 0.0196376, 4.56826, -8.54557, 10.6376 |
| sir_d18 / 260401 / float32 | iid_dual_cap / covariance_only | 0/0/2 | invalid | invalid |
| sir_d18 / 260401 / float32 | iid_dual_cap / original | 0/0/2 | invalid | invalid |
| sir_d18 / 260401 / float32 | iid_dual_cap / richer_marginal | 0/0/2 | invalid | invalid |
| sir_d18 / 260401 / float32 | iid_dual_cap / richer_pairwise | 0/0/2 | invalid | invalid |
| sir_d18 / 260401 / float32 | iid_dual_cap / guarded_pairwise | 0/0/2 | invalid | invalid |
| sir_d18 / 260401 / float64 | iid_dual_cap / covariance_only | 2/2/2 | -975.573545 | 884.173, -456.478, 571.353 |
| sir_d18 / 260401 / float64 | iid_dual_cap / guarded_pairwise | 1/2/2 | -1041.353940 | 1532.81, -884.054, 949.904 |
| predator_prey / 260402 / float32 | iid_dual_cap / covariance_only | 4/4/4 | -99.297594 | -21.1052, 0.0374908, 0.0192727, 1.53951, -6.71427, 8.29985 |
| predator_prey / 260402 / float32 | iid_dual_cap / original | 3/4/4 | -99.303041 | -21.1198, 0.0411079, 0.0199297, 1.4974, -6.85978, 8.4788 |
| predator_prey / 260402 / float32 | iid_dual_cap / richer_marginal | 4/4/4 | -99.287849 | -21.2183, 0.0345742, 0.0188289, 1.5292, -6.58847, 8.14454 |
| predator_prey / 260402 / float32 | iid_dual_cap / richer_pairwise | 4/4/4 | -99.287880 | -21.2214, 0.0345755, 0.0189681, 1.50847, -6.6117, 8.17372 |
| predator_prey / 260402 / float32 | iid_dual_cap / guarded_pairwise | 4/4/4 | -99.287128 | -21.2151, 0.0351281, 0.0189034, 1.52016, -6.60116, 8.16064 |
| predator_prey / 260402 / float32 | previous_inverse_cdf / covariance_only | 4/4/4 | -99.326353 | -20.9205, 0.0414378, 0.0196, 1.48792, -6.76877, 8.36709 |
| predator_prey / 260402 / float32 | previous_inverse_cdf / original | 4/4/4 | -99.319910 | -20.954, 0.0419956, 0.0198132, 1.48275, -6.82084, 8.43249 |
| predator_prey / 260402 / float32 | previous_inverse_cdf / richer_marginal | 4/4/4 | -99.314289 | -21.1016, 0.037298, 0.0188303, 1.5354, -6.58898, 8.1462 |
| predator_prey / 260402 / float32 | previous_inverse_cdf / richer_pairwise | 4/4/4 | -99.311543 | -21.1044, 0.0369014, 0.0189568, 1.53213, -6.62277, 8.18732 |
| predator_prey / 260402 / float32 | previous_inverse_cdf / guarded_pairwise | 4/4/4 | -99.312626 | -21.1131, 0.0362627, 0.0189238, 1.53028, -6.61176, 8.17402 |
| predator_prey / 260402 / float32 | repaired_permutation / covariance_only | 4/4/4 | -99.324743 | -20.9273, 0.0412008, 0.0194804, 1.49312, -6.74011, 8.33178 |
| predator_prey / 260402 / float32 | repaired_permutation / original | 4/4/4 | -99.320013 | -20.9614, 0.0411266, 0.0198753, 1.47876, -6.83382, 8.4473 |
| predator_prey / 260402 / float32 | repaired_permutation / richer_marginal | 4/4/4 | -99.315323 | -21.0874, 0.03769, 0.0187944, 1.53661, -6.57928, 8.13447 |
| predator_prey / 260402 / float32 | repaired_permutation / richer_pairwise | 4/4/4 | -99.312044 | -21.1322, 0.0353361, 0.0188162, 1.53183, -6.58705, 8.14263 |
| predator_prey / 260402 / float32 | repaired_permutation / guarded_pairwise | 4/4/4 | -99.312950 | -21.1045, 0.0365236, 0.0188795, 1.53292, -6.60219, 8.16208 |

SIR FP64 rows must also be inspected individually because their score variation is large:

| Arm / design seed | Log likelihood | Full score | Validity |
|---|---:|---|---|
| covariance_only / 260501 | -969.027666 | -671.392180, 393.814046, 706.036725 | passed |
| guarded_pairwise / 260501 | -1041.829818 | -165.242935, -186.132429, 963.395359 | trace-score parity veto |
| covariance_only / 260502 | -982.119424 | 2439.738135, -1306.769838, 436.668491 | passed |
| guarded_pairwise / 260502 | -1040.878062 | 3230.854953, -1581.974837, 936.413250 | passed |

## Conditional comparisons with simple arms

Constructed comparators were covariance-only (avoid higher-moment fitting), original capped correction (existing bounded design), and richer marginal-only correction (omit mixed-moment fitting). Comparisons remain conditional on each dataset and ancestry rule. All 25 available matched comparisons are retained in summary.json. The compact table below shows guarded pairwise versus covariance-only. Positive differences mean greater observed error relative to the approximate reference. Score L2 is unscaled across physical parameter units and explanatory only; the coordinate values above are the substantive evidence.

| Model / data / ancestry | Matched / both-valid | Change in absolute log-likelihood error | Change in unscaled score L2 error |
|---|---:|---:|---:|
| predator_prey / 260401 / iid_dual_cap | 2/2 | -0.046886 | -0.191488 |
| sir_d18 / 260401 / iid_dual_cap | 2/1 | 65.780395 | 388.675868 |
| predator_prey / 260402 / iid_dual_cap | 4/4 | -0.001317 | -0.033600 |
| predator_prey / 260402 / previous_inverse_cdf | 4/4 | -0.001798 | -0.287732 |
| predator_prey / 260402 / repaired_permutation | 4/4 | -0.002493 | -0.232324 |

No heuristic dominance or statistical ranking is established. The guarded method is not uniformly descriptively favorable across these situations. Untuned controls, small replication counts, an approximate reference, and missing untouched validation prohibit promotion. In particular, the SIR likelihood gap is hundreds of log units while the reference varies by hundredths at the largest rung. That observed discrepancy cannot be explained by the measured replication MCSE. A systematic reference implementation error is a separate possibility, bounded here by the derivative and linear-Gaussian checks but not logically excluded.

## Numerical localization and interpretation

At zero-based time index 3 (observation 4), both localized FP32 SIR arms fail the Contract E reset numerical check. Ancestry, UKF prediction, UKF update and callback checks pass; predicted/posterior covariances, flow states/tangents and posterior logits are finite. Covariance-only reset states are finite but reset tangents are not. The guarded arm has nonfinite reset states and tangents, and its moment-safety input check fails. The first observed failure is therefore in the reset, before a usable protected moment-correction input exists. This does not identify the exact offending factorization or establish a repair.

Two additional finite runs fail trace-score consistency without nonfinite outputs: original PP/IID/data260402/design260501 differs by 0.000646591 in score coordinate r; guarded SIR/FP64/design260501 differs by 1.70984e-7 in coordinate log-kappa. Their tolerances were not loosened and both vetoes remain. Directional value calls in the finite evaluations agreed within the recorded checks.

The SIR failures do not invalidate the research direction or the reference harness. They identify a reset precision defect and a separate approximation/calibration gap. Passing in FP64 does not establish that the FP32 default is repaired. Moment residuals and cap activity remain explanatory diagnostics, not substitutes for likelihood/score agreement.

## Decision and inference status

| Decision | Primary criterion | Vetoes | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain PP variants as diagnostic candidates | Finite likelihood/full scores; approximate reference available | One original-arm trace veto; no per-scope tuning | Few seeds, reference bias, unseen parameters/data | Scope-specific calibration/validation and untouched comparison | Repair superiority or default readiness |
| Reject current SIR FP32 evaluations | Nonfinite reset output/derivative | All ten screening runs invalid | Exact reset suboperation still to localize | Replay observation 4 and inspect reset factorization/derivative margins | Rejection of LEDH or higher-moment repair as a direction |
| Keep SIR FP64 as localization evidence only | Finite outputs but large reference discrepancy | One trace veto; no calibration; large errors | Approximation layers and warm-start controls | Diagnose reset, then calibrate at this exact scope with fresh partitions | FP64 as a new default or accuracy repair |
| Keep bootstrap/Fisher result as approximate reference | Exact target/data timing and derivative checks passed | No hard failure; finite-particle bias unresolved | Four replications per rung | Additional independent nonlinear cross-check before certification | Exact nonlinear oracle |

| Inference status | Finding |
|---|---|
| Hard veto screen | SIR FP32 reset failure; two finite trace-score parity failures |
| Statistically supported ranking | None |
| Descriptive-only differences | All means, score errors, paired arm differences and runtime comparisons |
| Default readiness | Not established; canonical implementation/default direction unchanged |
| Next evidence needed | Reset repair with unchanged healthy trajectories; per-scope calibration; replicated held-out accuracy and reference-bias assessment |

The next SIR repair should first reproduce observation 4 with the same source cloud and derivative, measure reset covariance/factorization conditioning and compare FP32/FP64 intermediate values. Evaluate any numerics-altering protection against non-harm and finite, flagged behavior before accuracy tuning. Then calibrate correction/OT controls separately on new data at the exact target scope; do not tune on these diagnostic datasets and call them untouched validation.

Engineering checks: 25 focused CPU reference tests passed, including exact data replay/hash rejection, both nonlinear complete-data score finite differences, Kalman comparison, shared canonical endpoint, and trace/no-trace parity. GPU devices were intentionally hidden for the CPU tests. GPU runs used the recorded memory-growth policy and fresh output directories. The new reference uses analytical derivatives and no pfor, resampling autodiff, or NumPy numerical path.

Post-run red team: the strongest alternative explanation is shared model-adapter error, since the reference and LEDH share the model definition. The exact Kalman fixture and fixed-state finite differences check the estimator and derivatives, not every intended nonlinear modeling assumption. An independent nonlinear reference disagreeing with this ladder would overturn the accuracy interpretation. The weakest evidence is repair ranking across only two/four design seeds at one parameter point; no ranking is claimed.

Plan: [ledh-nonlinear-execution-20261002.md](../plans/ledh-nonlinear-execution-20261002.md). Full artifacts: [comparison summary](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/summary.json), [all reference-attached rows](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/rows-with-reference.json).

Document verification: both chapter copies are identical. The monograph builds to 606 pages without unresolved references or citations; the new results and equations on PDF pages 218–221 were visually inspected after line-layout repairs. Build commands, hashes and inspection details are in `monograph-build/build-manifest.json` and `monograph-build/verification.json` under the artifact root.
