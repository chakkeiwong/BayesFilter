# SIR tuning results: existing controls retained

Evidence: [compact numerical record](../plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/terminal_evidence.json), [selection](../plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/selection.json), [score localization](../plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/score-localization-01/result.json).

The six-setting search found no replacement that passed every predeclared calibration screen at T=10,20,40,50. The guarded marginal baseline therefore remained selected at every horizon. The baseline itself fails the descriptive heuristic screen: on the two paired confirmation seeds, its score variation exceeds simpler comparators in some coordinates. This blocks promotion; two seeds do not establish a statistical ranking.

This campaign adds diagnostics and offline selection around the shared analytical LEDH evaluator. It does not change the filter equations or global controls. LGSSM, KSC SV and predator-prey were replayed at all four horizons with identical inputs; all 12 likelihoods and all score coordinates were exactly unchanged.

The scope is SIR d=18, N=1008, FP64 TensorFlow/XLA with TF32 disabled, guarded pairwise reset and marginal-mixture importance correction. Scores are derivatives with respect to the log multipliers of kappa, nu and observation scale. The observation seeds are 26100611/12/13 for calibration/validation/confirmation; the design partitions contain 4/4/8 seeds. These are three synthetic datasets from the current model, not an original-paper replication.

## Frozen confirmation results

Every entry below is mean (sample standard deviation) across eight complete random designs on one held-out dataset. Standard deviations measure design variability, not error against an oracle. MCSEs equal SD/sqrt(8); the full records are in terminal_evidence.json.

| T | log likelihood | score kappa | score nu | score scale |
|---:|---:|---:|---:|---:|
| 10 | -482.537 (33.092) | -499.824 (5896.601) | 144.928 (2582.780) | 694.433 (1979.734) |
| 20 | -825.825 (33.111) | -619.644 (5679.031) | 165.859 (2515.258) | 973.700 (2157.958) |
| 40 | -1505.941 (33.044) | -604.499 (5688.078) | 156.673 (2517.966) | 975.347 (2140.283) |
| 50 | -1845.160 (33.048) | -604.523 (5688.166) | 156.634 (2518.024) | 981.950 (2140.270) |

The nearly unchanged score dispersion from T=20 to T=50 is descriptive evidence that later observations do not remove the sensitivity already present early in these runs. The time-resolved diagnostics below test that explanation.

## Calibration trade-offs

Each vector is ordered as log likelihood, kappa score, nu score, scale score. Eligibility requires every variance ratio and every reset-RMSE ratio to be at most 1.10. Ratios compare paired designs with the exact guarded marginal baseline. Four calibration seeds cannot support a superiority claim.

| T | candidate | variance ratios | reset-RMSE ratios | screen |
|---:|---|---|---|---|
| 10 | baseline_marginal | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | eligible |
| 10 | flow4 | 0.807, 0.995, 0.590, 1.255 | 1.000, 0.500, 0.416, 0.843 | rejected |
| 10 | epsilon51 | 1.021, 0.252, 0.230, 0.239 | 1.000, 0.928, 0.966, 1.143 | rejected |
| 10 | weak_correction | 0.977, 2.227, 0.857, 0.089 | 1.079, 1.019, 0.622, 1.665 | rejected |
| 10 | epsilon204 | 0.976, 2.089, 2.311, 2.840 | 1.000, 1.004, 0.994, 0.978 | rejected |
| 10 | flow16_weak | 1.007, 3.403, 2.043, 4.892 | 1.074, 1.781, 1.140, 3.435 | rejected |
| 20 | baseline_marginal | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | eligible |
| 20 | flow4 | 0.815, 1.028, 0.608, 1.292 | 1.000, 0.499, 0.417, 0.845 | rejected |
| 20 | epsilon51 | 1.021, 0.246, 0.231, 0.236 | 1.000, 0.927, 0.966, 1.142 | rejected |
| 20 | weak_correction | 0.972, 2.261, 0.852, 0.088 | 1.080, 1.021, 0.623, 1.682 | rejected |
| 20 | epsilon204 | 0.976, 2.102, 2.310, 2.852 | 1.000, 1.005, 0.995, 0.980 | rejected |
| 20 | flow16_weak | 0.997, 3.459, 2.104, 5.216 | 1.076, 1.786, 1.145, 3.449 | rejected |
| 40 | baseline_marginal | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | eligible |
| 40 | flow4 | 0.814, 1.030, 0.609, 1.293 | 1.000, 0.499, 0.417, 0.845 | rejected |
| 40 | epsilon51 | 1.021, 0.246, 0.230, 0.236 | 1.000, 0.927, 0.966, 1.142 | rejected |
| 40 | weak_correction | 0.971, 2.261, 0.851, 0.088 | 1.080, 1.021, 0.623, 1.682 | rejected |
| 40 | epsilon204 | 0.976, 2.101, 2.310, 2.852 | 1.000, 1.005, 0.995, 0.980 | rejected |
| 40 | flow16_weak | 0.996, 3.437, 2.104, 5.241 | 1.076, 1.786, 1.145, 3.449 | rejected |
| 50 | baseline_marginal | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 | eligible |
| 50 | flow4 | 0.814, 1.030, 0.609, 1.293 | 1.000, 0.499, 0.417, 0.845 | rejected |
| 50 | epsilon51 | 1.021, 0.246, 0.230, 0.236 | 1.000, 0.927, 0.966, 1.142 | rejected |
| 50 | weak_correction | 0.971, 2.261, 0.851, 0.088 | 1.080, 1.021, 0.623, 1.682 | rejected |
| 50 | epsilon204 | 0.976, 2.101, 2.310, 2.852 | 1.000, 1.005, 0.995, 0.980 | rejected |
| 50 | flow16_weak | 0.996, 3.437, 2.104, 5.241 | 1.076, 1.786, 1.145, 3.449 | rejected |

Lowering epsilon to 51.2 reduced the observed calibration score variances, but increased at least one conditional reset-score RMSE beyond the allowed margin. Weakening the moment correction and changing the flow count also failed at least one screen. These outcomes reject these settings under this grid; they do not invalidate the general marginal-mixture method or exclude other settings.

## Analytical derivative check

The original central differences at 1e-4 and 5e-5 disagreed in the first two coordinates at every horizon. The predeclared smaller-step ladder resolves that discrepancy on the first confirmation design with two adjacent steps meeting the scaled-error threshold. No derivative code was changed. This checks the derivative of the finite value program at these points; it does not establish closeness to an exact filtering score.

| T | analytical score | smallest scaled error by coordinate | adjacent-step agreement |
|---:|---|---|---|
| 10 | -10739.751291, 4774.539565, -2097.123241 | 2.54e-08, 4.89e-08, 4.97e-09 | True |
| 20 | -10426.737176, 4627.845894, -1917.237945 | 2.83e-08, 5.08e-08, 7.03e-09 | True |
| 40 | -10425.255046, 4621.985411, -1898.168447 | 2.85e-08, 5.08e-08, 6.84e-09 | True |
| 50 | -10425.382560, 4622.058048, -1891.349610 | 2.84e-08, 5.09e-08, 6.94e-09 | True |

The complete six-step estimates, plus/minus values, absolute discrepancies, endpoint parity and base-repeat checks are retained in score-localization-01/result.json. A small minimum alone was not the acceptance rule.

## Conditional reset and weight diagnostics

The local reference integrates the next Gaussian transition and observation exactly, conditional on the realized incoming cloud. It measures log Z_after - log Z_before and its total parameter derivative. It is not a full SIR oracle. Per-time-step fields are retained for every run.

| T | average within-run reset RMSE: log likelihood; three scores | minimum relative ESS |
|---:|---|---:|
| 10 | 0.035040, 11.184814, 4.299153, 3.577222 | 0.0009923 |
| 20 | 0.024143, 7.772012, 2.990735, 2.475348 | 0.0009923 |
| 40 | 0.016857, 5.424729, 2.087481, 1.727764 | 0.0009923 |
| 50 | 0.015040, 4.839631, 1.862331, 1.541415 | 0.0009923 |

The following post-run grouping is explanatory only and did not affect selection. It pools the eight T=50 baseline designs. Reset time t compares clouds after observation t for prediction of observation t+1.

| reset times | pooled log/score reset RMS | minimum relative ESS |
|---|---|---:|
| 1--5 | 0.056293, 30.628351, 11.662395, 6.268757 | 0.0009923 |
| 6--19 | 0.001639, 4.518377, 1.814884, 2.916177 | 0.0010066 |
| 20--49 | 0.000465, 0.005902, 0.002222, 0.004288 | 0.4901110 |

Relative ESS near 1/1008 means approximately one effective particle. The early low-ESS regime coincides with large reset-score sensitivity; the later regime has much smaller reset-score errors. The decline of horizon-averaged reset RMSE partly reflects adding those quiet later steps, not repairing the early steps. This association does not establish that the reset caused all of the accumulated score variance.

## Heuristic falsification on paired designs

Each ratio below is baseline divided by heuristic, using the same first two confirmation seeds for both. A value above 1.10 in any component is a descriptive promotion veto. The four-component variance and reset vectors preserve coordinate-level trade-offs; no likelihood mean is used as a tuning objective.

| T | comparator | baseline/comparator variance ratios | baseline/comparator reset ratios |
|---:|---|---|---|
| 10 | ancestor_guarded | 0.022, 164.563, 134.834, 9.580 | 0.741, 1.167, 1.140, 10.123 |
| 10 | covariance_only | 3.421, 10.565, 12.925, 0.039 | 0.701, 1.013, 1.011, 2.646 |
| 10 | diagonal_only | 0.388, 0.030, 0.042, 0.005 | 0.742, 0.336, 0.345, 0.773 |
| 20 | ancestor_guarded | 0.017, 174.879, 162.740, 3.668 | 0.743, 1.166, 1.119, 10.123 |
| 20 | covariance_only | 1.609, 7.678, 9.317, 0.017 | 0.702, 1.011, 1.001, 2.623 |
| 20 | diagonal_only | 0.324, 0.031, 0.044, 0.002 | 0.743, 0.334, 0.343, 0.768 |
| 40 | ancestor_guarded | 0.016, 174.332, 162.184, 3.781 | 0.743, 1.166, 1.119, 10.122 |
| 40 | covariance_only | 1.390, 7.607, 9.256, 0.018 | 0.703, 1.011, 1.001, 2.623 |
| 40 | diagonal_only | 0.313, 0.031, 0.044, 0.002 | 0.743, 0.334, 0.343, 0.768 |
| 50 | ancestor_guarded | 0.015, 174.356, 162.158, 3.753 | 0.744, 1.166, 1.119, 10.122 |
| 50 | covariance_only | 1.282, 7.607, 9.256, 0.018 | 0.703, 1.011, 1.001, 2.623 |
| 50 | diagonal_only | 0.308, 0.031, 0.044, 0.002 | 0.743, 0.334, 0.343, 0.768 |

Paired mean-difference intervals and paired bootstrap variance-ratio intervals are retained in terminal_evidence.json. The two-seed t intervals use one degree of freedom; the corresponding bootstrap intervals are highly discrete and are not reliable ranking evidence. The selected setting is the baseline itself, so its validation/confirmation paired differences from baseline are identically zero; that is preservation, not an improvement.

## Every confirmation value and score

| T | setting | design seed | log likelihood | score kappa, nu, scale |
|---:|---|---:|---:|---|
| 10 | baseline_marginal | 261006411 | -484.670429 | -10739.751291, 4774.539565, -2097.123241 |
| 10 | baseline_marginal | 261006412 | -497.301697 | 7241.106419, -3395.100683, -1545.325551 |
| 10 | baseline_marginal | 261006413 | -412.810168 | 3877.427500, -1789.685539, 1357.628520 |
| 10 | baseline_marginal | 261006414 | -505.498539 | -5753.606706, 2277.978137, 1684.083004 |
| 10 | baseline_marginal | 261006415 | -473.653252 | 3686.858113, -1298.099116, 697.242994 |
| 10 | baseline_marginal | 261006416 | -502.369737 | 2150.967240, -980.240098, 144.060699 |
| 10 | baseline_marginal | 261006417 | -518.479563 | -3037.365712, 1152.276831, 1045.446164 |
| 10 | baseline_marginal | 261006418 | -465.512728 | -1424.230061, 417.751401, 4269.452319 |
| 10 | ancestor_guarded | 261006411 | -542.702079 | -2269.154467, 1150.833092, 418.726961 |
| 10 | ancestor_guarded | 261006412 | -627.886846 | -867.490976, 447.268626, 240.447365 |
| 10 | covariance_only | 261006411 | -491.184109 | -2923.053821, 1078.589666, 3108.394081 |
| 10 | covariance_only | 261006412 | -484.354534 | -8454.897229, 3350.987656, 301.905330 |
| 10 | diagonal_only | 261006411 | -477.398354 | 97240.410927, -36422.130978, -15938.685772 |
| 10 | diagonal_only | 261006412 | -497.677767 | -7296.889750, 3233.691208, -8160.942865 |
| 20 | baseline_marginal | 261006411 | -828.792059 | -10426.737176, 4627.845894, -1917.237945 |
| 20 | baseline_marginal | 261006412 | -839.967826 | 7519.446777, -3493.421594, -1586.253532 |
| 20 | baseline_marginal | 261006413 | -756.100529 | 2558.114811, -1575.754923, 3303.210651 |
| 20 | baseline_marginal | 261006414 | -849.728150 | -5570.006366, 2208.077073, 1632.238493 |
| 20 | baseline_marginal | 261006415 | -817.725487 | 3306.434146, -1115.408583, 816.545715 |
| 20 | baseline_marginal | 261006416 | -845.657600 | 2063.081079, -898.954612, 164.673907 |
| 20 | baseline_marginal | 261006417 | -861.048770 | -2945.957095, 1129.586976, 1065.255909 |
| 20 | baseline_marginal | 261006418 | -807.580226 | -1461.524744, 444.900463, 4311.167670 |
| 20 | ancestor_guarded | 261006411 | -885.293677 | -2278.834852, 1134.976155, 430.189960 |
| 20 | ancestor_guarded | 261006412 | -971.334058 | -921.763461, 498.361130, 257.360484 |
| 20 | covariance_only | 261006411 | -835.523665 | -2426.832601, 914.367741, 2952.915057 |
| 20 | covariance_only | 261006412 | -826.713480 | -8903.639687, 3575.010059, 401.268281 |
| 20 | diagonal_only | 261006411 | -821.500457 | 94913.674358, -35569.004188, -15448.548651 |
| 20 | diagonal_only | 261006412 | -841.132727 | -7128.084678, 3220.779221, -8112.476191 |
| 40 | baseline_marginal | 261006411 | -1509.078587 | -10425.255046, 4621.985411, -1898.168447 |
| 40 | baseline_marginal | 261006412 | -1519.890185 | 7498.053488, -3493.742855, -1560.865043 |
| 40 | baseline_marginal | 261006413 | -1436.373324 | 2662.384936, -1605.867922, 3208.603594 |
| 40 | baseline_marginal | 261006414 | -1529.942900 | -5575.299024, 2205.177590, 1646.964117 |
| 40 | baseline_marginal | 261006415 | -1497.878040 | 3351.563038, -1135.239953, 815.465129 |
| 40 | baseline_marginal | 261006416 | -1525.547764 | 2066.739799, -905.775921, 178.616083 |
| 40 | baseline_marginal | 261006417 | -1541.151109 | -2940.939228, 1122.645672, 1080.278062 |
| 40 | baseline_marginal | 261006418 | -1487.662792 | -1473.237490, 444.200327, 4331.885809 |
| 40 | ancestor_guarded | 261006411 | -1565.556740 | -2280.315351, 1130.484731, 445.241840 |
| 40 | ancestor_guarded | 261006412 | -1651.364005 | -922.848207, 493.214307, 271.780004 |
| 40 | covariance_only | 261006411 | -1515.766919 | -2429.522138, 911.454782, 2962.639653 |
| 40 | covariance_only | 261006412 | -1506.598108 | -8928.099417, 3579.066762, 420.207612 |
| 40 | diagonal_only | 261006411 | -1501.723770 | 94959.057593, -35591.006071, -15437.480873 |
| 40 | diagonal_only | 261006412 | -1521.049483 | -7030.821520, 3173.814757, -7999.394778 |
| 50 | baseline_marginal | 261006411 | -1848.430246 | -10425.382560, 4622.058048, -1891.349610 |
| 50 | baseline_marginal | 261006412 | -1859.042673 | 7498.005889, -3493.807517, -1554.416083 |
| 50 | baseline_marginal | 261006413 | -1775.567310 | 2662.786938, -1606.075551, 3215.057244 |
| 50 | baseline_marginal | 261006414 | -1869.101290 | -5575.585397, 2205.188519, 1653.522202 |
| 50 | baseline_marginal | 261006415 | -1837.028018 | 3351.502219, -1135.254576, 821.956796 |
| 50 | baseline_marginal | 261006416 | -1864.765242 | 2066.711770, -905.828872, 185.321871 |
| 50 | baseline_marginal | 261006417 | -1880.413124 | -2941.069839, 1122.656953, 1086.878776 |
| 50 | baseline_marginal | 261006418 | -1826.935279 | -1473.153246, 444.135332, 4338.629096 |
| 50 | ancestor_guarded | 261006411 | -1904.908627 | -2280.221986, 1130.450307, 452.190858 |
| 50 | ancestor_guarded | 261006412 | -1990.518533 | -922.841046, 493.117510, 278.271523 |
| 50 | covariance_only | 261006411 | -1855.122476 | -2429.551494, 911.463338, 2969.748569 |
| 50 | covariance_only | 261006412 | -1845.749907 | -8928.190020, 3579.007958, 426.725460 |
| 50 | diagonal_only | 261006411 | -1841.079420 | 94959.370668, -35591.110915, -15430.583237 |
| 50 | diagonal_only | 261006412 | -1860.200964 | -7030.528578, 3173.594633, -7992.620134 |

## Protected model replay

| model | T | unchanged log likelihood | unchanged score coordinates |
|---|---:|---:|---|
| lgssm | 10 | -25.360556061 | 0.530219798, 0.329834750, -0.366574804, -0.091681975 |
| lgssm | 20 | -48.956901542 | 1.357623969, -0.464555712, -1.350484439, -0.091681975 |
| lgssm | 40 | -103.570109589 | 0.731037613, 2.079305165, -0.032587116, -0.091681975 |
| lgssm | 50 | -132.945233765 | 1.619373636, 4.584960083, 1.450817268, -0.091681975 |
| ksc_sv | 10 | -23.198889801 | -0.382740964, 0.697429016 |
| ksc_sv | 20 | -54.712033369 | 0.764933815, 3.073530381 |
| ksc_sv | 40 | -99.505871413 | -0.389498735, 1.811841086 |
| ksc_sv | 50 | -123.644679830 | -0.621258247, 1.165590886 |
| predator_prey | 10 | -47.458050558 | 11.520485547, 1.157053365, 0.027211348, -2.268751090, -6.160391156, 7.429474304 |
| predator_prey | 20 | -93.951804506 | 11.700458064, 1.397145551, 0.039106038, -2.016105389, -9.635513320, 11.662810292 |
| predator_prey | 40 | -193.251880348 | 13.964524863, 4.090504485, 0.067443734, -11.891468183, -11.483174545, 13.931853573 |
| predator_prey | 50 | -240.168067337 | 13.940500752, 3.483098748, 0.071129155, -10.582395771, -13.241441493, 16.088719160 |

LGSSM and KSC SV use their existing ancestor policy in this replay because those model adapters do not expose the multicomponent Gaussian transition callback; predator-prey uses marginal weights. This check shows that the diagnostic extension preserved their existing computations. It does not establish oracle accuracy of the nonlinear cases.

## Decisions and inference status

| decision | primary criterion | vetoes | main uncertainty | next justified action | conclusion excluded |
|---|---|---|---|---|---|
| Retain existing controls | no alternative passes calibration | heuristic screen prevents SIR promotion | few designs and no full oracle | inspect the early low-ESS regime on fresh tuning data before a new search | improved SIR accuracy |
| Accept diagnostic implementation | 168 complete rows valid; protected replay exact | same-scalar derivative check: True | checks cover fixed parameter points | retain tests and complete evidence | universal score accuracy |
| Preserve caps | numerical guards remain active | finite values do not prove harmless reset distortion | early score sensitivity | evaluate any changed protection under a separate non-harm criterion | unbounded moment fitting is safe |

| inference status | conclusion |
|---|---|
| hard veto screen | finite rows: True; same-scalar derivative localization: True; heuristic promotion veto remains |
| statistically supported ranking | none; no superiority or non-inferiority established |
| descriptive-only differences | calibration variance/reset trade-offs, score dispersion, and two-design heuristic ratios |
| default-readiness | no new default or runtime tuning artifact admitted |
| next evidence needed | independent datasets and seeds in the early regime, frozen controls, untouched confirmation, and a full likelihood/score reference where feasible |

The strongest alternative explanation is random-design variability in a tiny calibration sample; the small grid may also miss useful controls. A fresh scope-specific search with adequate replication could overturn rejection of a setting. The weakest evidence is the two-seed heuristic variance estimate. Passing the local derivative check cannot overturn the absence of an accuracy reference.

## Execution and reproducibility

All main stages completed. Worker attempts used 11777.555 of 28800 authorized seconds, leaving 17022.445. Calibration T=50 attempt 01 was accidentally interrupted by the assistant; its partial data and charged time are preserved, and attempt 02 completed. No failed attempt was silently overwritten.

The combined focused suite passed 54 tests in the explicitly CPU-only diagnostic environment. The serious campaign and score localization used trusted GPU access, FP64/XLA, TF32 off, and verified memory growth. Per-attempt manifests preserve commands, seeds, observations, environment and source hashes. LaTeX build records and the compiled PDF are under documentation-02 (the earlier methods-only build is preserved under documentation-01). Tests, manifests and the compact terminal evidence accompany the result; full per-step rows and logs remain in the original attempt directories.

The final reporting revision corrects the Student-t critical value for two-seed intervals; this does not change filtering values, calibration decisions or experimental controls. The review was a recorded skeptical self-review; no independent peer-review claim is made.
