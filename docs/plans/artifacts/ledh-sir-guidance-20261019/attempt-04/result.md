# SIR guidance with fixed OT (1,5)

This comparison evaluates frozen cSMC quadratic and KDM-inspired kernel guidance in the existing optional three-density consumer. It does not run the original KDM neural learner or establish canonical LEDH admission.

Completed 96/96 rows; 1 invalid rows. The primary criterion is the paired change in squared L2 score error versus the equal-mixture unguided arm, separately for each dataset.

## Data 26100831

| Arm | Valid / failed seeds | Mean log κ score | Mean log ν score | Mean log noise score | Score RMSE |
|---|---:|---:|---:|---:|---:|
| Zhao–Cui reference | 2 fits | -157.110397 | 104.578310 | 56.860701 | — |
| plain | 8 / 0 | 239.910460 | -67.264443 | 1.563467 | 1334.566382 |
| csmc_quadratic | 8 / 0 | -668.333737 | 1130.234403 | 685.518421 | 8524.206227 |
| kernel | 8 / 0 | -20473.537138 | 6252.677552 | -451.762386 | 58237.428352 |
| prior | 8 / 0 | -214.433252 | 128.131605 | 66.261831 | 116.041841 |
| global | 8 / 0 | -215876.596386 | 77408.794221 | 397766.399336 | 1445125.015474 |
| local | 8 / 0 | -209.005619 | 121.832689 | 64.355627 | 127.691878 |

RMSE is the square root of the mean per-run squared L2 error; it includes bias and seed variability.

| Guided minus plain | Paired MSE difference | 98.75% paired bootstrap interval | Status |
|---|---:|---|---|
| csmc_quadratic | 70881024.36736752 | [2002904.851656271, 163655370.4768814] | conditional harm |
| kernel | 3389816993.669923 | [831450.4297130653, 13551941142.604206] | conditional harm |

| Arm | κ RMSE | ν RMSE | noise RMSE |
|---|---:|---:|---:|
| plain | 1155.740090 | 516.754698 | 422.252120 |
| csmc_quadratic | 7144.906401 | 4505.027519 | 1147.663437 |
| kernel | 55736.593796 | 16785.499126 | 1810.301687 |
| prior | 108.304811 | 39.993932 | 11.673139 |
| global | 779557.304574 | 283132.213204 | 1183432.663518 |
| local | 120.682651 | 39.966208 | 11.983980 |

Heuristic losses: 18 coordinate-specific descriptive losses across plain and guided arms. All are retained in analysis.json; any guided loss is a promotion veto, not a continuation veto.

## Data 26100832

| Arm | Valid / failed seeds | Mean log κ score | Mean log ν score | Mean log noise score | Score RMSE |
|---|---:|---:|---:|---:|---:|
| Zhao–Cui reference | 2 fits | -135.190814 | 56.696288 | 5.887474 | — |
| plain | 8 / 0 | -198.987794 | 75.462351 | 20.658543 | 123.297001 |
| csmc_quadratic | 8 / 0 | -334.053199 | 120.980115 | 32.894686 | 1730.432181 |
| kernel | 8 / 0 | -73.054680 | 48.967905 | -12.920077 | 606.027000 |
| prior | 8 / 0 | -114.921025 | 54.013102 | 8.565178 | 86.634371 |
| global | 7 / 1 | 3400.973617 | -39.086743 | 2175.712700 | not comparable: failed seed |
| local | 8 / 0 | -115.500900 | 57.216863 | 8.435370 | 80.916405 |

RMSE is the square root of the mean per-run squared L2 error; it includes bias and seed variability.

| Guided minus plain | Paired MSE difference | 98.75% paired bootstrap interval | Status |
|---|---:|---|---|
| csmc_quadratic | 2979193.383093987 | [74357.41003519551, 9816549.043482851] | conditional harm |
| kernel | 352066.5744676988 | [109115.89191869707, 577104.7669089604] | conditional harm |

| Arm | κ RMSE | ν RMSE | noise RMSE |
|---|---:|---:|---:|
| plain | 111.086718 | 44.997020 | 28.933714 |
| csmc_quadratic | 1618.438442 | 610.725954 | 45.457162 |
| kernel | 576.403544 | 177.655694 | 58.873882 |
| prior | 84.702369 | 17.538789 | 4.838781 |
| global | 42159.715190 | 16659.357704 | 20338.707430 |
| local | 79.229231 | 15.907053 | 4.142368 |

Heuristic losses: 18 coordinate-specific descriptive losses across plain and guided arms. All are retained in analysis.json; any guided loss is a promotion veto, not a continuation veto.

## Decision

| Decision field | Result |
|---|---|
| Primary criterion | Conditional gain on both datasets: {'csmc_quadratic': False, 'kernel': False} |
| Veto diagnostics | Invalid rows: 1; heuristic screen: {'plain': 'fail', 'csmc_quadratic': 'fail', 'kernel': 'fail'} |
| Execution completeness | All 96 keys accounted for: True; all numerically valid: False |
| Decision | No canonical/default/HMC promotion from this diagnostic |
| Main uncertainty | Up to eight evaluation seeds per dataset (counts above); one offline fit each; approximate reference uncertainty is not propagated |
| Next justified action | Inspect message approximation and consumer mechanism before a fresh, independently fitted canonical comparison |
| Not concluded | Exact model-score correctness, original KDM performance, cross-dataset superiority, or equal-compute advantage |

| Inference status | Evidence |
|---|---|
| Hard veto screen | 1 invalid rows; complete=False |
| Statistically supported ranking | Only the conditional contrasts explicitly labeled gain/harm above; bootstrap uncertainty is approximate with eight pairs |
| Descriptive-only differences | Mean scores, component RMSE, timings and heuristic comparisons |
| Default-readiness | Not established; scope tuning and canonical conformance are absent |
| Next evidence needed | Independent offline fits, fresh data, reference uncertainty and canonical implementation comparison |

## Interpretation and audit limits

Engineering correctness is supported by CPU regression tests and fitted-guide GPU/XLA finite differences. Numerical validity is checked independently from statistical comparisons. MathDevMCP could not encode the three derivation obligations. Its single-file implementation scan also reported a missing solve; inspection found the checked Cholesky solve in the actual imported call chain. The full reports and dispositions are in the parent audit.md; neither automatic report is proof certification. The manuscript gives the completing-square, projection and chain-rule derivations explicitly.

The strongest alternative explanation for any gain or loss is the approximation imposed on the fitted future message: many quadratic Hessians required positive-information projection, and the kernel activates at only a subset of times. A weak or misleading message can produce a numerically correct but statistically poor proposal. These are candidate failures, not rejection of controlled SMC or KDM as research directions.

A claim of general improvement would be overturned by independently fitted repetitions or a canonical consumer showing no advantage. The weakest evidence is one fitted guide per dataset and the approximate Zhao–Cui reference. The source audit distinguishes the local corrected log-mean-exp path calculation from the original author's displayed mean-log-weight statistic.

Artifacts: manifest.json, rows.jsonl, analysis.json, guides-*.json and smoke.json in this attempt directory. The parent directory contains CPU tests, GPU preflight and MathDevMCP audit results. GPU FP64/XLA is an explicit reference exception; TF32 is off and memory growth is verified. OT is (1,5) throughout.
