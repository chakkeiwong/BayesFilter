# Expanded marginal-weight verification — running results

Purpose: test the likelihood/analytical-score component needed for surrogate HMC. This campaign does not yet test posterior sampling.

Updated: 2026-10-08T08:25:17.098064+00:00. Campaign: INCOMPLETE.
Measured worker time so far: 48.00 hours of the 48.0-hour ceiling.
Completed scope jobs: 27/90. A rejected control pair counts as a completed attempt, never as a successful accuracy comparison.
Retained prior-source comparisons: 10; current-source completed comparisons: 17. Retained results describe their original implementation and are not new-source replications.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Engineering | Passed | Batch parity, finite-program score, T=1 reference checks | Larger scopes | Continue the same master | Model accuracy or HMC convergence |
| Precision | Passed | Nonfinite evaluations | Different datasets/scopes | Keep FP32 and inspect larger scopes | Agreement with true model likelihood |
| Fresh-data replication | 27/90 scope jobs complete | See per-scope records | Dataset sampling and reference error | Complete planned 4→8→16 looks within budget | Universal improvement |
| Nonlinear dimension extension | Pending | Model equations undefined | KSC is scalar; original range-bearing is 4D | Resolve requested new model definitions | Requested nonlinear dimension grid covered |

| Inference status | Current interpretation |
|---|---|
| Hard veto screen | Engineering and precision verdicts above; accuracy scopes below |
| Statistically supported ranking | Only the predeclared per-cell dataset intervals can support a ranking |
| Descriptive-only differences | Timing, precision deltas and heuristic point estimates; FP64 timing from attempt 01 is superseded |
| Default-readiness | Not established; ancestor remains the default |
| Next evidence needed | Fresh datasets, reference refinement, particle-count and matched-FP32 cost comparisons, then actual surrogate-HMC tests |

Independent accuracy authorities use FP64. In the active continuation simple comparators and LEDH use FP32/TF32/GPU/XLA. The equal-runtime bootstrap comparison measures likelihood accuracy; the bootstrap produces no score, while LEDH timing includes its analytical score. It is not an equivalent HMC-force workload. Source04 synchronizes value, score and validity and excludes report serialization from timed inference. Retained source03 timings include unequal report-serialization overhead. Actual cost mismatches remain explicit; no overall computational-value ranking is established.

Invalid comparison attribution (original evidence is preserved):

- m13-t20 N=1024 point 1: INVALID_BASELINE; counts {'ancestor': 5, 'marginal_mixture': 0, 'reference': 0, 'incomplete': 0}.
- m13-t10 N=4096 point 1: INVALID_BASELINE; counts {'ancestor': 2, 'marginal_mixture': 0, 'reference': 0, 'incomplete': 0}.

The selective covariance repair passed all 68 same-process public-consumer checks: healthy values/scores were unchanged and 12 formerly invalid high-dimensional rows recovered. An unchanged-source comparison against a saved earlier process nevertheless exceeded the no-harm margin on three M13 rows. Cross-process FP32/TF32 reproducibility remains unresolved; neither repeat checks within one graph nor this component campaign establish HMC readiness.

The GPU engineering fixtures used distinct parameter rows for all three models and both policies. Maximum batch/separate-call discrepancy: 0; maximum relative score/finite-difference discrepancy: 1.77863e-05. The T=1 range-bearing quadrature/PF comparison passed its recorded uncertainty check.

| N=4096 precision arm | Parameter point | Absolute log-value delta | Scaled score delta | Repeat spread |
|---|---:|---:|---:|---:|
| ancestor | 0 | 0.00478839 | 0.0010828 | 0 |
| ancestor | 1 | 0.00952085 | 0.00115918 | 0 |
| marginal_mixture | 0 | 0.00534372 | 0.000171734 | 0 |
| marginal_mixture | 1 | 0.00537527 | 0.00217071 | 0 |

Pilot result: {'status': 'COMPLETE', 'scope': 'lg20-t240', 'particles': 1024, 'warm_call_seconds': {'ancestor': [20.929362221155316, 12.213052537059411, 9.836073495913297], 'marginal_mixture': [15.983433053828776, 12.321284190984443, 20.22855982510373]}, 'four_dataset_minimum_candidate_seconds': 10827.450466901064, 'qualification': 'excludes compilation, reference refinement, other baselines and extensions to 8/16 datasets'}

The four-dataset pilot time estimate exceeds one worker's time cap. Long scopes may therefore remain partial at the cap; saved rows are retained. Numerical pilot validity is not a completed accuracy comparison.

| Scope | Complete fresh datasets | Completed paired parameter/design rows |
|---|---:|---:|
| ksc-t10-n4096 | 16 | 1536 |
| ksc-t120-n1024 | 16 | 1536 |
| ksc-t240-n1024 | 16 | 1536 |
| lg1-t10-n4096 | 16 | 1536 |
| lg1-t120-n1024 | 8 | 768 |
| lg1-t240-n1024 | 16 | 1536 |
| lg10-t10-n1024 | 16 | 1536 |
| lg10-t10-n4096 | 16 | 1536 |
| lg10-t120-n1024 | 6 | 610 |
| lg10-t20-n1024 | 16 | 1536 |
| lg10-t240-n1024 | 3 | 308 |
| lg10-t50-n1024 | 16 | 1536 |
| lg20-t10-n1024 | 16 | 1536 |
| lg20-t10-n4096 | 2 | 225 |
| lg20-t120-n1024 | 1 | 175 |
| lg20-t20-n1024 | 16 | 1536 |
| lg20-t240-n1024 | 2 | 192 |
| lg20-t50-n1024 | 8 | 768 |
| lg3-t10-n4096 | 8 | 768 |
| lg3-t120-n1024 | 14 | 1397 |
| lg3-t240-n1024 | 2 | 285 |
| m13-t10-n4096 | 16 | 1024 |
| m13-t120-n1024 | 4 | 256 |
| m13-t20-n1024 | 16 | 1024 |
| m13-t240-n1024 | 2 | 128 |
| m13-t50-n1024 | 16 | 1024 |

Latest predeclared interim looks are shown below. These are unfinished comparisons; the master extends inconclusive cells to the next planned dataset count.

| Current scope | Datasets in look | Point | Mean absolute value error, ancestor → marginal | Mean scaled score error, ancestor → marginal | Value interval | Score interval | Decision |
|---|---:|---:|---|---|---|---|---|
| lg10-t120-n1024 | 4 | 0 | 0.61043 → 0.41778 | 0.21046 → 0.094774 | [-1.362, 0.9762] | [-0.3631, 0.1317] | INCONCLUSIVE/INCONCLUSIVE |
| lg10-t120-n1024 | 4 | 1 | 1.2756 → 0.79878 | 0.45483 → 0.2167 | [-2.063, 1.11] | [-0.7964, 0.3202] | INCONCLUSIVE/INCONCLUSIVE |
| lg10-t120-n1024 | 4 | 2 | 0.60179 → 0.38437 | 0.43416 → 0.15124 | [-1.659, 1.224] | [-2.473, 1.907] | INCONCLUSIVE/INCONCLUSIVE |
| lg3-t120-n1024 | 8 | 0 | 0.18103 → 0.15874 | 0.04571 → 0.029136 | [-0.08903, 0.04446] | [-0.03321, 6.297e-05] | NONINFERIOR/NONINFERIOR |
| lg3-t120-n1024 | 8 | 1 | 0.29025 → 0.24359 | 0.068276 → 0.048876 | [-0.1612, 0.06788] | [-0.05349, 0.01469] | INCONCLUSIVE/NONINFERIOR |
| lg3-t120-n1024 | 8 | 2 | 0.15134 → 0.13032 | 0.062298 → 0.029841 | [-0.07474, 0.03271] | [-0.05516, -0.009759] | NONINFERIOR/NONINFERIOR |
| m13-t120-n1024 | 4 | 0 | 21.545 → 1.5679 | 5.3727 → 0.42049 | [-185.4, 145.5] | [-42.11, 32.21] | INCONCLUSIVE/INCONCLUSIVE |
| m13-t120-n1024 | 4 | 1 | 135.38 → 15.219 | 26.921 → 5.2656 | [-577.2, 336.9] | [-80.56, 37.25] | INCONCLUSIVE/INCONCLUSIVE |

Intervals concern marginal minus ancestor error, with the predeclared multiplicity and reference allowances. Lower observed means alone do not establish improvement.

Grid coverage; numbers in parentheses are completed datasets. A retained entry identifies earlier-source evidence, and a complete entry identifies a finished current-source comparison, regardless of whether its accuracy criterion passed.

| Model | N | T=10 | T=20 | T=50 | T=120 | T=240 |
|---|---:|---|---|---|---|---|
| lg1 | 1024 | retained (16) | retained (16) | retained (16) | complete (8) | complete (16) |
| lg3 | 1024 | retained (16) | retained (8) | retained (8) | partial (14) | partial (2) |
| ksc | 1024 | retained (16) | retained (16) | retained (16) | complete (16) | complete (16) |
| m13 | 1024 | retained (16) | complete (16) | complete (16) | partial (4) | partial (2) |
| lg10 | 1024 | complete (16) | complete (16) | complete (16) | partial (6) | partial (3) |
| lg20 | 1024 | complete (16) | complete (16) | complete (8) | partial (1) | partial (2) |
| lg1 | 4096 | complete (16) | failed attempt | pending | pending | pending |
| lg3 | 4096 | complete (8) | pending | pending | pending | pending |
| ksc | 4096 | complete (16) | pending | pending | pending | pending |
| m13 | 4096 | complete (16) | pending | pending | pending | pending |
| lg10 | 4096 | complete (16) | pending | pending | pending | pending |
| lg20 | 4096 | partial (2) | pending | pending | pending | pending |
| lg1 | 16384 | pending | pending | pending | pending | pending |
| lg3 | 16384 | pending | pending | pending | pending | pending |
| ksc | 16384 | pending | pending | pending | pending | pending |
| m13 | 16384 | pending | pending | pending | pending | pending |
| lg10 | 16384 | pending | pending | pending | pending | pending |
| lg20 | 16384 | pending | pending | pending | pending | pending |

| Scope | Source | N | Datasets | Parameter point | Value verdict | Score verdict | Heuristic screen |
|---|---|---:|---:|---:|---|---|---|
| lg1-t10 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | retained 03 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | retained 03 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | retained 03 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t10 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | retained 03 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t20 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t20 | retained 03 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t20 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | retained 03 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | retained 03 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t20 | retained 03 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | retained 03 | 1024 | 16 | 1 | INCONCLUSIVE | NONINFERIOR | PROMOTION_VETO |
| ksc-t20 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t50 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t50 | retained 03 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t50 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | retained 03 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | retained 03 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t50 | retained 03 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t50 | retained 03 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t50 | retained 03 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t50 | retained 03 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t10 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t10 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t10 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t20 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t20 | current 04 | 1024 | 16 | 1 | INVALID_BASELINE | INVALID_BASELINE | Not assessed |
| lg10-t20 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t20 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t20 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t20 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t20 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t20 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t120 | current 04 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t120 | current 04 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t120 | current 04 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t50 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t50 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t50 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t50 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t50 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t50 | current 04 | 1024 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t50 | current 04 | 1024 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg20-t50 | current 04 | 1024 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t120 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t120 | current 04 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t120 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t240 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t240 | current 04 | 1024 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t240 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t240 | current 04 | 1024 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t240 | current 04 | 1024 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t240 | current 04 | 1024 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | current 04 | 4096 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | current 04 | 4096 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg1-t10 | current 04 | 4096 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | current 04 | 4096 | 8 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | current 04 | 4096 | 8 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg3-t10 | current 04 | 4096 | 8 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | current 04 | 4096 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| ksc-t10 | current 04 | 4096 | 16 | 1 | INCONCLUSIVE | INCONCLUSIVE | PROMOTION_VETO |
| ksc-t10 | current 04 | 4096 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | current 04 | 4096 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| m13-t10 | current 04 | 4096 | 16 | 1 | INVALID_BASELINE | INVALID_BASELINE | Not assessed |
| lg10-t10 | current 04 | 4096 | 16 | 0 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | current 04 | 4096 | 16 | 1 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |
| lg10-t10 | current 04 | 4096 | 16 | 2 | NONINFERIOR | NONINFERIOR | PROMOTION_VETO |

Paired accuracy results at the terminal predeclared look for each completed scope. Entries show ancestor → marginal mean absolute error. Score error is 0.1 times physical-score RMS error. Intervals concern marginal minus ancestor and include the declared multiplicity and reference allowances.

| Scope / N | Source | Point | Value error | Value difference interval | Scaled score error | Score difference interval | Supported lower error |
|---|---|---:|---|---|---|---|---|
| lg1-t10 / 1024 | retained 03 | 0 | 0.019784 → 0.019833 | [-0.002221, 0.00232] | 0.0058194 → 0.0051698 | [-0.001564, 0.0002648] | none |
| lg1-t10 / 1024 | retained 03 | 1 | 0.05524 → 0.041009 | [-0.03742, 0.008962] | 0.034847 → 0.014599 | [-0.04024, -0.0002547] | score |
| lg1-t10 / 1024 | retained 03 | 2 | 0.10734 → 0.049638 | [-0.1161, 0.0006566] | 0.11856 → 0.022979 | [-0.1658, -0.02541] | score |
| lg3-t10 / 1024 | retained 03 | 0 | 0.073192 → 0.06035 | [-0.04236, 0.01668] | 0.021212 → 0.011532 | [-0.02253, 0.003174] | none |
| lg3-t10 / 1024 | retained 03 | 1 | 0.090885 → 0.079163 | [-0.03489, 0.01145] | 0.020925 → 0.014191 | [-0.01662, 0.003158] | none |
| lg3-t10 / 1024 | retained 03 | 2 | 0.085722 → 0.048463 | [-0.06092, -0.0136] | 0.04674 → 0.016358 | [-0.04727, -0.01349] | value, score |
| ksc-t10 / 1024 | retained 03 | 0 | 0.054939 → 0.048373 | [-0.02545, 0.01232] | 0.015302 → 0.012136 | [-0.01276, 0.006428] | none |
| ksc-t10 / 1024 | retained 03 | 1 | 0.19267 → 0.11579 | [-0.3001, 0.1463] | 0.10773 → 0.050348 | [-0.221, 0.1062] | none |
| ksc-t10 / 1024 | retained 03 | 2 | 0.042574 → 0.040314 | [-0.008404, 0.003885] | 0.0066663 → 0.0069476 | [-0.002815, 0.003378] | none |
| m13-t10 / 1024 | retained 03 | 0 | 2.6931 → 0.47569 | [-4.344, -0.09083] | 0.61699 → 0.11899 | [-0.9348, -0.06121] | value, score |
| m13-t10 / 1024 | retained 03 | 1 | 23.614 → 3.0285 | [-35.14, -6.035] | 5.956 → 0.96351 | [-8.04, -1.945] | value, score |
| lg1-t20 / 1024 | retained 03 | 0 | 0.03083 → 0.03251 | [-0.001012, 0.004371] | 0.0077436 → 0.0069834 | [-0.001822, 0.0003021] | none |
| lg1-t20 / 1024 | retained 03 | 1 | 0.06436 → 0.04926 | [-0.03359, 0.003393] | 0.038 → 0.015169 | [-0.04486, -0.0007961] | score |
| lg1-t20 / 1024 | retained 03 | 2 | 0.11813 → 0.056119 | [-0.1093, -0.0147] | 0.12839 → 0.027063 | [-0.1586, -0.04408] | value, score |
| lg3-t20 / 1024 | retained 03 | 0 | 0.078078 → 0.070561 | [-0.03887, 0.02384] | 0.018421 → 0.011673 | [-0.01503, 0.001533] | none |
| lg3-t20 / 1024 | retained 03 | 1 | 0.095309 → 0.088523 | [-0.05292, 0.03935] | 0.017911 → 0.012575 | [-0.01596, 0.005291] | none |
| lg3-t20 / 1024 | retained 03 | 2 | 0.09831 → 0.05598 | [-0.106, 0.02132] | 0.049652 → 0.018192 | [-0.06257, -0.0003523] | score |
| ksc-t20 / 1024 | retained 03 | 0 | 0.062912 → 0.058483 | [-0.02092, 0.01206] | 0.017376 → 0.015154 | [-0.008497, 0.004051] | none |
| ksc-t20 / 1024 | retained 03 | 1 | 0.14745 → 0.10902 | [-0.1716, 0.09471] | 0.057735 → 0.038199 | [-0.0778, 0.03873] | none |
| ksc-t20 / 1024 | retained 03 | 2 | 0.048482 → 0.047617 | [-0.003661, 0.001931] | 0.0083254 → 0.0082733 | [-0.0005701, 0.0004659] | none |
| lg1-t50 / 1024 | retained 03 | 0 | 0.051085 → 0.052516 | [-0.002048, 0.00491] | 0.012321 → 0.01167 | [-0.002195, 0.0008939] | none |
| lg1-t50 / 1024 | retained 03 | 1 | 0.097066 → 0.083631 | [-0.03141, 0.004543] | 0.050367 → 0.024876 | [-0.05021, -0.0007754] | score |
| lg1-t50 / 1024 | retained 03 | 2 | 0.15712 → 0.094585 | [-0.1307, 0.005641] | 0.15117 → 0.037819 | [-0.215, -0.01173] | score |
| lg3-t50 / 1024 | retained 03 | 0 | 0.1227 → 0.10805 | [-0.05314, 0.02385] | 0.031786 → 0.020737 | [-0.02237, 0.0002769] | none |
| lg3-t50 / 1024 | retained 03 | 1 | 0.18538 → 0.15953 | [-0.08643, 0.03474] | 0.043141 → 0.030679 | [-0.02263, -0.002291] | score |
| lg3-t50 / 1024 | retained 03 | 2 | 0.12007 → 0.088522 | [-0.08731, 0.02422] | 0.055134 → 0.023192 | [-0.08336, 0.01948] | none |
| ksc-t50 / 1024 | retained 03 | 0 | 0.10041 → 0.087117 | [-0.0476, 0.02102] | 0.025107 → 0.021727 | [-0.00944, 0.002679] | none |
| ksc-t50 / 1024 | retained 03 | 1 | 0.22933 → 0.14103 | [-0.2713, 0.09468] | 0.10376 → 0.048482 | [-0.2143, 0.1037] | none |
| ksc-t50 / 1024 | retained 03 | 2 | 0.080972 → 0.077083 | [-0.01344, 0.005662] | 0.013285 → 0.013735 | [-0.001726, 0.002625] | none |
| lg10-t10 / 1024 | current 04 | 0 | 0.25684 → 0.16009 | [-0.1464, -0.0471] | 0.12031 → 0.049279 | [-0.1053, -0.03674] | value, score |
| lg10-t10 / 1024 | current 04 | 1 | 0.3154 → 0.23244 | [-0.146, -0.01989] | 0.11212 → 0.062097 | [-0.07248, -0.02757] | value, score |
| lg10-t10 / 1024 | current 04 | 2 | 0.47759 → 0.15944 | [-0.4322, -0.2041] | 0.47057 → 0.16055 | [-0.4032, -0.2168] | value, score |
| lg20-t10 / 1024 | current 04 | 0 | 0.80598 → 0.40793 | [-0.6151, -0.1811] | 0.39072 → 0.15947 | [-0.3378, -0.1246] | value, score |
| lg20-t10 / 1024 | current 04 | 1 | 0.94094 → 0.52565 | [-0.6916, -0.139] | 0.39881 → 0.17425 | [-0.3452, -0.1039] | value, score |
| lg20-t10 / 1024 | current 04 | 2 | 1.6449 → 0.67353 | [-1.338, -0.605] | 1.6135 → 0.86167 | [-1.059, -0.4449] | value, score |
| m13-t20 / 1024 | current 04 | 0 | 3.1776 → 0.46669 | [-4.462, -0.9602] | 0.84884 → 0.12158 | [-1.235, -0.219] | value, score |
| lg10-t20 / 1024 | current 04 | 0 | 0.30689 → 0.18751 | [-0.2223, -0.01647] | 0.12403 → 0.052441 | [-0.1158, -0.0274] | value, score |
| lg10-t20 / 1024 | current 04 | 1 | 0.48226 → 0.31215 | [-0.2811, -0.05908] | 0.16122 → 0.081518 | [-0.121, -0.03836] | value, score |
| lg10-t20 / 1024 | current 04 | 2 | 0.50421 → 0.19234 | [-0.412, -0.2117] | 0.43267 → 0.13765 | [-0.417, -0.1731] | value, score |
| lg20-t20 / 1024 | current 04 | 0 | 0.84206 → 0.43067 | [-0.6347, -0.1881] | 0.406 → 0.15978 | [-0.3425, -0.15] | value, score |
| lg20-t20 / 1024 | current 04 | 1 | 1.3464 → 0.66894 | [-1.035, -0.3196] | 0.59473 → 0.24071 | [-0.5158, -0.1922] | value, score |
| lg20-t20 / 1024 | current 04 | 2 | 1.6297 → 0.70555 | [-1.255, -0.5936] | 1.6549 → 0.88577 | [-1.138, -0.4007] | value, score |
| lg1-t120 / 1024 | current 04 | 0 | 0.084964 → 0.08414 | [-0.01627, 0.01463] | 0.022238 → 0.019722 | [-0.007486, 0.002453] | none |
| lg1-t120 / 1024 | current 04 | 1 | 0.14893 → 0.12917 | [-0.03854, -0.0009928] | 0.062036 → 0.038123 | [-0.06309, 0.01526] | value |
| lg1-t120 / 1024 | current 04 | 2 | 0.19719 → 0.13535 | [-0.1408, 0.0171] | 0.14919 → 0.058659 | [-0.2085, 0.0274] | none |
| m13-t50 / 1024 | current 04 | 0 | 7.941 → 0.85338 | [-10.92, -3.258] | 2.1085 → 0.22591 | [-2.844, -0.9212] | value, score |
| m13-t50 / 1024 | current 04 | 1 | 69.181 → 10.097 | [-90.45, -27.72] | 15.455 → 3.7848 | [-18.81, -4.534] | value, score |
| lg10-t50 / 1024 | current 04 | 0 | 0.41467 → 0.27733 | [-0.2479, -0.02684] | 0.1468 → 0.070052 | [-0.1129, -0.04059] | value, score |
| lg10-t50 / 1024 | current 04 | 1 | 0.70231 → 0.46974 | [-0.4015, -0.06361] | 0.23833 → 0.1288 | [-0.1641, -0.05496] | value, score |
| lg10-t50 / 1024 | current 04 | 2 | 0.48827 → 0.23374 | [-0.3836, -0.1255] | 0.41777 → 0.16153 | [-0.3526, -0.1599] | value, score |
| lg20-t50 / 1024 | current 04 | 0 | 1.1534 → 0.61489 | [-1.11, 0.03327] | 0.53936 → 0.20875 | [-0.5633, -0.09792] | score |
| lg20-t50 / 1024 | current 04 | 1 | 2.5299 → 1.193 | [-2.39, -0.2841] | 1.1837 → 0.43513 | [-1.354, -0.1433] | value, score |
| lg20-t50 / 1024 | current 04 | 2 | 1.6514 → 0.74788 | [-1.271, -0.5365] | 1.5185 → 0.81366 | [-1.234, -0.1754] | value, score |
| ksc-t120 / 1024 | current 04 | 0 | 0.19929 → 0.18676 | [-0.06741, 0.04235] | 0.056801 → 0.048846 | [-0.01972, 0.003805] | none |
| ksc-t120 / 1024 | current 04 | 1 | 0.60337 → 0.39865 | [-0.5736, 0.1641] | 0.26016 → 0.12619 | [-0.3635, 0.09555] | none |
| ksc-t120 / 1024 | current 04 | 2 | 0.1484 → 0.14222 | [-0.03161, 0.01925] | 0.026103 → 0.028933 | [-0.005385, 0.01104] | none |
| lg1-t240 / 1024 | current 04 | 0 | 0.10955 → 0.11102 | [-0.007681, 0.01062] | 0.026474 → 0.023189 | [-0.007803, 0.001234] | none |
| lg1-t240 / 1024 | current 04 | 1 | 0.17556 → 0.16205 | [-0.04366, 0.01664] | 0.060173 → 0.042286 | [-0.02984, -0.005937] | score |
| lg1-t240 / 1024 | current 04 | 2 | 0.21867 → 0.1815 | [-0.08344, 0.009091] | 0.13877 → 0.055907 | [-0.1316, -0.0341] | score |
| ksc-t240 / 1024 | current 04 | 0 | 0.30472 → 0.23848 | [-0.1696, 0.03707] | 0.082376 → 0.061304 | [-0.048, 0.005855] | none |
| ksc-t240 / 1024 | current 04 | 1 | 1.0775 → 0.54043 | [-1.286, 0.2116] | 0.49497 → 0.18477 | [-0.7727, 0.1523] | none |
| ksc-t240 / 1024 | current 04 | 2 | 0.21929 → 0.19145 | [-0.06665, 0.01098] | 0.03489 → 0.034546 | [-0.006618, 0.00593] | none |
| lg1-t10 / 4096 | current 04 | 0 | 0.014908 → 0.015335 | [-0.001603, 0.002455] | 0.0038474 → 0.0036204 | [-0.0006459, 0.000192] | none |
| lg1-t10 / 4096 | current 04 | 1 | 0.031828 → 0.02558 | [-0.01667, 0.004175] | 0.021774 → 0.0097919 | [-0.02835, 0.004385] | none |
| lg1-t10 / 4096 | current 04 | 2 | 0.064522 → 0.028553 | [-0.07942, 0.00748] | 0.087549 → 0.016834 | [-0.1244, -0.01708] | score |
| lg3-t10 / 4096 | current 04 | 0 | 0.033398 → 0.025645 | [-0.03371, 0.0182] | 0.011425 → 0.0053055 | [-0.02131, 0.009072] | none |
| lg3-t10 / 4096 | current 04 | 1 | 0.038534 → 0.034057 | [-0.02217, 0.01321] | 0.0091899 → 0.0068131 | [-0.01233, 0.007576] | none |
| lg3-t10 / 4096 | current 04 | 2 | 0.053907 → 0.021764 | [-0.1093, 0.04506] | 0.034697 → 0.0077402 | [-0.07696, 0.02304] | none |
| ksc-t10 / 4096 | current 04 | 0 | 0.045352 → 0.041129 | [-0.01465, 0.006208] | 0.011164 → 0.0094123 | [-0.006282, 0.002779] | none |
| ksc-t10 / 4096 | current 04 | 1 | 0.13667 → 0.093175 | [-0.1498, 0.06281] | 0.067723 → 0.033772 | [-0.1255, 0.05756] | none |
| ksc-t10 / 4096 | current 04 | 2 | 0.032344 → 0.031205 | [-0.003629, 0.001351] | 0.0047019 → 0.004975 | [-0.0009098, 0.001456] | none |
| m13-t10 / 4096 | current 04 | 0 | 1.4664 → 0.16901 | [-2.186, -0.4084] | 0.36079 → 0.044889 | [-0.603, -0.02881] | value, score |
| lg10-t10 / 4096 | current 04 | 0 | 0.16386 → 0.083778 | [-0.1368, -0.02338] | 0.080191 → 0.02714 | [-0.09107, -0.01503] | value, score |
| lg10-t10 / 4096 | current 04 | 1 | 0.18722 → 0.11862 | [-0.1154, -0.02185] | 0.069134 → 0.03214 | [-0.06159, -0.0124] | value, score |
| lg10-t10 / 4096 | current 04 | 2 | 0.3324 → 0.084412 | [-0.3647, -0.1313] | 0.32432 → 0.080057 | [-0.3451, -0.1434] | value, score |

Simple-filter likelihood comparisons at the same particle count. These are descriptive mean absolute errors, not statistically established rankings. The exact Kalman comparator naturally supplies the zero-error LGSSM reference. A component non-inferiority result relative to ancestor weights does not establish that particle filtering is preferable to Kalman filtering for a linear Gaussian model.

| Scope / N | Source | Point | Marginal error | Bootstrap error | Prior-trajectory error | Gaussian-filter error |
|---|---|---:|---:|---:|---:|---:|---:|
| lg1-t10 / 1024 | retained 03 | 0 | 0.019833 | 0.13361 | 1.7085 | 7.8815e-07 |
| lg1-t10 / 1024 | retained 03 | 1 | 0.041009 | 0.11788 | 1.0929 | 1.3153e-06 |
| lg1-t10 / 1024 | retained 03 | 2 | 0.049638 | 0.1155 | 1.2568 | 9.2365e-07 |
| lg3-t10 / 1024 | retained 03 | 0 | 0.06035 | 0.39392 | 13.851 | 2.3858e-06 |
| lg3-t10 / 1024 | retained 03 | 1 | 0.079163 | 0.77793 | 9.947 | 4.4371e-06 |
| lg3-t10 / 1024 | retained 03 | 2 | 0.048463 | 0.32211 | 24.018 | 2.0205e-06 |
| ksc-t10 / 1024 | retained 03 | 0 | 0.048373 | 0.049926 | 0.10799 | 1.6957 |
| ksc-t10 / 1024 | retained 03 | 1 | 0.11579 | 0.076359 | 0.17342 | 1.7162 |
| ksc-t10 / 1024 | retained 03 | 2 | 0.040314 | 0.041931 | 0.079552 | 2.1211 |
| m13-t10 / 1024 | retained 03 | 0 | 0.47569 | 0.2516 | 1.7779 | 0.1896 |
| m13-t10 / 1024 | retained 03 | 1 | 3.0285 | 0.56891 | 1.2085 | 0.26579 |
| lg1-t20 / 1024 | retained 03 | 0 | 0.03251 | 0.22325 | 11.54 | 2.1937e-06 |
| lg1-t20 / 1024 | retained 03 | 1 | 0.04926 | 0.1559 | 6.5232 | 1.7504e-06 |
| lg1-t20 / 1024 | retained 03 | 2 | 0.056119 | 0.1491 | 7.0167 | 1.7985e-06 |
| lg3-t20 / 1024 | retained 03 | 0 | 0.070561 | 0.46706 | 42.406 | 7.1828e-06 |
| lg3-t20 / 1024 | retained 03 | 1 | 0.088523 | 0.5329 | 21.411 | 4.9481e-06 |
| lg3-t20 / 1024 | retained 03 | 2 | 0.05598 | 0.42824 | 79.51 | 7.0896e-06 |
| ksc-t20 / 1024 | retained 03 | 0 | 0.058483 | 0.073286 | 0.34536 | 1.4936 |
| ksc-t20 / 1024 | retained 03 | 1 | 0.10902 | 0.10269 | 0.60072 | 1.5416 |
| ksc-t20 / 1024 | retained 03 | 2 | 0.047617 | 0.060745 | 0.20576 | 1.4195 |
| lg1-t50 / 1024 | retained 03 | 0 | 0.052516 | 0.39327 | 59.118 | 7.9755e-06 |
| lg1-t50 / 1024 | retained 03 | 1 | 0.083631 | 0.26024 | 49.197 | 8.1753e-06 |
| lg1-t50 / 1024 | retained 03 | 2 | 0.094585 | 0.26514 | 54.365 | 7.7362e-06 |
| lg3-t50 / 1024 | retained 03 | 0 | 0.10805 | 0.96393 | 219.41 | 2.0511e-05 |
| lg3-t50 / 1024 | retained 03 | 1 | 0.15953 | 2.4421 | 138.93 | 2.4876e-05 |
| lg3-t50 / 1024 | retained 03 | 2 | 0.088522 | 0.80791 | 367.56 | 1.5458e-05 |
| ksc-t50 / 1024 | retained 03 | 0 | 0.087117 | 0.11571 | 2.2046 | 2.8489 |
| ksc-t50 / 1024 | retained 03 | 1 | 0.14103 | 0.16016 | 5.5048 | 2.6316 |
| ksc-t50 / 1024 | retained 03 | 2 | 0.077083 | 0.093027 | 0.98852 | 2.4523 |
| lg10-t10 / 1024 | current 04 | 0 | 0.16009 | 8.8401 | 115.86 | 0.00075724 |
| lg10-t10 / 1024 | current 04 | 1 | 0.23244 | 8.5725 | 67.055 | 0.0012797 |
| lg10-t10 / 1024 | current 04 | 2 | 0.15944 | 13.844 | 216.5 | 0.0016748 |
| lg20-t10 / 1024 | current 04 | 0 | 0.40793 | 64.683 | 295.76 | 0.0011664 |
| lg20-t10 / 1024 | current 04 | 1 | 0.52565 | 42.705 | 164.79 | 0.0039932 |
| lg20-t10 / 1024 | current 04 | 2 | 0.67353 | 119.99 | 593.62 | 0.0040406 |
| m13-t20 / 1024 | current 04 | 0 | 0.46669 | 0.40184 | 8.8626 | 0.26146 |
| lg10-t20 / 1024 | current 04 | 0 | 0.18751 | 17.193 | 312.7 | 0.00078177 |
| lg10-t20 / 1024 | current 04 | 1 | 0.31215 | 20.275 | 190.1 | 0.006267 |
| lg10-t20 / 1024 | current 04 | 2 | 0.19234 | 26.274 | 566.83 | 0.0032721 |
| lg20-t20 / 1024 | current 04 | 0 | 0.43067 | 129.12 | 729.64 | 0.0010254 |
| lg20-t20 / 1024 | current 04 | 1 | 0.66894 | 90.67 | 401.48 | 0.00985 |
| lg20-t20 / 1024 | current 04 | 2 | 0.70555 | 228.57 | 1467.2 | 0.0067008 |
| lg1-t120 / 1024 | current 04 | 0 | 0.08414 | 1.5739 | 235.2 | 2.7033e-05 |
| lg1-t120 / 1024 | current 04 | 1 | 0.12917 | 0.47921 | 235 | 3.4103e-05 |
| lg1-t120 / 1024 | current 04 | 2 | 0.13535 | 0.4174 | 269.11 | 2.8786e-05 |
| m13-t50 / 1024 | current 04 | 0 | 0.85338 | 0.70803 | 86.007 | 0.29433 |
| m13-t50 / 1024 | current 04 | 1 | 10.097 | 6.5315 | 85.346 | 0.4913 |
| lg10-t50 / 1024 | current 04 | 0 | 0.27733 | 41.278 | 961.47 | 0.0013007 |
| lg10-t50 / 1024 | current 04 | 1 | 0.46974 | 42.64 | 535.98 | 0.016039 |
| lg10-t50 / 1024 | current 04 | 2 | 0.23374 | 61.377 | 1866.9 | 0.0079901 |
| lg20-t50 / 1024 | current 04 | 0 | 0.61489 | 337.47 | 2185.8 | 0.0025977 |
| lg20-t50 / 1024 | current 04 | 1 | 1.193 | 241.99 | 1206.8 | 0.029718 |
| lg20-t50 / 1024 | current 04 | 2 | 0.74788 | 557.1 | 4557.1 | 0.01588 |
| ksc-t120 / 1024 | current 04 | 0 | 0.18676 | 0.21655 | 21.033 | 8.9061 |
| ksc-t120 / 1024 | current 04 | 1 | 0.39865 | 0.29153 | 56.667 | 8.2136 |
| ksc-t120 / 1024 | current 04 | 2 | 0.14222 | 0.15366 | 7.2221 | 6.8847 |
| lg1-t240 / 1024 | current 04 | 0 | 0.11102 | 1.3979 | 497.94 | 7.7214e-05 |
| lg1-t240 / 1024 | current 04 | 1 | 0.16205 | 0.61707 | 590.82 | 0.00011788 |
| lg1-t240 / 1024 | current 04 | 2 | 0.1815 | 0.59023 | 741.62 | 0.00011475 |
| ksc-t240 / 1024 | current 04 | 0 | 0.23848 | 0.29502 | 57.664 | 19.71 |
| ksc-t240 / 1024 | current 04 | 1 | 0.54043 | 0.39514 | 175.23 | 17.973 |
| ksc-t240 / 1024 | current 04 | 2 | 0.19145 | 0.22029 | 21.094 | 15.659 |
| lg1-t10 / 4096 | current 04 | 0 | 0.015335 | 0.11054 | 3.5933 | 7.6523e-07 |
| lg1-t10 / 4096 | current 04 | 1 | 0.02558 | 0.060047 | 0.86531 | 9.146e-07 |
| lg1-t10 / 4096 | current 04 | 2 | 0.028553 | 0.056928 | 0.73953 | 6.6712e-07 |
| lg3-t10 / 4096 | current 04 | 0 | 0.025645 | 0.15775 | 10.336 | 2.8709e-06 |
| lg3-t10 / 4096 | current 04 | 1 | 0.034057 | 0.31952 | 6.6383 | 4.3286e-06 |
| lg3-t10 / 4096 | current 04 | 2 | 0.021764 | 0.16254 | 17.972 | 1.4029e-06 |
| ksc-t10 / 4096 | current 04 | 0 | 0.041129 | 0.025582 | 0.05386 | 1.9817 |
| ksc-t10 / 4096 | current 04 | 1 | 0.093175 | 0.033352 | 0.075186 | 1.8483 |
| ksc-t10 / 4096 | current 04 | 2 | 0.031205 | 0.020128 | 0.038689 | 2.3579 |
| m13-t10 / 4096 | current 04 | 0 | 0.16901 | 0.1147 | 0.76911 | 0.20493 |
| lg10-t10 / 4096 | current 04 | 0 | 0.083778 | 4.1717 | 102.61 | 0.00073116 |
| lg10-t10 / 4096 | current 04 | 1 | 0.11862 | 5.0416 | 59.967 | 0.0021495 |
| lg10-t10 / 4096 | current 04 | 2 | 0.084412 | 6.1512 | 190.48 | 0.001607 |

Equal-runtime coverage is shown separately. Partial matches describe only the matched subset, and cannot establish an overall runtime ranking. Old-source timings include unequal serialization overhead; current-source timings exclude it.

| Scope / N | Source | Point | Matched rows / expected | Marginal error on matched rows | Bootstrap error on matched rows | Complete |
|---|---|---:|---:|---:|---:|---|
| lg1-t10 / 1024 | retained 03 | 0 | 340 / 512 | 0.019663 | 0.032371 | False |
| lg1-t10 / 1024 | retained 03 | 1 | 341 / 512 | 0.040191 | 0.028069 | False |
| lg1-t10 / 1024 | retained 03 | 2 | 340 / 512 | 0.046987 | 0.026921 | False |
| lg3-t10 / 1024 | retained 03 | 0 | 295 / 512 | 0.062677 | 0.051601 | False |
| lg3-t10 / 1024 | retained 03 | 1 | 311 / 512 | 0.081392 | 0.15873 | False |
| lg3-t10 / 1024 | retained 03 | 2 | 307 / 512 | 0.047691 | 0.038965 | False |
| ksc-t10 / 1024 | retained 03 | 0 | 26 / 512 | 0.042319 | 0.0064929 | False |
| ksc-t10 / 1024 | retained 03 | 1 | 27 / 512 | 0.10525 | 0.010339 | False |
| ksc-t10 / 1024 | retained 03 | 2 | 16 / 512 | 0.016811 | 0.0062861 | False |
| m13-t10 / 1024 | retained 03 | 0 | 16 / 512 | 0.39928 | 0.017444 | False |
| m13-t10 / 1024 | retained 03 | 1 | 17 / 512 | 3.4417 | 0.05832 | False |
| lg1-t20 / 1024 | retained 03 | 0 | 399 / 512 | 0.032162 | 0.057077 | False |
| lg1-t20 / 1024 | retained 03 | 1 | 401 / 512 | 0.049506 | 0.03595 | False |
| lg1-t20 / 1024 | retained 03 | 2 | 403 / 512 | 0.056303 | 0.036006 | False |
| lg3-t20 / 1024 | retained 03 | 0 | 204 / 256 | 0.072997 | 0.043597 | False |
| lg3-t20 / 1024 | retained 03 | 1 | 195 / 256 | 0.088176 | 0.067028 | False |
| lg3-t20 / 1024 | retained 03 | 2 | 210 / 256 | 0.055546 | 0.047896 | False |
| ksc-t20 / 1024 | retained 03 | 0 | 35 / 512 | 0.048819 | 0.012765 | False |
| ksc-t20 / 1024 | retained 03 | 1 | 47 / 512 | 0.086175 | 0.020689 | False |
| ksc-t20 / 1024 | retained 03 | 2 | 41 / 512 | 0.056342 | 0.0092021 | False |
| lg1-t50 / 1024 | retained 03 | 0 | 419 / 512 | 0.052542 | 0.10347 | False |
| lg1-t50 / 1024 | retained 03 | 1 | 422 / 512 | 0.085009 | 0.066875 | False |
| lg1-t50 / 1024 | retained 03 | 2 | 419 / 512 | 0.091528 | 0.062339 | False |
| lg3-t50 / 1024 | retained 03 | 0 | 210 / 256 | 0.11167 | 0.12107 | False |
| lg3-t50 / 1024 | retained 03 | 1 | 203 / 256 | 0.1566 | 0.46346 | False |
| lg3-t50 / 1024 | retained 03 | 2 | 204 / 256 | 0.088358 | 0.073658 | False |
| ksc-t50 / 1024 | retained 03 | 0 | 241 / 512 | 0.081115 | 0.011578 | False |
| ksc-t50 / 1024 | retained 03 | 1 | 256 / 512 | 0.12549 | 0.014345 | False |
| ksc-t50 / 1024 | retained 03 | 2 | 254 / 512 | 0.066006 | 0.010012 | False |
| lg10-t10 / 1024 | current 04 | 0 | 83 / 512 | 0.15159 | 2.694 | False |
| lg10-t10 / 1024 | current 04 | 1 | 285 / 512 | 0.22842 | 3.5796 | False |
| lg10-t10 / 1024 | current 04 | 2 | 298 / 512 | 0.1635 | 5.1613 | False |
| lg20-t10 / 1024 | current 04 | 0 | 118 / 512 | 0.40777 | 42.314 | False |
| lg20-t10 / 1024 | current 04 | 1 | 236 / 512 | 0.50068 | 30.842 | False |
| lg20-t10 / 1024 | current 04 | 2 | 288 / 512 | 0.70511 | 87.657 | False |
| m13-t20 / 1024 | current 04 | 0 | 355 / 512 | 0.47417 | 0.029779 | False |
| lg10-t20 / 1024 | current 04 | 0 | 252 / 512 | 0.18828 | 7.1351 | False |
| lg10-t20 / 1024 | current 04 | 1 | 386 / 512 | 0.31687 | 10.181 | False |
| lg10-t20 / 1024 | current 04 | 2 | 417 / 512 | 0.19224 | 11.995 | False |
| lg20-t20 / 1024 | current 04 | 0 | 314 / 512 | 0.41863 | 98.984 | False |
| lg20-t20 / 1024 | current 04 | 1 | 459 / 512 | 0.66227 | 68.449 | False |
| lg20-t20 / 1024 | current 04 | 2 | 443 / 512 | 0.70316 | 176.45 | False |
| lg1-t120 / 1024 | current 04 | 0 | 98 / 256 | 0.090753 | 0.86583 | False |
| lg1-t120 / 1024 | current 04 | 1 | 137 / 256 | 0.11899 | 0.15427 | False |
| lg1-t120 / 1024 | current 04 | 2 | 179 / 256 | 0.14249 | 0.12461 | False |
| m13-t50 / 1024 | current 04 | 0 | 0 / 512 | not assessed | not assessed | False |
| m13-t50 / 1024 | current 04 | 1 | 0 / 512 | not assessed | not assessed | False |
| lg10-t50 / 1024 | current 04 | 0 | 427 / 512 | 0.28116 | 22.316 | False |
| lg10-t50 / 1024 | current 04 | 1 | 453 / 512 | 0.46024 | 24.843 | False |
| lg10-t50 / 1024 | current 04 | 2 | 442 / 512 | 0.2375 | 35.562 | False |
| lg20-t50 / 1024 | current 04 | 0 | 180 / 256 | 0.61094 | 273.33 | False |
| lg20-t50 / 1024 | current 04 | 1 | 178 / 256 | 1.2576 | 202.27 | False |
| lg20-t50 / 1024 | current 04 | 2 | 194 / 256 | 0.75082 | 458.35 | False |
| ksc-t120 / 1024 | current 04 | 0 | 131 / 512 | 0.17629 | 0.02008 | False |
| ksc-t120 / 1024 | current 04 | 1 | 167 / 512 | 0.30976 | 0.026902 | False |
| ksc-t120 / 1024 | current 04 | 2 | 183 / 512 | 0.12969 | 0.014379 | False |
| lg1-t240 / 1024 | current 04 | 0 | 480 / 512 | 0.11114 | 0.46504 | False |
| lg1-t240 / 1024 | current 04 | 1 | 486 / 512 | 0.16157 | 0.18237 | False |
| lg1-t240 / 1024 | current 04 | 2 | 485 / 512 | 0.18196 | 0.15457 | False |
| ksc-t240 / 1024 | current 04 | 0 | 454 / 512 | 0.23819 | 0.030286 | False |
| ksc-t240 / 1024 | current 04 | 1 | 494 / 512 | 0.53984 | 0.038268 | False |
| ksc-t240 / 1024 | current 04 | 2 | 501 / 512 | 0.1916 | 0.019845 | False |
| lg1-t10 / 4096 | current 04 | 0 | 161 / 512 | 0.015617 | 0.020398 | False |
| lg1-t10 / 4096 | current 04 | 1 | 329 / 512 | 0.026296 | 0.0082558 | False |
| lg1-t10 / 4096 | current 04 | 2 | 165 / 512 | 0.026517 | 0.011027 | False |
| lg3-t10 / 4096 | current 04 | 0 | 226 / 256 | 0.025191 | 0.010898 | False |
| lg3-t10 / 4096 | current 04 | 1 | 194 / 256 | 0.032406 | 0.022063 | False |
| lg3-t10 / 4096 | current 04 | 2 | 151 / 256 | 0.022346 | 0.01251 | False |
| ksc-t10 / 4096 | current 04 | 0 | 262 / 512 | 0.033823 | 0.0020947 | False |
| ksc-t10 / 4096 | current 04 | 1 | 279 / 512 | 0.084169 | 0.0027921 | False |
| ksc-t10 / 4096 | current 04 | 2 | 278 / 512 | 0.029605 | 0.0014687 | False |
| m13-t10 / 4096 | current 04 | 0 | 0 / 512 | not assessed | not assessed | False |
| lg10-t10 / 4096 | current 04 | 0 | 408 / 512 | 0.087123 | 0.72729 | False |
| lg10-t10 / 4096 | current 04 | 1 | 397 / 512 | 0.1204 | 1.2723 | False |
| lg10-t10 / 4096 | current 04 | 2 | 461 / 512 | 0.083798 | 1.2446 | False |

Post-run red-team note: exact finite-program derivatives do not establish accuracy relative to the statistical model. Four datasets can leave large uncertainty; particle-reference refinement and MCSE do not bound all remaining bias. The diagonal/pairwise correction controls remain scope-specific hypotheses. Low process-noise range-bearing performance must be reported separately from the nominal parameter point. The current parameter regimes are likelihood-evaluation points on data generated at the recorded truth; they are not a factorial grid of generating parameters.

The next phase is part of the same authorized master. Consult next-phase.json and campaign.json for current execution and unfinished work. An active worker is not a completed result.
