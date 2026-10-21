# Range-bearing fresh mixed-arithmetic result, 2026-10-12

Selected candidate0 passes the1% numerical screen on both fresh validation
datasets and both parameter rows, with all20 observations completed. Its largest
scaled score error is0.0139245%. The comparator is the same finite filter in
uniform FP64, not the true model score. N4032/K2016,GPU/XLA,TF32off and memory
growth are unchanged. FP32 remains the default; FP64 recursion with centered
FP32 pair kernels is optional.

Calibration seeds26101231/32 and validation seeds26101233/34 are disjoint.
Candidate0 freezes8flow steps,40reset steps,epsilon0.5,direct contraction,
zero optional moment correction and equal three-component mixture weights.

| Candidate | Flow/reset/epsilon | Precision pass | Maximum scaled score error by validation dataset | Warm B2 seconds |
|---|---|---|---|---|
| 0 | 8/40/0.5 | True | 0.013924% / 0.009851% | 6.192 |
| 1 | 16/80/0.5 | False | 0.004117% / 38.801128% | 13.185 |
| 2 | 8/40/0.25 | True | 0.012680% / 0.013417% | 8.685 |

Errors use max(1,absolute FP64 component). Candidate1 fails materially on one
dataset despite healthy states. More iterations do not guarantee numerical
robustness. Candidates0/2 remain viable. Runtime and importance-weight variance
times runtime are descriptive nomination measures, not an HMC-efficiency ranking.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Freeze candidate0 for exploratory mechanics | Both comparisons below1% | Candidate1 rejected;0/2 finite,valid,complete | Two datasets and local rows | Cost localization and frozen-selection mechanics | Exact model score, posterior convergence, canonical LEDH admission |
| Continue performance repair | Warm B2 score6.19s | No primitive attribution yet | GPU contention, parameter dependence, XLA fusion | Staged versus fused first-observation diagnostic | Statistical speed ranking |

| Inference status | Finding |
|---|---|
| Hard veto screen | Candidate1 fails;0/2 pass local numerical screen |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, scaled errors and nomination proxy |
| Default-readiness | FP32 unchanged; mixed arithmetic optional |
| Next evidence needed | Frozen-selection mechanics, new independent data and cumulative posterior/reference checks |

Alternative explanation: success is local to these parameter rows and datasets;
a new frozen-selection precision failure overturns broader robustness. Posterior
coverage is the weakest evidence. Existing campaign data was used for diagnosis;
mechanics on it is exploratory, not untouched confirmation.

Source/environment/device/memory/command manifest:
../surrogate-hmc-mixed-scope-range-20261012-01/range_bearing/result.json.
Full trajectory checks, values and score components:
../surrogate-hmc-mixed-scope-range-20261012-01/range_bearing/filter-selection.json.

After range tuning: 25969.4729s used,10030.5271s unspent of36000s. Component diagnostics are next.
