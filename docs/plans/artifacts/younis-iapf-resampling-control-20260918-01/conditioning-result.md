# Fresh-stream conditioning safeguard: numerical non-harm passes

2026-09-18. The optional input-precision cutoff passed its predeclared safety
criterion on all five fixed datasets at N4096. Calibration and final particle
streams were new and disjoint from all earlier streams. The 192 calibration
replicates and 128 final replicates per dataset consumed 1600 filter calls and
no new fits. This stage tests numerical safety, not scientific superiority.

In affine dataset 1490, the cutoff removes the rounding-induced eighth
singular direction and retains rank seven, as required by the quadratic-score
derivation in the plan. The largest coefficient falls from 221863.637 to
.0186772. All corrected outputs are finite and the predeclared six-component
Bonferroni 99% Kalman mean-bias screen passes. In all four nonlinear datasets,
both coefficient arrays and all 128 final corrected score arrays are exactly
identical under the two cutoff rules. No MSE criterion or final-data tuning
was used. The optional safeguard is now validated on these scopes; the
combination factory's default remains unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Optional safeguard passes this check | 4/4 nonlinear coefficient/final-array equality; affine rank seven and Kalman bias screen | No numerical or safety veto | Other near-dependencies, longer horizons and targets | Use the explicit input-precision declaration in future scoped control diagnostics; record it | Universal rank threshold or a new default |
| Score candidate remains diagnostic | No performance-promotion criterion in this supplement | Earlier heuristic losses remain | Residual Gaussian-innovation variance and generalization | Derive additional zero-mean innovation controls or conditional integration | HMC/LEDH readiness or dominance |

| Inference status | Finding |
|---|---|
| Hard veto screen | All finite/trace/source/disjoint-stream checks pass; old performance vetoes remain. |
| Statistically supported ranking | None sought; exact nonlinear equality is numerical evidence, and the affine bias screen is a safety check. |
| Descriptive-only differences | MSE/variance and affine coefficient magnitudes are saved for explanation. |
| Default-readiness | No. This validates an explicit optional safeguard, not a new scientific or numerical default. |
| Next evidence needed | Further variance mechanism with independently frozen controls, fresh confirmation and conditional heuristic comparisons. |

The strongest alternative explanation is over-specialization to five fixed
fits. A resolved nonlinear correction changing, a failed affine bias screen,
or failure on a different near-null pattern would overturn this scope's safety
conclusion. The weakest evidence is transfer beyond these fixed T=2 examples.
The FP32 threshold is a recorded input-precision hypothesis, not a proved error
bound for every score implementation.

Code change: make_combination_kernels now accepts optional
control_input_dtype_name. It uses the larger of regression and declared input
roundoff when determining numerical rank. No ridge, coefficient clipping,
target change or change to the existing default was introduced. The 17 focused
CPU tests pass, including a contaminated-rank case and exact equality for a
resolved case. CPU tests hide GPU. The GPU comparison uses verified memory
growth and FP32/TF32/XLA filters, with FP64 diagnostic regression/application.
All regression and particle specializations trace once. Peak allocator bytes
in this follow-up are 8,531,968. Runtime is not a comparative performance claim.

Artifacts: [manifest](conditioning-confirmation01/manifest.json),
[results](conditioning-confirmation01/results.json),
[decision](conditioning-confirmation01/decision.json),
[terminal validation](final-validation.json), and
[log](conditioning-confirmation01.log). The manifest records exact commands,
environment, source hashes, inherited observations/fits, seed derivation and
the two declared source changes from the original comparison: this optional
regression argument and its reviewed plan amendment. All terminal source hashes
match. Independent saved-row arithmetic passes FP64 dot-product error bounds.
All 6400 new seed pairs are distinct from each other and the 11520 old pairs.

Final cumulative campaign use: 3/4 launches, 129.311595/1800 driver seconds,
5/8 fits, 6884/8000 filter charges, conservative 180/600 test/probe seconds.
Remaining: one launch, 1670.688405 seconds, three fits, 1116 filter charges
and 420 test/probe seconds. All planned comparisons and safety checks are
complete; no process is running. Preserve every final stream as held-out
evidence. A further mechanism needs its own pre-run evidence contract and
adequate fresh-stream budget; remaining capacity must not be hidden or reset.
