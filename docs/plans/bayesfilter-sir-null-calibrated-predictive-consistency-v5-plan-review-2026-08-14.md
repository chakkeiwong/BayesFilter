# V5 Plan Review: Null-Calibrated Predictive Consistency

Date: 2026-08-14  
Reviewed plan: `bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-2026-08-14.md`  
Verdict: `PASS_FOR_GPU_SMOKE_AFTER_IMPLEMENTATION_GATES`

## Summary

The plan correctly replaces V4's uncalibrated nine-cell precision veto with a
single joint null-predictive coverage target. The split-conformal threshold and
independent audit are appropriate for the user's stated 5% false-rejection
criterion. The first draft of the audit rule was too stringent: requiring a
95% upper confidence bound to be below 5% would reject too often even when the
true failure rate is exactly 5%. The corrected plan uses the calibration order
statistic for the coverage claim and the audit only as a lower-bound
falsification veto. The plan cannot be treated as a score-validation plan:
centering at the estimator's own null mean can make a biased or degenerate
estimator look predictively consistent. Execution is allowed only if the
implementation preserves that limitation and the zero-mean/head diagnostics
remain visible.

## Skeptical Audit

| Risk | Finding | Required disposition |
|---|---|---|
| Wrong target | Null coverage is not pathwise score accuracy | Keep `predictive consistency` in every promotion/result label |
| Proxy promotion | Conformal coverage could accept a constant output | Retain head signal checks and zero-mean simultaneous diagnostic; no score-correctness claim |
| Arbitrary margin | No numeric score margin is selected | `q` is an order statistic from an independent null-calibration split |
| Multiplicity | Nine cells are handled by one max/Mahalanobis statistic | Preserve joint prefixes and cross-cell covariance |
| Calibration leakage | Center/covariance, threshold, and audit could overlap | Enforce domains 70/80/90 and fail on seed/domain collision |
| Small audit | A handful of failures may be misleading | Predeclare `B_audit=500`; use lower-bound falsification only, with veto at `F>=34` |
| Threshold overfit | Audit could be used to tune q | Make audit read-only and terminal for the frozen bundle |
| Covariance singularity | SVD whitening can explode in null directions | Require declared rank rule, record rank, fail closed on inadequate rank |
| Training scope | A frozen-bundle guarantee excludes retraining variation | State conditional scope; make retraining stability secondary |
| Gaussian transfer | Exact Gaussian behavior does not prove SIR behavior | Use Gaussian for harness validation only; no transferred margin claim |
| Stale V4 context | V4 exact gate was a different estimand | Preserve V4 artifacts; do not mix V4 scores or thresholds into V5 |
| Resource drift | Repeated simulator construction caused retracing | Cache static-horizon compiled simulators as an implementation repair before full run |

## Mathematical And Statistical Review

For a fixed frozen bundle and fixed `mu,C`, the calibration nonconformity
scores are exchangeable with a future same-parameter score. The order statistic

`D_(ceil(0.95*(B_cal+1)))`

therefore gives finite-sample marginal coverage at least 95% under the stated
exchangeability assumptions. The audit's Clopper-Pearson upper bound is an
independent check of the realized failure rate, not part of threshold fitting.

The guarantee is conditional on the frozen bundle and on the simulator being
the same data-generating law. It is not a guarantee under parameter drift,
observation-model misspecification, or retraining unless those sources of
variation are included in calibration.

The proposed SVD geometry changes power, not coverage, provided it is fitted
only on the null-fit split. It must not be described as a scientifically
derived error margin. The zero-mean check follows from the regular score
identity `E_theta[s_theta(Y)] = 0`, but its finite-sample bootstrap region is a
diagnostic and does not recover the pathwise score. The audit's one-sided
Clopper-Pearson lower bound is an independent falsification check, not part of
threshold fitting; demanding an upper bound below 0.05 would be a stronger
claim and would recreate the over-stringency problem.

## Required Implementation Gates Before Execution

1. Add a pure TensorFlow/standard-library conformal nonconformity helper with
   deterministic order-statistic indexing and exact CP audit calculation.
2. Add tests for exchangeable toy coverage, calibration/audit separation,
   max-statistic dependence, SVD rank handling, and zero-mean reporting.
3. Cache the three static-horizon simulators instead of creating compiled
   functions inside the path-generation loop.
4. Add fresh-process source and loaded-module audits; preserve the existing
   no-filter/no-particle route veto.
5. Make the runner fail closed when a calibration or audit domain is reused.
6. Record the frozen-bundle scope and explicitly label all Gaussian exact-score
   comparisons as secondary diagnostics.
7. Run a small CPU smoke before any GPU campaign; no SIR execution occurs in
   the implementation-gate phase.

## Decision

The implementation gates were closed on 2026-08-14:

- deterministic marginal and tolerance-bound order ranks are tested;
- exact lower-bound binomial audit logic is tested at the `33/34` boundary;
- TensorFlow SVD rank handling and simultaneous zero-mean reporting are tested;
- selection, training, null-fit, null-calibration, and null-audit domains are
  distinct and enforced;
- static-horizon simulators are cached;
- full runs fail closed on frozen-head validity screens;
- the result schema labels the claim
  `joint_same_parameter_predictive_coverage_only`; and
- a deliberate CPU-only integration smoke completed end to end.

Focused gate result: `15 passed`. The corrected plan is approved for a trusted
GPU/XLA smoke. Full Gaussian execution remains conditional on that smoke; SIR
remains conditional on a valid full Gaussian harness result and retains only
the weaker predictive-consistency claim.
