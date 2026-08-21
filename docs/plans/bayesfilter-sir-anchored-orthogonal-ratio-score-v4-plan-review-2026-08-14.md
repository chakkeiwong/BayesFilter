# Thorough Review: Anchored-Orthogonal Ratio Score V4

Date: 2026-08-14  
Reviewed plan: `bayesfilter-sir-anchored-orthogonal-ratio-score-v4-plan-2026-08-14.md`  
Verdict: `PASS_WITH_IMPLEMENTATION_GATES`

## Mathematical Review

The density-ratio identity remains conditional balanced classification. The
anchored basis is valid because `phi0=r`, the curvature mode has zero derivative
at zero, and alpha makes the two modes orthogonal on the actual perturbation
design. Therefore the score is exactly the calibrated coefficient of `phi0`
divided by `2*delta_scale`; it is not a regression of noisy per-delta scores.

The basis is not claimed to make learned coefficient functions statistically
independent. That limitation is explicitly tested through replicate spread and
oracle error.

## Skeptical Audit

| Risk | Finding | Disposition |
|---|---|---|
| Wrong baseline | linear and MLP coefficient heads share the same anchored basis | pass |
| Proxy promotion | ECE/AUC are veto diagnostics; exact fixed-path score is primary | pass |
| Underidentified polynomial | two modes only; third mode explicitly excluded | pass |
| Leakage | fixed path evaluated only after all training/selection/calibration | pass |
| Unfair scope | controls selected per stage/horizon/coordinate with fresh domains | pass |
| Stale V3 context | V3 motivates basis change but contributes no controls or data | pass |
| Environment mismatch | tftwogpu, trusted GPU, memory growth, XLA, TF32-off recorded | pass |
| Missing stop condition | any exact-cell failure blocks SIR and further silent repairs | pass |

## Required Tests

1. alpha is computed from the declared six-point design;
2. discrete inner product of the two basis functions is zero;
3. `phi0'(0)=1` and `phi1'(0)=0` by finite difference;
4. basis condition number is recorded and bounded;
5. score source contains only `c0/(2*delta_scale)`;
6. conditional delta balance and no signed-delta leakage;
7. synthetic anchored-logit recovery without exact-score labels;
8. source/fresh-process dependency audits;
9. selection/final domains and paired prefixes are distinct;
10. SIR refuses to run without a passed exact oracle.

## Decision

The plan is a proportionate methodological revision to V3. It should execute
only after all implementation gates pass. A full exact-oracle failure is a
terminal result, not permission to widen the basis, alter ECE, or change the
perturbation grid after seeing outcomes.
