# LGSSM Gradient Failure Root-Cause Diagnostic Plan

Date: 2026-07-13

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Why does the full `d=3,T=50,N=10000`, seed-`81120` LGSSM LEDH score fail both its same-scalar derivative check and the exact Kalman likelihood-gradient check, especially for `q_scale`? |
| Claimed target | The gradient of the transition-first LGSSM observed-data log likelihood with respect to `(phi1, phi2, phi3, q_scale, r_scale)`. |
| Exact scientific oracle | TensorFlow autodiff of the float64 transition-first Kalman likelihood on the same observations and parameterization, checked by float64 central differences. |
| Implementation oracle | Finite differences or TensorFlow autodiff of the identical fixed-noise LEDH scalar evaluated by the compact/manual score route. |
| Candidate mechanisms | A forward/JVP mismatch at clipped float32 pairwise OT costs; other nonsmooth or precision-sensitive transport operations; deterministic barycentric OT covariance contraction; finite-`N` pathwise Monte Carlo slope variability. |
| Expected failure mode | The manual compact tangent can disagree with its own scalar under float32/TF32, while the correctly differentiated finite-`N` LEDH scalar can separately have a slope far from the Kalman oracle. |
| Promotion criterion | None. This is a debugging diagnostic, not an admission or default-policy run. |
| Promotion veto | Any same-scalar derivative mismatch blocks use of the compact score. Any material Kalman-gradient mismatch blocks claiming the LEDH score estimates the LGSSM likelihood gradient. |
| Continuation veto | Corrupted/mismatched observations, parameterization, fixed-noise identity, time convention, or inability to reproduce the recorded failure. |
| Repair trigger | A causal intervention that removes one discrepancy while preserving the compared scalar identifies the corresponding repair target. |
| Explanatory diagnostics | Prefix length, particle count, dtype, TF32 state, resampling policy, clipped-cost counts, covariance trace ratios, seed-to-seed variation, and FD step stability. |
| Forbidden conclusion | The accepted value bias does not establish gradient correctness. Same-scalar agreement does not establish agreement with Kalman. A one-seed slope does not establish systematic bias without replication. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Does the compact JVP differentiate the exact scalar emitted by the production LEDH value path? |
| Numerical/scientific question | Once correctly differentiated, does that finite-`N` LEDH scalar have the same local gradient as the exact Kalman likelihood within declared uncertainty? |
| Primary implementation criterion | Manual JVP, GradientTape, and stable central FD of the identical scalar agree coordinatewise in a smooth neighborhood. |
| Primary oracle criterion | Multi-seed LEDH gradient estimates, with uncertainty, are compatible with the differentiated Kalman score; a single close scalar value is explanatory only. |
| Hard vetoes | Nonfinite values, collapsed FD endpoints, changed resampling masks/noise across endpoints, mismatched time convention/model, or a manual tangent that omits an active forward operation. |
| Explanatory only | Scalar relative value bias, raw per-seed differences, runtime, memory, and isolated clipped-count totals without an intervention test. |
| Preserved artifact | This plan, focused JSON diagnostics under `docs/plans/artifacts/lgssm-gradient-root-cause-2026-07-13/`, and a result note in `docs/plans`. |
| Not concluded | Leaderboard admission, HMC readiness, posterior correctness, statistical superiority, or production readiness. |

## Skeptical Plan Audit

- Wrong-baseline check: use same-scalar FD/tape for implementation correctness and differentiated Kalman for the intended likelihood gradient. Do not substitute LEDH-versus-LEDH agreement for Kalman agreement.
- Proxy check: value bias is not a promotion criterion for gradient correctness.
- Hidden-assumption check: verify candidate-dependent stationary initial particles, transition-first convention, common random numbers, fixed resampling masks, parameter order, and identical observations before interpreting gradients.
- Precision check: compare float64, float32 with TF32 disabled, and production float32/TF32 where feasible. A float64 pass alone cannot clear the production route.
- Nonsmoothness check: localize `clip`, `max`, `floor`, and fixed branch masks. A causal mask-corrected comparison is required before blaming clipping.
- Statistical check: one fixed seed can diagnose same-scalar correctness but cannot establish systematic Kalman-gradient bias. Multi-seed uncertainty is required for that claim.
- Artifact check: each command must report the scalar, gradient, FD endpoints/steps, route, noise/mask identity, dtype/device, and intervention state needed to answer its stated question.
- Stop-condition check: do not run a full expensive ladder until a smaller prefix/particle diagnostic discriminates among mechanisms.

Audit status: `PASS_FOR_FOCUSED_CPU_ONLY_DIAGNOSTICS`. GPU/XLA work, if later required, needs trusted/escalated execution and a refreshed run manifest.

## Diagnostic Sequence

1. Reconstruct the exact dependency graph from `theta` through stationary initialization, transition, LEDH flow, importance correction, normalized weights, fixed resampling, OT reset, and accumulated likelihood.
2. Reproduce recorded `T=1,10,50` score/FD values and compare each coordinate with the prefix Kalman score.
3. Unit-test the clipped pairwise-cost JVP in float32 using deliberately cancellation-prone clouds. Compare current manual JVP, TensorFlow autodiff, and central FD of the exact clipped forward operation.
4. At small `N`, compare end-to-end `no-resampling` and `active-all` routes across `T=1,10,50`, float64 and float32, using manual JVP, tape where memory permits, and a stable FD-step ladder.
5. If a forward/JVP mismatch is isolated, apply the smallest diagnostic-only intervention and rerun the same case. Do not modify production behavior during diagnosis.
6. After same-scalar correctness is established, decompose Kalman-gradient error by resampling policy, time, and seed. Measure cloud covariance before/after OT and test a diagnostic moment-restored reset only as a causal intervention.
7. Write a result note separating implementation failure, numerical failure, finite-`N` stochastic uncertainty, and approximation bias.

## Stop Conditions

- Stop and repair the harness if the recorded artifacts cannot be reproduced with identical source/configuration identities.
- Stop the compact-score claim immediately if the manual tangent is shown not to differentiate an active forward operation.
- Stop attributing the oracle gap to OT if the no-resampling scalar has the same systematic multi-seed gradient gap.
- Stop attributing a systematic bias if multi-seed uncertainty has not been measured.
- Stop before any production patch, full GPU rerun, or leaderboard change; those require a separate reviewed repair plan after localization.
