# Actual-SV Overcomplete Analytical Chart Phase 5 Result

Date: 2026-07-17

Status: `PASS_PHASE_5_OWN_SCALAR_DERIVATIVE`

## Result

The selected fixed `K=23` route passes `T=2,10,100,1000` at the center and the
four exact `1e-5` finite-difference endpoints.  Every claim-bearing value,
manual score, and chart diagnostic is finite.  No time-748 or other positivity
failure occurs.  Warm replay is bitwise identical in every XLA result.

The manual score is the explicit two-direction total JVP of the same finite
scalar.  Central finite differences use only the owner-approved FD policy
`0.05*sqrt(2)=0.07071067811865477`:

| Horizon | Manual versus FD relative error | FD gate |
| ---: | ---: | --- |
| 2 | `2.323339358376968e-11` | pass |
| 10 | `1.6187801132686436e-10` | pass |
| 100 | `2.247653507298375e-9` | pass |
| 1000 | `5.7988816510053705e-8` | pass |

TensorFlow reverse-mode autodiff agrees at short horizons but returns NaN at
`T>=100` despite finite values and manual scores.  Focused repair attempts
showed this is an oracle implementation limitation associated with the
extreme tiny-weight reverse adjoint path; a custom-pullback experiment was
removed after it failed to repair the oracle.  TensorFlow eager
`ForwardAccumulator` differentiates the same unmodified scalar and gives:

| Horizon | Manual versus eager forward-AD relative difference | Float64 reference gate |
| ---: | ---: | --- |
| 10 | `1.298650134071988e-13` | pass |
| 100 | `4.447444400460831e-12` | pass |
| 1000 | `2.6364004959309425e-12` | pass |

The forward-AD gate is `sqrt(float64 epsilon)=1.4901161193847656e-8`.
Component-level teacher, continuation, feature, and projection derivatives
also agree with TensorFlow AD, and primitive JVP/VJP duality tests pass.

Automatic forward mode cannot be placed inside the compiled production graph:
XLA rejects a transformed-loop transpose permutation and non-JIT `tf.function`
hits a TensorList shape invariant.  Eager forward AD is therefore a labeled
CPU reference exception.  The production score remains the XLA-default manual
JVP.

## Verification

The final affected regression command passed `29` tests after removing the
unsuccessful custom-pullback experiment.  The broader pre-experiment Phase 2
suite passed `40` tests; all historical square and scalar-SV loop regressions
remain intact.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 5 | Same finite scalar has a finite explicit total JVP agreeing with FD and independent forward AD | No chart, FD, manual-JVP, or forward-AD veto fired | Scientific agreement with the dense target reference remains unknown | Run descriptive same-target comparison | No exact nonlinear likelihood, cross-method equivalence, GPU, HMC, canonical, or leaderboard claim |
