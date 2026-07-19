# SSL-LSTM NeuTra Phase 8 Terminal Orientation Repair Review

Date: 2026-07-17

Verdict: `AGREE_TERMINAL_ONLY_VALIDATION_REQUIRED`

## Finding

The target pilot failed for six G points on `STATUS_PROJECTION` only. The
bounded diagnostic reproduced those failures on GPU/XLA and found zero failures
when the exact raw covariance tensors were re-audited on CPU TensorFlow.
Every failing GPU raw covariance was finite, symmetric, and strictly positive
definite; factor reconstruction was far inside the existing gate. Projection
residuals were millions of `tau`, so increasing the multiplier would be an
invalid post-hoc threshold repair.

The code formed `Q diag(lambda) Q^T` unconditionally. The evidence pattern is
consistent only with the CUDA XLA batched eigensolver returning the equivalent
row-vector convention on those points. The repair computes both orientations
and chooses the one that reconstructs the input covariance more accurately,
then uses that orientation for the square root.

## Boundary Review

| Question | Disposition |
| --- | --- |
| Mathematical semantics | Preserved: principal eigen covariance/root after permitted negative-roundoff clipping |
| Threshold tuning | None; `tau`, projection `8*tau`, factor `16*tau`, and material-negative veto are unchanged |
| Target/forecast changes | None |
| Risk of hiding indefiniteness | None; eigenvalue and material-negative status are computed before orientation selection |
| New failure mode | Backend returns neither usable orientation; unchanged projection/factor gates reject it |
| Required evidence | CPU algebra tests plus exact G/H prefix terminal-only trusted GPU/XLA validation |
| Forbidden inference | No forecast, predictive equivalence, sampler, posterior, or model claim |

The current source edit is not sufficient evidence by itself. The next action
is the bounded exact-prefix terminal-only validation specified in the live
Phase 8 plan.

Post-validation addendum: exact-prefix GPU validation reproduced the same six
projection failures after the proposed orientation selection. The hypothesis
was falsified and the production edit was reverted. No tolerance was widened.
Further work must compare alternative decompositions on the exact raw
covariances before another implementation repair is proposed.
