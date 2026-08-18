# GenUT Feasible Trust-Region Repair Plan Review

Date: 2026-08-15

## Review Verdict

`PASS WITH EXPLICIT LIMITATIONS`

The plan asks the right numerical question: whether the non-finite higher-
moment correction can be replaced by a finite, differentiable map without
changing the declared empirical target. The baseline is the historical
unscaled normal-equation route, and the candidate is the column-scaled
Levenberg--Marquardt solve plus smooth RMS trust cap.

## Skeptical Audit

| Risk | Audit finding | Required handling |
|---|---|---|
| Wrong baseline | Historical route is retained as comparator; no hidden target clipping | Keep exact target and label legacy route historical |
| Literature overclaim | Ebeigbe constrained GenUT and Easley--Berry HOUT are checked direct competitors, but neither is a drop-in positive equal-weight differentiable reset | Describe the selected route as a local composition, not rediscovery |
| Proxy promotion | Finite values, residuals, and score magnitudes are diagnostics, not posterior or HMC evidence | Do not admit NeuTra/HMC from this plan |
| Missing stop conditions | Non-finite, scalar/batch mismatch, derivative failure, stale identity, and unavailable required inputs are vetoes | Preserve fail-closed status |
| Hidden route mismatch | The old step-169 checkpoint has a different target signature/source route in the current checkout | Do not call it an exact current-route replay |
| Small-count unfairness | `N=12` is invalid for Austria `d=18`; valid `N=36` is still a low-particle regime | Record model failures as scope diagnostics only |
| Hardware mismatch | CPU and RTX 5080 TensorFlow builds differ | Treat current GPU permission timeout as unavailable, not pass/fail numerical evidence |
| Unreviewed defaults | LM damping, scale floor, trust radius, correction count, and small-screen counts are hypotheses | Use diagnostic-only labels; require scope tuning before promotion |

## Execution Decision

The plan was safe to execute for documentation, focused mechanics, and bounded
CPU diagnostics. It was not safe to claim broad model validity, default
readiness, NeuTra training, HMC convergence, or GPU parity. The result note
records those boundaries and the next evidence required.
