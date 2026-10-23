# SIR safety checks: placement and repair design

22 October 2026; branch `ledh-refactor-with-policy-fix`, HEAD `90a3acebd`.
This answers whether the earlier consumer was equally safe and how to recover
speed. It is a source-reviewed design, not an implemented optimization or a new
GPU experiment. The retained candidate02 runtime remains unchanged.

## What the earlier implementation actually checked

The frozen pre-merge source is
`artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/before-source/`.
Its `bayesfilter/highdim/ledh_canonical_score_tf.py:839` combines UKF,
reset, callback, state/covariance and final value/score validity at observation
boundaries. Its trace at lines 891–897 exposes flow and reset finite checks;
`bayesfilter/score_study/moment_training_tf.py:91` consumes those checks in the
measured endpoint. The old program therefore already rejected many failures.

However, its flow at lines 1245 and 1324 uses raw Cholesky and QR, without the
current per-substep failure masks. The optimized plain and conditional consumers
do not execute the unused QR determinant or its derivative. These were not the
same checks relocated outside the loop. The current full-matrix symmetry check,
for example, detects corruption of an otherwise unused matrix triangle that a
raw triangular factorization and finite-output test need not detect.

Equal healthy outputs do not prove identical rejection on invalid inputs. Neither
"the old code had no safety" nor "the two versions are equally safe everywhere"
is supported. The [dependency trace](artifacts/ledh-sir-remaining-overhead-trace-20261022/result.md)
establishes added work, not that every added computation is necessary for safety.

## Put each check at the lifetime of its inputs

| Quantity or event | Appropriate placement |
| --- | --- |
| Shapes, dtype, static schedule, fixed reset design | Factory/configuration boundary |
| Parameter-dependent Q, R and their tangents | Once per parameter evaluation, outside the observation and flow loops |
| Predicted P and its tangent, fixed during one flow | Once per observation, before the flow; establish any needed definiteness premise explicitly |
| Moving H, D = R + lambda H P H^T, actual states/tangents | Check computed values/factors locally; these change at every substep |
| Failure history | Absorbing device-side booleans and first-failure location; a later finite value cannot erase an earlier failure |
| Host reporting and final acceptance | At the endpoint; no per-substep host synchronization |

Q/R factor checks are already outside the loops at current lines 342–343; moving
them again cannot explain or fix the measured regression. Current `valid_predict`
also cannot be assumed to certify the newly predicted P: in
`ledh_canonical_score_stages_tf.py:502–531` it originates from the input sigma-point
factor, before construction of the predicted covariance.

Retain local placeholders where an invalid input would otherwise enter an unsafe
solve or a model-domain operation. Separate that requirement from whether an
unrelated determinant diagnostic must gate the next valid state. Merely testing
the last output, or moving a final boolean reduction to Python, loses transient
failure information and does not remove the work needed to compute that boolean.

## Remove redundant work by proof and by the consumed density

The plain Gaussian consumer evaluates the transported Gaussian using its final
affine map and covariance; the fully conditional consumer uses its conditional
Gaussian density. Their density formulas do not need the accumulated substep
log determinant. The physical mixture with a nonzero LEDH component does need it
(`ledh_physical_proposal_tf.py:128` onward). Some other shared consumers need it
as well. Retain one shared implementation and select required quantities from the
repository-owned consumer definition, never a SIR-only shortcut or a caller's
claim that a check succeeded. Explicit determinant diagnostic requests must still
receive the actual diagnostic, not an invented zero.

There is a useful conditional proof for avoiding a second factorization solely
to test invertibility. Write h = 1/L, lambda_j = jh, S = H P H^T,
D = R + lambda_j S, K = P H^T D^{-1}, and M = I - h K H/2. Assume
P is positive semidefinite and R is positive definite. Then D is positive
definite and

    0 <= D^{-1/2} S D^{-1/2} < I/lambda_j.

Sylvester's identity and similarity give

    det M = det(I - h S D^{-1}/2)
          = det(I - h D^{-1/2} S D^{-1/2}/2) > 0,

because every eigenvalue of the last matrix exceeds
`1 - h/(2 lambda_j) = 1 - 1/(2j) >= 1/2`.
Thus, under these premises, each mathematical flow map is invertible without
requiring a fresh determinant factorization as an independent certificate.

This does not certify floating-point conditioning, finite analytical tangents,
or a covariance produced by a UKF with negative covariance weights. Check the
premises at their proper boundary and retain finite/factor checks on the actual
computed path. A successful Cholesky of D alone does not prove P is PSD. If the
premises or an equivalent rejection argument cannot be established, keep the
generic determinant guard. The current mask at lines 1470–1482 must not simply
be deleted on the strength of healthy-case timing.

For consumers that require determinants, reuse existing factorization work and
reduce redundant solves. The previously tested Cholesky-ratio candidate is
algebraically justified but has inconclusive timing for three conditions; it is
not a demonstrated speed fix. Combining tangent solve right-hand sides is also
a candidate, not an automatic floating-point-equivalent change.

## Bounded implementation order and acceptance

1. Specify value and force rejection separately for each real consumer; inventory
   existing boundary checks and transient failures before moving guards.
2. Keep cheap local validation and absorbing failure status. Decouple state
   progression from unrelated accumulated determinant diagnostics only where
   invalid-input handling remains defined and rejection is preserved.
3. Establish the above premises or an equivalent consumer-level rejection proof
   before pruning any unused determinant. Use the shared implementation and test
   every affected consumer, including determinant-bearing physical mixtures.
4. Verify healthy values, raw moments, analytical scores and flags against frozen
   candidate02. Inject nonfinite/indefinite/asymmetric covariances, bad tangents,
   invalid intermediate states and singular maps outside the proved domain.
   Every previously rejected relevant failure must remain rejected; any changed
   rejection domain needs an explicit mathematical explanation and review.
5. Inspect optimized HLO for the actual endpoint, then run paired timing with
   uncertainty against candidate02 and pre-merge under the same scope. Treat
   graph counts as explanatory only. Preserve OT (1,5), data, seeds and controls.
6. Investigate plain-score sensitivity independently. A faster implementation
   does not resolve the pre-merge discrepancy or prove that either version is
   more accurate. Keep the checked-reference task and existing tolerances.

Skeptical review: blanket hoisting fails because H/D change inside the flow;
boundary finiteness alone misses transient failures; trusting `valid_predict`
would use the wrong covariance; pruning without preserving the actual density
would change the target. The staged design addresses these flaws. It passes as
a repair design; it is not a finding that a reduced-check runtime is safe.

Future execution must freeze exact commands, fault cases, paired timing criteria
and attempt budget before launch. The previous candidate stage is complete;
this note launches no further attempt. Current work is read-only source review
plus documentation, consuming zero GPU time. Remaining allocation is 4,424.451901
local GPU-process seconds (4,460.221424 in the parent campaign). No default,
threshold, accepted density, runtime guard, training setting or scientific
promotion changes here.
