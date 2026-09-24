# BayesFilter Current Blockers - 2026-09-08

Branch: `ledh-refactor-with-policy-fix`
Baseline checkpoint: `76f09a6d`

## Current scientific status

The canonical single-cloud LEDH--OT--GenUT Contract-E dual-cap endpoint has an
analytical total directional derivative for its executed finite scalar,
`ATOM-FINITE`. Phase 4A is also fully differentiated for its different
kernelized-observation scalar, `KDM-FINITE`, but the authoritative Attempt 05
LGSSM result gives no promotion evidence: every positive bandwidth has worse
point MSE than ATOM-FINITE on the consumed holdout, and `rho=1.6` is
statistically worse.

Phase 4B is no longer blocked by an unspecified mixture law. The separate
`RESKDM-IWSG-FINITE` reference now implements the complete fixed-anchor,
all-components Gaussian-mixture resampling operation after Contract-E and both
GenUT caps. Its analytical derivative matches finite differences of that
declared scalar on CPU, and its state, raw `r/N` weight, and covariance-mark
outputs reach the next PF--PF step. A source and official-code audit found that
the first implementation had prematurely self-normalized those ratios. That
older route differentiated `RESKDM-SN-FINITE`, not the intended raw IWSG
normalizer. This does not establish that the repaired route estimates the
exact model score better.

## Active blockers

### 1. Phase 4B score quality

Status: scientific evidence missing, not an implementation failure.

The existing float64 and float32/no-TF32 GPU/XLA smokes predate the raw-ratio
repair and certify only `RESKDM-SN-FINITE`. Fresh smokes are required for the
repaired route. No paired Kalman-oracle campaign has measured bias, variance,
or MSE for `RESKDM-IWSG-FINITE`. The next
campaign must use fresh calibration and validation paths, a prospective power
calculation, the bootstrap PF and Phase 4A comparators, and a prespecified
responsibility-mean versus selected-label covariance-mark ablation.

### 2. Degenerate DSGE support

Status: not implemented for Phase 4B.

The current Phase 4B reference uses a full-rank common Gaussian bandwidth in
an ordinary LGSSM. A DSGE route needs a density with respect to the stochastic
innovation/chart measure and the total derivative of any parameter-dependent
chart, support map, or Jacobian. An ambient full-rank kernel is wrong for a
degenerate transition.

### 3. Batch-fused and NeuTra conformance

Status: separate implementation-conformance blocker.

`bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` still bypasses the full
Contract-E, GenUT, and dual-cap recurrence. Its passing batch tests establish
batch mechanics, not equivalence to the canonical algorithm. It cannot support
canonical, NeuTra-training, or HMC claims until a batch-native full recurrence
and executable parity test exist.

### 4. Numerical admission scope

Phase 4A remains vetoed under TF32 by its identity/parity thresholds. Phase 4B
has been checked only in float64 and float32 with TF32 disabled. Neither route
has HMC or production admission evidence.

## Resolved issues

- The original partial/conditional derivative was wrong relative to the stated
  total-score claim. The canonical endpoint now carries state, weight,
  covariance, observation-Jacobian, covariance-density, reset, and cap
  dependencies through the actual recurrence.
- Contract-E covariance carry is wired through the reset and consumed by the
  next UKF prediction.
- The Phase 4B proposal law, ancestry rule, denominator, and finite target are
  explicit. The active target is `RESKDM-IWSG-FINITE`, not `ATOM-FINITE`,
  `MODEL-IS`, or the superseded `RESKDM-SN-FINITE`.
- The raw IWSG correction now enters the next normalizer as `log(r/N)` with
  uncentered tangent `d log r`. Posterior normalization occurs after the next
  model factors, matching Younis Eqs. (14)--(15) and the released author code.
- The sequential Phase 4B API now enforces the common-bandwidth assumption in
  the LaTeX document and rejects asymmetric bandwidth/covariance-mark values
  or tangents rather than silently changing them.
- The dense-versus-streaming discrepancy was diagnosed as floating-point
  order sensitivity in the tested setting; it is not an active KDM blocker.

## Next step

Rerun the repaired Phase 4B GPU/XLA smokes for both covariance-mark policies.
Then write the paired LGSSM campaign amendment, audit its power and comparator
design, and run the smallest adequately powered cell. Do not port
the route, extend it to DSGE, or use it in HMC before that score-quality result.
Batch-fused conformance remains an independent engineering program and is not
repaired by any KDM result.
