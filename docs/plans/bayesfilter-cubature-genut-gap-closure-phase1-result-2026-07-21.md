# Gap Closure Phase 1 Result

Date: 2026-07-21

Status: `PASS_PHASE1_LOOP_NATIVE_CORE`

## Outcome

The candidate finite value/recursive-score core in
`bayesfilter/highdim/cubature_genut_filter.py` now uses TensorFlow control flow
throughout the traced computation:

- Sinkhorn balancing iterations use `tf.while_loop`;
- Contract E reset tangents are batched over the parameter axis;
- time recursion and per-time score/value histories use `tf.while_loop` and
  `tf.TensorArray`; and
- the runtime module contains no NumPy import, `.numpy()` call, or Python loop.

The core requires a static particle dimension per compiled trace, while the
horizon remains a TensorFlow dynamic dimension. This is intentional for XLA
shape stability and is documented as a route contract.

## Checks

| Check | Result |
|---|---|
| Existing candidate/filter/adapter tests | `13 passed` CPU-hidden |
| Source purity test | Passed: no Python loop, NumPy, or host conversion in `finite_value_score` |
| Multidimensional toy | `d=2`, `N=12`, finite value/score and reset residual |
| Trusted GPU/XLA smoke | Passed after static-particle shape repair |
| Canonical Contract E route | Unmodified |

## Decision Table

| Decision | Status |
|---|---|
| No Python loop in compiled candidate core | Passed |
| No NumPy/host numeric path in compiled candidate core | Passed |
| Same-scalar recursive score mechanics | Passed existing FD gates |
| XLA high-dimensional scaling | Not yet established beyond `d=2` toy |
| Identity source closure | Still incomplete; Phase 2 |
| Default/leaderboard readiness | False; policy unchanged |
| Next justified action | Runtime-closure audit and bounded `d=2/d=4` precision ladder |

## Nonclaims

This result does not establish exact nonlinear filtering, target-horizon
accuracy, FP64 agreement, score precision, method superiority, HMC readiness,
leaderboard admission, or default promotion.
