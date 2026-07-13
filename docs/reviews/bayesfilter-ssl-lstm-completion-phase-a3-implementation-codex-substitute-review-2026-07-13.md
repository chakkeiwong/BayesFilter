# BayesFilter SSL-LSTM Completion Phase A3 Implementation Review

Date: 2026-07-13

Review class: `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude.

Claude remained policy-unavailable. No A3 source was sent to Claude, and this
record must not be described as Claude convergence.

## Reviewed Production Paths

The implementation was reviewed as two separate bounded exact-path reviews:

| Path | Accepted SHA-256 | Final verdict |
| --- | --- | --- |
| `bayesfilter/testing/scalar_lgssm_forecast_oracle.py` | `74889d699e3575ee163c64d9a67325f0376e161106e9b36fb6b61453c3a5eb43` | `AGREE` |
| `bayesfilter/inference/predictive_equivalence.py` | `99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16` | `AGREE` after repair |

Bound focused tests:

| Path | SHA-256 |
| --- | --- |
| `tests/test_scalar_lgssm_forecast_oracle.py` | `977134cbc92b63ca6d8dab7a1e6ca25eb58137cb27430518a1aacc120cecfab8` |
| `tests/test_predictive_equivalence.py` | `5e6a137c12b3131c8ff7471d74abd4a877777ef6432a2c51f5c62cceedf9290d` |

## Oracle Review

The bounded oracle review found no material defect after earlier local repairs.
The accepted implementation separates state and observation covariance
symmetry/PSD diagnostics, exposes degenerate-variance and log-variance status,
uses TensorFlow/TFP `float64`, defaults algorithmic paths to XLA, and preserves
direct equation-level simulation independently of the predictive-statistics
implementation.

## Statistics Round 1

Verdict: `REVISE` on source SHA-256
`21f1664f66082ee056e396a4cdd131d95cc6d04c8f2727084f0a37e1ed8b454c`.

Material findings:

1. Public interval dataclasses could be directly constructed with arbitrarily
   narrow, internally symmetric bounds and then manufacture `PASS` without
   proving that their critical values came from the declared method and alpha.
2. The IID quadratic-MMD role was admitted from a caller-provided string, so a
   dependent or shared-bank rank-two sample could be mislabeled inferential.
3. Forbidden scale-floor use could fail open when `tf.get_static_value(...)`
   returned `None` for a dynamically valued Tensor.

## Visible Repair

- Feature intervals, cross-chain linear-MMD results, and MMD intervals now
  receive constructor-bound fingerprints plus process-local live-object
  authentication. Direct construction, `dataclasses.replace`, copy,
  reconstruction, mutation, replay, and stale Python object identity cannot
  authenticate or emit `PASS`.
- Fresh-process artifact verification must recompute constructors and
  classification from materialized tensors; serialized result containers are
  not authentication authority.
- Quadratic MMD U/V forms are unconditionally descriptive with
  `inference_admissible=False`, regardless of caller IID labels or booleans.
  Decision-bound MMD inference is exclusively the authenticated cross-chain
  linear estimator.
- Forbidden scale-floor use is enforced by a runtime TensorFlow assertion linked
  to the standardization execution path.
- A scale-aware degeneracy gate rejects zero, nonfinite, and roundoff-scale
  long-run MMD uncertainty. This uncovered and replaced an old nearly
  deterministic test fixture rather than weakening the veto.
- The repaired tests include dynamic graph-valued floor rejection, forged and
  copied result objects, descriptive-only quadratic MMD, authenticated decision
  branches, and roundoff-degenerate MMD uncertainty.

## Statistics Round 2

Verdict: `AGREE` on source SHA-256
`99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16`.

The fresh bounded reviewer found no material authentication, interval-algebra,
quadratic-MMD role, runtime scale-floor, XLA-default, or degeneracy-veto defect.
The review explicitly does not infer runtime success or statistical validity of
any future SSL-LSTM comparison.

## Local Verification

- Combined CPU-hidden oracle/statistics suite: `65 passed`.
- Statistics-only hardened suite: `44 passed`.
- In-memory compilation: passed for both production paths and both focused test
  paths.
- `git diff --check`: passed on the four implementation/test paths.
- Forbidden NumPy/PyTorch/JAX algorithmic backend and stateful-RNG scan: clean.
- No repository A3 bytecode or pytest cache was created.

## Claim Boundary

This review supports only the bounded A3 oracle/statistics implementation gate.
It does not establish SSL-LSTM predictive equivalence, posterior correctness,
sampler validity, HMC or NeuTra readiness, calibrated A4 constants, production
readiness, a public/default policy, or scientific validity.

VERDICT: AGREE
