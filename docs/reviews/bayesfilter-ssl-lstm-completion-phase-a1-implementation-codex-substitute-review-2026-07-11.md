# Phase A1 Implementation And CPU Evidence CODEX_SUBSTITUTE_REVIEW

Date: 2026-07-12

Review type: `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude review.
Claude remained policy-unavailable; no Claude process ran and no repository
content was sent.

## Exact Reviewed Files

| Role | Path | SHA-256 |
| --- | --- | --- |
| Production target | `bayesfilter/nonlinear/ssl_lstm_posterior_tf.py` | `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667` |
| Lazy exports | `bayesfilter/nonlinear/__init__.py` | `9bfbe2a912b6465e8338d61c48c51b91b2b30d1f11912a543772e5901998de68` |
| Focused tests | `tests/test_ssl_lstm_posterior_tf.py` | `9635074e50f47b321e946707770503480e43b2bad78d2963d155127569ac25ca` |
| Evidence harness | `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a1_masked_posterior_2026_07_11.py` | `94d232114395438b743f4cc06ff7a5b806df28c82016d1af9d9bea4da7061440` |
| Golden signatures | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/golden-signatures.json` | `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34` |
| Historical comparator | `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py` | `fea73716e1d972a5336e3bdedb733dfc31c4a0bb61cf40cdf877d577d68cbe28` |
| CPU evidence | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json` | `b6dc26637d584dbf6d62575a999af5cf43bb7bab35a5cf9eb6984d1cfaf6a068` |

CPU evidence signature:
`c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d`.

## Findings

No material findings.

- The implementation preserves the locked four-parameter historical SVD-UKF
  target and unnormalized prior.
- Nonfinite input rejection is graph-native and does not swallow finite filter
  failures.
- Production HMC and full-chain XLA readiness remain false.
- All requested hashes and the CPU evidence signature match.
- The CPU artifact records ten passing historical, finite-difference, and
  eager/CPU-XLA checks plus all three reject cases while retaining the required
  nonclaims.
- The harness binds source and boundary hashes, validates strict schemas, and
  supports proceeding only to the bounded ten-point trusted GPU/XLA canary.

## Boundary

This review authorizes only the plan's trusted ten-point GPU/XLA target canary.
It does not establish posterior correctness, HMC/NeuTra readiness, predictive
equivalence, calibration, model adequacy, performance, or default/product/
release readiness.

VERDICT: AGREE
