# Phase 8 Rung 0B Result: Tiny-Fixture Kalman Oracle Harness

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`TINY_ORACLE_HARNESS_PASSED_DESCRIPTIVE_COMPARISON_ONLY_TARGET_PREPARATION_BLOCKED`

## Outcome

A dedicated CPU-hidden/XLA Rung 0B harness now compares only the canonical
Contract E callable with the repository TensorFlow Kalman likelihood. It uses
the frozen Phase 5 `B=2,N=4,T=2` fixture, whose residual design, prepared ridge,
observations, noise, mask, and transport settings are source-bound literals.
The harness also computes an independent stationary joint-Gaussian likelihood
for the same observations and model.

The Kalman and joint-Gaussian values agree exactly and their physical gradients
agree to floating-point roundoff. The canonical tiny fixture is finite,
chart-valid, center-repeatable, and bound to one concrete value-and-score
callable. Canonical-versus-Kalman differences are recorded below as explanatory
only; no tiny-fixture equivalence margin is applied.

## Evidence

| Quantity | Contract E | Kalman oracle | Difference |
| --- | ---: | ---: | ---: |
| Value | `-5.3333917621985805` | `-5.312272244921968` | `-0.02111951727661232` |
| Relative value difference / `abs(Kalman)` | | | `-0.003975608986681846` |
| `phi1` physical score | `-0.44685258073797396` | `-0.38874016537108436` | `-0.0581124153668896` |
| `phi2` physical score | `-0.22403176890798468` | `-0.16527408007631963` | `-0.058757688831665045` |
| `phi3` physical score | `0.056158371488133435` | `0.1207573617896066` | `-0.06459899030147316` |
| `q_scale` physical score | `-3.552222320527654` | `-3.764369782283386` | `0.21214746175573174` |
| `r_scale` physical score | `-4.746981132539256` | `-4.804383870097376` | `0.05740273755812009` |

Declared HMC-coordinate differences are
`[-0.043584311525167174, -0.05508533327968598, -0.06056155340763109,
0.10607373087786587, 0.04305205316859029]`. They are not compared with the
unfrozen Phase 8 target margin.

The independent joint-Gaussian cross-check differs from Kalman by value `0.0`
and physical-gradient differences no larger than `8.88e-16` on this fixture.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Rung 0B harness correctness | finite, valid, repeatable canonical call plus independent oracle cross-check | Passed for this tiny fixture | No target-shape preparation or telemetry | Freeze target-shape residual/ridge preparation and telemetry design | General numerical correctness |
| Tiny Contract E versus Kalman | descriptive diagnostic only | No equivalence margin exists for this fixture | Finite-`N`, reset, ridge, and residual bias | Do not promote or reject from this result | `T=50,N=10000` equivalence |
| Advance to target Rung 1 | Requires prepared-input design and formal FD contract | Blocked | Residual design/ridge defaults and endpoint error bounds | Write/review Rung 1 preparation subplan | GPU, HMC, leaderboard |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Canonical and both oracle values/gradients finite; chart and one-callable checks pass |
| Statistically supported ranking | None; no stochastic target experiment was run |
| Descriptive-only differences | All canonical-versus-Kalman tiny-fixture differences |
| Default-readiness | Not established |
| Next evidence needed | Target-specific fixed residual/ridge preparation, telemetry, formal FD contract, then lower-rung T=1/T=10 comparisons |

## Artifacts

- Harness: `docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_rung0b.py`
- Tests: `tests/highdim/test_contract_e_phase8_rung0b_harness.py`
- Result JSON: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung0b-attempt1/tiny-fixture-kalman-comparison.json`
- SHA-256: `4bd0e4d35fe88ce835506660452b25372ffe27a0c9cd76173cb1338be6daadd2`

## Nonclaims

This result is not a target `d=3,T=50,N=10000` result, not a Kalman-gradient
equivalence decision, not a ridge/residual default, not formal Phase 1 FD
certification, and not GPU, HMC, admission, leaderboard, release, or integrity
readiness evidence.
