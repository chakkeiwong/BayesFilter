# Canonical LGSSM Balancing And Kalman Repair Phase 1 Result

Date: 2026-07-17

Status: `PASS_BALANCE_STEPS_50_SELECTED_AND_AUDITED`

Campaign ID: `canonical-lgssm-balancing-kalman-repair-20260717`

## Outcome

The canonical LGSSM caller now binds terminal balancing explicitly, includes
the consumed-plan marginal gate in reset validity, exposes post-quotient
telemetry, and binds the balance count into preparation identity.  The
marginal-only GPU/XLA ladder selected `balance_steps=50`, the first tested count
passing all eight frozen design seeds.  The same count passed all eight
untouched audit seeds once, without retuning.

No Kalman value or score was imported, computed, or inspected by the selection
harness.  This result closes the schedule-selection part of `CE-01`; it does
not establish Kalman agreement, scientific equivalence, a parameter region,
factory-issued canonical identity, HMC readiness, or leaderboard admission.

## Ladder

| Balance steps | Design pass | Max row residual | Max post-quotient column residual | Roundoff tolerance |
| ---: | --- | ---: | ---: | ---: |
| 0 | no | `2.3723e-2` | `2.6846e-2` | `4.8321e-12` |
| 1 | no | `1.0048e-2` | `7.9010e-3` | `4.8321e-12` |
| 2 | no | `5.6450e-3` | `3.9895e-3` | `4.8321e-12` |
| 5 | no | `9.8610e-4` | `6.9459e-4` | `4.8321e-12` |
| 10 | no | `5.3515e-5` | `3.7678e-5` | `4.8321e-12` |
| 20 | no | `1.5759e-7` | `1.1095e-7` | `4.8321e-12` |
| 50 | yes | `4.2188e-15` | `7.1054e-15` | `4.8321e-12` |

The audit maximum residuals were `1.2212e-15` row and `8.8818e-15`
post-quotient column, against maximum tolerance `7.4164e-12`.  The minimum
covariance-gap eigenvalues were positive: `0.0360572` on design and `0.0393230`
on audit.

## Attempt Record

| Attempt | Classification | Result |
| --- | --- | --- |
| 1 | localized harness initialization failure | TensorFlow constants initialized the GPU before the 8192 MiB logical-device cap; no candidate ran; structured failure preserved |
| 2 | valid unchanged experiment | imported project modules only after device configuration; selected 50 and passed untouched audit |

Attempt 1 SHA-256:
`2301ffafee244682f21ae00785c0e72d27041cc875d0487f344de8051a0261fb`.

Attempt 2 SHA-256:
`958a0596b3518f46e0c256f70dc5365230e6dadcf9ff720a996bc692c1309218`.

## Verification

- Selector source-policy tests: `3 passed`.
- Highest-risk focused wiring/derivative assertions: `6 passed`.
- Trusted GPU/XLA: RTX 4080 SUPER, exact `8192 MiB` logical limit, memory
  growth disabled, float64, XLA compiled.
- Attempt 2 total wall time: `161.82 s`.
- Both artifacts parse as JSON and scoped `git diff --check` passes.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Freeze `balance_steps=50` for this campaign | first design pass and untouched audit pass | no marginal, chart, covariance-gap, replay, device, or artifact veto | behavior at larger `N` and longer `T` | run paired Contract E/no-reset particle-count diagnostic | no mathematical optimum, Kalman accuracy, HMC, or leaderboard claim |

## Post-Run Red Team

The strongest alternative explanation is that 50 succeeds only for the frozen
`T=2,N=128` seed sets.  This would invalidate transfer to later rungs but not
the selection record.  The raw residual margin is large relative to the
implemented roundoff envelope, yet the envelope is an engineering backward-
error criterion rather than a general finite-Sinkhorn convergence theorem.
