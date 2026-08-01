# P4 Terminal-State Correction: Predator-Prey Tuning Admission

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `HISTORICAL_CORRECTION_RESOLVED_BY_P4_R4_REPAIR`

This note records the valid P7 attempt-01 downgrade. It is no longer the active
terminal state: the bounded repair below completed with fresh evidence and is
superseded by the refreshed P4 result and P7 attempt 02.

## Correction

The P7 exact-schema audit found that both predator-prey R4 runs selected their
fixed transported-HMC kernel directly from short probes by maximum minimum bulk
ESS. Their selected short probes had modern R-hat above the current fixed-
kernel admission limit and neither run performed a disjoint modern-R-hat
verification:

| Cell | Selected short-probe modern R-hat | Required disjoint verifier | Corrected state |
| --- | ---: | --- | --- |
| `PP-UKF` | `1.0324311905` | `<=1.01` | `EVIDENCE_BLOCKED_TUNING_ADMISSION` |
| `PP-SGQF` | `1.0454615203` | `<=1.01` | `EVIDENCE_BLOCKED_TUNING_ADMISSION` |

This correction supersedes the `NEUTRA_CONFIRMED` terminal states in
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p4-predator-prey-result-2026-07-16.md`
and the P4 phase-close ledger. Historical files and raw evidence remain
unchanged.

## What Remains Valid

The target, transport, training, comparator, archive, hash, and final-sampler
evidence reverified. Both final R4 retained runs had 4,000 draws per chain,
finite health/status, no declared energy divergences, modern R-hat below
`1.01`, bulk/tail ESS above thresholds, and passed their prospective six-mean
agreement rule. Those facts remain diagnostic evidence. They do not repair the
missing tuning admission or support terminal confirmation under the current
runbook.

## Repair Boundary

The earliest re-entry rung for both cells is R4 tuning admission. A valid
repair must use the same typed target, frozen transport, comparator, physical
mean rule, and thresholds; order candidate kernels using short probes only,
admit one through a disjoint 1,000-burn-in/1,000-draw modern-R-hat `<=1.01`
health-valid verifier, then run a fresh warm-up and retained confirmation under
new seeds and a fresh output root. Prior retained draws cannot be relabeled as
post-repair confirmation.

No target invalidity, transport invalidity, full-distribution agreement,
filter exactness, superiority, calibration, robustness, or readiness claim is
made.

## Repair Result

Both cells followed the boundary above under the unchanged scientific
contract. Hash-verified old probes were used only to order candidates. Step
`0.20` then passed a fresh disjoint 1,000-burn-in plus 1,000-draw verifier for
each cell, followed by fresh separately archived warm-up and retained draws.

| Cell | Verifier modern R-hat | Final modern R-hat | Final retained draws per chain | Active result SHA-256 | Active state |
| --- | ---: | ---: | ---: | --- | --- |
| `PP-UKF` | `1.0054056853` | `1.0008110775` | `4,000` | `d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf` | `NEUTRA_CONFIRMED` at six-mean scope |
| `PP-SGQF` | `1.0013382279` | `1.0003275699` | `4,000` | `a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4` | `NEUTRA_CONFIRMED` at six-mean scope |

Both final runs also passed bulk/tail ESS, health/status, zero declared energy-
divergence, and all six prospective simultaneous physical-mean bounds. This
closes only the historical tuning-admission defect; the original failed state
and attempt-01 P7 artifacts remain preserved as the trigger evidence.
