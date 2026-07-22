# Phase 3 Result: Batch-Native SVD/Eigh Graph-Status Kernel

Date: 2026-07-14

## Outcome

**PASS_PHASE3_AND_CONTINUE.** The new batch kernel evaluates SVD/eigh Kalman
likelihoods, first-order scores, and graph-status fields with one time-axis
`tf.while_loop`, no sample-axis loop/map, no NumPy, and no host callback.

## Evidence

| Check | Result |
| --- | --- |
| Phase 3 focused suite | `6 passed` |
| Phase 3 plus scalar LGSSM/SVD regressions | `25 passed` |
| Eager batch versus eager scalar value | max absolute difference `4.3e-14` |
| Eager batch versus eager scalar score | max absolute difference `3.6e-14` |
| CPU-XLA batch versus CPU-XLA scalar value | max absolute difference `1.6e-13` |
| CPU-XLA batch versus CPU-XLA scalar score | max absolute difference `9.6e-14` |
| Regular status fields | scalar parity pass |
| High-floor status | scalar parity pass |
| Mixed valid/invalid batch | `[0,2,0]`, valid rows bitwise unchanged |
| Row permutation | pass |
| Graph topology | exactly one `While`; no map or host callback |
| Source, compile, and diff hygiene | pass |

The initial test compared CPU-XLA batch output directly with eager scalar
output and observed approximately `3.8e-7` likelihood and `1.0e-6` score
differences over 120 steps. A focused diagnostic showed that eager batch matches
eager scalar and XLA batch matches XLA scalar at near-machine precision. The
difference is CPU-XLA execution ordering, not a batch formula mismatch. The
test was repaired to preserve both evidence classes rather than loosen one
mixed-regime tolerance.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit batch kernel | same-regime scalar value/score/status parity | no mathematical/status veto | trusted GPU XLA behavior not measured | bind exact adapter | no GPU speed or training claim |
| Preserve per-row guard | mixed invalid row does not contaminate valid rows | no cross-row contamination | invalid row payload itself remains diagnostic-only | apply scalar NaN gate in adapter | no validity for blocked rows |

## Handoff

Phase 4 may start under
`docs/plans/bayesfilter-neutra-batch-native-training-phase4-exact-adapter-integration-subplan-2026-07-14.md`.

