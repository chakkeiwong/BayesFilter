# LGSSM NeuTra Gap Closure Phase 3 - Frozen Transport Validation

Date: 2026-07-15  
Status: `COMPLETE_PASS`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Bind both fresh frozen payloads to the exact LGSSM target and prove stable
compiled forward, logdet, target value, and transformed score behavior across
trusted GPU/XLA and deliberate CPU-hidden/XLA execution before tuning.

## Entry Conditions

- Both 5,000-step training candidates passed their strict or recovered strict
  closure with immutable payload hashes recorded in the Phase 2 result.
- The target and adapter signatures remain the campaign constants.
- The TensorFlow-only HMC integration smoke passed in Phase 0.

## Required Artifacts And Checks

For each candidate, run the same deterministic `[4,18]` latent probe on:

- trusted GPU `/GPU:0`, memory growth, float64, XLA, no soft placement; and
- deliberate CPU-hidden `/CPU:0`, float64, XLA.

Record and compare transport output, log determinant, transformed target value,
transformed score, target status, artifact/transport/adapter signatures, file
hashes, devices, and second-call determinism. Tolerances are `1e-12` for
transport/logdet and `1e-8` for target value/score. All outputs must be finite,
all target statuses valid, and identities exact.

## Evidence Contract

Passing establishes that a particular frozen training artifact computes the
same checked finite transformed objective on both execution targets. It does
not establish chain movement, tuning, convergence, agreement, recovery, or
scientific superiority.

## Forbidden Claims And Actions

Do not tune or sample seriously before Phase 3 closes. Do not change payloads,
target, probe, tolerances, or device policy. Do not use a CPU/GPU difference to
rank training seeds. Do not fall back to eager or non-JIT execution.

## Handoff Conditions

Freeze candidate records and HMC seed families only after all probes finish.
At least one parity-valid candidate advances to Phase 4; a failed candidate is
rejected without stopping another valid candidate.

## Stop Conditions

Stop for target/payload identity corruption, common nonfinite/status failure,
or no candidate passing cross-device parity. A localized serialization or
device-harness failure triggers focused repair and retry under the same budget.

## Suitability Review

The probes directly test the boundary not answered by training parity, use the
actual fresh payloads, and cannot silently substitute a proxy metric for HMC
validity. Verdict: `SUITABLE_TO_EXECUTE`.
