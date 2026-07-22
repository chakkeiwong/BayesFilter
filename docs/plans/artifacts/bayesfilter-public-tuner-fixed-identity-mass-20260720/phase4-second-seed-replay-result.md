# Phase 4 Second-Seed Replay Result

Snapshot: `2026-07-20`

## Decision

`BLOCKED_RUNTIME_BEFORE_RETAINED_SAMPLING`

The new admitted-kernel artifact was successfully migrated and replay-validated
without tuning. Three direct-replay execution attempts then reached the
TensorFlow/XLA HMC path but terminated before retained samples or a terminal
result were written. No attempt produced posterior evidence.

## Valid Evidence

- Migrated artifact:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260720-replay-migration/admitted_kernel_mechanics.json`;
- mechanics fingerprint:
  `477f9b321817fd292f569913ac6d5233fedf0133df85a8fb0c7bd8c570be28cb`;
- frozen transport SHA-256 matched:
  `b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`;
- replay mechanics: step `0.7779889586003162`, leapfrog count `6`, fixed
  identity mass signature preserved;
- GPU/XLA preflight passed on NVIDIA RTX 4080 SUPER;
- archived warm-up chunks from the second retry are finite and satisfy the
  declared warm-up R-hat gate: maximum R-hat `1.0065277293 <= 1.05`;
- warm-up is explicitly excluded from posterior estimates.

## Attempts

| Root | Durable output | Classification |
| --- | --- | --- |
| `phase4-second-seed-replay` | warm-up chunk 0000 only | runtime interruption before warm-up completion |
| `phase4-second-seed-replay-retry` | warm-up chunks 0000 and 0001 | runtime interruption after valid warm-up, before retained sampling |
| `phase4-second-seed-replay-background-2` | launch log only | detached sandbox process killed after XLA compilation |
| retained-only continuation | no output root | process terminated during retained XLA HMC call |

No `result.json`, retained archive, convergence diagnostic, or truth-tail
artifact exists for the second seed. The partial warm-up tensors must not be
used as posterior or truth-tail evidence.

## Interpretation

This does not reject NeuTra, the target, the admitted kernel, or the tuning
choice. It identifies a runtime/process-boundary failure in the current
TensorFlow/XLA execution harness. The failure is not a scientific continuation
veto, but repeated retries under the same boundary are not productive.

The missing evidence remains exactly one retained second-seed run. A future
attempt should use a host/session arrangement that permits the TensorFlow HMC
call to complete and must preserve the same artifact, transport, seed offset,
and convergence/truth-tail contract. Do not tune again or interpret the
partial warm-up as a second-seed result.
