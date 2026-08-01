# Phase 4 Result: Exact Adapter And Trainer Integration

Date: 2026-07-14

## Outcome

**PASS_PHASE4_AND_CONTINUE_WITH_PROVENANCE_REPAIR.** The exact adapter now
exposes a direct `neutra_batch_log_prob_and_grad_status` method. The repository
binding accepts it, same-regime scalar parity passes, and the generic trainer
completes a real exact-target optimizer update through the bound batch method.
Scalar HMC/parity methods and their stable target/adapter signatures remain
unchanged.

## Evidence

| Check | Result |
| --- | --- |
| Direct real-adapter binding and same-regime parity | pass |
| One-step exact-target optimizer integration | pass; valid status and binding provenance emitted |
| Identity/binding/negative integration group | `17 passed` |
| Scalar LGSSM plus batch-kernel regression group | `15 passed` |
| Python compile and diff hygiene | pass |

The one-step test initially asserted guessed top-level metadata keys after the
optimizer update had succeeded. The canonical provenance already existed under
`runtime_metadata["batch_native_target"]`; the test was repaired to assert that
payload instead of adding duplicate schema fields.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit adapter/trainer integration | real method binds, matches scalar, and reaches an optimizer update | no integration or scalar-compatibility veto | helper source dependency closure not yet in binding identity | repair/certify in Phase 5 | no default-readiness or GPU performance |
| Preserve scalar signatures | existing identity and scalar regressions pass | no target drift | batch capability evolves separately | retain scalar authority and add batch closure hash | no claim that old scalar mapped training is eligible |

## Phase-Transition Finding

The Phase 0 binding records the direct method source hash but not the source of
the repository helper functions invoked by that method. This is adequate for
callable ownership but incomplete for reproducible batch implementation
identity. Phase 5 must derive a non-caller-supplied repository dependency
closure from direct global function calls and bind its hash and module hashes.
Default-readiness is withheld until that repair passes adversarial tests.

## Handoff

Phase 5 starts under
`docs/plans/bayesfilter-neutra-batch-native-training-phase5-correctness-certification-subplan-2026-07-14.md`.

