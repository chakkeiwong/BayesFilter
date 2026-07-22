# Phase 2 Subplan: Typed Identity Schemas And Canonical Hashing

Date: 2026-07-11

Status: `READY_FOR_REVIEW`

## Phase Objective

Implement typed, versioned transition, execution, provenance, and artifact
identity primitives from validated runtime objects, without yet changing the
Phase 6 serializer or Phase 7 validator.

## Entry Conditions

- Phase 0 passed with external-review limitation recorded.
- Phase 1 classified every current replay/runtime field and found no unknown
  consumer.
- The P7G legacy blocker remains active; no baseline is adopted.
- Core HMC identity source files have no unexpected concurrent edits.

## Required Artifacts

- New focused module `bayesfilter/inference/hmc_identity.py`.
- Public exports only where required by Phase 3 integration.
- Focused tests under `tests/test_hmc_identity.py`.
- Phase 2 result and reviewed Phase 3 integration subplan.

## Required Design

Implement:

1. `CanonicalArrayIdentityV1` from a validated array, with declared canonical
   dtype, shape, C-order byte rule, and bytes SHA-256.
2. `CanonicalFloat64V1` using the exact IEEE-754 big-endian bit pattern.
3. `FrozenHMCTransformIdentityV1` from a validated adapter transform, binding
   ordered runtime routes/base relation and exact center/factor identities.
4. `FrozenHMCTransitionIdentityV1` built from the reconstructed final adapter,
   final kernel mechanics, target-bearing fixture hash, target capability, and
   exactly two ordered transform layers for this lane.
5. `FrozenHMCExecutionContractV1` built from a transition identity plus a
   normalized Phase 7 execution/stopping-policy payload.
6. `SelectionProvenanceIdentityV1` over a declared provenance payload.
7. exact canonical artifact-integrity helpers.

All types must validate schema/version, finite values, dimensions, non-empty
signatures, and allowed fields. Payload methods must be deterministic and hashes
must be computed from those payloads.

The transition builder must traverse the actual reconstructed adapter chain.
It must not accept a caller-maintained list of JSON paths or ignored keys.
Historical full adapter signatures must be carried as validation cross-links
outside the transition hash because their current construction includes mass
provenance and nonclaims.

## Required Checks And Tests

- Unit tests for deterministic round-trip payloads and hashes.
- Array tests for dtype, shape, byte order, C/F layout, and one-bit changes.
- Float tests for exact adjacent float64 values and nonfinite rejection.
- Transform-order and adapter-signature mutation tests.
- Transition mutation tests for target, fixture, dtype, step, leapfrog count,
  transform center/factor, and kernel family.
- Provenance-only mutations must not change transition identity.
- Execution-only mutations must change execution identity without changing
  transition identity.
- Unknown/missing schema and extra-field payloads must fail closed.
- Python compile, focused pytest, forbidden-route scan, and scoped
  `git diff --check`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can typed identity primitives represent exactly the classified runtime semantics with deterministic, adversarially sensitive hashes? |
| Baseline | Phase 1 source-anchored field ledger and current validated replay object model. |
| Primary criterion | All required types validate and all mutation/round-trip tests pass. |
| Vetoes | JSON-key allowlist, raw-payload mechanical projection, dtype/byte-order loss, transform omission/reordering tolerance, unknown schema acceptance, or source edits outside scope. |
| Explanatory only | Hash values, test runtime, and module size. |
| Not concluded | No serializer/validator integration, legacy migration approval, Phase 7 readiness, convergence, or scientific claim. |

## Forbidden Claims And Actions

- Do not modify Phase 6 artifacts, serializer, or Phase 7 validator in this
  phase.
- Do not update legacy pins or create an adoption decision.
- Do not run HMC, smoke, serious sampling, Phase 8, or NeuTra.
- Do not claim old/new transition equality.

## Exact Handoff

Phase 3 may start only when identity primitives and focused adversarial tests
pass, the Phase 2 result records exact scope and residual risks, and the Phase
3 integration subplan passes a fresh findings-first review.

## Stop Conditions

- An identity cannot be derived from actual validated runtime objects.
- Canonicalization cannot distinguish a classified mechanical mutation.
- Tests reveal runtime/identity drift that requires revising Phase 1.
- Scoped files change unexpectedly.
- Review does not converge after five substantive rounds.
