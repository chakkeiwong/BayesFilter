# Phase 2 Result: Typed Identity Schemas And Canonical Hashing

Date: 2026-07-11

Status: `PASSED_TO_PHASE3_INTEGRATION_PLANNING`

## Direct Verdict

Phase 2 passed. BayesFilter now has strict, typed primitives that distinguish
the mathematical HMC transition, deterministic Phase 7 execution, tuning
selection provenance, canonical payload integrity, and exact file integrity.

The implementation does not change the Phase 6 serializer, the Phase 7
validator, any existing artifact, or the active legacy baseline. The P7G
blocker therefore remains active until later reviewed integration and explicit
human adoption.

## Implemented Artifacts

- `bayesfilter/inference/hmc_identity.py`
- `tests/test_hmc_identity.py`
- Iteration 1 implementation review:
  `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase2-implementation-codex-review-iter1-2026-07-11.md`
- Converged review:
  `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase2-implementation-codex-review-iter2-2026-07-11.md`

## Identity Design

### Transition

`FrozenHMCTransitionIdentityV1` binds fixed TFP HMC/integrator route constants,
the exact deterministic LGSSM target inputs consumed by the live base adapter,
float64 state, the two ordered affine transform routes and exact array bytes,
the exact float64 step bits, and leapfrog count.

Whole fixture hashes, full adapter signatures, mass provenance, stage lineage,
acceptance, timings, and nonclaims are not transition owners. Full historical
signatures remain required future reconstruction-integrity cross-links.

### Execution

`FrozenHMCExecutionContractV1` binds serious versus smoke mode, exact global
initial-state bytes, worker/chain partition, seed formula and stage indices,
compile-probe behavior, CPU/thread environment, TensorFlow/XLA route, derived
static chunk size, burn-in/retained schedules, diagnostic definitions and
thresholds, hard-veto/stopping/no-resume policies, wall cap, and runtime
versions.

### Provenance And Integrity

`SelectionProvenanceIdentityV1` stores typed hash-only lineage rather than an
arbitrary nested mapping. `canonical_artifact_payload_hash` hashes a strict
type-tagged canonical tree and exact float64 bits, including nonfinite
explanatory diagnostic values. `artifact_file_sha256` covers exact serialized
bytes.

## Checks Actually Run

Environment: deliberate CPU-hidden diagnostic execution with
`CUDA_VISIBLE_DEVICES=-1`; no HMC transition, GPU workload, sampler, or
experiment ran.

| Check | Result |
| --- | --- |
| Python compilation | Passed for implementation and tests. |
| Focused pytest | `49 passed`, two pre-existing TFP `distutils` deprecation warnings. |
| Endian/layout/bit mutation tests | Passed. |
| Target/execution/provenance adversarial tests | Passed. |
| Unknown/missing schema and strict digest tests | Passed. |
| Real Phase 7 config canonical hash dry run | Passed deterministically. |
| Real private replay canonical hash dry run | Passed deterministically, including explanatory `Infinity`. |
| Forbidden generic-payload/ignore-list scan | No implementation match. |
| Scoped `git diff --check` | Passed. |
| Independent Codex repair verification | `VERDICT: AGREE`. |

Run metadata:

| Field | Value |
| --- | --- |
| Git commit at check time | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Python | `3.11.14` |
| NumPy | `2.1.3` |
| TensorFlow | `2.19.1` |
| TensorFlow Probability | `0.25.0` |
| Device policy | CPU only, GPU hidden intentionally. |
| Final implementation file SHA-256 | `420cc6434a8e94641c26fad74c545ddec306beb3dd7e6c37f7901c4dd5a24bd1` |
| Final focused test file SHA-256 | `87383a386d7007a96e2a4f9b3e60a9d87bfa19fb706a400dcbf681f097153353` |

The TensorFlow import emitted duplicate CUDA factory-registration messages in
the CPU-hidden environment. Under repository policy these are environment
messages only and are not GPU-health evidence.

## Review Repair Loop

Iteration 1 returned `REVISE` for provenance-bearing target identity, loose
execution/provenance mappings, endian-sensitive dtype labels, caller route
overrides, incomplete validation, and missing adversarial tests. The same
implementation was visibly repaired.

The final independent audit found two remaining fail-closed issues:
`np.dtype(None)` resolving to float64 and an absent base-capability scope being
conditionally accepted. Both were patched, regression-tested, and re-reviewed.
The final verdict is `AGREE`. Stalled reviewer transports were interrupted and
were not counted as substantive review rounds or agreement.

## Decision Table

| Decision | Status |
| --- | --- |
| Primary criterion | Passed: typed primitives and adversarial tests represent the Phase 1 classifications. |
| Continuation vetoes | None fired within Phase 2. |
| Legacy P7G blocker | Still active; no baseline was adopted. |
| Transition equality between historical and refreshed artifacts | Not checked and not claimed. |
| Phase 7 readiness | Not established. |
| Next justified action | Review and execute a separate Phase 3 integration subplan. |

## Nonclaims

- No serializer or validator integration is complete.
- No historical/refreshed transition equality is established.
- No baseline adoption is authorized or performed.
- No Phase 7 smoke, serious sampling, Phase 8, or NeuTra work ran.
- No convergence, recovery, ranking, production, default, GPU, or scientific
  claim is made.
