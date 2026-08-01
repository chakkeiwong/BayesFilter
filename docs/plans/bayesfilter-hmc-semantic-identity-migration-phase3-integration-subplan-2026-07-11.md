# Phase 3 Subplan: Serialization, Replay, Redaction, And Validator Integration

Date: 2026-07-11

Status: `REVIEWED_READY_FOR_EXECUTION`

## Phase Objective

Integrate the reviewed Phase 2 identity primitives with the deterministic
LGSSM private replay serializer, replay reconstruction, public redaction, and
Phase 7 preflight so the current artifacts can produce and verify a candidate
identity bundle without rerunning Phase 6 or weakening the active legacy
blocker.

## Entry Conditions Inherited From Phase 2

- Phase 2 typed schemas and adversarial tests passed independent Codex review.
- The refreshed private replay and public kernel artifacts remain unmodified.
- The historical Phase 6AA pins remain the active Phase 7 vetoes.
- No baseline has been adopted; no Phase 7 smoke or serious runtime is
  authorized.
- Scoped HMC files have no unexpected concurrent edits. Unrelated LEDH/QR
  changes remain untouched.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Preserve historical pins and current refreshed artifact hashes side by side; do not replace either in this phase. |
| Proxy promoted | Replay reconstruction and identity equality are engineering evidence only, not convergence or scientific evidence. |
| Hidden transition input | Build identities from the same live base adapter and reconstructed two-transform replay used by the worker. |
| Serializer/runtime mismatch | Provide one shared identity-bundle builder used by serializer/reconstruction/preflight tests; do not independently project raw JSON keys. |
| Public disclosure | Public payload follows the exact field allowlist below; no dimensions, step size, transform arrays, observations, private paths, or tuning internals. |
| Premature gate switch | Candidate semantic validation runs in parallel with the existing legacy checks; legacy checks remain binding until Phase 4 approval. |
| Artifact mutation | Existing governed artifacts are read-only in Phase 3; new evidence writes only to the declared protected/public Phase 3 paths or test temporary paths. |
| Environment mismatch | All local integration checks use `CUDA_VISIBLE_DEVICES=-1`; no HMC execution is needed. |
| Unanswerable artifact | Structured candidate bundle and validation output include transition, execution, provenance, canonical payload, exact file, and reconstruction cross-link hashes. |

Audit verdict: `PASS_FOR_PHASE3_IMPLEMENTATION_AFTER_SUBPLAN_REVIEW`.

## Required Artifacts

- Shared typed identity-bundle/reconstruction helpers in the narrowest
  BayesFilter HMC identity module or a dedicated sibling module.
- An opt-in sidecar serializer for a versioned private identity bundle. It must
  not change the payload or bytes emitted by `build_private_tuning_replay_payload`.
- A public redacted identity reference schema.
- Persistent machine-readable candidate semantic validation output that records
  both candidate results and the still-binding legacy veto.
- A protected pre/post input-integrity manifest for every governed input read by
  Phase 3.
- A terminal output-integrity manifest, created after all other outputs, with
  exact hashes for the new sidecar, input manifest, and public record. The
  terminal manifest is not referenced by any file it hashes.
- Focused serializer/reconstruction/redaction/preflight tests.
- Phase 3 result and reviewed Phase 4 certificate/adoption subplan.

## Required Implementation

1. Export only the identity symbols needed by the existing driver/controller.
2. Add one shared builder that receives the validated base adapter,
   reconstructed replay, validated Phase 7 config, complete tuning payload,
   source tuning hash, named stage lineage, exact private replay bytes, and
   runtime versions.
3. Produce a separate private sidecar/envelope with:
   - transition identity payload/hash;
   - serious execution contract payload/hash;
   - smoke execution contract payload/hash;
   - selection provenance payload/hash;
   - canonical complete tuning-payload hash;
   - the immutable legacy private replay's canonical payload hash and exact
     file SHA-256/byte count;
   - historical full base/Phase 4/final adapter and mass signatures as
     reconstruction-integrity cross-links outside the transition hash.
   The sidecar must not contain its own file hash or byte count. After it is
   serialized, its exact file hash/size are recorded by the public reference
   and terminal output-integrity manifest, eliminating circular hashing.
4. Preserve the sidecar under the existing protected
   `private_diagnostics` convention as a new file. Do not modify the immutable
   legacy replay. Give the sidecar an embedded canonical artifact hash computed
   over its payload without that field; its exact serialized-byte hash is owned
   externally.
5. Produce a public redacted validation record with exactly these top-level
   fields and no others:
   - `schema`;
   - `status` and `decision`;
   - `transition_identity_schema` and `transition_identity_hash`;
   - `serious_execution_contract_schema` and hash;
   - `smoke_execution_contract_schema` and hash;
   - `selection_provenance_schema` and hash;
   - `candidate_checks` containing only named booleans;
   - `legacy_gate` containing only `passed`, a bounded public veto code, and
     `remains_binding`;
   - `legacy_private_replay_reference` containing its already-public embedded
     artifact hash, file SHA-256, and byte count;
   - `private_sidecar_reference` containing schema, embedded artifact hash,
     exact file SHA-256, byte count, and redaction booleans;
   - `input_integrity_manifest_hash` for the protected governed-input manifest;
   - `nonclaims`; and
   - the record's embedded `artifact_hash`.

   Nested schemas and values are also exact:
   - `status` is exactly `blocked_legacy_gate` in Phase 3;
   - `decision` is exactly
     `CANDIDATE_IDENTITIES_RECORDED_LEGACY_GATE_REMAINS_BINDING`;
   - `candidate_checks` has exactly these boolean keys:
     `transition_reconstructed`, `serious_execution_reconstructed`,
     `smoke_execution_reconstructed`, `selection_provenance_reconstructed`,
     `private_sidecar_round_trip`, `public_private_hashes_match`,
     `governed_inputs_unchanged`, and `public_redaction_passed`;
   - `legacy_gate` has exactly `passed=false`,
     `veto_code=LEGACY_WHOLE_PAYLOAD_HASH_MISMATCH`, and
     `remains_binding=true` for the current artifact set;
   - `legacy_private_replay_reference` has exactly `artifact_hash`,
     `file_sha256`, and `byte_count`;
   - `private_sidecar_reference` has exactly `schema`, `artifact_hash`,
     `file_sha256`, `byte_count`, `observations_publicized=false`,
     `transform_arrays_publicized=false`, `hmc_mechanics_publicized=false`,
     `seeds_publicized=false`, `runtime_versions_publicized=false`,
     `private_paths_publicized=false`, `stage_lineage_publicized=false`, and
     `adapter_mass_signatures_publicized=false`;
   - `nonclaims` is exactly the ordered tuple:
     `candidate semantic identity engineering evidence only`,
     `legacy whole-payload gate remains binding`,
     `not baseline adoption`,
     `not Phase 7 readiness or execution`,
     `not posterior convergence or recovery evidence`, and
     `not production, default, GPU, NeuTra, or scientific evidence`.

   `status`, `decision`, `veto_code`, all redaction booleans, all candidate
   check names, and all nonclaim strings are closed enums/tuples. The strict
   constructor/parser must reject any other nested key, value, ordering, or
   cardinality.

   Authority for the legacy replay file SHA-256 and byte count is the existing
   public kernel artifact's `private_replay_reference`, which already publishes
   those values. Authority for the new sidecar hash/size is this reviewed
   redacted-reference schema; hashes and byte counts reveal neither mechanics
   nor target data. No other file hash/size may be carried into the public
   record. The public record must not expose observations, transform payloads,
   step size, leapfrog count, seeds, runtime versions, private paths, stage
   lineage, adapter/mass signatures, or internal tuning diagnostics.
6. Add a strict public-record parser/constructor that rejects missing, extra,
   or incorrectly typed fields. Recursive secret scanning is a defense in
   depth test, not the primary redaction boundary.
7. Extend replay reconstruction/preflight to recompute the candidate transition
   from the live adapter and reconstructed replay, recompute execution identity
   from the validated config, verify selection/integrity hashes, and compare
   public/private references.
8. Evaluate candidate and legacy diagnostics independently. The aggregation
   order is:
   - snapshot all governed input file hashes;
   - compute candidate identities and candidate checks even when a known legacy
     mismatch exists;
   - evaluate the unchanged legacy validator in a captured result;
   - write the protected sidecar and protected pre/post input-integrity
     manifest;
   - verify the governed input hashes did not change; and
   - write the public redacted validation record, then write a terminal
     output-integrity manifest hashing the sidecar, input manifest, and public
     record; and
   - return or raise the legacy veto as the final Phase 7 decision.

   Candidate agreement must never suppress, replace, or downgrade the legacy
   veto. `validate_phase7_inputs` must retain its current fail-closed external
   behavior until Phase 4 adoption approval.
9. Persist the new protected sidecar and input-integrity manifest under the
   existing private-diagnostics convention and the redacted record plus
   terminal output-integrity manifest under
   `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/`. These are
   new Phase 3 evidence files, not rewrites of governed inputs. Phase 4 will
   consume them to construct a migration certificate.

## Required Checks, Tests, And Reviews

- Existing Phase 2 identity suite.
- Focused driver serializer tests.
- Focused Phase 7 preflight tests, including current expected legacy failure.
- Private identity-bundle round trip and live replay reconstruction.
- Mutation tests with this ownership oracle:
  - observation, target-contract, either transform, step, or leapfrog mutation:
    transition hash changes; execution hashes change only through their bound
    transition hash; provenance hash stays stable;
  - execution mode/count/seed/environment/version mutation: the applicable
    execution hash changes; transition and provenance stay stable;
  - selection-policy/lineage mutation: provenance changes; transition and
    execution stay stable;
  - JSON formatting/whitespace-only byte mutation: exact-file hash changes;
    canonical payload, transition, execution, and provenance hashes stay
    stable;
  - semantic serialized-payload mutation: canonical payload and the owning
    semantic identity change according to the above rules.
- Exact public allowlist parser/round-trip tests plus recursive defense-in-depth
  scans of keys, values, and serialized text.
- Public/private hash-reference mismatch tests.
- Unknown identity schema/extra field rejection tests.
- Python compilation, forbidden ignore-list scan, and scoped `git diff --check`.
- Fresh independent Codex implementation/result review because Claude remains
  unavailable after managed disclosure rejection.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the live serializer, replay reconstruction, redaction boundary, and preflight compute and verify the same typed candidate identities without changing the active baseline? |
| Exact baseline | Historical expected pins, current refreshed public/private artifacts, current Phase 7 config, and Phase 2 schemas. |
| Primary criterion | Protected sidecar and public validation record round-trip; live reconstruction produces the same transition identity; pre/post input hashes match; redaction and tamper tests pass; existing legacy blocker remains binding. |
| Continuation vetoes | Existing artifact write, legacy check removal/bypass, identity mismatch, reconstruction failure, public secret/mechanics disclosure, unknown schema, incomplete cross-links, scoped concurrent edit, or failed review. |
| Explanatory only | Candidate hash values, legacy hash values, acceptance, timings, and selection diagnostics. |
| Not concluded | No baseline adoption, historical/refreshed semantic equality certificate, Phase 7 readiness, convergence, recovery, sampler ranking, production/default, GPU, or scientific claim. |
| Preserving artifact | Persistent protected identity sidecar, protected pre/post input-integrity manifest, public redacted validation record, terminal output-integrity manifest, and Phase 3 result; existing governed inputs remain byte-for-byte unchanged. |

## Forbidden Claims And Actions

- Do not rerun Phase 6 or modify its existing artifacts.
- Do not update `expected_hashes` or make semantic hashes the active gate.
- Do not remove, ignore, downgrade, or bypass any current preflight mismatch.
- Do not run HMC, actual-target smoke, serious sampling, Phase 8, or NeuTra.
- Do not expose private mechanics in public artifacts or tests.
- Do not claim old/new transition equality or migration approval.

## Exact Next-Phase Handoff Conditions

Phase 4 certificate work may begin only when:

1. all Phase 3 implementation and focused checks pass;
2. current governed artifacts remain unchanged;
3. the Phase 7 legacy blocker is demonstrably still binding;
4. candidate private/public identities agree after live replay reconstruction;
5. public redaction tests pass;
6. the persistent pre/post input manifest proves governed inputs were unchanged;
7. Phase 3 result and Phase 4 subplan pass independent review.

Phase 4 may generate and review a migration certificate, but actual baseline
adoption remains an explicit human approval stop.

## Stop Conditions

- Any candidate transition identity cannot be reproduced from the live replay.
- Existing artifacts would need to be rewritten to complete Phase 3.
- A public payload exposes HMC mechanics, target data, or private lineage.
- A current legacy veto must be removed to make tests pass.
- A new unclassified runtime input appears.
- Scoped HMC files change unexpectedly.
- The same review blocker remains after five substantive repair rounds.
