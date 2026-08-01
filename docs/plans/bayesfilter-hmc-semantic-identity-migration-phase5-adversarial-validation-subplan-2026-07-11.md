# Phase 5 Subplan: Approved Baseline Materialization And Adversarial Validation

Date: 2026-07-11

Status: `APPROVED_PHASE5_IMPLEMENTATION_IN_PROGRESS`

## Phase Objective

If and only if the human explicitly approves the exact reviewed Phase 4
certificate, materialize that decision as a new versioned Phase 7 semantic
identity baseline, switch preflight to live typed-identity plus artifact-
integrity validation, and adversarially prove that each mismatch is owned by
the correct gate. Phase 5 is local implementation and test work only. It does
not authorize an HMC transition or sampler run.

## Entry Conditions Inherited From Phase 4

- The Phase 4 result, protected certificate, public proposal, terminal output
  manifest, and this subplan pass independent review.
- The seven Phase 4 source files and nine Phase 3 governed inputs still match
  their recorded exact hashes.
- The protected certificate artifact hash is exactly
  `sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`.
- The human explicitly approves
  `PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION` bound to that certificate hash and
  the bounded action in the Phase 4 result. Silence, a general request to
  continue, prior planning authority, or approval of another phase is not
  sufficient.
- Until that exact approval is recorded, this subplan is non-executable and
  the legacy gate remains binding.
- Phase 7 smoke, serious sampling, Phase 8, and NeuTra remain separately
  unapproved.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Bind the adoption record and new config to the reviewed certificate, proposal, terminal manifest, all refreshed typed hashes, and exact source references. |
| Destroyed evidence chain | Do not edit the historical Phase 7 config, Phase 3 artifacts, Phase 4 certificate/proposal, or governed tuning artifacts; create a new V2 config and separate adoption record. |
| Hash copy mistaken for live identity | Rebuild transition, execution, and provenance identities from the same validated live adapter/replay/config objects used by the Phase 7 worker; copied fields alone cannot pass. |
| Proxy promoted | Acceptance, timings, visible mechanics, and legacy whole-payload hashes remain explanatory/historical, not promotion criteria. |
| Legacy gate silently deleted | Retain a named legacy comparison in structured preflight output and tests; reclassify it as historical audit evidence only after approved adoption. |
| Artifact integrity weakened | Preserve embedded, canonical full-payload, exact-file SHA-256, byte-count, schema, cross-link, and unknown-field checks as hard vetoes. |
| Execution identity under-specified | Validate serious and smoke contracts separately, including transition hash, run mode, counts, seeds, worker topology, CPU hiding, TensorFlow/TFP/Python versions, XLA/JIT, dtype, and threads. |
| Compatibility regression | Keep the historical V1 config parseable and demonstrably blocked by its unchanged legacy validator; the new V2 path is explicit and versioned. |
| Runtime smuggled into validation | Monkeypatch/guard worker and transition entry points in focused tests; Phase 5 may reconstruct objects but must not call one HMC step, worker pool, or sampler. |
| Missing stop | Stop on absent approval, source drift, unknown field, typed mismatch, integrity mismatch, review blocker, or any need to rewrite historical evidence. |

Audit verdict: `PASS_FOR_PLAN_REVIEW_ONLY_UNTIL_EXACT_HUMAN_APPROVAL`.

## Approval Record And Execution Audit

The human explicitly approved, verbatim:

`I approve PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION bound to certificate sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f.`

The decision and certificate hash exactly match the reviewed Phase 4 proposal.
Entry revalidation then passed for all nine Phase 3 governed inputs, three
Phase 3 outputs, seven Phase 4 sources, and two Phase 4 outputs.

Pre-implementation skeptical audit verdict:
`PASS_FOR_PHASE5_LOCAL_IMPLEMENTATION_ONLY`.

The audit confirmed that the worker and Phase 3 currently reconstruct the same
adapter/replay through duplicated code. Phase 5 must extract that reconstruction
into one shared side-effect-free builder and make both the worker and V2
preflight consume it. The historical V1 config remains immutable and blocked.
The V2 config may become the default validation config only with
`runtime_authority=false`; `run_phase7` must refuse V2 execution before worker
creation until a separately reviewed Phase 6 approval mechanism exists.

## Required Artifacts

- A strict, separate terminal baseline-adoption record under
  `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/` that binds:
  - the exact human-approved decision identifier and protected certificate
    artifact/file hashes;
  - the public proposal and Phase 4 terminal-manifest hashes;
  - the historical V1 config exact file hash;
  - the new V2 config schema, exact file hash, and canonical config hash;
  - the adopted transition, serious/smoke execution, provenance, complete
    tuning-payload, and replay-integrity hashes;
  - `legacy_whole_payload_gate_status=historical_audit_only`;
  - `runtime_authority=false`; and
  - exact ordered nonclaims plus its own embedded artifact hash.
  The adoption record is written after the V2 config and owns the V2 config's
  exact file hash and byte count. The V2 config must not reference the adoption
  record's embedded or file hash, so the evidence graph remains acyclic.
- A new Phase 7 V2 config file. Do not rewrite
  `docs/benchmarks/configs/multidim_lgssm_phase7_burnin_sampling_2026_07_11.json`.
- Shared read-only live identity reconstruction used by evidence generation and
  the active V2 preflight, with no duplicated field projection.
- A strict V2 preflight report schema that separately reports transition,
  serious/smoke execution, provenance, canonical payload, exact-file,
  historical legacy comparison, and runtime authorization states.
- Focused ownership, mutation, tamper, compatibility, redaction, and
  no-runtime tests.
- Phase 5 result/close record.
- Drafted and reviewed Phase 6 preflight/tiny-smoke subplan, still blocked on a
  separate human smoke approval.

## Required Implementation

1. Preserve the historical V1 config and validator behavior as an explicit
   compatibility lane. Loading that exact config must still end at
   `public final kernel hash mismatch`.
2. Create a strict V2 config rather than editing the V1 evidence source. The V2
   config must contain separate closed sections for:
   - adopted typed transition identity;
   - serious and smoke execution identities;
   - selection provenance;
   - complete tuning-payload and replay canonical/exact-file integrity;
   - exact governed source references;
   - historical whole-payload pins labeled audit-only; and
   - the approved decision identifier plus exact certificate and public-
     proposal references, but no adoption-record hash or byte reference.
3. Reject unknown, missing, duplicated, mistyped, reordered fixed-contract, or
   unsupported schema fields in the V2 config and adoption record.
4. Refactor Phase 3 reconstruction only as needed to expose one side-effect-free
   live builder. It must receive the validated config and governed source
   paths, reconstruct both affine transforms and the retained HMC mechanics,
   and return typed transition, serious/smoke execution, provenance, and
   integrity objects. It must not write evidence or launch a transition.
5. Make V2 preflight rebuild identities from the live builder, then apply gates
   in this order:
   - source schema and exact artifact integrity;
   - live transition identity;
   - selected run-mode execution identity;
   - selection provenance and complete payload integrity;
   - adoption-record/certificate cross-links; and
   - historical legacy comparison as structured audit evidence only.
6. Treat transition, selected execution contract, artifact integrity, unknown
   schema, and adoption-reference mismatch as hard continuation vetoes.
7. Treat provenance mismatch as a fail-closed governance veto that requires a
   new review/certificate but does not imply a mechanical mismatch or require
   retuning by itself.
8. Preserve the current statistical thresholds, CPU/XLA policy, worker
   topology, chain schedule, target, transforms, mechanics, and artifact paths.
   Phase 5 cannot use baseline adoption to change any of them.
9. If `DEFAULT_CONFIG_PATH` is switched to the V2 config after approval, make
   that the only default-path change and test both explicit V1 and default V2
   behavior. This is a runbook-local execution-config migration, not a
   repository product/default-policy promotion.
10. Emit a structured preflight result that says exactly why V2 passes while
    the historical whole-payload comparisons differ. Do not erase or overwrite
    those historical differences.
11. Write the adoption record only after the V2 config is final. Treat the
    adoption record as terminal for this two-artifact edge: it hashes the V2
    config, while the V2 config binds the pre-existing certificate/proposal and
    approved decision but never hashes or references the adoption record.

## Required Checks, Tests, And Reviews

- Revalidate the Phase 3 and Phase 4 terminal manifests before any edit.
- Verify explicit human approval matches the exact decision and certificate
  hash before constructing the adoption record or V2 config.
- Historical V1 compatibility test: exact unchanged legacy failure.
- V2 real-artifact preflight test: live transition, run-mode execution,
  provenance, and integrity agree without launching HMC.
- Ownership mutation matrix:
  - observation, target, either transform, adapted mass, step size, or leapfrog
    mutation changes transition and bound execution identities;
  - run mode, counts, seeds, topology, environment, versions, XLA/JIT, dtype,
    or threads changes the owning execution identity but not transition or
    provenance;
  - selection policy, lineage, or tuning-payload mutation changes provenance
    and/or complete payload integrity but not transition;
  - whitespace-only file mutation changes exact-file integrity but not the
    canonical semantic payload;
  - unknown schema or field fails closed before comparison.
- Test that the current real legacy whole-payload differences are reported but
  are not the V2 mechanical gate after approved adoption.
- Test that a copied compatible source path, stale adoption record, altered
  certificate reference, rehashed approval-state tamper, or public private key
  fails closed.
- Guard tests proving worker creation, `one_step`, burn-in, retained sampling,
  and Phase 8 entry points are not called.
- Recursive public redaction scan plus exact allowlist parsers.
- Python compilation, combined Phase 2-5 focused pytest, forbidden
  ignore-list/projection/bypass scan, and scoped `git diff --check`.
- Write the Phase 5 result and draft/refresh the Phase 6 subplan.
- Obtain fresh bounded independent review. Managed Claude disclosure rejection
  remains binding; use a fresh Codex substitute audit and do not retry Claude.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | After exact human approval, can the refreshed typed baseline replace whole-payload mechanical gating without weakening live transition, execution, provenance, or artifact-integrity validation? |
| Exact baseline | Reviewed Phase 4 certificate `sha256:684c5ae2...1fc7f`, its seven exact sources, the immutable V1 config, and the adopted refreshed typed hashes. |
| Primary criterion | V2 preflight reconstructs and matches every adopted typed/integrity identity, adversarial ownership tests pass, V1 remains historically reproducible, and no runtime entry point executes. |
| Continuation vetoes | Missing/ambiguous approval, certificate/source drift, true transition mismatch, selected execution mismatch, provenance governance mismatch, artifact tamper, unknown schema/field, historical evidence rewrite, public disclosure, runtime call, or failed review. |
| Explanatory only | Legacy whole-payload mismatch, acceptance, timing, diagnostic summaries, and stage narrative. |
| Not concluded | No historical typed equality, HMC readiness, smoke success, convergence, recovery, sampler ranking, production/default, GPU, NeuTra, or scientific claim. |
| Preserving artifact | Immutable Phase 3/4 evidence; new adoption record, V2 config, strict preflight report/tests, Phase 5 result, and review record. |

## Forbidden Claims And Actions

- Do not begin without exact human approval of the named decision and
  certificate hash.
- Do not edit or regenerate the historical V1 config, refreshed kernel/private
  replay, Phase 3 evidence, Phase 4 certificate/proposal, or their manifests.
- Do not use an ignored-key allowlist, projected mechanical hash, copied hash
  field, or legacy comparison suppression as the V2 gate.
- Do not call an HMC transition, create worker processes, run actual-target
  smoke, burn in, sample, run Phase 8, or start NeuTra.
- Do not alter statistical thresholds, target, transforms, mechanics, CPU/XLA
  policy, worker topology, chain schedule, package environment, or product
  defaults.
- Do not claim historical/refreshed typed identity equality or use approval as
  scientific evidence.

## Exact Next-Phase Handoff Conditions

Phase 6 planning may begin only when:

1. exact human adoption approval is recorded and bound to the certificate;
2. Phase 3/4 evidence remains immutable and revalidates;
3. the V2 config and adoption record pass strict round-trip and cross-link
   checks;
4. live V2 preflight passes transition, selected execution, provenance, and
   integrity gates without a runtime call;
5. the full adversarial ownership/compatibility/redaction suite passes;
6. the Phase 5 result and Phase 6 subplan pass independent review; and
7. the supervisor stops before tiny smoke and requests separate explicit human
   smoke approval.

Phase 6 must not execute merely because Phase 5 passes.

## Stop Conditions

- Exact human baseline-adoption approval is absent, denied, ambiguous, or bound
  to a different certificate.
- Any Phase 3/4 source or evidence artifact changes.
- Live reconstruction exposes an unclassified transition or execution input.
- Any transition, execution, provenance, or integrity gate cannot be assigned
  to one owner without projection or ignored fields.
- Implementing V2 would require rewriting historical evidence or changing
  runtime/statistical policy.
- A public artifact would expose protected mechanics or target data.
- A runtime entry point executes during Phase 5.
- The same substantive review blocker remains after five repair rounds.
