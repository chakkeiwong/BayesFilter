# BayesFilter HMC Semantic Identity Migration Master Program

Date: 2026-07-11

Updated: 2026-07-13

Status: `PROGRAM_CLOSED_AT_PHASE7_DIAGNOSTIC_CAP_FAILURE`

Governance supersession, 2026-07-13: current execution follows the Academic
Research Governance Profile in `AGENTS.md` and
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`.
Historical approval, authority, claim, manifest, and review gates below remain
audit evidence but are not active execution prerequisites.

Terminal update: the academic campaign executed once and reached its declared
burn-in cap without passing all-parameter R-hat. Phase 7 is terminal at
`diagnostic_cap_failure`; Phase 8 documentation closeout is complete, while
Phase 8 scientific runtime and NeuTra did not execute and remain outside this
program.

## Objective

Repair the deterministic LGSSM Phase 7 handoff so BayesFilter distinguishes:

1. execution-affecting HMC transition identity;
2. deterministic run/execution identity;
3. selection and review provenance; and
4. exact artifact integrity.

The repair must derive typed, versioned identities from the same validated
objects used to construct the retained HMC transition. It must not use an
allowlist of ignored JSON keys or silently repin the Phase 6AA legacy hashes.

## Direct Problem Classification

Claimed identity in the blocked Phase 7 plan: the exact executable frozen HMC
transition selected by Phase 6AA.

Quantity compared by the blocked validator: whole stage and final-kernel
payload hashes that also include selection-policy provenance and stage lineage.

Historical verdict: wrong relative to the claimed identity. The validator
correctly failed closed under its declared rules, but those rules conflated
mechanics, execution policy, provenance, and file integrity. That engineering
schema/governance blocker was repaired by the approved typed V2 migration. It
was not evidence of HMC convergence failure, target failure, or
scientific-direction failure.

Historical procedural blocker: Phase 7 serious attempt 1 consumed its one-use authority and
claim, then terminated at an output-reservation infrastructure defect before
any worker or HMC transition. Its manifest
`sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330`
and approval are permanently non-actionable. A no-runtime attempt-2 repair now
passes local and bounded review. Its versioned proposal and terminal manifest
were materialized, reconstructed exactly, and separately reviewed. The new
terminal manifest is
`sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e`.
No attempt-2 authority or claim was needed under the academic profile. The
subsequent academic campaign executed the fixed typed candidate and terminated
at its burn-in diagnostic cap. The active next step is outside this program: a
new research/repair plan is required before any additional HMC campaign.

## Authority And Boundaries

Authorized by the 2026-07-11 user instruction:

- planning and documentation;
- source and test changes scoped to semantic identity and Phase 7 validation;
- CPU-hidden local tests, static checks, migration-artifact generation, replay
  reconstruction, and preflight validation;
- bounded Claude read-only review through the governed review gate.

The human approved the refreshed replay's typed V2 baseline adoption under
`PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION` bound to certificate
`sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`.
That approval was consumed by Phase 5 and grants no runtime authority.

The exact actual-target two-worker Phase 7 mechanics smoke was separately
approved and passed under the historical governance. Its authority granted
nothing further. Serious Phase 7 subsequently used one bounded academic
campaign plan and terminated at its diagnostic cap.

Phase 8 documentation closeout and boundary handoff are in this program and
were authorized by the request to continue the rest of the runbook. Phase 8
posterior-recovery/runtime work, NeuTra training, package installation, network
fetches, default-policy changes, and unrelated LEDH/QR work are outside this
program and did not execute.

## Identity Architecture

### Transition Identity V1

`transition_identity_hash_v1` covers every value that can change one HMC
transition:

- schema and HMC kernel/integrator family;
- target scope, target-route identity, target dimension, and computation dtype;
- target-bearing fixture identity;
- ordered base, Phase 4, and final adapter/transform stack identities;
- adapted-mass tensor identity including dtype, shape, byte order, and
  canonical bytes;
- exact canonical step-size representation;
- leapfrog count; and
- any additional value found by the Phase 1 consumer audit to affect the
  transition.

### Execution Contract V1

`execution_contract_hash_v1` covers transition identity plus deterministic run
semantics:

- initial-state policy;
- root seeds and seed derivation;
- chain count, worker partition, and stable chain ordering;
- burn-in and retained chunk schedules;
- CPU visibility, TensorFlow/TFP route, XLA/JIT, dtype, and thread policy; and
- every other value that affects deterministic Phase 7 reproduction without
  changing a single transition kernel.

### Selection Provenance V1

`selection_provenance_hash_v1` covers tuning and explanatory history:

- selection policy and candidate lineage;
- tuning budgets, seeds, screens, and diagnostics;
- acceptance observations and verification summaries;
- stage hashes, review metadata, timestamps, and nonclaims.

It is evidence about how the kernel was selected, not the kernel's transition
identity.

### Artifact Integrity

`artifact_integrity_hash` covers the exact serialized artifact or its canonical
complete payload. It detects corruption or tampering and does not establish
semantic equivalence by itself.

## Gate Semantics

| Mismatch | Classification | Required action |
| --- | --- | --- |
| Transition identity | Mechanical mismatch | Hard continuation veto; retune or approve an explicit mechanical migration. |
| Execution contract | Run-semantics mismatch | Hard veto or separately reviewed execution migration and fresh smoke. |
| Artifact integrity | Corruption/tamper mismatch | Hard veto; regenerate or investigate. |
| Selection provenance with equal approved transition identity | Governance/history mismatch | Review and migration certificate; do not retune automatically. |
| Legacy whole-payload hash after approved migration | Historical audit evidence | Preserve and report; do not use as the active mechanical gate. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can BayesFilter prove and enforce a versioned transition identity independently of provenance and file-integrity changes, then safely admit an approved replay to Phase 7? |
| Exact baseline | Phase 6AA committed artifact and private event, the 2026-07-11 refreshed public artifact/private replay, the blocked P7G result, and current Phase 7 config. |
| Primary program criterion | Typed identities are implemented from runtime-consumed objects; migration evidence is reviewed; adversarial tests pass; preflight and authorized runtime gates pass without weakening HMC/statistical criteria. |
| Continuation vetoes | Unclassified execution-affecting field, true transition mismatch, replay reconstruction failure, tamper/redaction failure, unknown schema, baseline adoption without human approval, unexpected scoped edits, or missing required evidence. |
| Explanatory only | Equal acceptance, elapsed time, stage lineage, policy text, legacy hashes, and observed mechanics fields that do not completely define the transition. |
| Not concluded | No byte identity with the unavailable old private replay; no convergence, recovery, sampler superiority, production/default readiness, GPU readiness, DSGE, NeuTra, or broad scientific claim. |
| Preserving artifacts | This master program, phase subplans/results, visible runbook/ledger/handoff, identity schemas, migration certificate, structured test/preflight/runtime artifacts, and review logs. |

## Phase Index

| Phase | Name | Subplan | Required result |
| --- | --- | --- | --- |
| 0 | Governance And Review Launch | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase0-governance-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase0-governance-result-2026-07-11.md` |
| 1 | Runtime Consumer Audit And Field Classification | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase1-consumer-audit-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase1-consumer-audit-result-2026-07-11.md` |
| 2 | Typed Identity Schemas And Canonical Hashing | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase2-schema-implementation-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase2-schema-implementation-result-2026-07-11.md` |
| 3 | Serialization, Replay, Redaction, And Validator Integration | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase3-integration-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase3-integration-result-2026-07-11.md` |
| 4 | Migration Certificate And Baseline Adoption | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase4-certificate-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase4-certificate-result-2026-07-11.md` |
| 5 | Adversarial Validation | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase5-adversarial-validation-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase5-adversarial-validation-result-2026-07-11.md` |
| 6 | Phase 7 Preflight And Tiny CPU/XLA Smoke | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-result-2026-07-11.md` |
| 7 | Serious Phase 7 Burn-In And Sampling | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md` |
| 8 | Closeout And Boundary Handoff | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-subplan-2026-07-11.md` | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-result-2026-07-11.md` |

Only the current research campaign requires an executable plan. Review is
advisory and used when it addresses a material risk; it is not a universal
phase-entry condition.

## Per-Phase State Machine

1. `PRECHECK`: verify inherited artifacts, restate the evidence contract, and
   run the skeptical plan audit.
2. `EXECUTE_MINIMAL`: perform only the smallest implementation or diagnostic
   that answers the phase question.
3. `ASSESS_GATE`: apply primary criteria and vetoes before interpreting
   explanatory evidence.
4. `WRITE_RESULT`: preserve commands, artifacts, direct verdicts, and nonclaims.
5. `REVIEW_IF_MATERIAL`: use one bounded review when it materially reduces
   scientific or engineering risk.
6. `REPAIR_LOOP`: patch localized failures, rerun focused checks, and retry
   within the unchanged campaign attempt/wall-time budget.
7. `ADVANCE_OR_STOP`: update ledger and handoff; advance when scientific and
   engineering gates pass and campaign budget remains.

## Claude Review Contract

Codex remains supervisor and executor. Claude is read-only and advisory. When
material review is useful, use a bounded one-path question and record its
status. Reviewer availability or a purely procedural disagreement is not an
execution gate for trusted local research when focused evidence is adequate.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Pin both historical Phase 6AA evidence and the refreshed replay; never rewrite historical artifacts to manufacture equality. |
| Proxy promoted | Equal acceptance, visible mechanics fields, and smoke success remain explanatory or engineering evidence only. |
| Missing stop condition | Typed mismatch gates, scientific/numerical vetoes, eight-hour/three-launch campaign caps, external/irreversible boundaries, and scoped-edit checks are explicit. |
| Hidden execution field | Phase 1 traces actual replay construction and Phase 7 runner consumption before defining schemas. |
| Hash/runtime drift | Runtime consumes typed contracts; identity projection is generated from those same validated objects. |
| Stale context | Each phase rechecks artifacts, hashes, source consumers, and scoped worktree state. |
| Environment mismatch | CPU hiding, TensorFlow/TFP, XLA/JIT, dtype, threads, and worker topology belong to execution identity. |
| Artifact cannot answer question | Migration certificate separates equal, different, unsupported, and not-checked fields and preserves all old/new identities. |

Historical Phase 0 audit verdict: `PASS_PHASE0_GOVERNANCE_ONLY`. Its review and
local-consistency conditions were satisfied before Phase 1 began.

## Current Program State

Phases 0 through 5 are complete. The approved typed V2 baseline is active for
validation while historical V1 evidence remains immutable and intentionally
fails its named compatibility gate.

Phase 6 attempt 1 consumed the V2 proposal-manifest approval
`sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b`
and its permanent claim, then failed before worker initialization with
`runtime_error:BrokenProcessPool`. It produced zero worker PIDs, no HMC
transition, no diagnostics, and no private sample bytes. The authority, claim,
failure/progress/output manifest, log, and empty reservations are immutable
attempt-1 evidence. This implementation failure does not reject the target,
HMC, XLA, or the scientific direction.

The localized attempt-2 repair passed `106` authority tests and the `251`-test
combined migration gate, then converged in independent frozen review. The V3
proposal and terminal manifest passed live 71-role verification and independent
exact-artifact review. The exact V3-bound approval was received and consumed by
one launch. Attempt 2 passed its mechanics-only gate with two persistent
workers, four chains, 4 burn-in and 8 retained transitions per chain, Host XLA,
finite protected samples, and a verified terminal output manifest.

The terminal result is
`sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d`;
the terminal output manifest is
`sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8`.
The smoke diagnostics are explanatory only and do not establish convergence or
serious readiness. Phase 6 runtime is terminal at the mechanics-only pass.
Phase 7 attempt 1 consumed its authority and claim, then failed before runtime
at `secure_output_reservation:historical_result_replacement`. Its terminal graph
and historical archive are immutable. The attempt-2 repair uses separate
schemas and versioned paths, pins the complete attempt-1 terminal graph, and
exclusively creates every active output without replacement. Its no-runtime
gates and bounded implementation review pass. The attempt-2 proposal and
terminal manifest were then materialized and byte-exactly reconstructed. Both
one-path reviews initially used ordinary JSON hashing, were corrected to the
project's type-tagged canonical rule, and converged at `VERDICT: AGREE`. The old
exact-approval gate is superseded by the Academic Research Governance Profile.
No attempt-2 authority or claim was required. The academic campaign preserved
the fixed target, identities, counts, thresholds, CPU-hidden Host-XLA route,
and budget. Attempt 1 executed two workers/four chains and reached the `16000`
burn-in cap. All diagnostics were finite and both ESS gates passed, but eight
parameters failed R-hat `<=1.01`; maximum R-hat was
`1.043456525609825`. Retained sampling did not begin. The checksum-verified
terminal classification is `diagnostic_cap_failure`, so retry is forbidden.
Phase 8 documentation closeout records this boundary. Phase 8 scientific
runtime and NeuTra did not execute.

## Terminal Conditions

The program closes when either:

1. all authorized phases pass, artifacts and reviews close, and execution is
   handed off at the next unapproved boundary; or
2. a structured blocker records the exact failed identity, evidence gap,
   preserved artifacts, and smallest justified repair without weakening gates.
