# BayesFilter HMC Semantic Identity Migration Visible Ledger

Date: 2026-07-11

Updated: 2026-07-13

Status: `PROGRAM_CLOSED_AT_PHASE7_DIAGNOSTIC_CAP_FAILURE`

## 2026-07-11 - Phase 0 PRECHECK

Evidence contract:

- Question: does the migration program separate transition, execution,
  provenance, and integrity identities while preserving the P7G fail-closed
  behavior and all human boundaries?
- Baseline: the committed Phase 6AA evidence, refreshed private replay, P7G
  blocker result, current Phase 7 config, and actual replay/runtime consumers.
- Primary criterion: master program, runbook, ledger, handoff, Phase 0 subplan,
  and Phase 1 draft exist, are internally consistent, and pass material review.
- Vetoes: allowlist-based hashing, silent repinning, runtime launch, baseline
  adoption, missing stop condition, or unsupported equivalence claim.
- Nonclaims: no implementation correctness, transition equality, Phase 7
  readiness, convergence, recovery, or scientific claim.

Skeptical audit:

- Wrong baseline: controlled by keeping both historical and refreshed evidence.
- Proxy promotion: prohibited; equal acceptance and visible mechanics fields do
  not establish complete identity.
- Missing stops: adoption, smoke, serious runtime, unknown fields, scoped edits,
  and review nonconvergence are explicit stops.
- Stale context: current source consumers and worktree were inspected before
  drafting.
- Environment mismatch: no runtime is authorized in Phase 0.
- Artifact coverage: every phase has a declared subplan/result path and the
  runbook requires next-plan review before advancement.

Gate status: `PASSED_TO_PHASE1_CONSUMER_AUDIT`.

Review status: governed Claude review was rejected before execution by managed
external-disclosure policy. Fresh Codex substitute reviews of the master and
Phase 1 subplan both ended `VERDICT: AGREE`.

Next action: execute only the read-only Phase 1 consumer audit.

## 2026-07-11 - Phase 1 ASSESS_GATE

Result:

- Consumer graph traced from fixture/base adapter through both mass transforms
  to the float64 fixed-size TFP HMC transition.
- All current consumed and signed fields classified as transition, execution,
  provenance, integrity, or excluded/derived.
- Material correction: both Phase 4 and final affine transforms are
  transition-bearing; final adapted mass alone is insufficient.
- No current execution-affecting field remains unclassified.
- Fresh Phase 1 result and Phase 2 subplan substitute reviews ended
  `VERDICT: AGREE`.

Gate status: `PASSED_TO_PHASE2_SCHEMA_IMPLEMENTATION`.

Next action: implement and test only the typed identity primitives declared in
the reviewed Phase 2 subplan.

## 2026-07-11 - Phase 2 ASSESS_GATE

Result:

- Strict target, transition, execution, provenance, canonical payload, and
  exact-file identity primitives implemented.
- Initial review returned `REVISE`; all target/provenance, execution typing,
  endian, validation, live-replay, and adversarial findings were repaired.
- A real private-replay dry run found explanatory `Infinity`; canonical hashing
  was repaired to use a type-tagged tree and exact float64 IEEE bits.
- Final focused suite: `49 passed`, with only existing TFP deprecation warnings.
- Final independent repair verification: `VERDICT: AGREE`.
- Serializer, validator, artifacts, legacy pins, and runtime remained unchanged.

Gate status: `PASSED_TO_PHASE3_PLAN_REVIEW`.

Next action: review the Phase 3 integration subplan. Do not edit integration
source until that review agrees.

## 2026-07-11 - Phase 3 PRECHECK

Skeptical review repaired circular sidecar hashing, legacy/candidate aggregation,
persistent evidence, mutation ownership, and exact nested public redaction.
Review iterations 1 and 2 ended `REVISE`; iteration 3 ended `AGREE`.

Gate status: `PHASE3_IMPLEMENTATION_AUTHORIZED_WITHIN_EXISTING_BOUNDARIES`.

Next action: snapshot governed inputs, then implement candidate identity
integration without changing serializer output, governed inputs, legacy pins,
or legacy validator behavior.

## 2026-07-11 - Phase 3 ASSESS_GATE

Result:

- Live replay reconstruction produced typed transition, serious execution,
  smoke execution, and seven-stage selection-provenance identities.
- Protected sidecar/input manifest and public validation/output manifest passed
  strict round-trip, cross-link, redaction, tamper, and exact-byte checks.
- All nine governed inputs remained byte-identical.
- Final focused gate: `84 passed, 11 deselected`, with only two existing TFP
  deprecation warnings.
- The unchanged legacy validator still raises exactly
  `public final kernel hash mismatch` after candidate evidence persistence.
- Fresh bounded independent review ended `VERDICT: AGREE`.

Gate status: `PASSED_TO_PHASE4_CERTIFICATE_DRAFTING_ONLY`.

Next action: construct and review only the migration certificate/proposal. Stop
before baseline adoption and request explicit human approval.

## 2026-07-11 - Phase 4 ASSESS_GATE

Result:

- Protected certificate, redacted public proposal, and terminal output manifest
  passed strict schema, classification, source-ownership, cross-link,
  redaction, approval-boundary, and exact-byte tamper checks.
- Classifications are seven `equal`, three `different`, two `unsupported`, and
  two `not_checked`. Historical typed transition and execution identity remain
  `unsupported` because the old private transition-bearing payload is absent.
- All nine Phase 3 governed inputs, three Phase 3 outputs, seven Phase 4
  sources, and two Phase 4 outputs revalidated.
- Phase 4 focused gate: `11 passed`; combined Phase 2-4/controller gate:
  `92 passed`, with only two existing TFP deprecation warnings.
- The unchanged legacy validator still raises exactly
  `public final kernel hash mismatch`.
- Fresh independent Codex substitute review and focused post-clarification
  re-review both ended `VERDICT: AGREE`.
- No baseline, config, expected pin, validator, or active gate changed. No HMC
  transition or runtime ran.

Gate status: `CERTIFICATE_ONLY_PASSED_AWAITING_HUMAN_BASELINE_ADOPTION_APPROVAL`.

Exact proposed decision:
`PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION`, bound to certificate artifact hash
`sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`.

Next action: stop. Phase 5 is reviewed but cannot execute until the human
explicitly approves that exact decision and certificate. Smoke and serious
runtime remain separate later approval gates.

## 2026-07-11 - Phase 5 PRECHECK

Approval received exactly for
`PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION`, bound to certificate artifact hash
`sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`.

Entry revalidation passed for all nine Phase 3 governed inputs, three Phase 3
outputs, seven Phase 4 sources, and two Phase 4 outputs. The certificate remains
proposal-only in its immutable Phase 4 bytes; this approval is recorded as a
new Phase 5 decision rather than rewriting the certificate.

Skeptical audit result: the V1 config must remain immutable and blocked; V2
must use one shared live replay reconstruction with the worker; and default V2
validation must remain runtime-inert through `runtime_authority=false` and a
pre-worker refusal in `run_phase7`.

Gate status: `PHASE5_LOCAL_IMPLEMENTATION_AUTHORIZED_NO_RUNTIME`.

## 2026-07-11 - Phase 5 ASSESS_GATE

Result:

- The typed V2 baseline, adoption record, live preflight, and terminal manifest
  passed strict local, compatibility, tamper, redaction, and no-runtime gates.
- V2 transition and smoke-execution identities matched live reconstruction.
- Historical typed equality remains `unsupported`; the named V1 validator
  still fails exactly with `public final kernel hash mismatch`.
- No transition or worker ran.

Gate status: `PASSED_TO_PHASE6_AUTHORITY_IMPLEMENTATION_NO_RUNTIME`.

## 2026-07-11 - Phase 6 PRECHECK And REPAIR_LOOP

Evidence contract: implement and review only a fail-closed, one-use,
manifest-bound authority path for a tiny mechanics smoke. Do not create
authority or runtime artifacts before a new exact human approval.

Skeptical audit and independent review found material trust-boundary defects in
the initial implementation. The repair loop closed incomplete source
inventory, forged-context, import/restore, benchmark-role, namespace-parent,
teardown, claim-mode, output-TOCTOU, emergency-evidence, serious-diagnostic,
and collision/race gaps. Final targeted, focused, and combined gates passed;
the final frozen implementation review ended `VERDICT: AGREE`.

Gate status: `PASSED_TO_PHASE6_PROPOSAL_MATERIALIZATION_ONLY`.

## 2026-07-11 - Phase 6 PROPOSAL_GATE

Result:

- Materialized only the pending authority proposal and its terminal manifest.
- Proposal embedded hash:
  `sha256:57b9434a54c3c2ac9c67ddf57a54caaf00feb9dcf9910a0fb41b03e44bad653a`.
- Terminal proposal-manifest embedded hash:
  `sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7`.
- Live verification passed all 533 exact implementation references, immutable
  Phase 5 references, typed identities, command, paths, and authority flags.
- Fresh independent exact-artifact review ended `VERDICT: AGREE`.
- Authority, permanent claim, outputs, log, private samples, workers, and HMC
  transitions remain absent.

Historical gate status at that time:
`AWAITING_HUMAN_PHASE7_SMOKE_APPROVAL`.

Original requested statement, later received but vetoed before authority
materialization by concurrent-lane inventory drift:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7.`

This original statement is superseded by the V2 gate below and is no longer
actionable. Serious Phase 7, Phase 8, and NeuTra remain separate human
boundaries.

## 2026-07-12 - Phase 6 APPROVAL_ATTEMPT_AND_INVENTORY_REPAIR

The exact approval bound to original manifest
`sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7`
was received. Authority materialization then failed before any write because
two unrelated tests from another active lane were outside the original 533-role
proposal. No authority, claim, output, log, private sample, worker, or HMC
transition was created.

The repository-wide inventory was repaired to a deterministic 71-role Phase 6
runtime/review closure. Targeted tests passed (`8 passed`); the combined gate
passed (`231 passed, 2 warnings`); implementation review converged; and the
refreshed versioned proposal pair passed live plus independent exact-artifact
verification.

Gate status: `AWAITING_HUMAN_PHASE7_SMOKE_APPROVAL_V2`.

Historical exact statement requested next at that V2 gate:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b.`

The old approval cannot authorize the new manifest. Stop before authority or
runtime until the new exact statement is received.

## 2026-07-12 - Phase 6 ATTEMPT1_TERMINAL

The exact V2-manifest-bound approval was received. The authority and permanent
claim were materialized, and attempt 1 was launched once. The claim consumed
the authority before any worker could initialize.

Terminal evidence:

- authority embedded hash:
  `sha256:6206888214a63b5b0c56a776d27f2a520880b0645ebf7e2afc280f000cfe4c58`;
- permanent claim embedded hash:
  `sha256:6cddfe35278935f9be30a65bf2b481ac53eaaba59818de56855b55b53d73b2ad`;
- failure embedded hash:
  `sha256:68bfd9078c9f187874d2a2334f8353fbc5d4f4736e52dd58c42764782bbcd275`;
- reason `runtime_error:BrokenProcessPool` at stage `preflight_passed`;
- zero worker PIDs, zero burn-in/retained checks, no diagnostics, and zero
  private sample bytes; and
- immutable terminal progress/output manifest/log plus two infrastructure and
  one private-sample zero-byte reservations.

Root cause: the retained child source loader omitted `module.__file__`, so a
child import raised `NameError` before worker initialization. Classification:
implementation failure. It is not evidence against the target, data, math,
HMC, XLA, or scientific direction.

Gate status: `ATTEMPT1_AUTHORITY_CONSUMED_REPAIR_REQUIRES_NEW_PROPOSAL_APPROVAL`.

## 2026-07-12 - Phase 6 ATTEMPT2_REPAIR_GATE

The repair added `module.__file__`, corrected zero-byte private reservation
handling, separated archival and live authorization checks, moved all retry
paths to V3/attempt 2, and introduced a retained exact 13-file attempt-1
integrity session. The session pins parent/file descriptors, rejects symlinks
and hard links, checks owner/group/mode, retains capture-time stat signatures,
double-reads and hashes exact bytes, verifies semantics, and spans proposal,
claim, reservation, controller, and teardown boundaries.

Typed attempt-1 drift permits no further write after detection, bypasses
controller failure classification and infrastructure sealing, and preserves
bounded worker teardown. Stage-specific adversarial tests cover preclaim,
postclaim, reservation, controller-entry, and post-progress outcomes.

Checks:

- stage-specific drift matrix: `3 passed`;
- complete authority module: `106 passed, 2 warnings in 312.64s`;
- combined eight-module gate: `251 passed, 2 warnings in 340.78s`;
- compilation, scoped whitespace, immutable 13-file integrity, and attempt-2
  absence checks: passed; and
- fresh frozen independent review: `VERDICT: AGREE`.

Gate status: `PASSED_TO_V3_PROPOSAL_MATERIALIZATION_ONLY`.

## 2026-07-12 - Historical Phase 6 V3_PROPOSAL_GATE

At that historical gate, only the pending V3 proposal and its terminal manifest
were materialized:

- proposal embedded hash:
  `sha256:d2aff98cb93b85527bd71a206af5244aa18e373ae8a3bd7897b8fc3c841d0395`;
- proposal raw SHA-256:
  `7a5c093a42d7b373d1711c29ed073eb46954f3517d4246878a5d1ff20df40880`;
- terminal manifest embedded hash:
  `sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0`; and
- terminal manifest raw SHA-256:
  `e15cd087fa40e91acb875d88d948fc185a0e6bf1eabc17841111aa9048a7d503`.

Live verification passed all 71 implementation references, exact attempt-2
command/paths, Phase 5 inputs, typed identities, and the immutable attempt-1
ledger. Independent exact-artifact review ended `VERDICT: AGREE`. Attempt-2
authority, claim, outputs, log, private sample, workers, and transitions remain
absent.

Historical gate status:
`AWAITING_HUMAN_PHASE7_SMOKE_APPROVAL_V3_ATTEMPT2`; later satisfied and
permanently consumed by the attempt-2 terminal launch below.

Historical exact statement, now received, consumed, and non-actionable:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SMOKE bound to Phase 6 authority proposal manifest sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0.`

At that time the original and V2 approval statements could not authorize
attempt 2, and execution stopped until the exact V3 statement was received.
No smoke approval can now be reused.

## 2026-07-12 - Phase 6 ATTEMPT2_TERMINAL

The exact V3-manifest-bound approval was received. Prelaunch verification
passed the 71-role live implementation closure, the immutable Phase 5 bundle,
all 13 attempt-1 evidence files, command/path constraints, output absence, and
process absence. The authority was materialized and the exact launcher ran
once. Its permanent claim consumed the authority before outputs or workers.

Terminal result:

- decision:
  `PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_STOP_BEFORE_SERIOUS_APPROVAL`;
- authority:
  `sha256:1f3b8f6b92fda72221fa5036ad752c997d75e4e975b0e0c83afe116eef5e0e9b`;
- permanent claim:
  `sha256:7c3b9ec793eb5dffc5f8b0471ba839cbda7684b2d794c172c51c7df50e93f5ca`;
- result:
  `sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d`;
- progress:
  `sha256:698818a54380c2f2207c35a122201c000111a63c8d52c9d256c98e9051370e05`;
- output manifest:
  `sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8`;
- two persistent workers, PIDs `21` and `121`, two chains each;
- 4 burn-in and 8 retained transitions per chain;
- deliberate CPU hiding, TensorFlow/TFP `2.19.1` / `0.25.0`, float64, Host
  XLA/JIT, and one compile trace per worker; and
- elapsed time `25.814051708206534` seconds.

Strict result/progress parsing, output-manifest reconstruction, authority/
claim/proposal/result/progress cross-links, file modes, empty emergency
reservations, all 13 attempt-1 files, and process teardown passed. The protected
NPZ hash is
`d46514c6fad6dd0b55f9563f9686fee0436034a448677e295062ec899c24393f`;
it contains finite retained samples `(8, 4, 18)` and final worker states
`(2, 2, 18)` with exact config and embedded private-replay provenance.

Observed retained maximum R-hat `3.685359225168008`, minimum bulk ESS
`13.697247858180793`, and minimum tail ESS `8.0` are explanatory only. Eight
draws per chain cannot support convergence or ranking. The unchanged serious
thresholds were not smoke promotion criteria.

Gate status: `PHASE6_MECHANICS_PASS_STOP_BEFORE_SERIOUS_AUTHORITY`.

## 2026-07-12 - Phase 7 PLAN_DRAFT_GATE

The Phase 7 serious subplan is drafted and non-executable. Its skeptical audit
identified two binding mechanical/governance requirements before serious
runtime:

1. the active V2 config remains `runtime_authority=false`, so serious mode
   requires a separate closed-schema one-use authority, permanent claim, and
   unforgeable launch context; and
2. the configured `burnin_sampling.json` path already contains the immutable
   pre-migration blocker. Its exact bytes must be archived and terminally bound
   before a human-approved proposal may authorize controlled replacement.

The serious contract remains unchanged: two persistent CPU-hidden Host-XLA
workers, four chains, burn-in `2000 + 1000` to cap `16000`, retained
`4000 + 2000` to cap `40000`, R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS
`>=400`, float64, and an eight-hour cap.

Gate status: `NON_EXECUTABLE_PENDING_PHASE6_CLOSEOUT_AND_PHASE7_PLAN_REVIEW`.

Next action: obtain fresh independent read-only agreement on the Phase 6
closeout and Phase 7 serious subplan. Then implement and test only the separate
serious pre-runtime authority boundary, materialize its proposal/terminal
manifest, review exact artifacts, and stop for a new exact manifest-bound human
approval. Do not execute serious Phase 7, Phase 8, or NeuTra.

## 2026-07-12 - Phase 6 CLOSEOUT_REVIEW_AND_PHASE7_PLAN_REVIEW

Fresh read-only Codex substitute reviews were used because the managed Claude
external-disclosure rejection remains binding.

- Phase 6 terminal closeout review initially returned `REVISE` for ambiguous
  historical pre-runtime wording and missing mandated decision/inference rows.
  Both were repaired; focused re-review ended `VERDICT: AGREE`.
- Coordination review initially returned `REVISE` for premature closeout,
  stale V3 approval wording, Phase 8 scope ambiguity, a stale reset-memo tuning
  instruction, and historical Phase 0 present tense. All were repaired; focused
  re-review ended `VERDICT: AGREE`.
- The bounded Phase 7 serious-subplan review found no material issues and ended
  `VERDICT: AGREE`. Two nonresponsive reviewer turns were terminated and were
  not counted as agreement.

Review records:

- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase6-terminal-closeout-codex-review-2026-07-12.md`;
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-codex-review-2026-07-12.md`; and
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase6-phase7-coordination-codex-review-2026-07-12.md`.

Gate status:
`PHASE6_CLOSED_PHASE7_SERIOUS_PRERUNTIME_AUTHORITY_IMPLEMENTATION`.

Next action: implement and test only the reviewed Phase 7 archive, separate
one-use serious authority, permanent claim, secure launcher/context, and
terminal proposal artifacts. Stop before serious runtime and request the exact
human approval bound to the reviewed terminal proposal manifest.

## 2026-07-12 - Phase 7 SERIOUS_PRERUNTIME_PROPOSAL_GATE

The historical blocker archive, separate one-use serious authority/claim
mechanics, secure launcher/context, source-bound worker cache seal, and
descriptor-backed output mechanics passed their no-runtime gates. The accepted
historical-retirement repair allows only the necessary sole-link retirement
metadata transition and strengthens archive, parent, byte, owner, mode, and
replacement-inode checks.

Checks:

- serious-authority module: `39 passed`;
- controller/cache-seal module: `28 passed`;
- smoke-authority compatibility module: `106 passed`;
- nine-module combined migration gate: `302 passed, 2 warnings in 387.19s`;
- compilation, whitespace, authority/bypass scans, inherited 20-artifact
  verification, proposal reconstruction, runtime-artifact absence, and process
  absence: passed; and
- bounded implementation, exact proposal, and exact terminal-manifest reviews:
  `VERDICT: AGREE`.

The terminal serious proposal-manifest hash is
`sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330`.
No authority, claim, worker, HMC transition, progress, private sample, log, or
runtime output manifest was created.

Gate status: `AWAITING_HUMAN_PHASE7_SERIOUS_APPROVAL`.

Consumed historical action, no longer valid:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS bound to Phase 7 authority proposal manifest sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330.`

This statement was later received and permanently consumed by attempt 1. It
must never be reused. Attempt 1 failed before runtime at
`secure_output_reservation:historical_result_replacement`; see the 2026-07-13
entry below. Phase 8 and NeuTra remain separately unauthorized.

## 2026-07-13 - Phase 7 SERIOUS_ATTEMPT1_TERMINAL_AND_ATTEMPT2_REPAIR_GATE

Attempt 1 consumed authority
`sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d`
and claim
`sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c`,
then terminated before worker creation at
`secure_output_reservation:historical_result_replacement`. This was a launcher
output-reservation implementation failure, not HMC, XLA, target, convergence,
or scientific evidence. Every attempt-1 artifact is immutable.

The attempt-2 repair:

- fixes same-inode/different-descriptor comparison while rejecting a different
  inode;
- hard-pins attempt-1 JSON artifacts, plans/result note, archive, empty
  reservations, modes, links, parents, and required absences;
- introduces separate attempt-2 schemas and authority/pass/block decisions;
- uses versioned paths for every proposal, authority, claim, result, progress,
  sample, log, ordinary manifest, and infrastructure artifact; and
- removes historical replacement from the active path in favor of exclusive
  creation of every attempt-2 output.

Checks were deliberate CPU-hidden no-runtime engineering checks:

- serious-authority module: `48 passed, 2 warnings in 149.91s`;
- controller/smoke compatibility: `134 passed, 2 warnings in 250.74s`;
- nine-module migration gate: `311 passed, 2 warnings in 422.72s`;
- compilation, whitespace, static mutation/authority scans, exact attempt-1
  integrity, attempt-2 absence, and in-memory proposal reconstruction: passed;
  and
- bounded Codex substitute implementation review: `VERDICT: AGREE` because
  Claude remains unavailable under the binding disclosure rejection.

Gate status: `PHASE7_SERIOUS_ATTEMPT2_PROPOSAL_MATERIALIZATION_PENDING`.

Next action: materialize and exactly review only the versioned attempt-2
proposal and terminal manifest. No valid approval hash exists yet. Stop before
authority, claim, output reservation, worker creation, HMC/XLA runtime, Phase
8, or NeuTra.

### Attempt-2 Pre-Materialization Skeptical Audit

Audit verdict: `PASS_FOR_PROPOSAL_AND_TERMINAL_MANIFEST_ONLY`.

- Wrong baseline: no; the exact attempt-1 terminal graph and unchanged Phase
  5/6 typed identities are pinned.
- Proxy promotion: no; tests and review are engineering evidence only and do
  not establish HMC feasibility or convergence.
- Missing stop: no; authority, claim, output reservation, workers, HMC/XLA
  runtime, Phase 8, and NeuTra remain forbidden.
- Hidden assumption or stale context: no; all active attempt-2 paths are
  versioned and absent, v1 authority is parser-incompatible, and current
  governance marks the v1 approval consumed.
- Environment mismatch: no; proposal construction deliberately hides CUDA and
  fixes the reviewed thread environment before framework import.
- Artifact insufficiency: no; the proposal binds exact source, runtime, paths,
  attempt-1 terminal evidence, and nonclaims, while the terminal manifest binds
  the exact proposal bytes.

The proposal builder may now run once. It cannot build authority, consume a
claim, reserve outputs, create workers, or execute the controller.

## 2026-07-13 - Phase 7 SERIOUS_ATTEMPT2_PRERUNTIME_GATE

The versioned attempt-2 proposal and terminal manifest were materialized under
the no-runtime repair authority. No attempt-2 authority, claim, output
reservation, worker, XLA compile transition, HMC transition, progress, private
sample, log, ordinary output manifest, or infrastructure terminal was created.

Exact artifacts:

- proposal embedded hash:
  `sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4`;
- proposal file SHA-256:
  `cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7`;
- proposal byte count/mode: `39904` / `0600`;
- terminal manifest embedded hash:
  `sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e`;
- terminal manifest file SHA-256:
  `e7aa19fb234dd3eff960e97c0c50a643c98663a6e87c98170a9c0f09c9a991b6`;
  and
- terminal manifest byte count/mode: `869` / `0600`.

The supervisor reconstructed both artifacts from one pinned
`SeriousInheritedEvidenceSession`, verified the proposal live candidate,
verified the manifest against exact proposal bytes, rebuilt identical
serialized bytes, checked project type-tagged embedded hashes, rejected
duplicate JSON keys, and confirmed filtered process absence.

The first reconstruction assertion compared JSON arrays with builder tuples
for `command` and `nonclaims`; object equality failed while serialized bytes
were identical. The gate was corrected to compare strict semantics,
authoritative hashes, and serialized bytes.

Both exact artifact reviews initially returned `REVISE` after applying ordinary
compact JSON hashing. BayesFilter's declared artifact hash first applies
type-tagged `_strict_json_value` normalization. The authoritative function
reproduced both embedded hashes exactly; focused re-reviews withdrew the
findings and ended `VERDICT: AGREE`. No artifact was edited or regenerated.

Review/result records:

- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-proposal-codex-review-2026-07-13.md`;
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-manifest-codex-review-2026-07-13.md`;
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-runtime-subplan-codex-review-2026-07-13.md`;
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-preruntime-result-2026-07-13.md`; and
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-runtime-subplan-2026-07-13.md`.

The next runtime subplan review found no blocking consistency, feasibility,
artifact-coverage, evidence-contract, or boundary-safety issue and ended
`VERDICT: AGREE`. This review grants no authority.

Gate status: `AWAITING_HUMAN_PHASE7_SERIOUS_ATTEMPT2_APPROVAL`.

The user's most recent statement bound to
`sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330`
is the already-consumed attempt-1 approval and is non-actionable. The only
acceptable next statement is:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS_ATTEMPT2 bound to Phase 7 attempt-2 authority proposal manifest sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e.`

Stop before authority creation, claim consumption, output reservation, worker
creation, or HMC/XLA runtime. Phase 8 and NeuTra remain unauthorized.

## 2026-07-13 - ACADEMIC_GOVERNANCE_MIGRATION

Owner direction replaced the prior production-style launch-security model with
proportional academic-research governance. Historical proposal, manifest,
authority, claim, descriptor/inode, reservation, and review artifacts remain
preserved, but their procedural gates are superseded for active work.

The retained controls are scientific and operationally relevant:

- fixed target, typed transition and execution identities;
- unchanged all-parameter convergence thresholds and numerical vetoes;
- exact commands, environment, seeds, hardware, wall time, and output paths;
- fresh versioned run directories and ordinary checksums;
- eight total wall-clock hours and at most three campaign launches;
- failure classification, focused regression, and attempt accounting; and
- explicit boundaries for external/public actions, secrets/privacy,
  destructive operations, broad environment changes, expanded compute, Phase
  8, and NeuTra.

Retired active requirements include exact hash-bound approval prose, one-use
authority/claim files, immutable empty reservations, inode/descriptor security
protocols, and mandatory review of each procedural artifact. A localized
infrastructure repair may be retried inside the unchanged campaign budget.

Active artifacts:

- `AGENTS.md`;
- `docs/plans/bayesfilter-academic-research-governance-simplification-2026-07-13.md`;
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`;
- refreshed master program, runbook, ledger, and stop handoff; and
- shared canonical policy under `~/python/claudecodex`.

Gate status: `PHASE7_ACADEMIC_CAMPAIGN_READY_FOR_PLAIN_LANGUAGE_RESUME`.

This policy-change turn does not launch Phase 7. A later plain-language request
to resume or execute the current Phase 7 campaign is sufficient. Phase 8 and
NeuTra remain separate campaigns.

## 2026-07-13 - Phase 7 ACADEMIC_CAMPAIGN_TERMINAL

The user requested continuation of Phase 7 and the rest of the runbook. Before
runtime, the skeptical implementation audit found and repaired four material
evidence issues: terminal progress cross-linking, full schedule/diagnostic
validation, source-drift failure classification, and transition-dispatch
accounting. The repaired focused gate passed `31` tests; the complete
no-runtime migration gate passed `319` tests; static checks passed; and the
independent implementation review returned `VERDICT: AGREE`.

Attempt 1 used the fixed V2 config, transition identity
`sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a`,
serious execution identity
`sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4`,
two CPU/XLA workers, four chains, and root seed `(20260711, 701)`.

The run reached the `16000` burn-in cap without starting retained sampling.
All diagnostic quantities were finite, there were no recorded hard vetoes,
minimum bulk ESS was `1243.2342193161846`, and minimum tail ESS was
`511.5036456092887`. Eight parameters failed R-hat, with maximum R-hat
`1.043456525609825`. The controller correctly wrote
`diagnostic_cap_failure`; attempt elapsed/cumulative time was
`541.4647207008675` seconds with no budget overrun.

Terminal result embedded hash:
`sha256:0724851756606956d2bf9d79fa62597fcef22a0c3c0737548d3383650306e076`.
Checksum-manifest embedded hash:
`sha256:41f6682abc28edd8c3b5650db19b4a6ee906bf2cd40a34a5fcedb303a6cc0b0b`.
Attempt history and checksums verified and no Phase 7 process remained.

Gate status: `PHASE7_DIAGNOSTIC_CAP_FAILURE_NO_RETRY`.

This rejects the fixed candidate under its declared convergence screen. It
does not reject the target or broad HMC direction. Retry, Phase 8 scientific
runtime, and NeuTra are not authorized.

## 2026-07-13 - Phase 8 DOCUMENTATION_CLOSEOUT

Phase 8 executed only the documentation and boundary-handoff work in
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-subplan-2026-07-11.md`.
The master program, runbook, ledger, and stop handoff were refreshed to the
terminal Phase 7 result. No HMC transition, retained sampling, posterior
recovery, NeuTra, package/default change, or new scientific experiment ran.

Gate status: `PROGRAM_CLOSED_NEW_RESEARCH_REPAIR_PLAN_REQUIRED`.
