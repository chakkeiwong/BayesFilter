# BayesFilter HMC Semantic Identity Migration Visible Runbook

Date: 2026-07-11

Updated: 2026-07-13

Status: `CLOSED_AT_PHASE7_DIAGNOSTIC_CAP_FAILURE`

Governance supersession, 2026-07-13: the active route is
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md`.
Earlier exact-approval and authority/claim text is historical only.

## Role Contract

Codex in the current conversation is supervisor and executor. Claude is a
read-only reviewer only and cannot edit, execute experiments, launch agents, or
authorize boundary crossings.

This is a visible, recoverable runbook. It must not use `codex exec`, detached
or background supervisors, `overnight_gated_launch.sh`, `nohup`, `setsid`,
detached `tmux`, or copied-workspace execution.

## Program Artifacts

- Master program:
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-master-program-2026-07-11.md`
- Ledger:
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-ledger-2026-07-11.md`
- Stop handoff:
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-stop-handoff-2026-07-11.md`
- Blocked predecessor result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md`
- Phase 7 academic result:
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md`
- Phase 8 closeout result:
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-result-2026-07-11.md`

## Phase Index

| Phase | Gate | Runtime authority |
| --- | --- | --- |
| 0 | Governance artifacts and launch review agree | Planning only |
| 1 | Every replay/runtime field classified with source anchors | Read-only audit only |
| 2 | Typed schemas and canonical hashes pass unit/mutation tests | Local source/tests |
| 3 | Serialization/replay/validator integration and redaction pass | Local source/tests and replay reconstruction |
| 4 | Migration certificate reviewed and baseline adoption explicitly approved | Human approval required for adoption |
| 5 | Adversarial/tamper/compatibility suite passes | Local tests |
| 6 | Preflight passes; tiny actual-target CPU/XLA smoke passes | Human approval required before smoke |
| 7 | Terminal `diagnostic_cap_failure`; no retry | Campaign closed after one launch |
| 8 | Documentation closeout and next-boundary handoff agree | Complete; no Phase 8 recovery or NeuTra |

## Quiet Visible Execution

Predeclare log and structured-artifact paths for long commands. Redirect full
output to the log, poll bounded public status, and inspect at most the final
20-40 lines on failure. Full logs remain artifacts. Never interpret partial
sampler diagnostics or modify gates during a run.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can a typed semantic identity replace whole-payload mechanical gating without weakening artifact integrity or runtime governance? |
| Baseline | Historical Phase 6AA evidence, refreshed private replay, P7G blocker, and current source consumers. |
| Program pass | Typed contracts drive runtime, migration/adversarial gates pass, and authorized Phase 7 gates close under unchanged statistical thresholds. |
| Vetoes | True transition mismatch, unknown execution field, corrupted artifact, failed replay/redaction, unauthorized adoption/runtime, or invalid result artifact. |
| Explanatory only | Legacy payload hashes, acceptance, timings, and selection lineage. |
| Nonclaims | No old-private byte equality, posterior recovery, ranking, production/default, GPU, DSGE, NeuTra, or broad scientific claim. |

## Visible State Machine

For each phase:

1. read the current subplan and verify entry artifacts;
2. record a skeptical audit and evidence contract in the ledger;
3. execute only the declared minimal actions;
4. write the phase result before advancing;
5. draft or refresh the next subplan;
6. run focused checks and request one review only when materially useful;
7. repair infrastructure failures and retry within the campaign budget;
8. advance after scientific, engineering, artifact, and budget gates pass.

## Review Protocol

When independent review materially reduces risk, use the governed
`claude_review_gate.sh` command with trusted permissions and one exact
path/question. Reviewer failure is recorded but does not block trusted local
research when focused local evidence is adequate. No review verdict changes a
scientific criterion or external/irreversible boundary.

## Repair Loop

A fixable infrastructure failure triggers a focused regression, scoped repair,
updated attempt record, and fresh versioned output directory. Retry without new
approval while the scientific contract and total campaign budget remain
unchanged. Stop if evidence is missing, a scientific/numerical continuation
veto fires, or a retry would change the contract or exceed budget.

## Human Stops

Stop before material baseline or scientific-direction changes unless the user
directs them. Stop for package/network/environment changes,
destructive operations, default-policy changes, unrelated dirty-file edits,
unknown transition inputs, Phase 8 runtime/recovery, or NeuTra work. Phase 8
closeout planning is in-program only after Phase 7 closes and still requires
its own scientific campaign plan.

## Current State

Phases 0 through 5 are complete. The human approved
`PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION` bound to certificate
`sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f`,
and the resulting typed V2 baseline/adversarial gates passed without runtime.
Historical V1 evidence remains immutable and intentionally fails its named
compatibility gate; it is preserved audit evidence rather than the active V2
mechanical gate.

Phase 6 pre-runtime implementation, no-runtime checks, frozen implementation
review, proposal materialization, live reference verification, and exact
proposal-artifact review passed. The original terminal proposal-manifest hash
was
`sha256:9db02019042769750a731dbc849746c5e3380a8883e03167553d7829acf0f1c7`.

The original exact approval was received, but pre-authority verification
failed safely after unrelated concurrent-lane test additions changed the
over-broad repository-wide inventory. The inventory was repaired to the
reviewed 71-role Phase 6 runtime/review closure, and versioned proposal
artifacts passed live and independent review. The V2 terminal proposal-manifest
hash was
`sha256:e8e913e005423da1da87bfa1f5a8e832f7b32d8a8a90172aa81fefea8607bc3b`.

The exact V2 approval was received and attempt 1 permanently consumed its
authority and claim. Attempt 1 failed before worker initialization with
`runtime_error:BrokenProcessPool`; it produced zero worker PIDs, no HMC
transition or diagnostics, and no private sample bytes. Its authority, claim,
failure/progress/output manifest, log, and empty reservations are immutable.

The repaired attempt-2 implementation and evidence-integrity gate passed local
checks plus independent frozen review. The reviewed V3 terminal
proposal-manifest hash is
`sha256:9f026fcf4382e77df5e5e4adff97ac63ceed918717e3be88f611eac7f1a2c3d0`.
The exact V3-bound approval was received, its authority was consumed once, and
attempt 2 passed the mechanics-only gate. Its terminal result is
`sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d`
and terminal output manifest is
`sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8`.
Two persistent workers executed four chains with Host XLA; protected samples
and all terminal cross-links independently verified. The smoke's eight-draw
R-hat/ESS values are explanatory only, not convergence evidence.

Phase 7 attempt 1 consumed authority
`sha256:dc3deaef659b4dffa07d1b45c8512e440828aec6faa01b6fcd88786ab2c1899d`
and claim
`sha256:4854dfd990dcb250ab976f51b66d14f54e83e68aa921941d36d5d74f42b6869c`,
then failed at `secure_output_reservation:historical_result_replacement`
before worker creation or any HMC/XLA transition. The v1 proposal manifest
`sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330`
and its approval are consumed historical evidence and must never be reused.

The attempt-2 no-runtime repair pins that complete terminal graph, uses new
schemas and `attempt2` paths, and exclusively creates every active output. The
serious-authority module passed `48` tests, controller/smoke compatibility
passed `134`, and the nine-module gate passed `311`; compilation, static,
integrity, in-memory proposal reconstruction, and bounded implementation review
also passed. The active V2 config remains `runtime_authority=false`.

The versioned attempt-2 proposal and terminal manifest are now materialized.
Exact reconstruction verified proposal artifact hash
`sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4`,
proposal file SHA-256
`cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7`,
and terminal manifest artifact hash
`sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e`.
Both exact one-path reviews converged at `VERDICT: AGREE` after correcting their
initial use of ordinary JSON hashing to BayesFilter's type-tagged canonical
rule. No proposal bytes changed during review.

The old exact manifest-bound gate is superseded. Its artifacts remain history
and must not be overwritten, but no new authority or claim is required.

Terminal gate: `PHASE7_DIAGNOSTIC_CAP_FAILURE_NO_RETRY`.

The academic campaign ran once. It reached `16000` burn-in transitions per
chain with finite diagnostics and passing bulk/tail ESS, but eight of 18
parameters failed R-hat `<=1.01`; maximum R-hat was
`1.043456525609825`. Retained sampling did not begin. Terminal checksums and
attempt history verified, and no worker/controller process remains.

Phase 8 performed documentation closeout only. No retry, posterior-recovery
runtime, retained sampling, Phase 8 scientific evaluation, or NeuTra work is
authorized by this runbook. Further HMC work requires a new research/repair
plan and user direction.
