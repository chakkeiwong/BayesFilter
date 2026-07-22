# Complete High-Dimensional Leaderboard Visible Execution Ledger

Date: 2026-07-11

Status: `PHASE0_PASSED_PHASE1_REVIEWED_SCHEMA_V7_WAIVER_REVIEW_PENDING_NOT_AUTHORIZED`

## Program

- Master:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-master-program-2026-07-11.md`
- Visible runbook:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-visible-gated-execution-runbook-2026-07-11.md`
- Detached supervisor plan:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-detached-overnight-supervisor-plan-2026-07-11.md`
- Exact launch manifest:
  `docs/plans/complete-highdim-leaderboard-exact-command-manifest-2026-07-11.json`

## 2026-07-11 - Phase 0 - PASSED

Evidence contract:

- Question: are six main rows, one sidecar, four algorithms, exact source
  identities, and current gaps frozen correctly?
- Baseline: July 3 non-LEDH JSON, July 6 historical LEDH-inclusive JSON, six
  July 7 forward artifacts, and current score/FD/transport source files.
- Primary criterion: generated freeze JSON plus focused tests and material
  reviews pass without editing shared source files.
- Vetoes: wrong hash/target/scope/shape/seed/parameters, stale admission,
  sidecar promotion, or dirty-work conflict.
- Nonclaims: no numerical admission, GPU result, complete leaderboard, or
  ranking.

Skeptical audit:

- The July 6 builder has stale hardcoded inputs and cannot be the current
  release baseline.
- The shared Sinkhorn fix requires current-source LEDH reruns.
- Detached execution needs separate namespace isolation and export controls.
- Later Zhao-Cui evaluator work cannot be pre-approved from Phase 0; each row
  needs checked paper and local author-source anchors.

Actions and evidence:

- Generated and independently audited the 24-cell Phase 0 freeze.
- Closed Phase 0 as `PASS_PHASE0_BOUNDARY_FREEZE`.
- Phase 0 JSON SHA-256:
  `4115ef55114ffd73255363f0c62c4a19dd85d7ca3241d002c48409cb9004f878`.
- Master and Phase 0 substitute reviews converged at iteration 5 after two
  trusted Claude health probes timed out.
- No GPU run, cell admission, detached launch, commit, push, or merge occurred.

Gate status: `PASSED`

## 2026-07-11 - Phase 1 - PRECHECK AND SUBPLAN REVIEW PASSED

Evidence contract:

- Question: can canonical target and Zhao-Cui availability pre-gates safely
  precede a six-row endpoint-rich LEDH harness repair?
- Baseline: Phase 0 freeze and the current five-nonlinear-row schema-v4
  harness.
- Primary criterion: canonical byte identities and six-row source availability
  both pass before harness edits; the later aggregate binds paired total
  values/scores for exactly seeds `81120..81124`; every seed and direction
  reconstructs FD from manifest-bound endpoints and passes the FD-only
  `0.05 * sqrt(p)` rule.
- Vetoes: target/source contradiction, unapproved Zhao-Cui invention, harness
  drift before pre-gate receipts, substituted seeds, off-center endpoints,
  stored-FD reuse, or mixed command/config/route identities.
- Explanatory only: CPU-hidden parser runtime, tiny values, and compile time.
- Nonclaims: no GPU evidence, row admission, HMC/posterior correctness,
  ranking, or complete leaderboard.

Skeptical audit:

- The initial subplan could have permitted harness edits after P1-A but before
  P1-B; this was a material sequencing flaw and was repaired.
- The initial phrase `byte-level` was not a deterministic serialization
  specification; the final plan now fixes frame prefix, byte order, canonical
  JSON, payload order, and checker-owned row field ledgers.
- An exact-command hash cannot be seed-invariant when seed and output path
  differ; the final plan separates a command-template-family hash from each
  per-seed exact-command hash.
- Real endpoint scalars alone do not establish the intended centered FD; the
  final plan binds score theta, endpoint roles, full endpoint vectors, steps,
  and directions to the per-seed manifest.
- The arbitrary historical `0.005` setting is explicitly forbidden. The
  threshold is scoped only to FD validation and is exactly
  `0.05 * sqrt(p)` for each seed and parameter direction.

Review trail:

- Iterations 1-4 returned `VERDICT: REVISE` and were visibly repaired.
- Iteration 5 returned `VERDICT: AGREE` for subplan SHA-256
  `ff75b73fdbc2f75c0d5f05c0ac835fdfec69cc7ccd1448b47c6f66b2d9ebb62b`.
- Receipt:
  `docs/reviews/bayesfilter-complete-highdim-leaderboard-phase1-subplan-review-receipt-2026-07-11.json`.

Gate status: `SUBPLAN_REVIEW_PASSED_EXECUTION_NOT_STARTED`

## 2026-07-11 - Detached Launch - PREFLIGHT

Evidence contract:

- Question: can one reviewed Codex supervisor start Phase 1 inside a copied,
  mount-isolated workspace for at most eight hours and leave a complete export
  without modifying the source workspace?
- Baseline: Phase 0 pass, SHA-bound reviewed Phase 1 subplan, current dirty
  source workspace, exact launch-state dependency manifest.
- Primary criterion: focused infrastructure tests pass; trusted GPU/XLA,
  noninteractive Codex, and fake-supervisor isolation artifacts pass; detached
  plan, supervisor/export code, and exact manifest receive fresh read-only
  substitute review; final schema-v6 manifest
  `--check --require-reviewed-static-readiness` passes; exact human launcher
  approval is then obtained.
- Vetoes: source-workspace mutation, stale dependency hash, missing export,
  failed GPU/XLA/TF32 device check, failed Codex health, failed namespace
  handoff, unreviewed active phase, or review nonconvergence.
- Explanatory only: probe runtime, log size, PID values, and which later phase
  is reached within eight hours.
- Nonclaims: successful launch is not numeric completion, source-faithfulness,
  posterior correctness, HMC readiness, ranking, or release.

Pre-mortem:

- A handoff could pass while writes still reach the source workspace; compare a
  full test-source snapshot before and after while requiring a marker only in
  the copied workspace.
- TensorFlow could enumerate a GPU without XLA placement; execute an actual
  `jit_compile=True` float32 matmul and require a GPU output device and TF32.
- Codex could exist but fail noninteractive authentication; require the exact
  final token `CODEX_PROBE_OK` under trusted execution.
- Export files could be hashed and then mutated; the supervisor now captures
  successful exporter stdout without appending it to a hashed file, and tests
  recompute every exported hash after exit.
- A successful smoke could be mistaken for program success; all preflight
  artifacts explicitly forbid that conclusion.

Checks and preflights completed:

- The pre-schema-v4 gate had `26 passed` across Phase 0 freeze, isolated export,
  supervisor, watchdog,
  launch manifest, and review-receipt tests with `CUDA_VISIBLE_DEVICES=-1`.
- Python compilation passed for all launch/preflight helpers.
- Supervisor and Codex-preflight shell syntax passed.
- Scoped `git diff --check` passed.
- Trusted GPU/XLA/TF32 preflight passed; artifact SHA-256:
  `98cd7e6bb624f16c8b51dfab1988ba05c91a469b8b27b6e3d7814a3ba98ff146`.
- Trusted noninteractive Codex probe passed with exact token
  `CODEX_PROBE_OK`; artifact SHA-256:
  `abfe78be4a3a450ed26c151164152406162f985dbd5b8ba234327cb5be769d62`.
- Production-equivalent fake-Codex namespace isolation v3 passed with a
  byte-identical source snapshot and changed isolated copy. The child had PID
  1, empty capability sets, `NoNewPrivs=1`, read-only handoff and pinned
  runtimes, hidden/unwritable sibling home and `/mnt`, and a finite TensorFlow
  GPU/XLA/TF32 computation. Artifact SHA-256:
  `d76e0808c3e28e772bbceb6d93eef5df7a616e8c1d738e3a874a0fdd41f8920a`.
- The schema-v4 builder keeps legacy `ready_except_human_approval=false` and
  `launch_authorized=false`. Static technical evidence is reported separately
  as `static_evidence_ready_for_launch_review`; it cannot authorize launch.

Launch review iteration 1:

- Three fresh read-only substitute reviews returned `VERDICT: REVISE`.
- The detached-plan review found authority precedence, source handoff,
  out-of-process recovery, and SHA-bound review defects.
- The implementation review found signal forwarding, pre-baseline export,
  namespace proof, Git failure, baseline identity, and total-runtime defects.
- The manifest review found concrete-identity, timeout-enforcement,
  unqualified-executable, stale-preflight, and human-approval-binding defects.
- Repairs now use a concrete run identity, constrained absolute executables,
  absolute launch-epoch deadlines, immediate watchdog fallback, post-export
  recomputation, exact-SHA receipts, manifest-bound external/runtime hashes,
  and selected-runtime-only model visibility.

## 2026-07-12 - Detached Launch - Iteration 2 Repair

Iteration-2 plan and manifest reviews returned `REVISE`; the implementation
review stalled and produced no valid verdict. The defects were missing Claude
disclosure, late live-source closure, no outer timeout over preparation,
unstable producer-log sealing, incomplete reviewer authority limits,
incomplete whole-workspace/runtime binding, contradictory readiness state,
weak fresh-probe identity/freshness rules, inconsistent clocks, and an unbound
relative supervisor path.

Repairs now use schema v4, a complete frozen-source inventory, aggregate
runtime fingerprint, hash-bound ledger/stop overlays, recursively read-only
live mounts before probes/copy, exact approval identity, absolute frozen paths,
and the common-clock deadline ladder `26400, 26700, 27360, 27600, 28180,
28200, 28230, 28720, 28740, 28800`. Intermediate exports exclude live
producer/control files and a finalizer hashes every run-prefixed file after
supervisor/watchdog closure.

Focused validation after these edits: Python compilation passed, shell syntax
passed, scoped diff checking passed, and `28` tests passed. One manifest test
is intentionally pending because the exact frozen-source inventory has not yet
been created. The reviewed master, Phase 0 subplan, and Phase 1 subplan remain
byte-identical. No fresh model/GPU probe, detached launch, commit, push, merge,
or source merge-back occurred.

Gate status: `ITERATION2_REPAIRED_SNAPSHOT_AND_ITERATION3_REVIEW_PENDING`

Next action: freeze and verify the exact source snapshot, regenerate/check the
schema-v4 manifest, run the complete local gate, then run fresh bounded plan,
implementation, and exact-manifest reviews. Only after convergence and an
exact launch receipt may the user be asked for launcher/disclosure/GPU/model
approval.

## 2026-07-12 - Detached Launch - Iteration 3 Repair

Fresh bounded plan review returned `REVISE`. It identified ambiguous writable
handoff mount ordering, a wall-clock origin captured after the outer timeout
started, incomplete Codex-readable disclosure scope, fallback export without a
proved writer-quiescence prerequisite, and a final seal whose self-exclusion
and completion authority were not explicit. The manifest-only review returned
structural `AGREE`, but it was superseded by the ensuing plan/code repairs and
cannot bind current bytes. The implementation review was interrupted because
its target bytes changed and is not counted.

Repairs now:

- preserve the fresh handoff through a unique per-run `/tmp` bind, recursively
  close the live repository, rebind only the handoff, and probe both sides;
- capture wall and monotonic origins before GNU `timeout`, use wall time only
  for freshness, and enforce every deadline from monotonic elapsed time;
- hide dynamic handoff contents from Codex and expose only the frozen inventory,
  two exact overlays, and four manifest-bound read-only support files;
- require termination and confirmed absence of every known writer process group
  before watchdog fallback export;
- validate the post-export verification chain and foreground outcome before a
  self-excluding final seal, then make the handoff read-only and probe that no
  later write succeeds.

Focused post-repair validation currently passes `32` tests plus Python compile,
shell syntax, and scoped diff checks. The frozen snapshot, isolation GPU/XLA
preflight, manifest, and iteration-3 reviews are now stale by design and must be
regenerated against final bytes. No detached launch, real Codex disclosure,
Claude call, commit, push, merge, or source merge-back occurred.

Gate status: `ITERATION3_REPAIRED_REFREEZE_FULL_GATE_AND_REVIEW_PENDING`

## 2026-07-12 - Detached Launch - Iteration 4 Repair

The schema-v5 plan and implementation reviews returned `REVISE`. The
manifest-local review returned `AGREE` but cannot bind changed bytes. The
consolidated blockers were an undefined launcher manifest variable, incomplete
Claude technical disclosure, process-group-only producer closure, expired
budget rounding and a fixed launcher wait, stale approval use across probes and
copy, asserted rather than enumerated primary-only export, receipt verification
after handoff creation, and mount discovery not checked against an independent
kernel inventory.

Schema v6 now verifies the exact manifest and review receipt before creating
the live handoff; rechecks approval expiry at every disclosure/copy/launch
boundary; derives waits from one monotonic origin; rejects an expired budget;
cross-checks `findmnt` against `/proc/self/mountinfo`; requires the entire outer
PID namespace to be quiescent before outcome and seal; and independently scans
for unapproved export artifacts. Claude disclosure now states that Claude
inherits OS-level read/write access to the isolated copied repository and
private temporary storage. At that iteration, the disclosure described
read-only/single-path scope as prompt/settings/tool policy rather than
filesystem isolation; iteration 5 later found that wording still overclaimed
settings/tool enforcement because edit and command capabilities remain
technically available.

Focused validation passes `29` tests plus Python compilation, shell syntax, and
scoped diff checking. The immutable master, Phase 0 subplan, and Phase 1 subplan
remain at their reviewed hashes. No real restricted Codex probe, Claude call,
detached launch, leaderboard run, commit, push, merge, or merge-back occurred.

Gate status: `ITERATION4_REPAIRED_SCHEMA_V6_REFREEZE_FULL_GATE_AND_FINAL_REVIEW_PENDING`

Next action: regenerate and verify the frozen snapshot, run trusted GPU/XLA
isolation preflight, regenerate/check schema v6, run the full launch-control
gate, then perform the final bounded plan/implementation/manifest review round.
Only an all-`AGREE` final round permits writing a launch-readiness receipt and
asking for exact human approval.

### Iteration-4 Trusted Preflight Repair

The first schema-v6 trusted isolation attempt exited `1` because the synthetic
preflight preparer omitted the new `approval_not_after_epoch` field from its
launch-preparation receipt. The namespace entrypoint failed closed before GPU
execution or isolated-workspace mutation; the structured artifact recorded
launcher exit `22`, source workspace unchanged, sibling-home probe unchanged,
and no boundary/terminal result. This invalidated the preflight harness bytes,
not the GPU, scientific target, data, or leaderboard candidate.

The synthetic preparer now supplies a one-hour bounded approval expiry, with a
focused static regression assertion. Because the preflight script is
snapshot/hash-bound, the snapshot, inventory, and failed preflight artifact
must be regenerated before the attempt can count.

The second trusted attempt also failed closed before GPU execution. Its log
reported `findmnt home-tree inventory was partial or inconsistent`; source and
sibling-home probes again remained unchanged and no terminal/boundary artifact
was produced. The independent inventories differed because stacked bind mounts
can produce duplicate mount IDs at one target in `/proc/self/mountinfo`, while
`findmnt` reports the reachable target once. This is an inventory-normalization
defect, not evidence against the GPU or leaderboard implementation.

The proof now compares unique target sets from `findmnt` and
`/proc/self/mountinfo` in both outer and home boundaries. It still fails if
either source omits any reachable target, while correctly treating duplicate
mount IDs at the same target as one reachable path. This byte change again
invalidates the snapshot and trusted preflight artifact.

The third trusted attempt passed. Artifact SHA-256:
`a8b1699715647a5a98238cac56a4c4b1ab5ae63b23e6a06c8d5a4a83cae1e6f7`;
preflight-script SHA-256:
`5c1ec6b2b2ae7c01c3a5b3da286b53248f0ce034a6f1cfdd5cda524028670db3`.
The structured result records real TensorFlow GPU placement, XLA JIT, TF32,
finite output, empty effective/bounding capabilities, `NoNewPrivs=1`, hidden
handoff and staging aliases, verified support hashes, source preservation,
primary export verification, and whole inner PID-namespace quiescence. This is
infrastructure evidence only; it does not authorize launch or support a
leaderboard, posterior, HMC, ranking, or scientific claim.

### Final Schema-v6 Static Gate

- Frozen source inventory: `11,920` entries, `1,117,534,453` bytes,
  entries SHA-256
  `43b2aa1895dfef056c1cc85593812487c084dbf55ef2829e17dd431b8e3a44eb`,
  inventory-file SHA-256
  `64a8ef1ad3c7c97fd391dcae19e3f16cf7dfbd2f5a7def41339ab51ad8761722`.
- Runtime fingerprint SHA-256:
  `598f9b031eee92c7cf6a51e09ae1a84a7ea4c395f0e1a6f1786a0fddf03cf616`.
- Trusted isolation preflight SHA-256:
  `a8b1699715647a5a98238cac56a4c4b1ab5ae63b23e6a06c8d5a4a83cae1e6f7`.
- All nine launch-control modules passed `43/43` tests. Python compilation,
  shell syntax, frozen snapshot verification, runtime fingerprint verification,
  exact manifest check, and scoped diff hygiene passed.
- Immutable reviewed hashes remain master
  `e8edb25929a0c6448440d1f841a880227f272683727d78263e6063ca82ad8a05`,
  Phase 0
  `60602e00923e6637d7d40fb762ddc50a8f57eefb3407ec99b17673a3a0faa18e`,
  and Phase 1
  `ff75b73fdbc2f75c0d5f05c0ac835fdfec69cc7ccd1448b47c6f66b2d9ebb62b`.
- Schema-v6 static evidence is ready for final launch review. Human approval,
  post-approval probes, conditional copy/launch authority, and scientific
  authority remain false.

Gate status: `FINAL_PLAN_IMPLEMENTATION_MANIFEST_REVIEWS_PENDING_NOT_AUTHORIZED`

## 2026-07-12 - Detached Launch - Owner Waiver And Schema v7 Preparation

The owner accepted the five final review findings for run
`complete-highdim-leaderboard-20260711-221500` only. This is a one-run
continuation decision, not a technical repair and not repository policy. The
binding amendment is
`docs/plans/bayesfilter-complete-highdim-leaderboard-run-risk-acceptance-amendment-2026-07-12.md`.

Accepted limitations:

1. Claude's bound worker/settings technically expose edit and command tools;
   read-only behavior is an instruction/prompt contract.
2. Claude may read the ephemeral private Codex `auth.json` through inherited
   `CODEX_HOME`.
3. Launch-time validators do not independently require the complete five-file
   primary export set.
4. The seal precedes alias lock. The new post-lock receipt rehashes the final
   state but does not prove no transient write occurred before lock.
5. Trusted preflight covers the synthetic inner GPU/Codex boundary, not the
   exact outer production wrapper/seal route.

Schema v7 binds those limitations as
`owner_accepted_run_scoped_limitations`, sets repository-default/reuse/technical-
repair fields false, binds the waiver plus the post-run audit plan/helper, and
keeps final-launch and completion/release authority false. The generic phase
review verifier remains capped at iterations `1..5`; only the run-scoped
launch-readiness receipt may use owner-authorized iteration `6`.

The mandatory post-run plan is
`docs/plans/bayesfilter-complete-highdim-leaderboard-post-run-integrity-audit-plan-2026-07-12.md`.
Its structural helper cannot alone issue the full audit pass: exported Phase
8/9 completeness evidence, all 24 cells, six LEDH five-seed/FD records, and
sidecar separation must also pass before the separate result can record
`PASS_POST_RUN_INTEGRITY_AUDIT`. No completion or release claim is permitted
before that result.

Skeptical audit: the accepted risks are not proxy promotion criteria and are
not encoded as repaired guarantees. The exact baseline remains the reviewed
Phase 0/Phase 1 state. The continuation veto is now sixth-round nonconvergence,
waiver/audit inconsistency, stale identity, check failure, unsafe approval
expiry, or missing fresh final-command approval. Trusted preflight remains
explanatory for the accepted outer-route gap and valid only for its inner
boundary scope.

Focused CPU-hidden schema/waiver tests passed `14/14`; Python compilation and
scoped diff checks passed. This is preparation evidence only. Snapshot,
inventory, runtime fingerprint, trusted preflight, schema-v7 manifest, sixth
review, readiness receipt, and final command presentation remain pending. No
live handoff, launch workspace, real restricted Codex probe, detached launch,
or leaderboard execution occurred.

The full CPU-hidden structural gate then passed `49/49` tests across all ten
launch-control modules, plus bound Python compilation, shell syntax, and diff
hygiene. The trusted TensorFlow device probe passed on the RTX 4080 SUPER with
GPU placement, XLA JIT, TF32 enabled, and finite output; artifact SHA-256:
`eef214487dea37010f7c8accba2cababf6a00ce9aa61bc9329ec55cbf466b845`.
The refreshed trusted scoped-inner isolation preflight also passed; artifact
SHA-256:
`28df28f0e724346a0d94dd2e374a6c4ad055e194b50c61438f04269ebe754a48`;
preflight-script SHA-256:
`323a5cf4a90cad1bd44ba0422f1d936c15c823499fe1e60f621f681b9877d3c2`.
It records real GPU/XLA/TF32 work, hidden handoff/support verification, empty
capabilities, `NoNewPrivs=1`, and inner PID-namespace closure. Its nonclaims
explicitly preserve the accepted gap: it does not cover the exact outer
wrapper, production preparer, watchdog, finalizer, alias lock, or post-lock
seal validation. The real run paths remained absent after both probes.

Before definitive refreeze, the post-run structural auditor was hardened to
require read-only file and directory modes, exact waiver/audit/iteration-6
receipt hashes, absence of extra export-schema artifacts, safe archive members,
the complete primary file set, full seal rehash, producer/PID and mount closure,
and the namespace/watchdog/outcome/approval/conditional-authorization chain.
Its `PASS_STRUCTURAL_POST_RUN_INTEGRITY` remains only the structural stage; it
cannot issue `PASS_POST_RUN_INTEGRITY_AUDIT` without exported Phase 8/9 numeric
completion checks. The final CPU-hidden structural gate passes `51/51` tests.

Gate status: `SCHEMA_V7_WAIVER_BOUND_REFREEZE_AND_SIXTH_REVIEW_PENDING_NOT_AUTHORIZED`

## 2026-07-12 - Iteration 6 Audit/Disclosure Repair

The audit/disclosure slice returned `REVISE` while the waiver/authority and
manifest/command slices returned `AGREE`. The material gaps were explicit pass
predicates for the structural and Phase 8/9 validators, credential leak
scanning, preserved Claude tool-use evidence with actual state-changing use as
a veto, semantic diff/status inspection, and post-lock alias identity plus seal
rehashing.

The repair keeps all five owner-accepted limitations unresolved and run-scoped.
It replaces only the exposed Claude support worker with a run-specific
stream-auditing wrapper that preserves raw events, stderr, and parsed tool use;
adds an outside-seal receipt after the canonical, snapshot, and staging aliases
are locked read-only; and extends the structural auditor to require that
receipt, scan current credential values in memory without persisting values or
hashes, reject malformed or non-read-only Claude evidence, and bind a separate
semantic inspection receipt. A matching post-lock receipt detects final drift
but does not prove the absence of a transient pre-lock write.

The full audit pass predicate is now explicit: structural helper exit `0` and
`PASS_STRUCTURAL_POST_RUN_INTEGRITY`; semantic inspection pass; zero credential
matches; no observed non-read-only Claude tool use; post-lock receipt pass; and,
when numeric completion is claimed, exported Phase 8/9 validator exit `0` with
every completeness check passing. No launch, completion, or release authority
is granted. Local checks, trusted scoped-inner preflight, refreeze, regenerated
manifest, and the focused iteration-6 rereview remain pending.

Gate status: `ITERATION6_AUDIT_DISCLOSURE_REPAIR_IN_PROGRESS_NOT_AUTHORIZED`

Focused adversarial tests then covered audited Claude success/TERM metadata,
missing invocation evidence, malformed streams, observed `Bash`/`Edit`, current
credential-value leak detection without echoing the value, safe change-manifest
path classes, snapshot-alias normalization, and post-lock ordering. The full
CPU-hidden launch-control suite passed `61/61`; bound Python compilation, shell
syntax, and diff hygiene passed.

The required trusted scoped-inner fake-Codex preflight then passed on the RTX
4080 SUPER with TensorFlow GPU placement, XLA JIT, TF32, finite output, PID 1,
zero effective/bounding capabilities, `NoNewPrivs=1`, source and sibling-home
preservation, support hash verification, and visibility of the run-specific
audited Claude wrapper. Artifact SHA-256:
`e7259b2c8eebf4ac3e128998539309b85114e898560d0107ce1f1ed79a7af0de`.
It explicitly does not cover the exact outer wrapper, production preparer,
watchdog, finalizer, alias lock, or post-lock seal validation. The real launch
root, live handoff, staging alias, and post-lock receipt remain absent.

Gate status: `ITERATION6_REPAIR_CHECKS_PASS_REFREEZE_REVIEW_PENDING_NOT_AUTHORIZED`

## 2026-07-12 - Iteration 6 Convergence And Exact Launch Handoff

The final audit/disclosure rereview found two additional fail-open details: the
Claude worker and independent auditor both ignored syntactically valid
non-object JSONL values, and the audit-time writable-mount check omitted the
frozen-snapshot handoff alias. Both were repaired. Every nonblank Claude stream
line must now parse as a JSON object in both paths, and the mount veto includes
the canonical, frozen-snapshot, and staging alias targets. Focused adversarial
tests cover a top-level JSON list containing hidden `Bash` use and a writable
snapshot-alias mount.

The final CPU-hidden launch-control suite passed `66/66`. Bound Python
compilation, shell syntax, diff hygiene, frozen-snapshot verification, runtime
fingerprint verification, and schema-v7 manifest verification all passed. The
trusted scoped-inner preflight was rerun because the audited Claude worker is a
hash-bound support file. It passed on the RTX 4080 SUPER with GPU placement,
XLA JIT, TF32, finite output, PID/capability isolation, and support hash
verification. It retains the explicit nonclaim that it does not cover the
outer wrapper, production preparer, watchdog, finalizer, alias lock, or
post-lock validation.

Stable non-overlay identities at the overlay-bind checkpoint:

- frozen inventory SHA-256:
  `080eab22cfffa073faaba48b6e44c21908fce056aa758a1386930ed4cd86669f`;
- frozen entries: `11,929`; entries SHA-256:
  `08d7800d55be9054c92f472f7200e17e19043f13398e4e364f6c6c3d985bb140`;
- runtime fingerprint SHA-256:
  `598f9b031eee92c7cf6a51e09ae1a84a7ea4c395f0e1a6f1786a0fddf03cf616`;
- trusted scoped-inner preflight SHA-256:
  `ce3f75e9693f004d318b5d9ec9b89594781178e2c91e65df76277732294971d7`.

This ledger and the visible stop handoff are manifest-bound overlays. They
cannot embed the final manifest hash without changing their own hashes and
invalidating that manifest. Final manifest, packet, substitute-review, and
schema-v2 receipt identities therefore live only in the dynamically excluded
review packet, convergence record, and readiness receipt. Any earlier values
for those four identities are pre-overlay checkpoints and are stale.

Claude produced no usable synchronous verdict; the exact availability record
is preserved, and all substitute evidence must be labeled weaker. Final static
readiness requires three bounded Codex slices against one post-overlay packet:
waiver and authority, manifest and exact command, and repaired
audit/disclosure. The external schema-v2 receipt is valid only if the
reviewed-static-readiness manifest check and frozen launcher's
`--verify-only-before-handoff` check both pass without creating a real run path.

All five owner-accepted risks remain unresolved, run-scoped, non-default, and
non-reusable. Any valid readiness receipt must keep final launch, conditional
launch, completion, release, and scientific authority false. After final
external validation, the next permitted action is to present the exact command
and stop for fresh explicit owner approval. After any launch, no completion or
result-release claim is allowed until the separate post-run integrity audit
passes.

Gate status: `OVERLAY_BOUND_NO_LAUNCH_AUTHORITY_CONSULT_EXTERNAL_READINESS_RECEIPT`
