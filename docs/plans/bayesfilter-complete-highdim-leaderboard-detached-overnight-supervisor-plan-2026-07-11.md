# Complete High-Dimensional Leaderboard Detached Overnight Supervisor Plan

Date: 2026-07-11

Status: `OWNER_WAIVER_BOUND_SCHEMA_V7_SIXTH_REVIEW_PENDING_NOT_AUTHORIZED`

## Purpose And Precedence

Run one Codex supervisor/executor for no more than eight hours after visible
governance, exact-command, local-preflight, read-only review, and exact human
approval gates pass. Claude remains a bounded read-only reviewer.

For this run only,
`docs/plans/bayesfilter-complete-highdim-leaderboard-run-risk-acceptance-amendment-2026-07-12.md`
governs the five owner-accepted launch limitations and the authorized sixth
launch-readiness review. The amendment does not change repository defaults or
authorize launch. Its separate post-run integrity-audit hold takes precedence
over any in-copy use of `complete`, `release`, or `NUMERICALLY_COMPLETE`.

This plan is the narrow exception to the visible runbook's detached-execution
ban. The exception applies only to the one concrete wrapper argv in the exact
manifest and only after the human approves that argv, including its external
Codex workspace disclosure. It does not authorize nested Codex supervisors,
background phase runners, additional detached launches, a different run ID,
or a different copy root. Within the detached program, nested detachment
remains forbidden.

## Concrete Launch Identity

- Run ID: `complete-highdim-leaderboard-20260711-221500`.
- Source root: `/home/chakwong/BayesFilter`.
- Copy root:
  `/tmp/complete-highdim-leaderboard-20260711-221500-workspace`.
- Frozen source snapshot:
  `/home/chakwong/BayesFilter/.complete-highdim-source-snapshot-complete-highdim-leaderboard-20260711-221500`.
- Frozen source inventory:
  `docs/plans/artifacts/complete-highdim-leaderboard/source-snapshot-inventory-2026-07-11.json`.
- Runtime fingerprint:
  `docs/plans/artifacts/complete-highdim-leaderboard/runtime-fingerprint-2026-07-11.json`.
- Fresh source-side handoff:
  `docs/plans/logs/complete-highdim-leaderboard-20260711-221500`.
- Copy-sentinel nonce: `53a8d896f02a35096f2bb8ff28bfb3fa`.
- One-time approval instance: `d34c0b4076dd7b1e9cdf225a785fa58c`.
- Approval expiry: epoch `1783882800`.
- External launcher:
  `/home/chakwong/python/claudecodex/scripts/overnight_gated_launch.sh`.
- Concrete wrapper: `scripts/launch_complete_highdim_leaderboard.py`.

The run ID, paths, nonce, approval instance/expiry, exact wrapper argv,
manifest SHA, source-inventory SHA, and runtime-fingerprint SHA must be bound
by the launch review receipt. The outer boundary invokes the frozen launcher in
verification-only mode before creating the handoff directory; the launcher
loads and validates the exact schema-v7 manifest plus its receipt and repeats
validation after the read-only repository boundary is active. A changed
manifest, snapshot, runtime, path, nonce, argv, approval identity, or receipt
vetoes launch.

## Filesystem And Process Boundary

The source workspace is not generally writable. The sole source-side write
surface is the new per-run handoff directory above. It must not exist before
the wrapper starts; the wrapper creates it mode `0700` and all producers use
exclusive creation. Existing files are never overwritten.

Before execution:

1. Visible preparation freezes the exact disclosed source snapshot from a
   complete inventory. Model/session state, dynamic control artifacts, live
   logs, the manifest, runtime fingerprint, and mutable ledger/stop records are
   excluded. The required per-run handoff mountpoint exists empty inside the
   snapshot. Special files and escaping symlinks are rejected.
2. The exact frozen wrapper records one wall-clock origin and one monotonic
   origin before starting GNU `timeout`. It derives the outer timeout's
   remaining budget from that monotonic origin and passes both origins through
   every child. Wall time is used only for timestamps/freshness; all deadline
   enforcement uses the common monotonic origin.
3. The boundary creates the fresh handoff and first preserves it through an
   independently named per-run bind under `/tmp`. It then recursively remounts
   the entire live repository and every nested mount read-only, rebinds only
   the preserved handoff into the frozen snapshot's empty mountpoint, proves a
   live-source write fails, and proves a handoff write succeeds. No other live
   repository path remains writable.
4. After exact human approval, the wrapper reruns the scoped inner GPU/Codex
   boundary probe and the real restricted Codex token probe under the exact
   run ID, nonce, and approval instance. The latter is an external
   model-service disclosure and cannot run before that approval. Approval
   expiry is checked immediately before each probe and again before copy and
   detached-supervisor launch; expiry vetoes the next action.
5. The external launcher creates the empty copy root and invokes the reviewed
   copy preparer. The preparer verifies the frozen inventory and runtime
   fingerprint, materializes only inventory-listed snapshot entries, and then
   installs the manifest-bound ledger and stop-handoff overlays with exclusive
   creation and exact SHA checks. The live worktree is never a bulk copy source.
6. The preparer verifies every manifest-bound repository file in the copy,
   creates a nonce-bound sentinel, freezes the baseline, and places SHA-bound
   exporter/status-writer/review helpers in the handoff before Codex can run.
7. The namespace entrypoint bind-mounts the copy over the source path, retains
   the fresh handoff submount, makes the rest of `/home/chakwong` read-only,
   hides `/mnt` behind a private read-only tmpfs, and uses private tmpfs mounts
   for temporary state.
8. The trusted Python supervisor owns handoff writes and the only admissible
   post-Codex export. Codex runs in a second private mount/PID namespace. The
   copied workspace is staged with a non-recursive bind so the nested handoff
   mount cannot follow it into a `/tmp` alias. The canonical dynamic handoff and
   every temporary staging alias are covered by empty read-only mounts before
   capabilities are dropped. The four support files are copied, checked against
   exact manifest hashes and sizes, and individually re-exposed as read-only
   regular-file mounts in a fresh allowlisted directory after the dynamic
   handoff is hidden. A post-drop verifier requires exactly those four files,
   rechecks their hashes/sizes, and proves the canonical handoff and staging
   aliases are empty and unwritable before `exec` of Codex.
9. Codex-readable repository/user data is exactly the inventory-bound frozen
   snapshot, two hash-bound overlays, the frozen supervisor prompt, and four
   hash-bound support files. Execution also exposes the full TensorFlow/TFP/CUDA
   conda tree and Node/Codex/Claude tree read-only; the runtime fingerprint binds
   selected package trees, libraries, executables, and metadata but is not a
   byte-complete inventory of every file in either mounted tree;
   ordinary Linux system libraries/executables, private proc/tmp filesystems,
   network transport, and approved NVIDIA device/driver interfaces; an
   ephemeral private copy of the existing Codex authentication state; and the
   configured Anthropic credential channel for bounded Claude review. Secret
   values are neither logged nor hashed, and unrelated inherited environment
   variables are stripped. Unrelated sibling-home paths, mounted host drives,
   dynamic producer logs, approval records, and exports remain hidden.
10. Claude runs as a descendant inside the same isolated process surface. It
    therefore has OS-level read/write access to the copied repository and
    private temporary storage, plus read access to the exposed runtime/support
    files, system/network interfaces, configured Anthropic credential channel,
    and ephemeral private Codex `auth.json` through inherited `CODEX_HOME`.
    The bound worker/settings technically permit edit and command tools. Its
    read-only role and single-path-first scope are instruction and prompt
    contracts, not deny-by-construction tool or filesystem boundaries. Human
    approval must cover this complete technical surface.

The detached Codex may read and write the copied repository and private
temporary storage. It cannot read unrelated sibling-home paths and cannot write
the handoff, pinned runtimes, host-mounted Windows drives, or the hidden source
workspace. It must not commit, push, merge, delete, or copy changes back.
Claude remains review-only by policy; that role is not claimed as an OS-level
write restriction on the isolated copy.

## Runtime And Recovery Controls

- Codex soft cutoff: `26,400` seconds (`7h20m`).
- Codex process-group termination deadline: `26,700` seconds (`7h25m`).
- Source-host watchdog primary-verification deadline: `27,360` seconds
  (`7h36m`).
- Namespace outer timeout: `27,600` seconds (`7h40m`).
- Primary export timeout: `600` seconds inside the reserved window.
- Supervisor/watchdog hard-work deadline: `28,200` seconds.
- Producer-close deadline: `28,230` seconds.
- Outer finalizer deadline: `28,720` seconds.
- Outer wrapper TERM deadline: `28,740` seconds.
- Outer wrapper KILL deadline and absolute cap: `28,800` seconds (`8h`).

Signals are forwarded to the nested Codex process group. The private PID
namespace uses PID 1 plus `--kill-child`, and the supervisor refuses export
until the `unshare` process returns and its entire process group has no
non-zombie member. The post-drop verifier proves the untrusted process is PID 1
of that private namespace with empty capabilities and `NoNewPrivs=1`; Linux
terminates every remaining namespace member when that PID 1 exits, so the
returned `unshare` process plus the absent enclosing process group proves no
descendant retains copy access, including descendants that created a new
session or process group. The supervisor then writes a namespace-close receipt,
performs the sole primary export, recomputes every bound export hash, and writes
the post-export verification receipt. The source-host watchdog has no path to
the copied workspace and no export authority; it only independently validates
the primary export and namespace-close receipts. There is no fallback export.
Missing quiescence, primary export, verification, or watchdog `PASS` vetoes the
outcome and final seal.
The namespace entrypoint, supervisor, watchdog, every post-approval probe,
preparer subprocess, launcher wait, and finalizer compute remaining budgets
from the same launch epoch. No expired budget is rounded up to one second. The
outer GNU timeout
covers all boundary work and sends TERM at `28,740`, followed by KILL at
`28,800`. Preparation time therefore consumes the same eight-hour budget; it
cannot silently shift the scientific run or export window later.

These controls cover normal completion, Codex failure, timeout, signals,
supervisor failure, and supervisor `SIGKILL`. No in-process trap can guarantee
recovery after host crash, power loss, kernel failure, storage loss, or failure
of the watchdog itself. Such events are infrastructure failure, not scientific
evidence, and must not be represented as a complete export.

## Export Contract

The baseline excludes `.git`, generated caches, model-review scratch state,
and the shared handoff path. After the Codex PID namespace is proven closed,
the supervisor-owned primary export creates, without overwrite:

- terminal/status evidence when available;
- Codex events, stderr, and final message when available;
- baseline snapshot;
- changed/new/deleted manifest relative to launch state;
- compressed archive of changed/new files;
- tracked Git diff and status;
- intermediate SHA-256/size records that exclude producer/control files still
  capable of changing;
- a supervisor-written post-export verification receipt that recomputes every
  file bound by the primary export-hash artifact and binds the namespace-close
  receipt;
- an independent watchdog receipt that validates the same primary artifacts
  without reading or writing the copied workspace;
- a final handoff seal, written only after supervisor and watchdog producers
  close, that hashes every pre-existing run-prefixed regular single-link
  handoff file while explicitly excluding the seal from its own ledger.

The exporter validates the baseline schema and canonical root identity, fails
on `git diff` or `git status` error, rejects symlinks/hardlinks/special files in
the handoff, and refuses all output collisions. Only the `primary` label is
admissible. The handoff is for visible inspection and manual merge-back only;
no automatic application is allowed.

For the exact waived run, the launch-time supervisor, watchdog, and finalizer
do not independently require all four primary payloads plus the primary hash
ledger. This remains the accepted `PRIMARY_EXPORT_COMPLETENESS` limitation; the
separate post-run audit must require and rehash the complete five-file set.

The finalizer validates the post-export verification receipt, its bound export
hash ledger, and every transitively bound file; validates the foreground
outcome; and refuses sealing if either completion input is absent or invalid.
The supervisor and watchdog each write a descriptor binding role, PID, process
group, session, start-time ticks, PID-namespace inode, exact frozen command path,
and command hash. Missing, malformed, wrong-role, or wrong-identity descriptors
veto sealing. Closure additionally requires two observations that the complete
outer private PID namespace contains no live process except the finalizer and
its ancestor chain, so a descendant that creates a new session or process group
cannot evade the gate. The launcher applies the same namespace-wide quiescence
gate before writing the foreground outcome. After the self-excluding seal is written, the outer
boundary removes write bits, remounts the snapshot and unique staging aliases
read-only (thereby closing the canonical alias to the same inode), probes that
subsequent writes fail, and writes the external three-alias post-lock receipt. A
missing verification receipt, missing/invalid seal, or failed post-seal lock is
an incomplete wrapper result even when the scientific program wrote a terminal
status. For the exact waived run, the seal is still written before the aliases
are locked read-only. The post-lock receipt rehashes the final state, but cannot
prove that no transient write occurred in the accepted interval. This remains
the accepted `SEAL_LOCK_TOCTOU` limitation.

## Phase Review Binding

Every active subplan and every material repair amendment requires a JSON review
receipt with schema
`bayesfilter.complete_highdim_leaderboard.review_receipt.v1`, exact artifact
path and SHA-256, `VERDICT: AGREE`, reviewer type, and iteration `1..5`.
`scripts/verify_complete_highdim_leaderboard_review_receipt.py` must run
immediately before execution. Any mutation invalidates the receipt and returns
the phase to review. When Claude is unavailable after the bounded probe policy,
a fresh Codex read-only substitute may review and must be labeled weaker;
neither Claude nor Codex can authorize launch, product/default promotion,
scientific claims, source-faithfulness, release, funding, runtime expansion, or
another human boundary. Five nonconvergent rounds stop the phase.

The only exception is launch-readiness iteration `6` authorized by the owner
for `complete-highdim-leaderboard-20260711-221500`. It may review only the
schema-v7 waiver, post-run audit, disclosure, and exact-command package. Its
receipt must use the run-scoped launch-readiness schema and must record both
`final_exact_launch_command_authorized: false` and
`completion_or_release_authority_granted: false`. It does not change the
ordinary five-round limit above.

## Detached Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the reviewed program make valid progress inside the isolated copy without crossing source, review, runtime, or evidence boundaries? |
| Baseline | Phase 0 freeze, exact-SHA reviewed Phase 1 subplan, concrete schema-v7 launch manifest, run-scoped waiver, post-run audit plan, complete frozen-source inventory, and runtime fingerprint. |
| Primary criterion | Fresh restricted preflights pass; exact wrapper and the full Codex/Claude-readable data, runtime, system-interface, credential-channel, and model/API disclosure are human-approved; handoff succeeds; each phase verifies its exact-SHA receipt immediately before execution; the Codex namespace closes; terminal, primary export, supervisor and watchdog verification, exact producer closure, valid self-excluding final seal, and read-only enforcement on every handoff alias complete within the hard deadline. |
| Vetoes | Any general source write, existing/unsafe handoff, stale receipt/hash, boundary failure, missing snapshot/export, unreviewed phase, invalid GPU provenance, target/source mismatch, or false completion claim. |
| Explanatory only | Number of phases reached, wall time, logs, and incomplete candidate results. |
| Nonclaims | Launch success is not leaderboard completion or release, post-run integrity, Zhao-Cui source-faithfulness, posterior/HMC validity, ranking, or scientific validity. |

## Preflight And Launch Gate

Before launch require:

1. master, Phase 0, and exact-SHA Phase 1 review receipts pass;
2. supervisor, exporter, watchdog, receipt-verifier, and manifest tests pass;
3. trusted `nvidia-smi` and TensorFlow GPU/XLA/TF32 probe pass;
4. the scoped synthetic inner Codex/GPU boundary probe passes; this is not
   evidence for the exact outer wrapper, production preparer, watchdog,
   finalizer, alias lock, or post-lock seal route;
5. exact manifest binds every plan, prompt, script, test, preflight, external
   helper, complete frozen-source inventory, overlay, runtime executable and
   package fingerprint, deadline, path, nonce, approval identity, and concrete
   wrapper argv;
6. the owner-authorized sixth review converges on the exact schema-v7 manifest,
   five accepted limitations, one-run/non-default scope, audit hold, disclosures,
   exact command, and absence of launch/release authority;
7. the iteration-6 launch review receipt binds the exact manifest SHA, waiver
   and audit hashes, limitation IDs, and concrete run identity;
8. the user explicitly approves the exact escalated wrapper command; restricted
   Codex disclosure of the frozen snapshot, two exact overlays, frozen prompt,
   four exact support files (including the run-specific audited Claude stream
   wrapper in place of the generic worker), two pinned read-only runtime trees, required Linux
   system/proc/tmp/network/NVIDIA interfaces, ephemeral private Codex auth copy,
   and configured Anthropic credential channel while unrelated home/mounted
   drives, dynamic handoff contents, and staging aliases remain hidden; Claude's
   inherited OS-level read/write access to the isolated copied repository and
   private temporary storage plus its exposed runtime/support/system/network,
   Anthropic credential, and ephemeral Codex-auth surfaces; Claude's technically
   available edit/command tools; the read-only and single-path instruction
   contract, explicitly not a tool or filesystem boundary; all five accepted
   limitations and the mandatory post-run audit hold; and trusted GPU/XLA plus
   Codex/Claude model/API calls;
9. the wrapper reruns both restricted preflights after approval and before the
   real copy.

Claude or Codex review cannot authorize launch. A technical pass cannot do so
either. Status remains not authorized until the exact human approval is
granted.

Even after launch approval and wrapper success, all results remain provisional.
No completion or release claim is permitted until
`docs/plans/bayesfilter-complete-highdim-leaderboard-post-run-integrity-audit-plan-2026-07-12.md`
is executed separately and records `PASS_POST_RUN_INTEGRITY_AUDIT`. That pass
removes only the run-integrity hold; scientific release gates remain separate.

The exact outer boundary must write an external post-lock receipt only after the
canonical, frozen-snapshot, and staging handoff aliases all reject writes and
are mounted read-only. It must bind their common device/inode identity and
rehash the self-excluding seal and every sealed file. This detects a final
mismatch but does not repair or disprove the accepted pre-lock race. The full
outside audit additionally requires zero current credential-value matches, no
observed non-read-only Claude tool use in preserved stream metadata, a passing
semantic inspection of export diff/status and Claude evidence, structural
helper exit `0` plus `PASS_STRUCTURAL_POST_RUN_INTEGRITY`, and Phase 8/9
validator exit `0` with every required check passing.

## Iteration-2 Skeptical Repair Record

Iteration 2 returned `REVISE`. The plan review found missing Claude disclosure,
live-source closure occurring too late, no outer timeout over preparation,
unstable producer logs in intermediate hashes, and incomplete reviewer-authority
limits. The manifest review found incomplete whole-workspace binding,
contradictory readiness state, incomplete disclosure/authority fields, weak
fresh-probe identity/freshness rules, incomplete runtime closure, inconsistent
deadline clocks, incomplete replay rules, and an unbound relative supervisor
path. The implementation review did not return a valid verdict and is not
counted.

The schema-v4 repair uses a complete frozen-source inventory, aggregate runtime
fingerprints, an absolute frozen supervisor path, exact parent identity and
freshness checks, recursively read-only live mounts before probes/copy,
hash-bound overlays, the common-clock ladder above, stable intermediate export
hashes, and a post-producer final seal. These repairs remain unapproved until
fresh bounded plan, implementation, and exact-manifest reviews converge.

## Iteration-3 Skeptical Repair Record

Iteration 3 returned `REVISE`. The plan review found that the fallback exporter
could not prove quiescence of descendants that escaped known process groups and
that the disclosure omitted the pinned runtimes and did not specify post-hide
support mounts. The implementation review additionally found a readable
handoff staging alias, a missing-support path misclassified as read-only,
fail-open missing producer descriptors, a writable post-seal staging alias,
unchecked recursive mount discovery, and an expired monotonic deadline that
could still start one second of work.

Schema v5 removes fallback export entirely, makes namespace closure a
precondition of the sole primary export, makes the watchdog verification-only,
requires exact producer identity descriptors, hides and verifies every staging
alias, mounts each support file individually after handoff hiding, checks every
mount-discovery/readonly result, rejects an expired monotonic budget, and names
the full runtime/system-interface/credential disclosure. These repairs remain
unapproved until the trusted isolation preflight, full local gate, and fresh
bounded plan, implementation, and manifest reviews all converge.

## Iteration-4 Skeptical Repair Record

The schema-v5 plan and implementation reviews returned `REVISE`; the
manifest-local review returned `AGREE` but is stale after this repair. Findings
were: an undefined manifest dereference; materially incomplete Claude technical
disclosure; producer closure limited to the original PID/process group; an
expired finalizer budget rounded up to one second; a fixed launcher wait not
derived from the common clock; approval expiry checked only once; primary-only
export asserted rather than observed; receipt verification after handoff
creation; and mount inventories that did not prove `findmnt` completeness.

Schema v6 repairs these defects by loading and identity-checking the exact
manifest before use; verifying manifest and receipt before live handoff
creation; checking expiry immediately before every model/GPU probe,
materialization, copy, watchdog, and detached-supervisor boundary; deriving
subprocess waits from the common monotonic origin without positive rounding;
cross-checking `findmnt` inventories against `/proc/self/mountinfo`; requiring
complete outer-PID-namespace quiescence before outcome and seal; and scanning
the handoff for fallback-labelled, wrong-label, or otherwise unapproved export
artifacts in the launcher, watchdog, and finalizer. The human disclosure now
states Claude's full inherited OS-level surface and labels read-only/single-path
scope as policy rather than filesystem enforcement.

Focused tests pass `30/30`, including adversarial `setsid()` descendant and
fallback-export cases. This does not authorize launch. Refreeze, trusted
GPU/XLA isolation preflight, the full local gate, and one final bounded review
round remain. A material `REVISE` in that final round triggers a blocker result
under the maximum-five-round rule rather than silent launch.

The first two trusted schema-v6 isolation attempts failed closed before GPU
execution: first because the synthetic preflight receipt omitted approval
expiry, then because duplicate mount IDs were compared as a list rather than a
unique reachable-target set. Both recorded the source and sibling-home probes
unchanged and produced no scientific evidence. After those harness repairs,
the trusted scoped inner fake-Codex attempt passed with TensorFlow
GPU/XLA/TF32 placement, finite output, PID 1 in the untrusted namespace, empty
effective and bounding capabilities, `NoNewPrivs=1`, hidden/unwritable handoff
and staging aliases, verified read-only support files, source preservation,
primary export verification, and whole inner PID-namespace quiescence. Artifact
SHA-256: `a8b1699715647a5a98238cac56a4c4b1ab5ae63b23e6a06c8d5a4a83cae1e6f7`.
This is launch-infrastructure evidence only and grants no launch or scientific
authority. It does not cover the exact outer wrapper, production preparer,
watchdog, finalizer, alias lock, or post-lock seal validation.

## Owner Waiver And Schema-v7 Record

After iteration-5 nonconvergence, the owner accepted the five documented risks
for `complete-highdim-leaderboard-20260711-221500` only and authorized one
sixth launch-readiness review. The binding amendment and post-run audit plan
are dated 2026-07-12. The findings remain technically true; schema v7 records
them as `owner_accepted_run_scoped_limitations`, not repaired guarantees.

The schema-v7 implementation binds the exact limitation IDs, amendment, audit
plan, structural audit helper, run identity, and iteration-6-only receipt
contract. It preserves `final_exact_launch_command_approved: false` and false
completion/release authority. The generic phase receipt verifier remains
unchanged at iterations `1..5`.

The final CPU-hidden launch-control suite passes `51/51` tests. Bound Python
compilation, shell syntax, and diff hygiene pass. The refreshed trusted device
probe passed with GPU placement, XLA JIT, TF32, and finite output; artifact
SHA-256:
`eef214487dea37010f7c8accba2cababf6a00ce9aa61bc9329ec55cbf466b845`.
The refreshed scoped-inner isolation preflight passed; artifact SHA-256:
`28df28f0e724346a0d94dd2e374a6c4ad055e194b50c61438f04269ebe754a48`;
script SHA-256:
`323a5cf4a90cad1bd44ba0422f1d936c15c823499fe1e60f621f681b9877d3c2`.
That artifact explicitly does not cover the exact outer wrapper, production
preparer, watchdog, finalizer, alias lock, or post-lock seal validation.

The bound post-run helper is a structural stage only. It requires read-only
handoff modes, the complete five-file primary export set, exact ledger/archive/
seal rehashes, closure and completion-chain consistency, and absence of extra
export schemas. It also requires the external post-lock receipt, zero current
credential-value matches, complete Claude tool-event metadata with no observed
state-changing use, and a separately bound semantic-inspection receipt. It
cannot issue the full audit verdict without the exported Phase 8/9 completeness
validator exiting `0` with all checks passing, exactly 24 main cells, six LEDH
five-seed/FD records, and sidecar separation checks required by the audit plan.

The real launch root, live handoff, and staging alias remain absent. Refreeze,
runtime fingerprint, exact schema-v7 manifest, sixth bounded review, readiness
receipt, and fresh owner approval of the final exact command remain required.
No completion or release claim is permitted before the separate post-run audit
passes.
