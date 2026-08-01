# Complete High-Dimensional Leaderboard Launch Review Nonconvergence Blocker

Date: 2026-07-12

Status: `BLOCKED_REVIEW_NONCONVERGENCE_NO_LAUNCH`

Historical status note: this record preserves the iteration-5 blocker. The
owner subsequently authorized a run-scoped waiver and one sixth review round;
see the transition record below. The five findings remain technically true.

## Decision

Do not create a launch-readiness receipt, request approval for the current exact
wrapper, run the real restricted Codex probe, create the live handoff/copy, or
start the leaderboard program. The fifth allowed launch-readiness review round
returned material `REVISE` verdicts for both plan and implementation. The
master-program stop condition requires escalation rather than a silent sixth
repair/review round.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the current schema-v6 detached harness be approved to execute the reviewed leaderboard program? |
| Primary criterion | `FAIL`: final plan and implementation reviews did not converge. |
| Hard vetoes | Incomplete Claude technical disclosure/tool restriction; incomplete required primary export set; seal/remount TOCTOU; trusted-preflight scope mismatch. |
| Explanatory evidence | `43/43` local control tests, trusted inner GPU/XLA isolation pass, snapshot/runtime/manifest checks. These do not override the hard vetoes. |
| Decision | Block detached launch and ask for human direction on a sixth repair/review round or a visible non-detached alternative. |
| Nonclaims | No leaderboard cell, Phase 1 scientific hypothesis, target, data, FD policy, algorithm, Zhao-Cui route, ranking, HMC, posterior, or release claim was tested or rejected by this blocker. |

## Final Review Findings

1. The bound Claude worker/settings do not technically enforce read-only
   behavior. The worker can use bypass permissions and the settings allow edit
   and command tools. The current prompt-only restriction was overdescribed.
2. Claude can technically read the ephemeral Codex `auth.json` via inherited
   `CODEX_HOME`; that credential access is absent from the human disclosure.
3. Export-ledger validators verify listed records but do not require all four
   primary payload artifacts plus the primary hash ledger, so a partial export
   can be accepted and sealed.
4. The finalizer writes the seal before the outer boundary locks both handoff
   aliases and does not revalidate the sealed bytes after lock, leaving a
   write-before-lock race.
5. The trusted preflight exercises a synthetic inner launch and cannot validate
   the exact outer wrapper, production preparer, watchdog, finalizer, or final
   seal. Its pass is valid only for the inner GPU/Codex isolation scope.

## Evidence Preserved

- Git commit at audit time:
  `d269f5bbd8531b878d4f25897a357fbc8f172488`.
- Exact manifest reviewed SHA-256:
  `2b6b522f6b8fa04b9968ba6576182048eae9930e2a2f9622cf5fcbbdf5bd0c44`.
- Frozen inventory SHA-256:
  `64a8ef1ad3c7c97fd391dcae19e3f16cf7dfbd2f5a7def41339ab51ad8761722`.
- Runtime fingerprint SHA-256:
  `598f9b031eee92c7cf6a51e09ae1a84a7ea4c395f0e1a6f1786a0fddf03cf616`.
- Trusted inner isolation artifact SHA-256:
  `a8b1699715647a5a98238cac56a4c4b1ab5ae63b23e6a06c8d5a4a83cae1e6f7`.
- Static gate: `43/43` tests; Python compilation, shell syntax, frozen
  snapshot, runtime fingerprint, preliminary manifest, and scoped diff checks
  passed.
- Immutable reviewed master/Phase 0/Phase 1 hashes remain
  `e8edb25929a0c6448440d1f841a880227f272683727d78263e6063ca82ad8a05`,
  `60602e00923e6637d7d40fb762ddc50a8f57eefb3407ec99b17673a3a0faa18e`,
  and `ff75b73fdbc2f75c0d5f05c0ac835fdfec69cc7ccd1448b47c6f66b2d9ebb62b`.

The exact manifest is now stale because post-review overlay/blocker records were
written, intentionally making accidental launch fail closed. The launch
readiness receipt, live handoff, launch root, and handoff staging path are
absent.

## Research Guardian Classification

- Harness validity: blocked for detached-launch promotion.
- Implementation: launch-control defects remain.
- Scientific target and data: not invalidated.
- Current candidate algorithms and FD validation rule: not evaluated by this
  launch review.
- Leaderboard research direction: remains viable but unexecuted.
- Continuation veto: the review-round limit, not a candidate failure.

## Smallest Justified Repair Program

1. Bind a genuinely read-only Claude worker/settings surface, or disclose that
   Claude has edit/command capability and accept that boundary explicitly;
   independently prevent Claude from reading `CODEX_HOME/auth.json`, or disclose
   and explicitly approve that credential access.
2. Require the exact primary manifest/archive/diff/status payload set and exact
   hash-ledger membership in supervisor, watchdog, and finalizer tests.
3. Make sealing atomic with alias lock: lock all aliases first, revalidate the
   complete handoff, then write a seal through a separately controlled path, or
   write an interim ledger and have the outer boundary emit the final seal only
   after read-only remount plus post-lock revalidation.
4. Add an exact-wrapper trusted dry run that uses the production preparer,
   watchdog, finalizer, and post-lock seal validation with a fake Codex payload
   plus real GPU/XLA boundary work.
5. If the owner authorizes a sixth review round, refreeze, rerun the full static
   and trusted gates, and obtain fresh plan/implementation/manifest reviews. Do
   not reuse the stale manifest or iteration-5 `AGREE`.

## Stop Condition

Stopped for the binding five-round review nonconvergence condition. Resume only
with explicit human direction that authorizes either a sixth repair/review
round for this detached harness or a materially different visible execution
route.

## Owner-Authorized Transition, 2026-07-12

The owner explicitly accepted all five documented risks for run
`complete-highdim-leaderboard-20260711-221500` only, prohibited their promotion
to repository defaults, authorized one sixth launch-readiness review solely to
bind the waiver and exact launch package, required a separate passing post-run
integrity audit before any completion or release claim, and required a fresh
approval request before the final exact launch command.

Binding amendment:
`docs/plans/bayesfilter-complete-highdim-leaderboard-run-risk-acceptance-amendment-2026-07-12.md`.

This transition changes the continuation decision, not the technical verdict:

- `BLOCK_CLAUDE_READONLY_OVERCLAIM` becomes the disclosed, owner-accepted
  `CLAUDE_TOOL_CAPABILITY` limitation for this run only;
- `BLOCK_CLAUDE_CODEX_AUTH_DISCLOSURE` becomes the disclosed, owner-accepted
  `CLAUDE_CODEX_CREDENTIAL_ACCESS` limitation for this run only;
- `BLOCK_PRIMARY_EXPORT_COMPLETENESS` becomes an accepted launch limitation
  with a mandatory fail-closed post-run complete-export audit;
- `BLOCK_SEAL_TO_REMOUNT_TOCTOU` remains an accepted race with a mandatory
  post-run rehash, not a repaired atomicity guarantee; and
- `BLOCK_PREFLIGHT_SCOPE_MISMATCH` remains an explicit scope limitation; the
  trusted preflight must be described only as inner GPU/Codex boundary evidence.

The stale schema-v6 manifest and iteration-5 `AGREE` remain non-authoritative.
The live handoff, launch workspace, readiness receipt, and final launch
authority remain absent until schema-v7 preparation and sixth review converge.
