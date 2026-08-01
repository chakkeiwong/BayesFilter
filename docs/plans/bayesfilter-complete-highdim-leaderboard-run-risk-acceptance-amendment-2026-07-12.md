# Complete High-Dimensional Leaderboard Run Risk-Acceptance Amendment

Date: 2026-07-12

Status: `OWNER_APPROVED_RUN_SCOPED_WAIVER_REVIEW_AND_LAUNCH_APPROVAL_PENDING`

## Binding Owner Authorization

The repository owner authorized the following on 2026-07-12:

> For run complete-highdim-leaderboard-20260711-221500 only, I accept the
> documented Claude tool, credential-access, export-completeness, seal-race,
> and preflight-coverage risks. These are owner-approved one-run exceptions and
> must not become repository defaults. Authorize a risk-acceptance amendment
> and a sixth review round solely to bind this waiver and prepare the exact
> launch. Do not claim completion or release results until a separate post-run
> integrity audit passes. Ask me again before executing the final exact launch
> command.

This amendment records that authorization. It does not itself authorize the
final launch command.

## Exact Scope

- Run ID: `complete-highdim-leaderboard-20260711-221500`.
- Launch root:
  `/tmp/complete-highdim-leaderboard-20260711-221500-workspace`.
- Live handoff:
  `docs/plans/logs/complete-highdim-leaderboard-20260711-221500`.
- Copy-sentinel nonce: `53a8d896f02a35096f2bb8ff28bfb3fa`.
- Existing approval instance: `d34c0b4076dd7b1e9cdf225a785fa58c`.
- Review exception: one launch-readiness round numbered `6`, solely for the
  schema-v7 waiver/audit/command package for this exact run.

The ordinary maximum-five-round rule remains unchanged for every phase,
subplan, blocker, future run, and repository workflow. This amendment must not
be copied into `AGENTS.md`, generalized in code, or treated as precedent.

## Owner-Accepted Run-Scoped Limitations

These are accepted risks, not repaired guarantees:

| ID | Accepted limitation | Exact technical meaning |
| --- | --- | --- |
| `CLAUDE_TOOL_CAPABILITY` | Claude tool risk | The bound Claude worker/settings technically permit edit and command capabilities. Read-only behavior is an instruction and role contract, not a deny-by-construction tool or filesystem boundary. |
| `CLAUDE_CODEX_CREDENTIAL_ACCESS` | Claude credential-access risk | Claude runs under the same isolated UID and may read the ephemeral private `CODEX_HOME/auth.json` used by Codex, in addition to the configured Anthropic credential channel. Secret values must not be logged or hashed. |
| `PRIMARY_EXPORT_COMPLETENESS` | Export-completeness risk | Launch-time supervisor/watchdog/finalizer validators verify records listed in the primary hash ledger but do not independently require the complete five-file primary export set. A partial ledger can pass those validators. |
| `SEAL_LOCK_TOCTOU` | Seal-race risk | The finalizer writes the self-excluding seal before the outer boundary removes write permission and remounts the canonical, snapshot, and staging aliases read-only. A separate post-lock receipt rehashes the final seal and sealed files after all three aliases are locked, so a final mismatch vetoes the audit; it still cannot prove that no transient write occurred during the pre-lock interval. |
| `TRUSTED_PREFLIGHT_OUTER_COVERAGE` | Preflight-coverage risk | The trusted GPU/XLA preflight covers the inner Codex/GPU isolation route with a synthetic preparer. It does not execute the exact outer wrapper, production preparer, watchdog, finalizer, alias lock, or post-lock seal validation. |

The manifest, launcher disclosure, readiness receipt, and final approval request
must identify all five items as `owner_accepted_run_scoped_limitations`. They
must not call them fixed, mitigated away, proven safe, repository policy, or
technical guarantees.

## Mandatory Post-Run Integrity Gate

The binding audit plan is
`docs/plans/bayesfilter-complete-highdim-leaderboard-post-run-integrity-audit-plan-2026-07-12.md`.
After the wrapper returns, the handoff remains provisional until an independent
post-run audit:

1. enumerates and rehashes every handoff file and every seal record;
2. requires the complete primary manifest/archive/diff/status/hash-ledger set;
3. compares primary hash-ledger membership, archive contents, and change
   manifest contents;
4. verifies producer/namespace closure evidence and all handoff aliases are no
   longer writable or mounted writable, requires the external post-lock receipt
   to prove all three aliases identified the same directory at lock time, and
   requires its seal/file rehashes to match the later canonical handoff;
5. verifies any claimed complete result has exactly 24 main cells, all six LEDH
   five-seed records for seeds `81120..81124`, and no parameterized-SIR sidecar
   contamination, with the exported Phase 8/9 validator exiting `0` and every
   required check passing;
6. scans current Anthropic credential values and sensitive string values loaded
   from the Codex auth source in memory across every handoff file and safe
   archive member, persisting neither secret values nor their hashes, and vetoes
   any match;
7. requires preserved Claude event/tool metadata to parse completely and vetoes
   any observed edit, command, or other non-read-only tool use;
8. requires a separately written semantic inspection receipt binding the change
   manifest, diff, status, Claude raw streams, stderr, and tool metadata; and
9. writes a separate audit result with a direct `PASS` or `FAIL` verdict only
   after the structural helper exits `0` with
   `PASS_STRUCTURAL_POST_RUN_INTEGRITY`.

Any missing, inconsistent, mutable, uninspectable, or unbound artifact fails
the audit. Because the seal race is accepted, a post-run match can detect
inconsistency but cannot prove the absence of an unobserved transient write;
the result must retain that limitation.

## Authority State

- Sixth launch-readiness review: authorized for this amendment package only.
- Refreeze, static checks, trusted GPU/XLA preflight, and bounded Claude review:
  authorized preparation actions.
- Exact final launch command: `NOT_AUTHORIZED` until the owner is asked again
  and explicitly approves that exact command after seeing its disclosure.
- Live handoff, launch workspace, post-approval real Codex probe, detached
  supervisor, and leaderboard execution: must not be created or started during
  preparation.
- Completion/release authority: `NOT_GRANTED_BY_LAUNCH`. It may be considered
  only after the separate post-run integrity audit passes and all scientific
  release gates independently pass.

## Stop Conditions

Stop without launching if the exact run identity changes, any accepted-risk
disclosure is absent, the waiver becomes reusable, the post-run audit is not
hash-bound, static/trusted checks fail, the sixth review returns `REVISE`, the
approval expiry is unsafe, or the owner has not approved the final exact
command. Stop without completion or release claims if the post-run audit is
missing or does not pass.
