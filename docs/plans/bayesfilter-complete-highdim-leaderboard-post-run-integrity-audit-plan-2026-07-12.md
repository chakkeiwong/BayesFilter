# Complete High-Dimensional Leaderboard Post-Run Integrity Audit Plan

Date: 2026-07-12

Status: `REQUIRED_BEFORE_COMPLETION_OR_RELEASE_CLAIM`

## Phase Objective

Independently audit the sealed handoff for run
`complete-highdim-leaderboard-20260711-221500` after the exact wrapper returns.
The audit decides only whether the exported handoff is internally complete,
immutable at audit time, and sufficient to inspect the governed program's
claimed result. It does not retroactively repair the five owner-accepted
launch limitations or replace the scientific release gates.

## Entry Conditions

- The final exact launch command was separately approved and has returned.
- The live handoff exists at
  `/home/chakwong/BayesFilter/docs/plans/logs/complete-highdim-leaderboard-20260711-221500`.
- No launch, supervisor, watchdog, finalizer, or Codex namespace process for
  this run remains live.
- The exact schema-v7 manifest, launch-readiness receipt, waiver amendment, and
  this plan are available at their reviewed hashes.
- The audit is run outside the detached copied workspace and does not mutate
  the handoff.

If any entry condition cannot be established, write a failing audit result and
do not claim completion or release.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the post-run handoff complete and internally consistent enough to support later inspection of this run's result despite the five accepted launch limitations? |
| Exact baseline | Schema-v7 manifest and readiness receipt for the exact run; final self-excluding seal; complete five-file primary export contract; master-program 24-cell contract. |
| Primary criterion | Every required handoff and export artifact exists, is a regular single-link file, rehashes correctly, agrees with the seal/export/archive ledgers, and any numeric-completion claim passes the 24-cell/six-LEDH/five-seed/sidecar checks. |
| Hard vetoes | Nonzero structural-helper exit or missing `PASS_STRUCTURAL_POST_RUN_INTEGRITY`; live producer or namespace; writable/mismatched handoff alias; missing/invalid post-lock receipt; missing primary file; seal/hash/archive disagreement; unsafe path/link; any current credential-value match; malformed Claude event metadata or observed non-read-only Claude tool use; failed semantic inspection; missing/nonzero Phase 8/9 validator or any failed validator check; fewer/more/duplicate main cells; LEDH seed mismatch; sidecar in main matrix; unreviewed manifest identity; or an audit exception. |
| Explanatory only | Wrapper exit status, number of phases reached, runtime, log size, descriptive cell values, and provisional scientific outcomes. |
| Not concluded | A matching post-lock receipt does not prove no transient write occurred during the accepted seal race; preserved tool events do not prove Claude containment beyond observed events; a zero-match scan does not prove confidentiality against unscanned encodings or past credentials; no statistical superiority, posterior correctness, HMC readiness, source-faithfulness, production-readiness, or release claim. |
| Preserved result | `docs/plans/bayesfilter-complete-highdim-leaderboard-post-run-integrity-audit-result-2026-07-12.md` plus a structured JSON receipt written outside the sealed handoff. |

## Required Artifacts

The handoff must contain the exact run-prefixed primary export set:

- `complete-highdim-leaderboard-20260711-221500-primary-isolated-change-manifest.json`;
- `complete-highdim-leaderboard-20260711-221500-primary-isolated-changed-files.tar.gz`;
- `complete-highdim-leaderboard-20260711-221500-primary-isolated-tracked.diff`;
- `complete-highdim-leaderboard-20260711-221500-primary-isolated-git-status.txt`;
- `complete-highdim-leaderboard-20260711-221500-primary-export-sha256.json`.

It must also contain the final seal, terminal status, namespace-close receipt,
post-export verification, watchdog status, foreground outcome, human approval,
conditional launch authorization, launch preparation, launcher handoff, both
producer descriptors, baseline snapshot, Codex events/stderr/final message,
and the fresh post-approval boundary and restricted-Codex preflight artifacts.
The outer wrapper must also leave the read-only receipt
`/tmp/complete-highdim-leaderboard-20260711-221500-post-lock-integrity.json`
outside the self-excluding seal. The audit must create, review, and bind the
outside-hand-off semantic receipt
`docs/plans/artifacts/complete-highdim-leaderboard/post-run-semantic-inspection-2026-07-12.json`.
If the program claims `NUMERICALLY_COMPLETE`, the exported archive must contain
the final dependency manifest, completeness-validator evidence, final JSON and
Markdown leaderboard, Phase 8 result, Phase 9 release result, and final review
evidence named by the executed subplans. Their exact names are discovered from
the exported Phase 8/9 plans and dependency manifest, not guessed or silently
omitted.

## Required Checks

1. Require all files in the primary hash ledger to be regular single-link files
   in the exact handoff and recompute size/SHA-256.
2. Require the hash ledger to include the exact primary manifest, archive,
   diff, and status files. Extra primary/fallback export schemas or labels veto.
3. Parse the change manifest and compare its `added` plus `modified` regular
   file paths exactly with the safe archive member set; verify deleted paths are
   absent from the archive; reject duplicate, absolute, parent-traversing,
   symlink, hardlink, device, or other special archive members.
4. Rehash every file named by the final seal and compare the seal's complete
   file-name set with all pre-seal run-prefixed handoff files. Verify the seal
   itself is a regular single-link file and was not included in its own ledger.
5. Verify namespace-close, producer descriptors, foreground outcome, watchdog,
   post-export verification, approval instance, waiver hash, audit-plan hash,
   manifest hash, and readiness-receipt hash all agree.
6. Check `/proc`, PID identity records, process groups, and mount state. No live
   process may match the recorded producer or Codex identities; no handoff alias
   may be mounted writable. A write probe is forbidden because the audit must
   not mutate the handoff.
7. Require the external post-lock receipt to bind the canonical, frozen-snapshot,
   and staging aliases to one device/inode identity after all three were mounted
   read-only. Rehash the current canonical seal and compare every recorded
   post-lock sealed-file tuple. Retain the nonclaim that this cannot prove the
   absence of a transient write before lock.
8. Extract the archive only into a fresh audit-only directory under `/tmp` with
   traversal and special-file protections. Compare the extracted bytes to the
   manifest and status/diff evidence.
9. Before the structural pass, load the current nonempty
    `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` and sensitive string values
    from `/home/chakwong/.codex/auth.json` in memory. Scan every handoff file and
    every safe regular archive member as bytes. Persist neither values nor
    hashes. Any match fails; record only counts and zero matches.
10. Require every `.complete_highdim_claude_audit/*-metadata.json` record in the
    archive to have its matching raw stream and stderr bytes, parse completely,
    and report no tool outside `Read`, `Glob`, `Grep`, or `LS`. Independently
    reparse the raw JSONL and require its event count and tool-use list to equal
    the metadata. Actual edit, command, malformed-stream, metadata/raw mismatch,
    or missing-event evidence fails even though the technical tool-capability
    risk remains accepted.
11. Independently inspect the primary change manifest, tracked diff, Git status,
    Claude raw streams, stderr, and tool metadata. Write the exact semantic
    receipt above with their SHA-256 bindings and
    `PASS_SEMANTIC_POST_RUN_INSPECTION` before invoking the structural helper.
12. Run `scripts/audit_complete_highdim_leaderboard_post_run_integrity.py` in
    read-only mode against the exact handoff, manifest, readiness receipt,
    post-lock receipt, already-written semantic-inspection receipt, and current
    Codex auth source. Require exit status `0` and verdict
    `PASS_STRUCTURAL_POST_RUN_INTEGRITY`; either condition without the other
    fails. The helper must reject a missing, stale, or nonpassing semantic
    receipt.
13. If terminal/final artifacts claim numeric completion, run the exported
   completeness validator with `--require-complete`, require exit status `0`,
   require every emitted check to pass, and require exactly six frozen
   row IDs times four frozen algorithm IDs, verify finite total values and
   correctly dimensioned total scores, and require canonical target/dependency
   bindings.
14. For all six LEDH main cells, require exactly seeds
    `81120,81121,81122,81123,81124`, paired same-scalar value/score evidence,
    five individual FD endpoint reconstructions, the FD-only threshold
    `0.05*sqrt(p)`, and trusted GPU/XLA/float32/TF32 provenance.
15. Verify
    `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` occurs only in
    a sidecar object/table and never among the 24 main keys.
16. Review the structured audit result at one exact path. A reviewer may assess
    audit correctness but cannot authorize release or erase an accepted risk.

## Phase Result / Close Record

Write the result file named above with:

- exact command and exit status;
- git commit and environment;
- manifest, receipt, waiver, audit-plan, seal, and export-ledger hashes;
- complete expected/observed artifact tables;
- process and mount closure evidence;
- 24-cell and LEDH seed/FD summaries when applicable;
- strongest alternative explanation and weakest evidence;
- direct verdict `PASS_POST_RUN_INTEGRITY_AUDIT` or
  `FAIL_POST_RUN_INTEGRITY_AUDIT`;
- explicit release eligibility `PROVISIONAL_PENDING_SCIENTIFIC_RELEASE_GATES`
  or `BLOCKED`.

## Forbidden Claims And Actions

- Do not edit, remount, unseal, merge, apply, or delete the handoff or launch
  workspace during the audit.
- Do not infer missing primary exports from logs or the copied workspace.
- Do not treat owner risk acceptance as a passing audit result.
- Do not call the run complete, release results, merge exports, or publish a
  leaderboard before the audit passes.
- A passed audit does not by itself prove scientific validity or authorize
  release; it only removes this additional run-integrity hold.

## Handoff And Stop Conditions

On pass, hand the immutable audit receipt and result to a separate scientific
release decision. On any failure, preserve the handoff, write the failing
record, classify the affected evidence, and stop. Do not repair artifacts in
place or rerun only failed cells under the same run ID.
