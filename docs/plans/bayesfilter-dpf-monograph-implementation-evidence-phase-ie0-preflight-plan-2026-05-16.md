# Phase IE0 plan: preflight, inventory, and plan audit

## Date

2026-05-16

## Purpose

Establish the implementation-and-evidence lane before any executable diagnostic
work starts.  IE0 verifies scope, worktree state, artifact inventory, source
availability, and plan consistency.

## Allowed Write Set

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-*`;
- reviewer-grade reset memo continuity section only, if IE0 executes.

No experiment code should be created in IE0 unless the result records a
specific need for an empty directory placeholder.

Read-only review of this plan is not IE0 execution.  IE0 execution requires
writing the IE0 result artifact and then updating reset-memo continuity.  A
read-only reviewer may return `ACCEPT` for execution readiness without creating
those artifacts.

IE0 records its required governance evidence in the IE0 Markdown result
artifact only.  Machine-readable JSON records begin at IE1 if the evidence root
already exists, and otherwise at IE2.  This Markdown-only IE0 result satisfies
the master program governance requirement for IE0 because IE0 is a planning,
inventory, and readiness gate rather than a numerical diagnostic phase.

## IE0 Evidence Contract

Question:

- Is the implementation-and-evidence program execution-ready as a bounded,
  clean-room, non-production evidence lane?

Comparator/baseline:

- current master program;
- IE1--IE8 taxonomy;
- reviewer-grade reset memo continuity requirements;
- Chapter 26 diagnostic contract.

Promotion criterion:

- IE0 may exit `ie0_plan_accepted` only if the program has no lane-drift,
  governance, source-review, artifact, or execution-order blocker.

Vetoes:

- the veto diagnostics listed below.

Explanatory-only diagnostics:

- dirty-file inventory;
- Chapter 26 diagnostic-to-phase map;
- ResearchAssistant status, unless it reveals a source-review/network/mutation
  blocker.

Non-conclusions:

- IE0 does not validate mathematical derivations, code correctness, diagnostic
  numerical behavior, DPF-HMC targets, empirical claims, banking claims, model-
  risk claims, or production readiness.

## Inputs

- reviewer-grade reset memo dated 2026-05-15;
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-reset-memo-2026-05-15.md`;
- reviewer-grade final readiness report:
  `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-2026-05-15.md`;
- Chapter 26 debugging crosswalk:
  `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
- this master program and IE1--IE8 subplans.

## Tasks

1. Record branch and `git status --short`.
2. Classify dirty files by lane.
3. Verify IE1--IE8 subplans exist and match the master taxonomy.
4. Verify no student-baseline plan is referenced as validation evidence.
5. Inspect ResearchAssistant status using only local, offline, read-only MCP
   queries and record whether source intake is authorized or deferred:
   - `ra_workspace_status`;
   - `ra_privacy_status`;
   - `ra_parser_tool_matrix`.
   Do not trigger network, ingestion, fetch, or mutation workflows; defer those
   to IE1 with `ie1_source_review_deferred` unless a separate grant exists.
6. Inspect Chapter 26 diagnostic tables and map them to IE2--IE7.
7. Run a mandatory read-only Claude/Codex skeptical plan audit and record the
   pass/fail note in the IE0 result.  The audit must check for wrong
   baselines, proxy metrics promoted improperly, missing stop conditions,
   unfair comparisons, hidden assumptions, stale context, environment mismatch,
   and artifact/question mismatch.
8. Record the clean-room import decision: no production `bayesfilter/` imports
   are allowed unless a later transition plan is accepted.
9. Create a research-intent ledger for the full program: diagnostic questions,
   intended use, prohibited use, and claim boundaries.

## Primary Criterion

The program is executable as a bounded, non-production evidence lane with clear
phase gates and no student-lane or production-lane drift.

## Veto Diagnostics

- any subplan requires production edits before evidence exists;
- any phase depends on student code as validation evidence;
- any phase lacks an exit label or artifact contract;
- IE2 is not a prerequisite for executable diagnostics;
- source-review requirements imply network/fetch work without authorization.
- the program lacks a skeptical audit, research-intent ledger, or clean-room
  import decision before executable diagnostics.

## Expected Result Artifact

`docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie0-preflight-result-{YYYY-MM-DD}.md`

The IE0 result artifact must contain:

- branch and dirty-file inventory, classified by lane;
- skeptical audit result;
- research-intent ledger for the full program;
- clean-room import decision;
- ResearchAssistant local/offline/read-only status;
- Chapter 26 diagnostic-to-phase map;
- IE1--IE8 subplan inventory;
- primary criterion and veto diagnostic review;
- exit label.

After IE0 executes, the reviewer-grade reset memo must receive a short IE0
continuity update under `## Next recommended program`, immediately after the
paragraph ending with "sensitivity unless a later actual-target-instantiation
phase is accepted." and before the paragraph beginning "Until that program
passes".  The reset memo is not edited during read-only review.

## Exit Labels

- `ie0_plan_accepted`;
- `ie0_plan_needs_revision`;
- `ie0_plan_blocked`.
