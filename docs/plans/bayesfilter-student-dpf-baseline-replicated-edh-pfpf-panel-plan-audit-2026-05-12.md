# Audit: student DPF replicated EDH/PFPF panel plan

## Date

2026-05-12

## Scope

This audit reviews
`docs/plans/bayesfilter-student-dpf-baseline-replicated-edh-pfpf-panel-plan-2026-05-11.md`
as if by a separate developer before execution.

The lane is the quarantined student DPF experimental-baseline stream only.  It
does not cover DPF monograph writing, enrichment, production BayesFilter code,
or HMC target construction.

## Drift Review

No active drift is present in the plan after tightening.

The plan correctly stays on:

- student experimental baselines;
- adapter-owned calls into vendored student snapshots;
- EDH/PFPF proxy comparison only;
- explicit reporting of ESS, resampling, runtime, finite-output, and RMSE
  diagnostics.

The plan explicitly excludes:

- monograph reset memos and enrichment/rebuild plans;
- `docs/chapters/ch19*.tex`;
- `docs/references.bib`;
- production `bayesfilter/`;
- vendored student-code edits;
- promotion of student code into production;
- HMC, neural OT, differentiable resampling, stochastic flow, kernel PFF, dPFPF,
  and plotting/notebook claims.

Known repository drift hazard:

- other agents may have dirty monograph-lane files in the same worktree.  The
  execution must use path-scoped staging and must update only the student DPF
  baseline reset memo.

## Audit Findings

### Finding 1: fixed seeds needed

Original issue: "multiple random seeds" was directionally correct but not fully
executable.

Resolution: the plan now fixes seeds to `[17, 23, 31]`.

Status: closed.

### Finding 2: runtime threshold needed

Original issue: runtime was bounded conceptually, but the warning threshold was
not written into the plan body.

Resolution: the plan now sets a 30 second per-run warning threshold.

Status: closed.

### Finding 3: artifact expectations needed

Original issue: output locations were described generally, which could cause
artifact drift.

Resolution: the plan now names the JSON, summary JSON, and Markdown report
artifacts for 2026-05-12.

Status: closed.

### Finding 4: reset-memo ownership needed

Original issue: the user has repeatedly identified lane contamination between
student experimental work and monograph writing.

Resolution: the plan now states that only
`docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md` may be
updated as the reset memo for this phase.

Status: closed.

### Finding 5: comparison semantics are correctly constrained

The plan keeps ESS/resampling semantics implementation-specific and explicitly
prohibits correctness claims from student agreement.  This is necessary because
the two implementations expose different diagnostics and likelihood fields.

Status: acceptable.

### Finding 6: exit gates are sufficient

The phase has concrete veto diagnostics:

- edits outside the student lane;
- vendored student-code edits;
- production imports;
- target changes to force a result;
- unbounded runtime;
- dominant nonfinite outputs;
- oversized artifacts;
- unclassifiable decision.

Status: acceptable.

## Execution Recommendation

Proceed with RP0 through RP3 if the RP0 lane guard passes.

Do not proceed if the panel requires changing vendored student snapshots,
production `bayesfilter/`, monograph files, or fixture targets.

The expected final decision must be one of:

- `replicated_panel_ready`;
- `replicated_panel_ready_with_caveats`;
- `needs_targeted_debug`;
- `blocked_or_excluded`.
