# Audit: student DPF full-horizon EDH/PFPF confirmation plan

## Date

2026-05-12

## Scope

This audit reviews
`docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-confirmation-plan-2026-05-12.md`
as if by a separate developer before execution.

The lane is the quarantined student DPF experimental-baseline stream only.  It
does not cover DPF monograph writing, enrichment, production BayesFilter code,
or HMC target construction.

## Drift Review

No active drift is present after tightening.

The plan correctly stays on:

- the already-selected EDH/PFPF pair;
- adapter-owned calls into vendored student snapshots;
- full-horizon confirmation over additional seeds;
- fixed pragmatic settings rather than broad grid expansion;
- proxy metrics and implementation-specific diagnostics;
- comparison-only interpretation.

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

### Finding 1: run matrix is appropriately narrow

The plan uses 30 planned records:

- low-noise fixture: 5 seeds x 2 implementations x 1 flow setting;
- moderate-noise fixture: 5 seeds x 2 implementations x 2 flow settings.

This is narrow enough for confirmation and avoids repeating the full sensitivity
grid.

Status: acceptable.

### Finding 2: low-noise degradation threshold needed tightening

Original issue: "materially worse" was not operational.

Resolution: the plan now defines material degradation using 75 percent of the
reference median ESS, 150 percent of reference median RMSE, and a resampling
count ceiling of 20.

Status: closed.

### Finding 3: moderate-noise flow-step policy needed tightening

Original issue: the plan listed possible policies but did not define when each
policy should be selected.

Resolution: the plan now defines bounded runtime as a 20/10 median runtime ratio
of at most 2.5 and gives explicit rules for `moderate_use_10_steps`,
`moderate_use_20_steps`, and `moderate_keep_both_as_diagnostic`.

Status: closed.

### Finding 4: clean-room boundary is protected

The plan asks for fixture, metric, setting, and caveat recommendations only.  It
does not recommend copying student implementation code or editing production
BayesFilter modules.

Status: acceptable.

### Finding 5: exit gates are sufficient

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

Proceed with C0 through C3 if C0 passes.

Do not proceed if the panel requires changing vendored student snapshots,
production `bayesfilter/`, monograph files, or fixture targets.

The expected final decision must be one of:

- `confirmation_ready_for_clean_room_spec`;
- `confirmation_ready_with_caveats`;
- `needs_targeted_debug`;
- `blocked_or_excluded`.
