# Audit: student DPF full-horizon EDH/PFPF sensitivity plan

## Date

2026-05-12

## Scope

This audit reviews
`docs/plans/bayesfilter-student-dpf-baseline-full-horizon-edh-pfpf-sensitivity-plan-2026-05-12.md`
as if by a separate developer before execution.

The lane is the quarantined student DPF experimental-baseline stream only.  It
does not cover DPF monograph writing, enrichment, production BayesFilter code,
or HMC target construction.

## Drift Review

No active drift is present in the plan.

The plan correctly stays on:

- the already-selected EDH/PFPF pair;
- adapter-owned calls into vendored student snapshots;
- full-horizon and bounded particle/flow-step sensitivity;
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

### Finding 1: run matrix is bounded

The plan uses 32 planned records:

- 2 fixtures;
- 2 seeds;
- 2 particle counts;
- 2 flow-step counts;
- 2 implementations.

This is large enough to test the next hypotheses and small enough for routine
local execution, assuming the 45 second per-run warning threshold is enforced.

Status: acceptable.

### Finding 2: full horizon is the correct next stressor

The preceding replicated panel succeeded at horizon 8.  Extending to the full
fixture horizon directly tests whether the panel remains useful without opening
new algorithm lanes.

Status: acceptable.

### Finding 3: sensitivity claims are scoped correctly

Particle-count and flow-step comparisons are constrained to fixed fixtures,
implementations, and matching parameter groups.  The plan prohibits claiming
superiority without metric support.

Status: acceptable.

### Finding 4: ESS and resampling semantics are protected

The plan requires implementation-specific labels for ESS and resampling counts.
This prevents the known false-common-metric risk.

Status: acceptable.

### Finding 5: lane boundary is explicit

The plan names the student reset memo as the only reset memo to update and
excludes monograph/prod paths.

Status: acceptable.

## Execution Recommendation

Proceed with FH0 through FH3 if FH0 passes.

Do not proceed if the panel requires changing vendored student snapshots,
production `bayesfilter/`, monograph files, or fixture targets.

The expected final decision must be one of:

- `full_horizon_sensitivity_ready`;
- `full_horizon_sensitivity_ready_with_caveats`;
- `needs_targeted_debug`;
- `blocked_or_excluded`.
