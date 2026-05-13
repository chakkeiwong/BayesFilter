# Result: student DPF clean-room controlled-baseline specification

## Date

2026-05-13

## Scope

This result completes the clean-room controlled-baseline specification phase in
the quarantined student DPF experimental-baseline lane.

No experiments were executed.  No production `bayesfilter/` code was edited.
No vendored student code was edited.  No monograph rebuild/enrichment files were
edited.

## Deliverables

Created or updated:

- source specification plan:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- completion plan:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`;
- clean-room specification:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`;
- independent audit:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-audit-2026-05-13.md`;
- result note:
  `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-result-2026-05-13.md`;
- student reset memo:
  `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
- student master program next-move pointer:
  `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`.

## Goals

The phase goal was to convert confirmed student-baseline evidence into a
BayesFilter-owned clean-room controlled-baseline specification without copying
or importing student implementation code.

The specification now defines:

- fixture contract;
- target settings;
- metric contract;
- result schema;
- acceptance and veto gates;
- clean-room prohibitions;
- moderate-noise flow-step caveat;
- later implementation outline.

## Gaps closed

### Gap 1: fixture contract

Status: closed for planning.

The specification defines fixture names, state layout, observation layout,
transition model, observation model, covariance regimes, initial distribution,
horizon, fixture-generation seeds, and algorithm seed policy.

### Gap 2: metric contract

Status: closed for planning.

The specification defines required metrics, optional diagnostics,
null-handling rules, structured blocker handling, and interpretation limits.

### Gap 3: moderate-noise flow-step policy

Status: closed as caveated diagnostic policy.

The specification preserves both moderate N128/steps10 and N128/steps20 as
diagnostic variants.  It does not claim that steps20 is uniformly superior.

### Gap 4: clean-room boundary

Status: closed for planning.

The specification prohibits student imports, student adapter calls, copied
student implementation code, vendored snapshot edits, production edits without a
separate plan, and monograph-lane edits.

### Gap 5: later execution gates

Status: closed for planning.

The specification defines preflight, implementation, execution, and commit
gates for a later clean-room implementation phase.

## Hypothesis results

| Hypothesis | Result | Interpretation |
| --- | --- | --- |
| H1: fixture contract can be specified without student code | Supported | The contract is written from BayesFilter-owned fixture definitions and reports. |
| H2: target settings are fixed enough | Supported with caveat | Low-noise N128/steps20 is fixed; moderate N128/steps10 and N128/steps20 remain diagnostic variants. |
| H3: metrics preserve proxy-only interpretation | Supported | Required metrics are proxy metrics; ESS, resampling, and likelihood remain optional or implementation-specific. |
| H4: boundary is enforceable | Supported | Mechanical import searches and code-review gates are specified. |
| H5: next decision can be made | Supported | The next move is clear: implementation planning, not immediate production or monograph work. |

## Decision

`clean_room_spec_ready_for_implementation_plan`

The result is not `clean_room_spec_ready_with_caveats` because the caveats are
explicitly captured in the specification and do not block writing the next
implementation plan.  The caveats do block immediate promotion to production or
uncaveated correctness claims.

## Interpretation

The student evidence remains comparison-only and proxy-only.  The clean-room
specification is strong enough to plan a BayesFilter-owned experimental
implementation, but not enough to implement without a phase-specific plan and
audit.

The next phase should stay in the student baseline lane and should not touch
monograph rebuild/enrichment files, production `bayesfilter/`, references, or
vendored student snapshots.

## Next hypotheses to test

### I1: clean-room fixture generation can match the BayesFilter-owned arrays

Test:

- implement or script fixture generation from the specification;
- compare generated arrays against the existing BayesFilter-owned fixtures;
- record exact match or numerical differences.

Pass criterion:

- exact match, or documented differences small enough to justify using stored
  fixture arrays for the first controlled comparison.

### I2: a minimal flow-assisted clean-room baseline can run the fixed target grid

Test:

- implement a BayesFilter-owned experimental flow-assisted particle baseline
  with explicit flow-step control;
- run only low-noise N128/steps20 and moderate N128/steps10 plus N128/steps20.

Pass criterion:

- all planned records complete or fail as structured blockers;
- successful records have finite required metrics.

### I3: comparison to student aggregates remains caveated and useful

Test:

- compare clean-room metrics to student aggregate medians without using student
  agreement as a correctness certificate.

Pass criterion:

- result classifies whether the clean-room baseline is in the same qualitative
  regime, worse, better, or blocked, while preserving proxy-only interpretation.

### I4: mechanical clean-room gates catch boundary violations

Test:

- run import-boundary searches;
- inspect code for student imports, copied control flow, vendored edits, and
  production-code drift.

Pass criterion:

- no student vendor imports, no copied student implementation code, no vendored
  edits, and no production edits.

## Recommended next phase

Write a student-lane clean-room implementation plan under `docs/plans/`, audit
it independently, and only then implement under `experiments/controlled_dpf_baseline/`.

The implementation plan should be deliberately narrow: fixture generator,
result schema, one minimal flow-assisted baseline, fixed target grid, comparison
report, reset memo update, and path-scoped commit.
