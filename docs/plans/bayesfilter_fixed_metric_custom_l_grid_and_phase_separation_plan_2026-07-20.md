# BayesFilter Custom Fixed-Metric Grid And Phase Separation Plan

Date: 2026-07-20

Status: `IN_PROGRESS`

## Objective

Make the fixed-metric HMC search accept a caller-supplied trajectory-length
grid while retaining the existing six-point grid as the default. Preserve a
clear boundary between tuning/nomination and fresh confirmation.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | Can target-specific DZ5 trajectory grids be evaluated without changing candidate semantics or confusing tuning with confirmation? |
| Candidate mechanism | Configurable integer `L` grid; each candidate independently tunes epsilon and runs discarded fixed-kernel screens. |
| Baseline | Existing default `L=(3,5,9,13,18,25)` and serial grid semantics. |
| Promotion criterion | Engineering parity for the default grid, correct execution of a custom grid, and preserved candidate-derived seed/lineage records. |
| Promotion veto | Invalid grid, duplicate/out-of-range values, changed seed identity, changed candidate order semantics, or confirmation silently invoked by the grid runner. |
| Continuation veto | None for this API change; a focused test failure requires repair before any real-target run. |
| Repair trigger | Custom-grid refinement, round-summary, or candidate-boundary test failure. |
| Explanatory diagnostics | Grid size, sorted candidate values, refinement values, and execution topology. |
| Must not conclude | HMC convergence, sampler superiority, mass adequacy, or DZ5 readiness. |

## Evidence Contract

| Item | Contract |
| --- | --- |
| Engineering question | Does a custom grid behave like the existing grid, with independent candidate seeds and no implicit confirmation? |
| Exact comparator | Existing default-grid serial tests and deterministic callbacks. |
| Primary pass criterion | Default payloads and seed behavior remain unchanged; custom grids complete with the requested candidates and correct midpoint refinement. |
| Veto diagnostics | Schema mismatch, candidate identity mismatch, seed collision, invalid grid acceptance, or hidden confirmation execution. |
| Explanatory only | Candidate count, runtime, and ordering. |
| Nonclaims | No stochastic ranking, convergence, posterior validity, or final-kernel handoff. |
| Artifact | This plan, focused BayesFilter tests, and the implementation diff. |

## Skeptical Audit

- **Wrong baseline:** rejected. The default grid remains unchanged and is the
  semantic oracle.
- **Proxy promotion:** rejected. Grid completion is an engineering result; no
  acceptance statistic promotes a kernel.
- **Missing stop conditions:** rejected. Invalid values fail at configuration
  time; the existing candidate and shared-failure boundaries remain active.
- **Unfair comparison:** rejected. Candidate-derived seeds include `L`, so
  changing grid order does not change a candidate's random stream.
- **Hidden assumptions:** the configurable grid must contain at least three
  distinct integer values, may include `L=1` or `L=2` for target-specific
  diagnostics, and may not exceed the existing `L=25` runtime ceiling. A
  six-to-eight point grid is recommended for DZ5, but is not silently forced by
  the library.
- **Stale context:** the old DZ5 grids `(2,7,13)` and `(4,5,9)` were fixed-step
  trajectory-only diagnostics. They are not promoted as defaults.
- **Artifact sufficiency:** private candidate records retain every requested
  candidate; public summaries remain aggregate and non-replayable.
- **Phase boundary:** `run_fixed_metric_grid_search` performs tuning/nomination
  only. `confirm_fixed_metric_candidate` remains a separate caller-owned fresh
  phase. Evidence extensions are labeled as tuning evidence and are not fresh
  confirmation.

Review decision: `PASS_FOR_IMPLEMENTATION`

## Implementation

1. Generalize grid validation and payloads while retaining the default.
2. Lower the library's minimum accepted `L` to one so target-specific grids can
   include the historical DZ5 baseline `L=2`; do not change the default grid.
3. Generalize midpoint refinement and public planning counts to use the active
   configured grid rather than the default constant.
4. Add focused tests for six-, seven-, and eight-point custom grids, `L=2`,
   custom-grid refinement, order-independent seeds, and explicit confirmation
   separation.
5. Require an adapter-declared target-domain classification before treating a
   TensorFlow `InvalidArgumentError` as evidence to shrink epsilon. Unclassified
   TensorFlow errors fail loudly as runner/contract failures.
6. Run the focused suite, compile checks, and diff checks with accelerator
   devices hidden. No scientific GPU run is authorized by this plan.

## Numeric Provenance

| Choice | Provenance | Status |
| --- | --- | --- |
| Default grid `(3,5,9,13,18,25)` | Existing reviewed BayesFilter API | preserved default |
| Minimum accepted `L=1` | HMC validity and need to represent historical DZ5 `L=2`; not a preferred default | configurable diagnostic bound |
| Maximum accepted `L=25` | Existing serious tuning/runtime policy | preserved safety bound |
| Recommended custom grid size 6-8 | DZ5 robustness objective supplied by user | caller recommendation, not library hard gate |
| Confirmation separate from tuning | research-stage separation | required phase boundary |

## Execution Closeout Requirements

Record test commands and outcomes in the result note. The result must state
whether the default API is unchanged, whether custom-grid behavior passed, and
that no target-specific HMC run was performed.

## Closeout

Implementation and focused verification are complete. The default six-point
grid remains unchanged, target-specific grids with six to eight distinct values
are accepted, `L=2` is representable, and midpoint refinement uses the active
configured grid. The grid runner does not invoke fresh confirmation; confirmation
remains a separate explicit API call. No target-specific HMC run was performed.

See:
`docs/plans/bayesfilter_fixed_metric_custom_l_grid_and_phase_separation_result_2026-07-20.md`.
