# Audit: student DPF clean-room controlled-baseline specification

## Date

2026-05-13

## Scope

This audit reviews:

- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-plan-2026-05-12.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-completion-plan-2026-05-13.md`;
- `docs/plans/bayesfilter-student-dpf-baseline-clean-room-controlled-baseline-spec-2026-05-13.md`.

The audit is limited to the quarantined student DPF experimental-baseline lane.
It does not review or authorize monograph rebuild/enrichment work, production
BayesFilter edits, vendored student-code edits, or new experiment execution.

## Audit result

Decision: `passed_with_nonblocking_tightening`.

Blocking findings: none.

Nonblocking tightening applied:

- the 2026-05-12 source plan was updated to point to the concrete 2026-05-13
  execution artifacts;
- the 2026-05-13 completion plan was updated to reflect the current user
  instruction: execute the phase, make one path-scoped commit, and do not push;
- the specification was tightened with an algorithm-scope rule so a future
  non-flow sanity baseline cannot be reported as satisfying the N128/steps10 or
  N128/steps20 controlled target.

## Checklist

| Check | Result | Evidence |
| --- | --- | --- |
| Lane boundary | Pass | All planned writes are student-baseline plan/reset/master files. |
| Monograph drift | Pass | Spec forbids monograph rebuild/enrichment edits, `docs/chapters/ch19*.tex`, and `docs/references.bib`. |
| Production drift | Pass | Spec forbids production `bayesfilter/` edits without a separate production plan. |
| Vendor edit risk | Pass | Spec forbids editing vendored student snapshots. |
| Student import risk | Pass | Spec forbids importing vendor snapshots, `advanced_particle_filter`, `2026MLCOE`, or student `src.*` modules. |
| Student code copying risk | Pass | Spec forbids copying classes, functions, control flow, tuning tricks, or numerical shortcuts. |
| Fixture completeness | Pass | State layout, observation layout, transition, observation law, covariance regimes, initial distribution, horizon, fixture seeds, and algorithm seeds are specified. |
| Metric completeness | Pass | Required metrics, optional diagnostics, null handling, failure records, and interpretation limits are specified. |
| Moderate caveat | Pass | Moderate N128/steps10 and N128/steps20 remain diagnostic variants. |
| Proxy-only interpretation | Pass | Spec states that state, position, and observation-proxy metrics are simulated-fixture proxy metrics, not correctness certificates. |
| Future gates | Pass | Preflight, implementation, execution, and commit gates are listed. |
| Scope of next implementation | Pass after tightening | Added rule that the controlled target needs a flow-step or integration-step control; non-flow references must be labeled separately. |

## Hypothesis audit

### H1: clean-room fixture contract can be specified without student code

Result: supported.

The specification gives enough model detail for a future developer to implement
the fixtures without reading student vendor source.  The specification cites
BayesFilter-owned fixture definitions and reports, not student code, as the
authority.

### H2: target settings are fixed enough for a first clean-room baseline

Result: supported with caveat preserved.

The low-noise target is N128/steps20.  The moderate target keeps N128/steps10
and N128/steps20 as diagnostic variants.  The specification does not expand the
grid and does not present moderate steps20 as uniformly superior.

### H3: metrics preserve proxy-only interpretation

Result: supported.

The metric contract separates required simulated-fixture metrics from optional
implementation-specific diagnostics.  ESS, resampling count, and likelihood are
not treated as correctness certificates.

### H4: implementation boundary is enforceable

Result: supported.

The specification includes mechanical import-boundary searches and review
checks against student imports, vendor edits, copied student code, and
production-code drift.

### H5: next decision can be made from the spec result

Result: supported.

The audit supports a result decision of
`clean_room_spec_ready_for_implementation_plan`, with caveats that the next
phase must first write and audit an implementation plan before code execution.

## Residual risks

- The future clean-room algorithm itself is not yet specified; this phase only
  fixes fixture, metric, and boundary contracts.
- Exact fixture-array reproduction can depend on random generator behavior; the
  specification requires either exact reproduction against the existing
  BayesFilter-owned arrays or explicit numerical-difference reporting.
- Observation proxy RMSE is unwhitened and should remain a proxy diagnostic.
- The moderate-noise flow-step policy remains unresolved by design.

## Recommendation

Proceed to a clean-room implementation planning phase in the student baseline
lane.  Do not implement immediately without that phase-specific plan and audit.
