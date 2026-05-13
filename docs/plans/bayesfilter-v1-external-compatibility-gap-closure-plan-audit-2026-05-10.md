# Audit: BayesFilter v1 External-compatibility Gap Closure Plan

## Date

2026-05-10

## Plan Under Audit

```text
docs/plans/bayesfilter-v1-external-compatibility-gap-closure-plan-2026-05-10.md
```

## Auditor Stance

Treat the plan as if another developer wrote it.  Check whether the pivot away
from immediate MacroFinance switch-over is reflected consistently, whether the
phases are executable inside the BayesFilter lane, and whether the plan avoids
coupling BayesFilter v1 to client repositories too early.

## Verdict

Approved.

The plan correctly changes the objective from client switch-over to external
compatibility certification.  It keeps MacroFinance and DSGE independent until
BayesFilter v1 has a stable API, local fixtures, diagnostics, benchmark
artifacts, and a later integration checklist.

The plan is executable in this workspace as a documentation/certification pass
because all ten phases can be completed as BayesFilter-local artifacts.  No
phase should edit MacroFinance, edit DSGE, run GPU commands, or claim HMC
readiness.

## Missing-point Review

The plan includes the important controls:

- lane-specific reset memo instead of shared monograph reset memo;
- explicit deferral of MacroFinance switch-over;
- BayesFilter production independence from MacroFinance and DSGE;
- stable local fixture requirement;
- optional live external test policy;
- read-only DSGE inventory;
- CPU benchmark planning before GPU benchmarking;
- escalated GPU policy;
- target-specific HMC gate;
- late SVD/eigen derivative need assessment;
- future v1 integration decision checklist.

Additional controls to enforce during execution:

1. Do not stage `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.
   It has another agent's uncommitted notes.
2. Do not stage unrelated untracked files:
   - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
   - `docs/plans/templates/*:Zone.Identifier`;
   - `singularity_test.png`.
3. Do not edit `/home/chakwong/MacroFinance`.
4. Do not edit `/home/chakwong/python`.
5. Do not run GPU commands in this pass.
6. Do not add production dependencies from BayesFilter to client projects.
7. If an optional live external test is run, label it as optional and do not
   require it for BayesFilter CI.

## Phase Gate Assessment

### Phase 1: Lane Isolation And Pivot Record

Approved for automatic execution.  It is already mostly satisfied by the
lane-specific reset memo.

### Phase 2: v1 API Freeze Criteria

Approved for automatic execution as documentation.  The artifact must
distinguish stable public API from internal/testing-only helpers.

### Phase 3: MacroFinance External Compatibility Matrix

Approved for automatic execution as documentation.  It must not imply that
MacroFinance will use BayesFilter before v1.

### Phase 4: Stable Local Fixture Gap Audit

Approved for automatic execution as read-only audit of BayesFilter tests.  It
may recommend future test implementation, but should not edit tests in this
planning pass unless the missing fixture is trivial and clearly BayesFilter
local.

### Phase 5: Optional Live External Test Policy

Approved for automatic execution as documentation.  Live external tests may be
described and optionally run later, but they are not required in this pass.

### Phase 6: DSGE Read-only Target Inventory Plan

Approved for automatic execution as documentation only.  Do not inspect or edit
DSGE source beyond already-known plan context unless a later pass explicitly
authorizes a read-only inventory.

### Phase 7: CPU Benchmark Harness Plan

Approved for automatic execution as benchmark-planning documentation.  Do not
claim benchmark results unless benchmarks are actually run in a later pass.

### Phase 8: Escalated GPU/XLA-GPU Gate Plan

Approved for automatic execution as policy documentation.  Do not run GPU
commands in this pass.

### Phase 9: HMC Readiness Gate Plan

Approved for automatic execution as gate documentation.  Do not run HMC or claim
readiness.

### Phase 10: Future v1 Integration Decision Plan

Approved for automatic execution as documentation.  It should be a checklist,
not an adapter implementation.

## Audit Outcome

Proceed through all ten phases as BayesFilter-local planning and certification
artifacts.  Stop if execution would require external repository edits, staged
shared-reset-memo changes, GPU execution, HMC execution, or client default
changes.
