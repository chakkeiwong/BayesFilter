# Phase 1 Repair and Refresh Note

Status: `PASS_GATE_REPAIRED`

Use the common inter-phase protocol. Record each fixture failure separately as
`HARNESS_FAIL_REPAIR`, `CANDIDATE_FAIL_REPAIR`, or `REAL_BLOCKER`; never turn a
moment residual or mode imbalance into a continuation veto without an exact
contract contradiction.

Before Phase 2, refresh its subplan with:

- fixture receipt/hash and exact residuals;
- frozen schedule/protocol identity;
- support and tail findings;
- measured runtime and remaining budget;
- q=20 controls retained as hypotheses versus controls justified by fixtures;
- the smallest q=20 pilot that can distinguish missing support from insufficient
  particle count or mutation.

## Actual repair

Attempt 1 failed with `ModuleNotFoundError: bayesfilter` before any target or
fixture ran. The wrapper now resolves `Path(__file__).resolve().parents[2]`
before importing the package, and the focused wrapper test passes. Attempt 2
passed affine density, frozen protocol, known mass, mode-missing, mutation,
defensive-tail, and metadata-parity fixtures. This was a harness repair, not a
scientific result.
