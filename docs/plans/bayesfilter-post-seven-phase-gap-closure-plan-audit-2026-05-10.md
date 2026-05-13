# Audit: Post Seven-phase Filtering Gap Closure Plan

## Date

2026-05-10

## Plan Under Audit

```text
docs/plans/bayesfilter-post-seven-phase-gap-closure-plan-2026-05-10.md
```

## Auditor Stance

Pretend the plan was written by another developer.  Check whether its execution
order can close the remaining gaps without confusing BayesFilter generic
capabilities with client-specific readiness, GPU performance claims, or HMC
readiness.

## Verdict

Approved with one execution-boundary correction.

The plan is correctly ordered:

1. push/sync the BayesFilter baseline;
2. audit MacroFinance switch-over before editing MacroFinance;
3. pilot MacroFinance static linear QR switch-over before any SVD value option;
4. inventory DSGE targets before adapter work;
5. keep SGU blocked until a causal local target exists;
6. run CPU benchmark harnesses before GPU/XLA-GPU claims;
7. require target-specific derivative branch audits before HMC;
8. implement linear SVD/eigen derivatives only if a real client needs them.

The main correction is operational rather than mathematical: Phase B2 edits the
MacroFinance repository, which is outside the BayesFilter writable root in this
workspace.  Therefore automatic execution can proceed through Phase B1 in this
repository, but Phase B2 must stop for an explicit edit/permission boundary
unless the workspace grants write access to MacroFinance.

## Missing-point Review

The plan includes the key missing pieces:

- remote synchronization of the new BayesFilter implementation commit;
- client-side parity and rollback before MacroFinance default changes;
- DSGE target inventory before nonlinear adapter work;
- explicit SGU causal-locality gate;
- benchmark metadata for dimensions, ranks, point counts, compile time, and
  steady-state time;
- escalated GPU device proof before GPU claims;
- target-specific HMC branch diagnostics;
- conditional, not automatic, linear SVD/eigen derivative work.

Additional controls required during execution:

1. Do not stage unrelated untracked files:
   - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
   - `docs/plans/templates/*:Zone.Identifier`;
   - `singularity_test.png`.
2. Do not edit MacroFinance during Phase B1.
3. Do not add MacroFinance as a BayesFilter production dependency.
4. If Phase B1 finds time-varying derivatives are needed for the first
   MacroFinance switch-over target, insert a Phase B1C and stop before B2.
5. If Phase B1 identifies a static path, B2 remains justified but requires a
   MacroFinance write/permission boundary.
6. Do not begin DSGE adapter work until MacroFinance B2/B3 are either complete
   or explicitly deferred with a reason.
7. Do not run GPU probes or GPU benchmarks without escalated permissions.
8. Do not promote HMC readiness without target-specific derivative branch
   evidence.

## Phase-by-phase Gate Assessment

### Phase A1

Approved for automatic execution.

Primary criterion:

- `origin/main` contains commit `68e1792` or a non-rewritten descendant.

Stop if:

- remote has moved with conflicts;
- the push would include unrelated untracked files;
- local tests need rerun after conflict resolution.

### Phase B1

Approved for automatic execution as read-only MacroFinance audit.

Primary criterion:

- identify one smallest MacroFinance static switch-over target, parity tests,
  import strategy, and rollback boundary.

Stop if:

- first target requires time-varying derivative tensors;
- conventions disagree on masks, jitter, or derivative order;
- client integration requires BayesFilter production imports from
  MacroFinance.

### Phase B2

Conditionally justified after B1, but not automatically executable in this
workspace unless MacroFinance write access is explicitly granted.

Stop if:

- the pilot requires editing MacroFinance outside the writable root without
  permission;
- Phase B1 does not prove a static target;
- rollback is not clear.

### Later Phases

The later phases are sound, but should not be executed in this pass unless B2
and B3 complete or are explicitly deferred:

- DSGE adapter work depends on selecting a non-SGU structural target.
- GPU/XLA-GPU benchmarking depends on benchmark scripts and escalated probes.
- HMC depends on target-specific derivative branch evidence.
- Linear SVD/eigen derivatives depend on a demonstrated client need.

## Audit Outcome

Proceed automatically through Phase A1 and Phase B1.

If Phase B1 passes and identifies a static MacroFinance target, record that
Phase B2 is justified but requires a MacroFinance write/permission boundary in
this workspace.  Stop there unless write access is explicitly granted.
