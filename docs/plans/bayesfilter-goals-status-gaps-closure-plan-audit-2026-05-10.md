# Audit: BayesFilter Goals, Status, Gaps, and Gap-closure Plan

## Date

2026-05-10

## Plan Under Audit

```text
docs/plans/bayesfilter-goals-status-gaps-closure-plan-2026-05-10.md
```

## Auditor Stance

Treat the plan as if another developer wrote it.  Check whether it closes the
right gaps in the right order, whether the phase gates are measurable, and
whether automatic execution can proceed without leaving the BayesFilter lane or
stepping on other agents.

## Verdict

Approved with an execution-boundary correction.

The plan correctly identifies the main remaining work:

1. client-side MacroFinance switch-over;
2. optional MacroFinance SVD/eigen value decision;
3. DSGE target inventory and non-SGU structural adapter;
4. benchmark and escalated GPU/XLA evidence;
5. target-specific HMC readiness;
6. linear SVD/eigen derivative need assessment.

The ordering is sound because it starts with the nearest real client and keeps
GPU/HMC/SVD-derivative claims behind client-side value and derivative parity.

The execution correction is operational: Phase 1 edits
`/home/chakwong/MacroFinance`, which is outside the BayesFilter writable root
for this workspace.  Therefore automatic execution in this pass can audit the
plan and record the stop boundary, but it cannot implement Phase 1 unless a
MacroFinance write boundary is explicitly granted.

## Missing-point Review

The plan includes the key controls:

- BayesFilter production code remains TensorFlow/TensorFlow Probability only;
- NumPy remains test/reference only;
- MacroFinance and DSGE switch-over happens through client-side adapters;
- MacroFinance defaults do not change before parity and rollback are proven;
- SGU remains blocked until the DSGE client supplies a causal local filtering
  target;
- GPU/CUDA/XLA-GPU claims require escalated probes;
- HMC readiness is target-specific;
- linear SVD/eigen derivatives are conditional on real client need.

Additional execution controls required:

1. Do not stage unrelated untracked files:
   - `docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md`;
   - `docs/plans/templates/*:Zone.Identifier`;
   - `singularity_test.png`.
2. Do not edit MacroFinance in this pass without explicit authorization.
3. Do not edit `/home/chakwong/python` in this pass.
4. Do not begin Phase 7 benchmarks before the client switch-over boundary is
   either completed or explicitly deferred.
5. Do not run GPU probes or benchmarks without escalated permissions.
6. Do not add BayesFilter production imports from MacroFinance or DSGE.
7. If another agent has moved `origin/main`, fetch and review before committing
   or pushing.

## Phase-by-phase Gate Assessment

### Phase 1: MacroFinance Dense QR Value Pilot

Mathematically justified, but not automatically executable in this BayesFilter
workspace.

Primary criterion is measurable:

- BayesFilter-backed dense QR value equals MacroFinance existing dense QR value
  on the selected MacroFinance tests.

Stop condition:

- executing this phase requires editing `/home/chakwong/MacroFinance`;
- therefore stop here unless MacroFinance write access is explicitly authorized.

### Phase 2: MacroFinance Masked QR Value Pilot

Sound after Phase 1 passes.

Do not execute before Phase 1 because masked-value switch-over adds mask
semantics and should not be the first client wrapper.

### Phase 3: MacroFinance QR Score/Hessian Pilot

Sound after Phase 1, and preferably after Phase 2 or an explicit masked-path
deferral.

Do not execute before dense value switch-over because derivative adapter changes
touch MAP/HMC behavior.

### Phase 4: MacroFinance SVD/eigen Value Decision

Sound after MacroFinance QR switch-over.  It is a decision phase, not an
automatic implementation phase.

The veto against derivative requests is essential and should remain.

### Phase 5: DSGE Target Inventory

Sound, but should wait until MacroFinance Phase 1 is completed or explicitly
deferred.  DSGE work is a different client lane and risks mixing two client
switch-over efforts.

### Phase 6: DSGE Non-SGU Structural Adapter

Sound only after Phase 5 chooses a target.  It must be implemented in DSGE or a
test-only bridge, not in BayesFilter production economics.

### Phase 7: CPU Benchmark Harness

Sound as a BayesFilter-only fallback if MacroFinance edits are deferred.  If
run before MacroFinance switch-over, label the benchmark as generic and not
client-readiness evidence.

### Phase 8: Escalated GPU/XLA-GPU Benchmarks

Sound only after Phase 7 produces exact CPU benchmark shapes.  GPU commands
must use escalated permissions on this machine.

### Phase 9: Target-specific HMC Readiness Gate

Sound only after a concrete model/backend pair has passed value and derivative
parity.  Generic filter tests are not enough.

### Phase 10: Linear SVD/eigen Derivative Need Assessment

Sound as a late decision gate.  It should not become implementation work unless
Phase 4 or a DSGE singular target proves a client need.

## Audit Outcome

Proceed only through this audit and the reset-memo update in the current
BayesFilter workspace.

Do not execute Phase 1 automatically because its first real action requires
MacroFinance edits outside the declared writable root.  Commit the scoped
BayesFilter planning, audit, source-map, and reset-memo artifacts, then request
direction on whether to authorize MacroFinance Phase 1 or defer to the
BayesFilter-only benchmark fallback.
