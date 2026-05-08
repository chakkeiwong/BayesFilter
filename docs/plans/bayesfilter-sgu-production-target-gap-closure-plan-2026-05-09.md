# Plan: SGU production-target gap closure

## Date

2026-05-09

## Question

What remains before SGU can move from diagnostic structural evidence to a
production BayesFilter filtering target, and what implementation path can close
or honestly retire that gap?

## Master Program Context

The master program plan already exists:

```text
docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md
docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-audit-2026-05-06.md
```

That master plan orders the work by dependency:

```text
model semantics
  -> model-specific residual evidence
  -> derivative and Hessian certification
  -> compiled static-shape parity
  -> HMC diagnostics
  -> documentation and release provenance
```

The SGU-specific successor plans are:

```text
docs/plans/bayesfilter-structural-sgu-goals-gaps-next-plan-2026-05-08.md
docs/plans/bayesfilter-sgu-causal-control-anchor-closure-plan-2026-05-08.md
```

This document is the next implementation plan after the causal-control-anchor
diagnostic pass.  It narrows the remaining work to one question: can SGU earn a
causal, local, residual-closing production target, or should SGU remain
state-support-complete plus diagnostic-only?

## Current Implementation Status

Closed implementation evidence:

- SGU state-identity completion is implemented in the DSGE client and passes
  for the selected deterministic support equations.
- The residual source is diagnosed: the active obstruction is in the
  Euler/static/control equations, not in the selected state identities.
- Pruned-state comparison does not rescue the quadratic residual gate.
- Two-slice state-control projection closes residuals, but only as an offline
  or smoothing diagnostic.
- Causal projection with the full second-order current-control anchor fails.
- Alternative causal current-control anchors were tested:
  `quadratic_without_constant` and `static_foc` close residuals, but both fail
  locality.
- The DSGE client has a committed structural-gap hypothesis test pass.
- BayesFilter adapter metadata gates pass, and no BayesFilter backend/filter
  code change is currently justified by SGU evidence.

Allowed SGU evidence:

```text
sgu_state_identity_completion_passed
sgu_two_slice_projection_diagnostic_passed
sgu_causal_control_anchor_residual_closure_diagnostic_passed
```

Active SGU blockers:

```text
blocked_sgu_causal_projection_residual_not_closed
blocked_sgu_causal_anchor_projection_nonlocal
blocked_sgu_two_slice_projection_not_filter_transition
blocked_sgu_projection_is_new_target_not_gate_b
blocked_sgu_expectation_averaging_does_not_rescue_quadratic_gate
blocked_sgu_quadratic_policy_not_better_than_linear_residual
blocked_nonlinear_equilibrium_manifold_residual
```

Key latest diagnostic:

```text
static_foc_projected_max     = 1.256486389166e-08
static_foc_next_control_norm = 3.147909874517e-02
locality_gate                = 2.000000000000e-02
```

Interpretation: SGU now has evidence that the current-control anchor matters,
but no tested anchor satisfies both residual closure and locality.

## Remaining Gaps

### Gap 1: Timing-Consistent Causal Control Map

Residual-closing anchors exist, but the best one is nonlocal.  The missing
object is a derivation-backed current-control map that is causal, equation
consistent, and local.

Hypotheses:

- H1a: `static_foc` is close because it solves the right current equations, but
  it is not yet coupled to a predictive law of motion in the correct variables.
- H1b: the nonlocal next-control move is caused by one or two equations in
  `H[3]` through `H[6]`, not by the whole residual system.
- H1c: SGU needs an augmented predictive state containing lagged controls,
  shadow controls, or volatility-correction bookkeeping.
- H1d: if a derivation cannot make the residual-closing anchor local, SGU
  should remain diagnostic-only for nonlinear residual closure.

### Gap 2: Model-Owned Implementation Surface

The strongest causal-anchor evidence currently lives in diagnostic tests.  It
is not yet a model-owned production API.

Hypotheses:

- H2a: if the derivation succeeds, the DSGE client should own an explicit
  `current_control_anchor` or production projection option.
- H2b: if the derivation fails, the diagnostic helper should stay out of the
  production model API to avoid accidental promotion.

### Gap 3: Two-Slice Semantics

Two-slice projection closes residuals by moving current controls and therefore
is not a predictive filtering transition.

Hypotheses:

- H3a: two-slice projection can be formalized as a local consistency or
  smoother diagnostic.
- H3b: it should never route through BayesFilter predictive filter metadata
  unless a separate smoother likelihood is derived.

### Gap 4: Volatility/Expectation Residual Gate

The quadratic perturbation policy fails the pointwise residual-improvement
gate at default volatility, and simple expectation averaging does not rescue
it.

Hypotheses:

- H4a: the current pointwise residual gate is too strict for the stochastic
  second-order SGU target and should be paired with an expectation-consistent
  gate.
- H4b: if no source-backed expectation gate improves the result, the
  quadratic-over-linear target remains blocked.

### Gap 5: BayesFilter Promotion, Derivatives, JIT, and HMC

BayesFilter can represent current metadata, but there is no production SGU
value target to promote downstream.

Hypotheses:

- H5a: no BayesFilter backend/filter change is needed unless SGU earns a
  causal production target or a generic diagnostic/smoother metadata field.
- H5b: derivative, compiled, and HMC gates remain blocked until the same
  model/backend pair has value evidence first.

## Phase Plan

### Phase 0: Evidence Freeze and Master-Plan Alignment

Motivation:
- prevent duplicate SGU plans from drifting away from the master program plan.

Implementation:
- record current BayesFilter and DSGE client commits and dirty files;
- link this plan from the reset memo and, if desired, `docs/source_map.yml`;
- confirm the current client evidence includes the causal-anchor diagnostic
  result and the structural-gap hypothesis tests.

Tests:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_structural_gap_hypotheses.py \
  tests/contracts/test_sgu_causal_control_anchor_gate.py
```

Exit gate:
- continue only if existing diagnostics reproduce.

### Phase 1: Equation-Level Timing Derivation for `H[3]` Through `H[6]`

Motivation:
- identify why residual-closing anchors require nonlocal next-control moves.

Implementation:
- in the DSGE client, write a derivation note that lists each equation in
  `H[3]` through `H[6]`;
- classify every variable as current, predetermined, next, stochastic, jump,
  control, or volatility-correction term;
- decompose residual contributions for `second_order`,
  `quadratic_without_constant`, `first_order`, and `static_foc` anchors;
- produce a ranked table of which equation drives the
  `static_foc_next_control_norm` excess.

Success:
- a specific equation or variable-timing mismatch explains the nonlocality.

Stop rule:
- if no equation-level source can justify a new timing map, keep SGU production
  blocked and skip production API work.

### Phase 2: Locality-Repair Candidate Design

Motivation:
- test whether the nonlocality is repairable before changing production code.

Implementation:
- design at most three candidates:
  1. static-FOC anchor with predictive next-control consistency;
  2. augmented state with lagged or shadow controls;
  3. expectation-consistent volatility bookkeeping for the current-control
     anchor;
- define variables, equations, rank conditions, and locality thresholds before
  implementation;
- explicitly forbid two-slice current-control adjustment in any predictive
  candidate.

Success criteria:

```text
projected_max <= 1e-7
projected_rms <= 1e-8
next_control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
current_control_anchor_error <= 1e-10
```

Stop rule:
- if all candidates need two-slice current-control movement, keep them as
  smoother diagnostics only.

### Phase 3: Diagnostic Implementation in DSGE Client

Motivation:
- test the candidate without contaminating BayesFilter or production APIs.

Implementation:
- add focused DSGE contract tests for each candidate;
- keep helpers local to tests unless a candidate passes;
- test default grid, sign-flipped grid, small stress grid, and at least one
  parameter perturbation near the default calibration;
- record residual max/RMS, per-equation residuals, Jacobian rank, condition
  indicators, and adjustment norms.

Tests:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_causal_control_anchor_gate.py \
  tests/contracts/test_dsge_structural_gap_hypotheses.py
```

Success:
- exactly one candidate earns a new production-candidate label, or all
  candidates receive named blocked labels.

### Phase 4: Promotion Decision and Model-Owned API

Motivation:
- separate diagnostic success from production implementation.

Implementation:
- if a candidate passes, implement the narrowest model-owned API in
  `/home/chakwong/python`;
- require result labels and timing metadata to identify the target as causal
  filtering, not two-slice smoothing;
- if no candidate passes, do not add production API and write a blocked result
  note instead.

Success:

```text
sgu_causal_filtering_target_passed
```

only if residual, locality, timing, rank, and stress gates all pass.

Stop rule:
- residual-only success remains diagnostic and must not promote.

### Phase 5: Two-Slice Smoother/Diagnostic Containment

Motivation:
- preserve useful residual closure evidence without confusing it with a
  predictive filter.

Implementation:
- write a short semantics note for two-slice projection;
- add or preserve tests that assert:

```text
sgu_two_slice_projection_diagnostic_passed
blocked_sgu_two_slice_projection_not_filter_transition
```

- route two-slice evidence only to documentation or future smoother plans.

Success:
- two-slice diagnostics remain available but cannot unlock BayesFilter
  filtering, derivative, JIT, or HMC gates.

### Phase 6: BayesFilter Integration Gate

Motivation:
- keep BayesFilter generic and avoid SGU economics in the core library.

Implementation:
- run the BayesFilter DSGE adapter guard;
- if a causal production target passes, add only generic metadata or result
  fields needed to represent the target;
- if no target passes, update only planning/provenance documents.

Tests:

```bash
cd /home/chakwong/BayesFilter
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

Stop rule:
- no causal production target means no BayesFilter backend/filter code change.

### Phase 7: Derivative, JIT, and HMC Gate Decision

Motivation:
- downstream numerical work is expensive and meaningful only for a value target.

Implementation:
- if Phase 4 passes, plan value parity, implicit derivative, compiled parity,
  and HMC diagnostics as separate follow-on plans;
- if Phase 4 fails, record the downstream blockers and stop.

Stop rule:
- do not run HMC until value, derivative, and compiled parity gates pass for
  the same model/backend pair.

### Phase 8: Documentation, Reset Memo, and Commit

Motivation:
- make the decision auditable even if the result is a blocker.

Implementation:
- update the BayesFilter reset memo with phase-by-phase results;
- update DSGE client result notes and reset memo if client files change;
- update `docs/source_map.yml` only for durable artifacts;
- run YAML parse and diff hygiene;
- commit scoped files only.

Final checks:

```bash
cd /home/chakwong/BayesFilter
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check

cd /home/chakwong/python
git diff --check
```

## Interpretation Rule

- If a causal candidate passes residuals and locality, promote it only after
  timing, rank, stress, and metadata checks pass.
- If residuals close but locality fails, record diagnostic progress and keep
  production blocked.
- If two-slice projection is the only residual-closing route, SGU remains an
  offline diagnostic or smoother candidate, not a BayesFilter predictive
  filtering target.
- If no production SGU target exists, derivative, JIT, and HMC remain blocked.

## Immediate Next Action

Run Phase 0 and Phase 1.  The highest-value next evidence is an equation-level
timing derivation for `H[3]` through `H[6]` explaining why `static_foc` closes
residuals but misses the locality gate.
