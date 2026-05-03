# Plan: DSGE structural-state filtering refactor

## Date
2026-05-03

## Motivation

The current source DSGE filtering path separates pruned first-order and
second-order components, but it does not explicitly separate shock-driven
exogenous states from endogenous, predetermined, or accounting states that are
deterministic conditional on lagged states and current shocks.  This is mostly
harmless for the smallest NK model whose state vector is only exogenous shocks.
It is a structural design issue for Rotemberg NK, SGU, EZ, and NAWM-scale
models when using nonlinear sigma-point or particle filters.

The next work should therefore be treated as a mathematical and software
contract refactor, not as a numerical patch.  The goal is to make it impossible
for a future nonlinear DSGE filter to silently propagate an artificial
full-state Gaussian model when the structural model requires:

```text
m_t = T_m(m_{t-1}, eps_t; theta)
k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)
x_t = (m_t, k_t)
```

The final implementation must preserve the valid Kalman path for linear
Gaussian models while requiring structural propagation for nonlinear DSGE
filters.

## Scope

Primary codebase:
- `/home/chakwong/python/src/dsge_hmc`
- `/home/chakwong/python/tests`

Documentation home:
- `/home/chakwong/BayesFilter/docs`
- `/home/chakwong/BayesFilter/docs/chapters/ch18b_structural_deterministic_dynamics.tex`

Key source files to audit:
- `src/dsge_hmc/filters/_svd_filters.py`
- `src/dsge_hmc/filters/_svd_core.py`
- `src/dsge_hmc/filters/CUTSRUKF.py`
- `src/dsge_hmc/models/base.py`
- `src/dsge_hmc/models/canonical.py`
- `src/dsge_hmc/models/small_nk.py`
- `src/dsge_hmc/models/rotemberg_nk.py`
- `src/dsge_hmc/models/sgu.py`
- `src/dsge_hmc/models/epstein_zin.py`
- `src/dsge_hmc/solvers/perturbation.py`

## Non-goals

- Do not remove exact linear Kalman filtering.
- Do not claim the SVD sigma-point filter is production-correct until the new
  structural tests pass.
- Do not hide deterministic states by adding larger nugget noise and calling
  the result structural filtering.
- Do not start with NAWM.  NAWM should be a design target, not the first
  implementation target.

## Phase 0: Freeze and inventory

### Plan

Record the current code behavior before changing anything.

### Execution instructions

1. Record `git status` and current branch for `/home/chakwong/python`.
2. List all filtering classes and all code paths that call DSGE filters.
3. Identify which code paths use:
   - exact linear Kalman filtering;
   - augmented second-order Kalman filtering;
   - generic nonlinear SSM sigma-point filtering;
   - default DSGE SVD sigma-point filtering;
   - old square-root UKF/CUTSRUKF paths.
4. Record current tests that exercise NK, Rotemberg NK, SGU, EZ, and generic
   nonlinear SSM filters.

### Test gate

No code changes yet.  The phase passes when the inventory is written to a
result note and all relevant files/tests are identified.

## Phase 1: Re-derive structural DSGE filtering equations

### Plan

Derive the state transition equations with an explicit structural partition:

```text
x_t = (m_t, k_t)
m_t = T_m(m_{t-1}, eps_t; theta)
k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)
y_t = H_y(m_t, k_t; theta) + measurement noise
```

This phase must reconcile three layers:

1. Structural DSGE timing: exogenous vs endogenous state.
2. Perturbation order: first-order component `x_f` and pruned correction
   `x_s`.
3. Filtering approximation: Kalman, sigma-point, or particle.

### Execution instructions

Write a derivation note under `/home/chakwong/BayesFilter/docs/plans` or as a
draft section in the monograph.  It must include:

1. Linear first-order case:
   - show when the collapsed representation
     `x_t = h_x x_{t-1} + eta eps_t` is equivalent to the structural
     transition for exact Kalman filtering;
   - explicitly state why singular or rank-deficient `eta eta'` is structural,
     not a bug.
2. Nonlinear sigma-point case:
   - derive prediction when sigma points are generated in the stochastic
     shock/exogenous integration space and lifted through deterministic
     endogenous transitions;
   - contrast with the wrong full-state additive Gaussian sigma-point
     propagation.
3. Particle case:
   - derive propagation where shocks are sampled and deterministic states are
     completed by `T_k`;
   - state how proposal noise in deterministic coordinates would require a
     proposal/target correction or an approximation label.
4. Pruned second-order case:
   - combine `(x_f, x_s)` with `(m, k)`;
   - state which quantities are stochastic integration variables and which are
     deterministic corrections.
5. Model-specific maps:
   - SmallNK: state is exogenous only, so existing collapsed nonlinear path is
     not the serious case.
   - Rotemberg NK: identify `R, g, z, dy`; document that `dy` is deterministic
     by identity and should not be given artificial process noise.
   - SGU: identify exogenous `a, zeta, mu` and endogenous/predetermined
     `d, k, r, riskprem` style coordinates.
   - EZ: identify available model timing and any missing metadata.
   - NAWM design target: describe required metadata without implementing NAWM.

### Test gate

The derivation phase passes only when another-agent audit can answer:

- Which variables are stochastic integration variables?
- Which variables are deterministic completions?
- Which collapsed Kalman equations remain valid?
- Which nonlinear filtering equations must be rewritten?
- What approximation is being made by each filter backend?

### Derivative-audit gate

Before any SVD sigma-point filter is promoted as an HMC target, run a separate
derivative audit.  The 2026-05-03 field test in
`docs/plans/svd-sigma-point-derivative-tool-field-test-2026-05-03.md` found
that MathDevMCP and ResearchAssistant are useful for provenance, source lookup,
assumption prompts, and conservative abstention, but they do not yet certify
SVD sigma-point gradients or Hessians.

The derivation pass must therefore:

1. use ResearchAssistant for primary-source lookup of SVD/eigen derivative
   claims, especially Matrix Backprop spectral-gap formulas;
2. write small labeled BayesFilter equations rather than one monolithic
   Hessian;
3. use MathDevMCP `audit-derivation-v2-label` as an abstention and missing
   assumption detector;
4. treat `unverified` as a blocker for production derivative claims;
5. require finite-difference, autodiff, JVP/VJP, eager/compiled, and
   spectral-gap stress tests before using any SVD sigma-point gradient inside
   HMC.

## Phase 2: Define the structural filtering contract

### Plan

Add an explicit software contract before modifying filter algorithms.

### Proposed API objects

Add one or more dataclasses, names to be refined during implementation:

```python
@dataclass(frozen=True)
class StatePartition:
    exogenous_indices: tuple[int, ...]
    endogenous_indices: tuple[int, ...]
    deterministic_indices: tuple[int, ...]
    shock_dim: int

@dataclass(frozen=True)
class StructuralTransitionSpec:
    partition: StatePartition
    stochastic_dim: int
    integrates_over: Literal["shock", "exogenous_state", "full_state"]
    supports_pruned_second_order: bool
```

Add model methods or adapter methods:

```python
model.state_partition()
model.structural_transition_spec(theta, sol)
model.propagate_exogenous(...)
model.complete_endogenous_state(...)
model.observation_from_structural_state(...)
```

Exact method names should follow existing repo style.

### Design requirements

- Exact Kalman filters may continue to consume collapsed `(F, Q, H, R)` state
  space objects.
- Nonlinear DSGE filters must either:
  - consume a structural transition spec; or
  - explicitly opt into a named collapsed approximation.
- Models with mixed exogenous/endogenous states must not silently use the old
  default DSGE sigma-point adapter.
- The smallest NK model can implement a trivial partition where all state
  indices are exogenous.

### Test gate

Add contract tests that fail when a nonlinear DSGE filter is called on a mixed
state model without structural metadata or an explicit approximation flag.

## Phase 3: Audit existing filters against the contract

### Plan

Classify each filter as safe, conditionally safe, or structurally unsafe.

### Audit instructions

Create a table covering:

| Filter | Current behavior | Safe for linear Kalman? | Safe for nonlinear DSGE? | Required action |
| --- | --- | --- | --- | --- |
| `SVDKalmanFilter` | collapsed linear Gaussian | yes, if state-space moments correct | not applicable | keep |
| `SVDAugmentedKF` | augmented pruned linear system | conditionally | not a nonlinear structural filter | document limits |
| `SVDSigmaPointFilter` generic SSM | user-supplied transition points | yes if user supplies structural map | yes with structural transition | keep/extend |
| `SVDSigmaPointFilter` default DSGE | collapsed `h_x`, `eta eta'`, pruned `x_s` | ok for SmallNK/linear smoke | unsafe for mixed-state nonlinear DSGE | rewrite/gate |
| old UKF/CUTSRUKF paths | inspect | inspect | inspect | classify |

### Test gate

The audit phase passes when each filter path has a disposition and no
mixed-state DSGE nonlinear path remains unclassified.

## Phase 4: Implement structural DSGE adapter

### Plan

Implement a structural adapter that lets nonlinear filters propagate only
declared stochastic variables and complete deterministic DSGE states through
the structural transition.

### Implementation instructions

1. Add metadata for SmallNK, Rotemberg NK, SGU, and EZ.
2. Implement a structural transition adapter for toy DSGE-like models first.
3. Use the adapter in `SVDSigmaPointFilter` when `model` supplies structural
   metadata.
4. Preserve current generic SSM path.
5. Preserve linear Kalman and augmented Kalman behavior.
6. Gate or rename the old default DSGE sigma-point path:
   - allowed for all-exogenous SmallNK;
   - blocked by default for mixed-state DSGE models;
   - available only under an explicit `collapsed_gaussian_approx=True` style
     flag with diagnostics.

### Design detail

For sigma-point filtering, the adapter should support two modes:

1. Shock-space prediction:
   - sigma/quad points represent shocks or current exogenous innovations;
   - each point is lifted through `T_m` and `T_k`.
2. Exogenous-state prediction:
   - sigma/quad points represent current exogenous states conditional on the
     lagged filter state;
   - endogenous states are deterministically completed.

The implementation should choose the simplest correct mode first and document
the approximation.

### Test gate

The structural adapter phase passes only when toy structural tests pass before
Rotemberg/SGU/EZ are attempted.

## Phase 5: Test ladder

### Required tests

1. Metadata tests:
   - partition dimensions match `n_states()` and `n_shocks()`;
   - shock-impact nonzero rows match exogenous declarations where applicable;
   - deterministic rows are declared.
2. Linear recovery tests:
   - structural adapter equals exact Kalman likelihood on linear Gaussian
     examples.
3. Toy nonlinear DSGE tests:
   - `m_t = rho m_{t-1} + sigma eps_t`;
   - `k_t = f(k_{t-1}, m_{t-1}, m_t)`;
   - compare structural sigma filter to dense quadrature or high-particle
     reference;
   - show old collapsed filter differs in the expected way.
4. Manifold tests:
   - propagated sigma points/particles satisfy deterministic identities.
5. Rotemberg NK tests:
   - verify `dy' = y' - y` identity is preserved in structural propagation;
   - verify old default nonlinear path is blocked or labeled.
6. SGU tests:
   - verify shock-driven vs endogenous state classification;
   - run small likelihood/gradient finite checks.
7. XLA and gradient tests:
   - eager/compiled parity for structural adapter;
   - finite gradient on controlled short data;
   - no `tf.py_function` in production path.
8. HMC smoke tests:
   - only after value and gradient tests pass;
   - label as smoke/sampler-usable, not converged.

### Acceptance criteria

- No nonlinear DSGE HMC target can be created without a declared structural
  filtering contract.
- Every mixed-state DSGE nonlinear filter test either uses the structural
  adapter or explicitly opts into a named approximation.
- The monograph and reset memo document the result.

## Phase 6: Documentation and release gates

### Documentation updates

Update:

- BayesFilter chapter 18b with final equations and implementation outcome.
- BayesFilter production checklist.
- Source project docs under `/home/chakwong/python/docs` only after the
  derivation and implementation pass.
- Reset memos in both repos as needed.

### Release gate

Before any future claim that SVD or sigma-point filtering works for DSGE HMC,
the report must state:

- model partition;
- filter backend;
- stochastic integration variables;
- deterministic completion map;
- approximation label;
- value tests;
- gradient tests;
- compiled-path tests;
- HMC smoke or convergence status.

## Phase 7: Commit policy

Commit only after:

1. Derivation note is written and audited.
2. Code contract tests pass.
3. Structural toy tests pass.
4. Existing Kalman/SVD generic tests still pass.
5. Reset memos are updated.

Do not commit generated PDFs unless explicitly requested.

## Risks

- Some current model classes may not expose enough timing metadata to complete
  the structural transition without additional derivation.
- Rotemberg NK and SGU may require separate structural transition adapters
  rather than one generic formula.
- Pruned second-order structural propagation may require careful handling of
  whether the deterministic completion uses lagged or current first-order
  components.
- If raw autodiff through the structural adapter is unstable, custom gradients
  may still be required.

## Recommended immediate next action

Run Phase 0 and Phase 1 only.  Do not rewrite filters until the structural
equations and model-specific partitions are written and audited.
