# Plan: structural SGU goals, remaining gaps, and next closure path

## Goal Summary

The broad goal is to make BayesFilter an honest Bayesian estimation framework
for structural state-space models, while keeping model economics in the owning
client repos.  For SGU specifically, the goal is not merely to pass adapter
metadata; it is to identify a value target that can support estimation without
overclaiming exact nonlinear equilibrium, derivative readiness, compiled
readiness, or HMC readiness.

The current SGU ladder has four intended goals:

1. **State support.**  Deterministic SGU coordinates must be completed
   pointwise so sigma/filter points remain on the declared structural support.
2. **Residual quality.**  A nonlinear or second-order target should improve
   full canonical residuals relative to the linear benchmark before it is
   promoted as a better perturbation-policy value target.
3. **Timing-correct filtering.**  Any production filtering transition must be
   causal: current controls cannot be adjusted using future stochastic
   candidates unless a separate timing/smoothing derivation justifies it.
4. **Downstream promotion.**  BayesFilter adapter/backend, derivative,
   compiled, and HMC work should start only after a causal value target passes
   value-level tests.

## Current Status

Closed:

- SGU state-identity completion passes for selected state equations
  `H[7]`, `H[8]`, `H[10]`, and `H[11]`.
- The residual source is diagnosed: the full quadratic failure is driven by
  volatility-correction effects in Euler/static/control FOC equations
  `H[3]` through `H[6]`, not by selected state identities.
- Pruned-state comparison was tested and does not rescue the quadratic gate.
- Joint state-control projection was classified:
  - causal predictive mode fails the residual/locality gate;
  - two-slice/offline mode closes residuals locally and on stress grids.

Allowed labels:

```text
sgu_state_identity_completion_passed
sgu_two_slice_projection_diagnostic_passed
```

Blocked labels:

```text
blocked_sgu_causal_projection_residual_not_closed
blocked_sgu_two_slice_projection_not_filter_transition
blocked_sgu_projection_is_new_target_not_gate_b
sgu_quadratic_policy_residual_improvement_passed
sgu_combined_structural_approximation_target_passed
blocked_nonlinear_equilibrium_manifold_residual
```

Recent evidence:

```text
causal_projected_max               = 8.561333092931e-02
causal_projected_rms               = 3.113537564923e-02
causal_next_control_adjustment_norm = 4.039154171952e-01
two_slice_projected_max            = 1.256486389166e-08
two_slice_projected_rms            = 3.543960161723e-09
two_slice_current_control_norm     = 7.219385132598e-03
two_slice_next_control_norm        = 6.860787546701e-03
two_slice_deterministic_state_norm = 3.758583443181e-03
two_slice_min_jacobian_rank        = 12
```

## Remaining Gaps

### Gap 1: Causal SGU Filtering Target

The current causal projection fixes `y_t = policy(x_t)` and preserves
`x'_S = (a', zeta', mu')`, but it leaves full residual max around `8.56e-02`.
It also requires a large next-control adjustment.  Therefore SGU does not yet
have a causal nonlinear filtering transition.

Hypotheses:

- H1a: the obstruction is the current control anchor, not state completion.
- H1b: equations `H[3]` through `H[6]` require a current-period FOC-consistent
  control map rather than the volatility-corrected perturbation-policy anchor.
- H1c: causal closure may require augmenting the filtering state with lagged
  controls or shadow controls.

### Gap 2: Two-Slice Projection Semantics

The two-slice projection closes residuals, but it adjusts current controls
using a two-time-slice solve.  It is useful as an offline diagnostic or
smoother candidate, not a predictive filter.

Hypotheses:

- H2a: two-slice projection can be formalized as a smoother/local consistency
  diagnostic with explicit timing.
- H2b: if used in estimation, it would require a different likelihood or
  smoothing target, not the existing predictive filtering recursion.

### Gap 3: Perturbation-Policy Residual Improvement

The second-order/quadratic perturbation policy is still worse than linear in
full residuals when default volatility corrections are included.

Hypotheses:

- H3a: the volatility correction terms are correct but not aligned with the
  canonical residual comparison used by the gate.
- H3b: the residual comparison should use an explicitly derived stochastic
  expectation target rather than pointwise deterministic residuals.
- H3c: if neither reinterpretation holds, the quadratic residual-improvement
  target should remain blocked for SGU.

### Gap 4: BayesFilter Promotion

BayesFilter should not add SGU-specific backend/filter logic while the only
passing residual-closure target is two-slice/offline.

Hypotheses:

- H4a: existing BayesFilter metadata is sufficient to represent the current
  SGU status.
- H4b: new metadata is justified only if a causal projection target passes or
  if BayesFilter needs to represent diagnostic-only smoother targets.

### Gap 5: Derivative, Compiled, and HMC Readiness

No derivative/JIT/HMC work is justified until a causal value target exists.

Hypotheses:

- H5a: eager SciPy projection diagnostics are not differentiable production
  targets.
- H5b: an implicit-function derivative path may be possible only after the
  causal projection equations and rank conditions are settled.

## Next Closure Plan

### Phase 1: SGU Timing and Equation Audit

Objective:
- determine whether the causal obstruction is a timing/anchor issue or a
  deeper impossibility.

Implementation:
- write a client-owned derivation note for equations `H[3]` through `H[6]`;
- classify which variables are current, next, predetermined, or jump controls;
- compare three current-control anchors:
  - existing second-order policy;
  - quadratic policy without constant volatility correction;
  - current-period FOC-consistent control solve.

Success:
- a causal residual test improves materially without moving current controls.

Stop:
- if all anchors leave residuals near `8e-02`, causal production filtering
  remains blocked.

### Phase 2: Causal Projection Variant

Objective:
- test whether a causal projection can close residuals with a better
  current-control anchor.

Implementation:
- extend `project_sgu_state_control(mode="causal")` with an explicit
  `current_control_anchor` option;
- test default, state stress, and parameter stress grids;
- keep stochastic next coordinates fixed.

Success:

```text
projected_max <= 1e-7
projected_rms <= 1e-8
next_control_adjustment_norm <= 2e-2
current_control_anchor_error <= 1e-10
```

Stop:
- if causal residuals remain above threshold or moves are nonlocal.

### Phase 3: Two-Slice Smoother Semantics

Objective:
- decide whether the passing two-slice target should become a documented
  offline/smoother diagnostic.

Implementation:
- write a timing note explaining what likelihood or smoother objective the
  two-slice solve would support;
- add tests that the two-slice target is never routed through predictive
  filter metadata.

Success:
- diagnostic/smoother semantics are explicit and cannot be confused with
  filtering.

Stop:
- if no coherent smoothing interpretation exists, keep it as diagnostic only.

### Phase 4: Perturbation Residual Gate Reassessment

Objective:
- decide whether the quadratic-over-linear residual gate is mathematically the
  right gate for SGU with volatility corrections.

Implementation:
- compare pointwise residuals to expectation-consistent residual diagnostics;
- isolate per-equation effects of `g_ss` and `h_ss`;
- document whether a stochastic-expectation residual gate should replace or
  supplement the pointwise gate.

Success:
- a revised gate is mathematically justified and tested without weakening the
  existing blocker.

Stop:
- if the current pointwise gate remains the honest comparison, keep Gate B
  blocked.

### Phase 5: BayesFilter Integration Gate

Objective:
- decide whether any BayesFilter metadata/API change is justified.

Implementation:
- run `tests/test_dsge_adapter_gate.py`;
- if causal projection passes, add generic metadata only, not SGU economics;
- if only two-slice passes, keep BayesFilter unchanged or add
  diagnostic-only labels if needed.

Success:
- BayesFilter can represent the target honestly.

Stop:
- if causal projection fails, no backend/filter promotion.

### Phase 6: Derivative/JIT/HMC Readiness Gate

Objective:
- start downstream numerical promotion only after a causal value target passes.

Implementation:
- define value parity tests first;
- then implicit derivative tests if projection equations are well ranked;
- then compiled parity;
- only then HMC smoke and convergence diagnostics.

Success:
- each promotion layer has value, derivative, and compiled evidence.

Stop:
- no causal value target means no derivative/JIT/HMC phase.

## Immediate Recommendation

Run Phase 1 next: a focused SGU timing and equations audit for `H[3]` through
`H[6]`.  That is now the highest-leverage gap.  State identities and two-slice
residual solvability are no longer the active blockers; causal timing is.
