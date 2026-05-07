# Experiment plan: SGU joint state-control projection design

## Question

Can the SGU joint state-control projection pilot be turned into an honest,
testable nonlinear projection target without reusing the failed
quadratic-over-linear Gate B claim?

The decision this phase must make is narrower than "make SGU pass":

```text
Should SGU remain state-identity-only, become an offline residual diagnostic,
or become a causal predictive filtering transition based on a local nonlinear
projection?
```

## Motivation

The previous SGU diagnostic pass resolved the immediate blocker:

- selected SGU state identities close to roundoff;
- the full quadratic perturbation residual is worse than the linear residual
  once constant volatility corrections are included;
- a pruned state comparison does not rescue the quadratic policy gate;
- a least-squares joint projection can reduce the full canonical residual from
  roughly `4.39e-02` RMS to `3.54e-09` RMS with small local moves.

That last point is promising but dangerous.  The pilot solved over current
controls, next controls, and deterministic next states.  If current controls
are adjusted using the next stochastic candidate, the result is a two-slice
or offline residual projection, not automatically a causal filtering
transition.  The next phase therefore has to decide the target class before
any backend, derivative, compiled, or HMC promotion.

## Mechanism being tested

The pilot projection fixed:

```text
x_t
x'_{S} = (a', zeta', mu')
theta
```

and optimized:

```text
y_t                  8 current controls
y'                   8 next controls
x'_{D} = (d',k',r',riskprem')  4 deterministic next states
```

against the 15 canonical SGU residual equations:

```text
H[0], ..., H[14]
```

This creates a 20-unknown, 15-residual local solve.  Residual closure alone is
not enough, because the solve is underidentified unless the target includes a
minimum-move or economics-based selection rule.  This phase compares three
projection targets:

1. **State-only baseline.**  Keep the existing state-identity completion over
   `x'_D`.  This is already known to close `H[7], H[8], H[10], H[11]` but not
   the full canonical residual.
2. **Causal predictive projection.**  Fix `y_t` at the policy value implied by
   `x_t`; optimize only `y'` and `x'_D`.  This is the only projection variant
   that can become a standard filtering transition without look-ahead.
3. **Two-slice/offline projection.**  Optimize `y_t`, `y'`, and `x'_D`, but add
   an explicit minimum-move selection rule.  This can be a diagnostic or
   smoother target, but it must not be labeled as a causal filtering
   transition unless a separate timing derivation justifies it.

## Scope

- Variant: SGU joint state-control nonlinear projection design.
- Objective: decide whether H3 can become a production-candidate value target
  or only an offline diagnostic.
- Seed(s): deterministic local grids first; optional randomized holdout grids
  only after deterministic grids pass.
- Training steps: none.
- HMC/MCMC settings: none.
- XLA/JIT mode: none.  Initial implementation is eager SciPy/NumPy only.
- Expected runtime: targeted DSGE projection tests under one minute for the
  default grid; stress grids may be split into a slower contract suite.

## Ownership

Implementation belongs in `/home/chakwong/python`.

Likely client files:

- `/home/chakwong/python/src/dsge_hmc/models/sgu.py`;
- `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`;
- `/home/chakwong/python/tests/contracts/test_sgu_joint_projection_target.py`;
- `/home/chakwong/python/docs/plans/sgu-joint-state-control-projection-result-YYYY-MM-DD.md`;
- `/home/chakwong/python/docs/plans/dsge-structural-completion-gap-closure-reset-memo-2026-05-06.md`.

BayesFilter should remain generic.  BayesFilter changes are justified only if
the projection exposes a missing generic metadata field or adapter status that
cannot be represented by existing approximation labels.

## Inputs

Required evidence from the completed diagnostic pass:

- `docs/plans/bayesfilter-sgu-residual-source-pruned-projection-plan-2026-05-07.md`;
- `docs/plans/bayesfilter-sgu-residual-source-pruned-projection-audit-2026-05-07.md`;
- `/home/chakwong/python/docs/plans/sgu-residual-source-pruned-projection-result-2026-05-07.md`;
- `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`.

Pilot facts to preserve:

```text
before_rms = 4.389182552998e-02
before_max = 8.595595430720e-02
projected_rms = 3.543965164610e-09
projected_max = 1.256486389166e-08
current_control_adjustment_norm = 7.704118037390e-03
next_control_adjustment_norm = 6.362048017869e-03
deterministic_state_adjustment_norm = 3.753306747968e-03
max_nfev = 41
```

Labels that must remain blocked during this phase:

```text
sgu_quadratic_policy_residual_improvement_passed
sgu_combined_structural_approximation_target_passed
blocked_sgu_quadratic_policy_not_better_than_linear_residual
blocked_sgu_pruned_state_comparison_not_residual_improving
blocked_sgu_projection_is_new_target_not_gate_b
blocked_nonlinear_equilibrium_manifold_residual
```

## Hypotheses

H1. A causal predictive projection can close the full canonical SGU residual
locally while preserving the supplied stochastic next coordinates and without
adjusting current controls.

H2. If H1 fails but the two-slice projection succeeds, the projection is useful
only as an offline diagnostic or smoother-style target unless a separate
timing argument is written.

H3. The two-slice projection is underidentified unless it includes an explicit
minimum-move selection rule relative to the perturbation policy and
state-identity completion.

H4. A projection target is useful only if it is local, continuous on heldout
grids, and fails closed with structured diagnostics when the local solve is
singular or nonfinite.

H5. No derivative, compiled, HMC, or generic BayesFilter backend promotion is
justified until a value-level projection target passes its own grid,
conditioning, and metadata tests.

## Target definitions

### State-only baseline

Fixed:

```text
x_t, x'_S, theta
```

Unknown:

```text
x'_D
```

Objective:

```text
close H[7], H[8], H[10], H[11]
```

Expected label:

```text
sgu_state_identity_completion_passed
```

This target is already allowed, but it does not close the full canonical
residual.

### Causal predictive projection

Fixed:

```text
x_t
x'_S
y_t = policy(x_t)
theta
```

Unknown:

```text
y'
x'_D
```

Objective:

```text
minimize H(y', y_t, x', x_t, theta)
```

with optional local regularization:

```text
lambda_y_next * ||y' - policy(x')||_2
lambda_x_next * ||x'_D - state_identity_completion_D||_2
```

This is the only candidate target that can be promoted to a filtering
transition in this phase.

### Two-slice/offline projection

Fixed:

```text
x_t
x'_S
theta
```

Unknown:

```text
y_t
y'
x'_D
```

Objective:

```text
minimize [
  residual_scale^{-1} * H(y', y_t, x', x_t, theta),
  lambda_y_current * (y_t - policy(x_t)),
  lambda_y_next * (y' - policy(x')),
  lambda_x_next * (x'_D - state_identity_completion_D)
]
```

This target must report both residual closure and adjustment sizes.  It may
earn a diagnostic label, but it must not become a filtering transition unless
current-control look-ahead is resolved explicitly.

## Success criteria

### Design gate

The design gate passes only if the result note states:

- which projection target is being pursued;
- which variables are fixed and which are optimized;
- which residual equations enter the objective;
- which selection rule resolves underidentification;
- whether the target is causal filtering, two-slice smoothing, or diagnostic
  only.

### Causal projection pilot gate

The causal projection earns:

```text
sgu_causal_joint_projection_pilot_passed
```

only if all default-grid cases satisfy:

```text
all_success is True
projected_max_residual <= 1e-7
projected_rms_residual <= 1e-8
next_control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
max_nfev <= 500
```

and the implementation preserves:

```text
x'_S = (a', zeta', mu')
y_t = policy(x_t)
```

If this gate passes, the next work may consider BayesFilter adapter metadata.
It still does not unblock derivative, compiled, or HMC promotion.

### Two-slice diagnostic gate

The two-slice projection earns:

```text
sgu_two_slice_projection_diagnostic_passed
```

only if all default-grid cases satisfy:

```text
all_success is True
projected_max_residual <= 1e-7
projected_rms_residual <= 1e-8
current_control_adjustment_norm <= 2e-2
next_control_adjustment_norm <= 2e-2
deterministic_state_adjustment_norm <= 1e-2
max_nfev <= 500
```

The required paired blocker remains:

```text
blocked_sgu_two_slice_projection_not_filter_transition
```

unless a separate timing derivation proves that current-control adjustment is
valid inside the intended filtering recursion.

### Stress gate

A production-candidate value target must pass the default grid plus heldout
stress grids:

- shock scale `0.0`;
- shock scale `0.25`;
- shock scale `0.5`;
- shock scale `1.0`;
- sign-flipped local shocks;
- at least two near-boundary but valid parameter perturbations.

Required stress-grid criteria:

```text
projected_max_residual <= 1e-6
projected_rms_residual <= 1e-7
max_adjustment_norm <= 5e-2
finite_conditioning_diagnostics is True
failure_count == 0
```

If the target passes only the default grid, it is a local pilot, not a
production candidate.

## Diagnostics

Primary:

- residual RMS and max before and after projection;
- per-equation residual table for `H[0]` through `H[14]`;
- current-control adjustment norm and max;
- next-control adjustment norm and max;
- deterministic next-state adjustment norm and max;
- preservation error for stochastic next coordinates;
- current-control anchoring error for causal mode.

Secondary:

- least-squares success flag, status, message, cost, optimality, and
  evaluations;
- numerical rank and condition estimate of the local residual Jacobian;
- singular values of the residual Jacobian where available;
- regularization weights and regularization residual contributions;
- residual scaling policy;
- projection continuity diagnostics on neighboring grid points;
- timing label: `causal_filtering`, `two_slice_smoothing`, or
  `offline_diagnostic`.

Sanity checks:

- no artificial deterministic process noise is introduced;
- stochastic coordinates `(a,zeta,mu)` are never optimized;
- exact nonlinear equilibrium-manifold blocker remains present;
- failed projection returns a structured failure status, not a silently
  completed state;
- no derivative, JIT, compiled, HMC, or posterior-convergence label is minted.

## Expected failure modes

- Causal projection cannot close full residuals without moving `y_t`.  Then
  SGU can have only an offline/two-slice projection target from this pass.
- Two-slice projection closes residuals but is nonlocal.  Then the result is
  blocked as an economically implausible projection.
- The residual system is rank-deficient and solutions change materially across
  solver tolerances or initial guesses.  Then the target is blocked until the
  selection rule is strengthened.
- Default-grid projection passes but heldout stress grids fail.  Then the
  target remains local-pilot-only.
- The projection succeeds numerically but requires SciPy eager execution.  Then
  the target remains value-only and compiled/HMC blocked.
- Any implementation weakens the existing SGU quadratic-over-linear blocker.
  That invalidates the pass.

## What would change our mind

- If the causal predictive projection passes default and stress gates with
  local adjustments, SGU can move from state-identity-only support to a
  projection-value-target candidate.
- If only the two-slice projection passes, SGU can gain an offline diagnostic
  or smoother target, but BayesFilter filtering integration remains blocked.
- If both projection targets fail locality, conditioning, or heldout grids, SGU
  should remain state-identity-only until the economic derivation is revised.
- If a generic metadata gap appears in BayesFilter, add only the minimal
  metadata/status field needed to represent the projection target honestly.

## Implementation phases

### Phase 0: preflight and audit

Plan:

- record `/home/chakwong/python` and BayesFilter status;
- protect unrelated dirty files;
- independently audit this plan before code changes.

Execute:

```bash
cd /home/chakwong/python
git status --short --branch
git log -5 --oneline --decorate

cd /home/chakwong/BayesFilter
git status --short --branch
git log -5 --oneline --decorate
```

Audit checklist:

- no step relabels the old Gate B as passed;
- causal and two-slice timing are separated;
- underidentification is addressed before promotion;
- BayesFilter changes are optional and evidence-triggered.

### Phase 1: write the projection target decision note

Plan:

- write a DSGE client note fixing the target variants, timing convention,
  unknowns, residuals, and selection rule.

Execute:

- create `/home/chakwong/python/docs/plans/sgu-joint-state-control-projection-design-YYYY-MM-DD.md`;
- include the exact `z` vector layout for each mode;
- include objective scaling and regularization choices.

Test:

- doc-only review;
- verify labels and blockers are present in the note.

Audit:

- the note must state whether the target is causal filtering, two-slice
  smoothing, or offline diagnostic.

Next phase justified if:

- the note can specify a target without contradicting SGU timing.

### Phase 2: factor the pilot into reusable diagnostics

Plan:

- move the projection helper out of a single test body and into a client-owned
  reusable diagnostic/helper path.

Execute:

- add a value-only helper such as:

```python
project_sgu_state_control(
    previous_state,
    stochastic_next,
    theta,
    sol=None,
    *,
    mode,
    policy_order="second",
    regularization_weights=None,
    residual_scale=None,
)
```

- return a structured result with:

```text
projected_state
current_controls
next_controls
residuals_before
residuals_after
adjustment_metrics
solver_diagnostics
status
timing_label
allowed_labels
blocked_labels
```

Test:

- preserve existing projection pilot numbers within tolerance;
- preserve the existing state-identity completion tests.

Audit:

- helper must fail closed on nonfinite inputs, nonfinite residuals, solver
  failure, or excessive adjustment.

Next phase justified if:

- the factored helper reproduces the previous local pilot.

### Phase 3: compare causal and two-slice targets

Plan:

- run both projection variants on the same default grid and compare residuals,
  locality, and timing semantics.

Execute:

- add tests for:
  - causal predictive projection;
  - two-slice/offline projection;
  - state-only baseline.

Test:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-sgu \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_joint_projection_target.py \
  tests/contracts/test_dsge_strong_structural_residual_gates.py
```

Audit:

- causal mode must keep `y_t = policy(x_t)`;
- two-slice mode must carry the paired blocker
  `blocked_sgu_two_slice_projection_not_filter_transition`.

Next phase justified if:

- at least one target closes residuals locally and the target class is clear.

### Phase 4: conditioning and uniqueness diagnostics

Plan:

- determine whether the projection is well-conditioned enough to be a stable
  value target.

Execute:

- compute residual-Jacobian rank and condition estimates at default-grid
  points;
- compare solutions from at least two starting points:
  - perturbation-policy start;
  - state-identity-completion start plus small deterministic perturbation;
- report sensitivity to regularization weights.

Test:

- projected residual remains below threshold under allowed starts;
- adjustment norms remain stable within a documented tolerance;
- rank/conditioning diagnostics are finite.

Audit:

- if solutions differ materially across starts, the target is not uniquely
  defined and must stay blocked.

Next phase justified if:

- local uniqueness and conditioning are acceptable for the chosen target.

### Phase 5: heldout stress grids

Plan:

- test whether the projection is only a default-grid artifact.

Execute:

- build deterministic grids for shock scales `0.0`, `0.25`, `0.5`, and `1.0`;
- add sign-flipped local shocks;
- add two near-boundary but valid parameter perturbations.

Test:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-sgu \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_joint_projection_target.py
```

Audit:

- stress failures should narrow the allowed validity region rather than being
  hidden behind default-grid success.

Next phase justified if:

- the chosen target passes the stress gate or the result note accepts a
  deliberately local validity region.

### Phase 6: BayesFilter integration decision

Plan:

- decide whether BayesFilter needs any generic adapter metadata change.

Execute:

- if causal projection passes, check whether current metadata can represent:

```text
deterministic_completion = "sgu_joint_projection"
approximation_label = "nonlinear_projection_closure"
differentiability_status = "value_only_projection_eager"
compiled_status = "eager_scipy"
hmc_status = "blocked"
```

- if current metadata is enough, do not edit BayesFilter code;
- if metadata is missing, add only a generic field and tests.

Test:

```bash
cd /home/chakwong/BayesFilter
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

Audit:

- BayesFilter must not import SGU economics or SciPy projection logic.

Next phase justified if:

- metadata can represent the target honestly.

### Phase 7: result note, reset memo, and provenance

Plan:

- record the outcome and preserve blocked claims.

Execute:

- write `/home/chakwong/python/docs/plans/sgu-joint-state-control-projection-result-YYYY-MM-DD.md`;
- update the DSGE reset memo;
- update this BayesFilter reset memo only if BayesFilter-local provenance or
  adapter metadata changes;
- update `docs/source_map.yml` if BayesFilter planning/provenance files are
  added.

Test:

```bash
cd /home/chakwong/BayesFilter
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
```

Audit:

- result note must say explicitly whether derivative, compiled, HMC, and
  BayesFilter backend work are still blocked.

Next phase justified if:

- the result selects a production-candidate causal target.  Otherwise stop
  before backend/HMC work.

## Command bundle

Primary client command:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-sgu \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_joint_projection_target.py \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter adapter command, only if metadata is touched:

```bash
cd /home/chakwong/BayesFilter
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 \
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

## Interpretation rule

- If causal projection passes default and stress gates, SGU can become a
  value-side nonlinear projection target candidate.
- If only two-slice projection passes, SGU gains an offline diagnostic target
  and remains blocked as a filtering transition.
- If projection closure requires nonlocal moves, unstable conditioning, or
  hidden look-ahead, SGU remains state-identity-only.
- In all cases, the old quadratic-over-linear Gate B remains failed unless a
  separate perturbation-policy residual comparison later passes on its own
  terms.
