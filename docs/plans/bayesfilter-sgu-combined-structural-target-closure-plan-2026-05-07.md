# Experiment plan: SGU combined structural target closure

## Question

Can SGU be advanced from `blocked_nonlinear_equilibrium_manifold_residual` to
a narrower, honest structural approximation target that:

1. closes deterministic state identities for `(d,k,r,riskprem)`; and
2. shows the quadratic/second-order perturbation policy improves residuals
   relative to the linear/first-order policy on declared grids?

## Mechanism being tested

The current SGU BayesFilter metadata is mixed structural:

```text
states = (d, k, r, a, riskprem, zeta, mu)
stochastic = (a, zeta, mu) = (3, 5, 6)
deterministic = (d, k, r, riskprem) = (0, 1, 2, 4)
```

The existing linear bridge fills deterministic coordinates from `h_x` and is
useful as a metadata gate, but it does not close the nonlinear canonical
equilibrium residual.  Prior evidence in
`/home/chakwong/python/docs/plans/sgu-nonlinear-deterministic-completion-derivation-2026-05-06.md`
shows:

- selected deterministic state equations `(7,8,10,11)` can be solved to
  roundoff;
- the full canonical residual remains around `8.6e-2`;
- the largest residuals are in Euler/static/control FOC equations, not merely
  state identities.

Therefore the target should not be "exact nonlinear equilibrium manifold".
The target should be a two-gate approximation contract:

```text
Gate A: state identities close
Gate B: quadratic policy residual improves over linear policy residual
```

This plan uses the repo experiment-template structure, but expands it into an
implementation handoff because the work spans `/home/chakwong/python` and
BayesFilter provenance.

## Motivation

The structural filter needs deterministic coordinates to be completed
pointwise, otherwise sigma points can leave the intended model support.  For
SGU, however, enforcing only deterministic next-state identities is too weak:
it does not make the full nonlinear canonical model hold.

At the same time, requiring exact nonlinear canonical residual closure is too
strong for the current second-order perturbation solver.  A perturbation
policy is an approximation target.  The honest scientific question is whether
the second-order/quadratic policy behaves better than the linear policy on the
same local structural grids.

The combined target keeps both truths:

- deterministic state support is a hard structural requirement;
- quadratic improvement is an approximation-quality requirement, not an exact
  equilibrium proof.

## Scope

- Variant: SGU combined structural target, client-owned implementation.
- Objective: close SGU as a value-side structural approximation target without
  claiming exact nonlinear equilibrium, derivative readiness, compiled parity,
  or HMC readiness.
- Seed(s): deterministic grids; no random seed required unless optional
  randomized stress grids are added.
- Training steps: none.
- HMC/MCMC settings: none.
- XLA/JIT mode: none required; eager reference first.
- Expected runtime: targeted pytest suite under one minute on current local
  hardware.

## Ownership

Implementation belongs in `/home/chakwong/python`.

Likely client files:

- `src/dsge_hmc/models/sgu.py`;
- `tests/contracts/test_dsge_strong_structural_residual_gates.py`;
- `tests/contracts/test_dsge_structural_completion_residuals.py`;
- `tests/contracts/test_structural_dsge_partition.py`;
- `docs/plans/sgu-combined-structural-target-result-YYYY-MM-DD.md`.

BayesFilter should stay generic.  BayesFilter changes are justified only if
the client work exposes a missing generic metadata/result-label abstraction.

## Inputs

Existing client-side artifacts:

- `/home/chakwong/python/docs/plans/sgu-structural-target-selection-plan-2026-05-06.md`;
- `/home/chakwong/python/docs/plans/sgu-perturbation-policy-residual-harness-plan-2026-05-06.md`;
- `/home/chakwong/python/docs/plans/sgu-nonlinear-deterministic-completion-derivation-2026-05-06.md`;
- `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`.

Those notes are useful but should be updated by this plan's stricter combined
target.  In particular, a pure perturbation-policy target is not enough; state
identity completion is also required.

## Success criteria

Gate A earns:

```text
sgu_state_identity_completion_passed
```

only if the deterministic coordinates `(d,k,r,riskprem)` satisfy the selected
state/timing equations on declared grids:

```text
7:  exp(r') = r_w + riskprem'
8:  riskprem' = zeta_t + psi(exp(d' - d_bar) - 1)
10: ca_t = d'/GDP_t - d_t/GDP_t
11: exp(k') = exp(kfu_t)
```

Gate B earns:

```text
sgu_quadratic_policy_residual_improvement_passed
```

only if, on the same declared grids:

```text
rms_residual_quadratic < rms_residual_linear
max_residual_quadratic <= max_residual_linear
```

Recommended stronger acceptance margin:

```text
rms_residual_quadratic <= 0.8 * rms_residual_linear
```

unless a result note explains why the no-margin strict inequality is more
appropriate for a specific near-boundary grid.

The combined target earns:

```text
sgu_combined_structural_approximation_target_passed
```

only when both Gate A and Gate B pass.

## Diagnostics

Primary:

- max and RMS residuals for Gate A state identities;
- max and RMS full canonical residuals for linear policy;
- max and RMS full canonical residuals for quadratic policy;
- residual ratio table:

```text
rms_quadratic / rms_linear
max_quadratic / max_linear
```

Secondary:

- per-equation residual table for all 15 canonical SGU equations;
- finite checks for state, control, residual, and completion outputs;
- shock-scale grid diagnostics for `0.0, 0.25, 0.5, 1.0`;
- `||g_ss||` and `||h_ss||` scaling with shock scale;
- near-boundary but valid parameter-grid diagnostics.

Sanity checks:

- preserve stochastic coordinates `(a,zeta,mu)` from the stochastic candidate;
- do not add artificial process noise to deterministic coordinates;
- keep the existing exact nonlinear blocker test:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

because the combined target is an approximation target, not exact canonical
residual closure.

## Expected failure modes

- The state-identity solve closes `(7,8,10,11)` but changes stochastic
  coordinates.  This fails Gate A.
- The state-identity solve closes selected equations but causes nonfinite
  controls or residuals.  This fails Gate A.
- Quadratic residual improves RMS but worsens max residual materially.  This
  fails Gate B unless the result note defines and justifies a per-equation
  exception before implementation.
- Quadratic residual is worse than linear residual on boundary-adjacent grids.
  The target must remain grid-limited or blocked for those regions.
- The implementation weakens or deletes the existing exact nonlinear residual
  blocker.  This invalidates the pass.
- The result is labeled as derivative-ready, compiled-ready, HMC-ready, or
  exact nonlinear filtering.  This is an evidence-category error.

## What would change our mind

- If Gate A cannot close the state identities without nonfinite values or
  stochastic-coordinate mutation, SGU should remain `adapter_ready_only`.
- If Gate B fails on the local default grid, the second-order policy should not
  be promoted as an improvement over the linear policy.
- If Gate B passes only at the default parameter point but fails near valid
  boundaries, the allowed label should be restricted to the tested local
  validity region.
- If a joint state-control projection cleanly closes the full canonical
  residual with stable failure semantics, a future plan may promote that
  different target.  That would be a separate nonlinear projection backend, not
  this perturbation-policy target.

## Implementation plan

### Phase 0: preflight

Record `/home/chakwong/python` and BayesFilter status:

```bash
cd /home/chakwong/python
git status --short --branch
git log -5 --oneline --decorate

cd /home/chakwong/BayesFilter
git status --short --branch
git log -5 --oneline --decorate
```

Protect dirty files.  At the time this plan was written, `/home/chakwong/python`
had unrelated/uncommitted SGU plan and reset-memo work.  Do not overwrite it.

### Phase 1: Gate A design

Write a short derivation/result note in `/home/chakwong/python/docs/plans` that
fixes the exact Gate A equations, unknowns, knowns, and timing convention.

Required decisions:

- Does `ca_t` and `kfu_t` use current policy controls evaluated at the
  completed next state or previous-period controls?
- Are Gate A equations checked at first order, second order, or both?
- What tolerance is used for exact state identity closure?
- What parameter grids are inside the declared validity region?

### Phase 2: Gate A implementation

Implement a model-owned SGU helper, for example:

```python
bayesfilter_state_identity_completion(...)
```

or extend `bayesfilter_deterministic_completion(...)` with an explicit policy
argument, such as:

```python
completion_target="state_identity"
```

Requirements:

- preserve stochastic candidate coordinates `(a,zeta,mu)`;
- solve only deterministic coordinates `(d,k,r,riskprem)`;
- fail closed with a named error/status if the local solve is singular or
  nonfinite;
- keep the existing linear bridge available for backward compatibility and
  comparison tests.

### Phase 3: Gate A tests

Add or update:

```text
tests/contracts/test_sgu_structural_completion_residuals.py
```

Minimum assertions:

- finite completion output;
- stochastic coordinates preserved;
- selected state residuals `(7,8,10,11)` below tolerance;
- linear bridge remains labeled separately;
- missing/failed state-identity completion does not earn the combined target.

### Phase 4: Gate B residual comparison harness

Add a deterministic residual-comparison harness in:

```text
tests/contracts/test_dsge_strong_structural_residual_gates.py
```

The harness should evaluate both:

- linear/first-order policy residuals;
- quadratic/second-order perturbation policy residuals.

Use the same state grid and parameter grid for both.  Record max and RMS
residuals.

Suggested helper names:

```python
canonical_residual_summary(...)
compare_linear_quadratic_policy_residuals(...)
```

### Phase 5: Gate B tests

Assert:

```text
rms_quadratic < rms_linear
max_quadratic <= max_linear
```

and, for the default local grid if supported:

```text
rms_quadratic <= 0.8 * rms_linear
```

Keep a separate test proving the exact nonlinear residual is not claimed as
closed at default volatility.

### Phase 6: result note and provenance

Write:

```text
/home/chakwong/python/docs/plans/sgu-combined-structural-target-result-YYYY-MM-DD.md
```

The result note must include:

- grid definition;
- tolerance values;
- residual tables;
- pass/fail interpretation;
- allowed labels;
- blocked labels that remain blocked;
- whether derivative or compiled work is justified next.

If the result is positive, update BayesFilter reset/provenance only after the
client commit exists.

## Command

Run the client tests:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_structural_completion_residuals.py \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

Run BayesFilter adapter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

Run cleanup:

```bash
cd /home/chakwong/python
git diff --check
```

If BayesFilter docs/provenance are touched:

```bash
cd /home/chakwong/BayesFilter
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
pytest -q tests/test_dsge_adapter_gate.py
```

## Interpretation rule

- If Gate A and Gate B pass, SGU earns:

```text
sgu_combined_structural_approximation_target_passed
```

for the tested grids and validity region.

- If Gate A passes but Gate B fails, SGU earns only:

```text
sgu_state_identity_completion_passed
```

and remains blocked for perturbation-policy promotion.

- If Gate A fails, SGU remains:

```text
adapter_ready_only
```

with the existing blocker:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

- In every case, SGU remains not derivative-certified, not compiled-parity
certified, and not HMC-ready until later gates pass.
