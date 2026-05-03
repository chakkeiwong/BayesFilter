# Plan: BayesFilter structural state partition core

## Date

2026-05-04

## Status

Canonical design and implementation plan for:

```text
/home/chakwong/BayesFilter/docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md
```

The DSGE-side plan in `/home/chakwong/python/docs/plans` should be treated as
an adapter/integration handoff, not the source of truth.

## Motivation

BayesFilter should own the common filtering machinery for state-space models
with structural or degenerate transitions.  The need is not specific to DSGE.
It also appears in simple AR(p) systems, macro-finance term-structure models,
companion-form lag stacks, stochastic-volatility augmentations, accounting
identities, and any model where some state coordinates are deterministic
functions of stochastic coordinates and lagged states.

A canonical example is an AR(2):

```text
m_t = phi_1 m_{t-1} + phi_2 m_{t-2} + sigma eps_t
l_t = m_{t-1}
x_t = (m_t, l_t)
```

Only `m_t` receives new innovation noise.  The lag coordinate `l_t` is a
deterministic shift.  A nonlinear sigma-point or particle filter that treats
both coordinates as freely shock-driven is integrating over states the model
does not generate.

The same pattern appears in DSGE models:

```text
m_t = T_m(m_{t-1}, eps_t; theta)
k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)
x_t = (m_t, k_t)
```

where `m_t` denotes exogenous or shock-driven states and `k_t` denotes
endogenous, predetermined, accounting, lag, or deterministic-completion
states.  MacroFinance models also need the same distinction for affine state
blocks, lag stacks, yields-only measurement states, no-arbitrage recursions,
and analytic Kalman derivative infrastructure.

Therefore the implementation should be in BayesFilter, with DSGE and
MacroFinance as clients through adapters.

## Architectural decision

BayesFilter is the source of truth for:

- structural state partition metadata;
- degenerate transition semantics;
- stochastic integration space selection;
- deterministic completion maps;
- SVD/QR/Cholesky square-root sigma-point filters;
- exact and square-root Kalman filters with singular process covariance;
- particle filters with deterministic completion;
- derivative/Hessian validation contracts;
- HMC-ready likelihood API contracts.

DSGE, MacroFinance, and any future model project should supply:

- model-specific state names and timing;
- structural transition and completion maps;
- observation equations;
- priors and HMC targets;
- adapter-level tests showing that model metadata maps correctly into
  BayesFilter.

BayesFilter should not contain DSGE economics or MacroFinance financial-model
logic.  It should contain reusable filtering abstractions and tested numerical
backends.

## Core mathematical contract

A structural state-space model should expose:

```text
x_t = pack(m_t, k_t)
m_t = T_m(m_{t-1}, eps_t; theta)
k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)
y_t = H(x_t; theta) + u_t
```

where:

- `eps_t` is the innovation;
- `m_t` is the stochastic/exogenous block;
- `k_t` is the deterministic-completion block;
- `u_t` is measurement noise;
- `pack` and `unpack` are explicit and tested.

For exact linear Kalman filtering, a collapsed moment representation may be
valid:

```text
x_t = F x_{t-1} + G eps_t
Q = G Sigma_eps G'
```

including singular `Q`.  For nonlinear sigma-point and particle filtering,
the filter must know whether it is integrating over:

- innovation/shock space;
- current stochastic-state space;
- full state space as an explicitly labeled approximation.

The default for mixed structural models must not be unlabeled full-state
integration.

## Proposed BayesFilter API

Exact names can change during implementation, but the contract should remain
stable.

### Partition metadata

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class StatePartition:
    state_names: tuple[str, ...]
    stochastic_indices: tuple[int, ...]
    deterministic_indices: tuple[int, ...]
    auxiliary_indices: tuple[int, ...] = ()
    innovation_dim: int = 0

    @property
    def state_dim(self) -> int:
        return len(self.state_names)
```

Validation rules:

- `state_names` length is the state dimension.
- Index sets are disjoint.
- The union of stochastic, deterministic, and auxiliary indices covers all
  states, unless a model explicitly marks an index as external.
- `innovation_dim` is positive for stochastic models and may differ from the
  number of stochastic coordinates.
- The partition is metadata, not an inference from `Q` alone.

### Structural transition protocol

```python
class StructuralStateSpaceModel(Protocol):
    partition: StatePartition

    def initial_mean(self, theta): ...
    def initial_cov(self, theta): ...
    def innovation_cov(self, theta): ...
    def observation_cov(self, theta): ...

    def unpack_state(self, x): ...
    def pack_state(self, stochastic_state, deterministic_state): ...

    def propagate_stochastic(self, stochastic_prev, innovation, theta, context): ...
    def complete_deterministic(
        self,
        deterministic_prev,
        stochastic_prev,
        stochastic_next,
        theta,
        context,
    ): ...

    def observe(self, x_points, theta, context): ...
```

`context` should be a typed or dataclass carrier for model solution objects,
time-varying inputs, pruned second-order corrections, known regressors, and
measurement masks.  Avoid arbitrary dictionaries in the public API unless the
codebase already uses them consistently.

### Filter configuration

```python
@dataclass(frozen=True)
class StructuralFilterConfig:
    integration_space: Literal["innovation", "stochastic_state", "full_state"]
    deterministic_completion: Literal["required", "none", "approximate"]
    approximation_label: str | None = None
    allow_full_state_for_mixed: bool = False
```

Rules:

- Mixed models default to `deterministic_completion="required"`.
- `integration_space="full_state"` on a mixed model requires
  `allow_full_state_for_mixed=True` and a nonempty approximation label.
- Result objects must record the integration space and approximation label.

### Result metadata

Every likelihood/filter result should expose:

```python
FilterRunMetadata(
    filter_name=...,
    partition=...,
    integration_space=...,
    deterministic_completion=...,
    approximation_label=...,
    differentiability_status=...,
    compiled_status=...,
)
```

This is necessary so downstream HMC diagnostics cannot confuse a collapsed
approximation with structural filtering.

## Filter backends to support

### Exact and square-root Kalman

Requirements:

- Accept singular or rank-deficient process covariance when mathematically
  valid.
- Preserve exact linear Gaussian likelihood semantics.
- Support collapsed companion-form representations for AR(p) and affine
  MacroFinance models.
- Expose derivative hooks without forcing nonlinear structural completion.

### SVD sigma-point filter

Requirements:

- Generate sigma points in declared integration space.
- Lift stochastic points through deterministic completion before observation.
- Preserve structural manifold identities pointwise.
- Support additive innovation covariance and later state-dependent extensions.
- Reject mixed full-state mode unless explicitly labeled as approximation.

### Particle filters

Requirements:

- Sample shocks or stochastic-state proposals.
- Complete deterministic coordinates exactly.
- If deterministic coordinates are proposed with artificial noise, require
  proposal/target correction or an approximation label.

### Derivative/Hessian support

Requirements:

- Do not certify SVD/eigen gradients by prose alone.
- Use proof-carrying workflow from MathDevMCP:

  ```text
  /home/chakwong/MathDevMCP/docs/proof-carrying-derivation-agent-guide.md
  ```

- Split obligations into small labeled derivations.
- Require finite-difference, JVP/VJP, eager/compiled, and spectral-gap stress
  tests before HMC promotion.

## Implementation phases

Each phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

### Phase 0: repository and dependency inventory

Actions:

1. Record BayesFilter status:

   ```bash
   cd /home/chakwong/BayesFilter
   git status --short
   find . -maxdepth 3 -type f | sort
   ```

2. Record client status:

   ```bash
   cd /home/chakwong/python
   git status --short
   rg -n "SVDSigmaPointFilter|transition_points|Kalman|particle" src tests
   ```

3. Identify MacroFinance source files for analytic Kalman derivatives and
   state-space abstractions.  Do not copy code yet; record candidate APIs and
   tests.

Pass gate:

- Inventory note is appended to the BayesFilter reset memo.
- No implementation changes.

### Phase 1: write structural contracts and toy examples in docs

Actions:

1. Update or extend:

   - `docs/chapters/ch02_state_space_contracts.tex`
   - `docs/chapters/ch16_sigma_point_filters.tex`
   - `docs/chapters/ch18_svd_sigma_point.tex`
   - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`

2. Add examples:

   - AR(2) lag-shift model;
   - local-level plus deterministic accumulated sum;
   - Rotemberg NK `dy` identity;
   - SGU exogenous shocks vs endogenous completions;
   - MacroFinance affine/lag-stack example.

3. Mark source provenance explicitly.

Pass gate:

- The monograph states that structural deterministic completion is a generic
  BayesFilter contract, not a DSGE-only patch.

### Phase 2: create BayesFilter core interfaces

Actions:

1. Add a BayesFilter package skeleton if one does not yet exist.
2. Add structural metadata dataclasses.
3. Add validation helpers.
4. Add minimal result metadata.

Suggested module layout:

```text
bayesfilter/
  __init__.py
  structural.py
  filters/
    __init__.py
    kalman.py
    sigma_points.py
    particles.py
  testing/
    structural_fixtures.py
```

If the actual package layout differs, follow the repo's existing style.

Tests:

- partition validation;
- invalid overlapping indices;
- missing coverage;
- approximation label required for mixed full-state integration.

### Phase 3: AR(p) and toy structural fixtures

Actions:

1. Implement AR(2) as the first structural fixture.
2. Implement a nonlinear toy mixed-state fixture:

   ```text
   m_t = rho m_{t-1} + sigma eps_t
   k_t = alpha k_{t-1} + beta tanh(m_t)
   y_t = m_t + k_t + noise_t
   ```

3. Add dense quadrature or exact references where possible.

Tests:

- AR(2) exact Kalman likelihood agrees with collapsed companion-form Kalman.
- Structural sigma points preserve lag-shift identity exactly.
- Nonlinear toy structural sigma likelihood is finite and close to dense
  quadrature reference.
- Full-state approximation is blocked or explicitly labeled.

### Phase 4: implement structural SVD sigma-point backend

Actions:

1. Port or reimplement the generic SVD sigma-point primitives only after the
   interface and tests are stable.
2. Generate sigma points in `innovation` or `stochastic_state` space.
3. Complete deterministic states pointwise.
4. Evaluate observations on completed states.
5. Return run metadata.

Tests:

- AR(2) linear recovery.
- Nonlinear toy reference.
- deterministic manifold preservation.
- finite gradients for small differentiable examples.
- no unlabeled full-state approximation.

### Phase 5: exact/square-root Kalman with degenerate Q

Actions:

1. Implement or wrap exact Kalman code that accepts singular `Q` safely.
2. Add square-root/SVD factor variants.
3. Add analytic derivative hooks where available.

Tests:

- LGSSM reference;
- AR(p) companion-form reference;
- singular `Q` smoke;
- analytic derivative vs finite difference on small cases.

### Phase 6: MacroFinance adapter pilot

Actions:

1. Inspect `/home/chakwong/MacroFinance` analytic Kalman derivative code.
2. Do not port wholesale.
3. Build a minimal adapter around one model or fixture.
4. Reuse BayesFilter Kalman/derivative contracts.

Tests:

- existing MacroFinance derivative fixture reproduces prior result;
- BayesFilter wrapper matches MacroFinance reference likelihood/gradient;
- no duplicate filter implementation is created in MacroFinance.

### Phase 7: DSGE adapter pilot

Actions:

1. In `/home/chakwong/python`, adapt SmallNK, Rotemberg, and SGU metadata into
   BayesFilter structural contracts.
2. Keep DSGE economics and perturbation solution in `dsge_hmc`.
3. Route filtering through BayesFilter when structural nonlinear filtering is
   requested.
4. Fail closed for mixed-state DSGE models that do not yet have structural
   maps.

Tests:

- SmallNK all-exogenous partition works.
- Rotemberg `dy` identity is preserved.
- SGU exogenous rows are `a,zeta,mu`.
- Existing exact Kalman tests are unchanged.
- Old collapsed sigma path is blocked or labeled.

### Phase 8: HMC readiness gates

Actions:

1. Eager likelihood tests.
2. Eager gradient tests.
3. Eager/compiled parity.
4. Tiny HMC smoke.
5. Medium HMC only after value/gradient/compiled gates.

Pass labels:

- `filter-correct`: value and gradient tests pass.
- `sampler-usable`: tiny/medium HMC produces finite chains and diagnostics.
- `converged`: strict multi-chain posterior diagnostics pass.

Do not call a structural filter converged just because the code compiles.

## Required test ladder

Minimum BayesFilter tests:

```text
tests/test_structural_partition.py
tests/test_structural_ar_p.py
tests/test_structural_sigma_points.py
tests/test_degenerate_kalman.py
tests/test_filter_metadata.py
tests/test_derivative_validation_smoke.py
```

Client integration tests:

```text
/home/chakwong/python/tests/contracts/test_bayesfilter_dsge_adapter.py
/home/chakwong/MacroFinance/<appropriate adapter tests>
```

Expected checks:

- partition dimension and disjointness;
- deterministic identity preservation;
- linear recovery against exact Kalman;
- nonlinear reference against quadrature;
- singular covariance support;
- no unlabeled mixed full-state integration;
- finite gradients;
- compiled/eager parity when a compiled backend exists.

## Documentation deliverables

BayesFilter docs should include:

- structural state-space contract chapter;
- AR(p) example;
- DSGE endogenous/exogenous state chapter;
- MacroFinance adapter example;
- filter choice table;
- HMC target checklist;
- derivative validation checklist.

The DSGE-side docs should point back to this plan and describe only:

- DSGE model metadata;
- perturbation solution integration;
- observation equation mapping;
- test fixtures for NK/Rotemberg/SGU/EZ.

## Stop rules

Stop and ask for direction if:

- BayesFilter package layout is not yet decided.
- MacroFinance derivative code cannot be inspected without changing that repo.
- DSGE structural completion maps require new model derivations not present in
  docs.
- A test can only pass by adding noise to deterministic coordinates.
- A full-state approximation is being used without a label.
- A gradient claim depends on unverified SVD/eigen derivative assumptions.

## Reset handoff

At the start of the next implementation session:

```bash
cd /home/chakwong/BayesFilter
git status --short
sed -n '1,240p' docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md
sed -n '1,220p' docs/chapters/ch18b_structural_deterministic_dynamics.tex
```

Then inspect client-side context:

```bash
cd /home/chakwong/python
sed -n '1,220p' docs/plans/svd-structural-dsge-state-partition-implementation-plan-2026-05-04.md
rg -n "SVDSigmaPointFilter|_build_dsge_sigma_filter_components|transition_points" src tests
```

Do not treat `/home/chakwong/python` as the source of truth for the generic
filter design.  It is a DSGE client and integration test site.

## One-sentence principle

BayesFilter owns structural filtering; model projects own structural maps.
