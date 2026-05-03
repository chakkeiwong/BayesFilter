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

## Non-negotiable execution rules

These rules are mandatory.  They are written to prevent shortcutting,
misclassification of mathematically distinct cases, accidental greenfield
rewrites of already-implemented machinery, and silent approximation creep.

1. Math audit before implementation.
   No BayesFilter implementation work may begin until the relevant DSGE and
   MacroFinance derivations have been audited and reconciled against the
   proposed BayesFilter contract.

2. Reuse only after audit.
   Existing filtering, square-root, Kalman, derivative, and adapter machinery in
   `/home/chakwong/python` and `/home/chakwong/MacroFinance` should be treated
   as candidate source material, not as automatically correct code and not as
   disposable code to be reimplemented casually.

3. No silent approximation promotion.
   If an existing implementation is exact only for linear-Gaussian collapsed
   moments, it must stay labeled as such.  It must not be presented as a
   structural nonlinear filter for mixed-state models.

4. Shared API does not mean identical propagation algorithm.
   BayesFilter should present a common likelihood/filtering contract to client
   projects, but it may and should dispatch to different mathematically valid
   backends depending on whether the model is:
   - pure linear Gaussian / collapsed LGSSM;
   - nonlinear but fully stochastic;
   - nonlinear with stochastic plus deterministic-completion states.

5. Prefer extraction/wrapping of audited common machinery over greenfield
   rewrites.
   Reimplementation is allowed only when the existing code fails the audit,
   cannot be disentangled from client-specific logic, or cannot satisfy the
   BayesFilter contract and tests.

6. Fail closed when metadata or derivations are missing.
   If a model does not yet provide the structural partition or deterministic
   completion maps required for nonlinear structural filtering, BayesFilter must
   refuse the structural nonlinear path rather than guess.

7. Tooling is advisory for provenance and obligation checking, not a substitute
   for mathematical judgment.
   ResearchAssistant should be used for source retrieval and citation
   verification.  MathDevMCP should be used for obligation-level audit and
   assumption surfacing.  Neither tool should be treated as an automatic proof
   certificate for the full filter unless the generated evidence actually
   supports that claim.

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

## Common API versus backend semantics

To avoid ambiguity: the goal is a common estimator-facing API, not a claim that
all models should use the same internal propagation algorithm.

The common API should allow downstream code to call a single family of
likelihood, filtering, derivative, and HMC-target functions across model
projects.  Internally, BayesFilter should dispatch according to mathematically
validated structure:

- Pure linear-Gaussian or collapsed LGSSM models may use exact or square-root
  Kalman backends.
- Nonlinear models whose full state is genuinely stochastic may use nonlinear
  sigma-point or particle backends over the declared stochastic state.
- Mixed structural models with deterministic-completion coordinates must use
  integration over innovation or stochastic-state space plus deterministic
  completion.

Therefore “same filter” in this project means the same BayesFilter contract and
higher-level estimation API, not necessarily the same quadrature/sampling space
or the same transition parameterization in every model class.

## Mandatory mathematical source audit

Before any implementation phase that changes BayesFilter code, write a
mathematical audit note under `docs/plans` that reconciles the relevant source
material with the BayesFilter contract.

Primary source sets:

- DSGE monograph root:

  ```text
  /home/chakwong/python/docs/monograph.tex
  ```

  Relevant included chapters should cover at least DSGE model timing,
  perturbation structure, Kalman filtering, square-root UKF/SVD filtering, and
  any model-specific endogenous/exogenous state definitions used by SmallNK,
  Rotemberg, SGU, EZ, or future NAWM-style targets.

- MacroFinance monograph root:

  ```text
  /home/chakwong/latex/CIP_monograph/main.tex
  ```

  Relevant included chapters should cover at least state-space recursions,
  Kalman filtering, nonlinear filtering, multi-country state construction,
  differentiable particle filtering, and analytical validation.

Audit objectives:

1. Identify which formulas are already derived in the DSGE monograph.
2. Identify which formulas are already derived in the MacroFinance monograph.
3. State which formulas are mathematically equivalent after relabeling of state
   blocks and notation.
4. State which formulas are valid only for collapsed linear-Gaussian moment
   filtering.
5. State which formulas are required for structural nonlinear propagation with
   deterministic completion.
6. State which claims remain unverified, approximate, or model-specific.
7. State which existing implementations are candidates for extraction or
   wrapping into BayesFilter.

Required tool usage during the audit:

- Use ResearchAssistant to locate source labels, equations, theorem statements,
  appendix references, and citations when the derivation claim depends on a
  paper or monograph source.
- Use MathDevMCP to check obligation-level derivation steps, expose hidden
  assumptions, and flag unsupported or under-specified steps.
- Record provenance for each nontrivial mathematical claim: monograph section,
  chapter/equation label, paper label, or explicit derivation in BayesFilter
  notation.

The audit must not rely on:

- memory of what the derivation “probably meant”;
- code behavior alone as mathematical evidence;
- prose statements that are not backed by equations, labels, or explicit
  derivations.

Pass gate for the audit:

- A written audit note exists in `docs/plans`.
- The note identifies exact versus approximate paths explicitly.
- The note identifies reusable existing code paths and suspected risk points.
- The note lists unresolved mathematical gaps and blocks implementation on those
  gaps where they affect correctness.
- No BayesFilter backend implementation starts before this note exists.

## Reuse-and-audit migration rule

The intent of this project is not to casually rewrite filtering code that
already exists elsewhere.  The intent is to centralize audited common machinery
in BayesFilter while keeping model-specific logic in client repos.

Therefore, for every candidate backend or derivative path:

1. Audit the math source.
2. Audit the existing code path.
3. Classify the code path as one of:
   - exact and reusable as-is after extraction/wrapping;
   - reusable with localized fixes;
   - reusable only as a labeled approximation;
   - not reusable because it is mathematically wrong, too entangled, or too
     poorly specified.
4. Only then decide whether to extract, wrap, refactor, or reimplement.

This rule applies especially to:

- exact and square-root Kalman implementations;
- singular-`Q` handling;
- SVD sigma-point primitives;
- derivative and Hessian support;
- DSGE sigma-point adapter logic;
- MacroFinance analytic derivative providers.

A migration should preserve:

- mathematical semantics;
- approximation labels;
- existing validated regression fixtures where those fixtures are relevant;
- client-side ownership of economics/finance model logic.

A migration should not preserve:

- silent assumptions that were never documented;
- client-specific hacks that violate the BayesFilter contract;
- convenience noise added to deterministic coordinates to make a filter run.

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
- Zero rows in an impact matrix may be used as a diagnostic but must not be the
  sole source of truth for partition classification.

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
- Pure LGSSM or genuinely all-stochastic models must not be forced through
  deterministic-completion machinery when an exact or simpler validated backend
  exists.

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
- Prefer audited extraction/wrapping of existing Kalman or square-root code
  over greenfield reimplementation.

### SVD sigma-point filter

Requirements:

- Generate sigma points in declared integration space.
- Lift stochastic points through deterministic completion before observation.
- Preserve structural manifold identities pointwise.
- Support additive innovation covariance and later state-dependent extensions.
- Reject mixed full-state mode unless explicitly labeled as approximation.
- Reuse audited generic SVD sigma-point primitives where valid; rewrite only
  the parts that fail the structural audit or cannot be separated from
  client-specific logic.

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

- Use ResearchAssistant when source-backed literature claims or monograph-source
  reconciliation is required.
- Split obligations into small labeled derivations.
- Require finite-difference, JVP/VJP, eager/compiled, and spectral-gap stress
  tests before HMC promotion.
- Do not migrate derivative code into BayesFilter merely because it already
  exists in a client repo; migrate it only after the derivative path is audited
  mathematically and numerically.

## Implementation phases

Each phase follows:

```text
plan -> audit math -> audit code -> execute -> test -> audit -> tidy -> update reset memo
```

No phase may skip the math-audit or code-audit components when correctness
depends on them.

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

3. Record MacroFinance status:

   ```bash
   cd /home/chakwong/MacroFinance
   git status --short
   rg -n "Kalman|filter|derivative|state_space|state-space|sigma|particle" .
   ```

4. Identify candidate source files for:
   - exact Kalman backends;
   - square-root or SVD Kalman backends;
   - nonlinear sigma-point primitives;
   - derivative/Hessian providers;
   - client-specific adapter logic.

5. Record which monograph chapters and appendices appear to contain the primary
   derivations for each candidate backend.

Pass gate:

- Inventory note is appended to the BayesFilter reset memo or a dedicated plan
  note.
- Candidate code paths and derivation sources are listed explicitly.
- No implementation changes.

### Phase 1: mathematical source audit and derivation reconciliation

Actions:

1. Read the relevant DSGE monograph chapters from:

   ```text
   /home/chakwong/python/docs/monograph.tex
   ```

2. Read the relevant MacroFinance monograph chapters from:

   ```text
   /home/chakwong/latex/CIP_monograph/main.tex
   ```

3. Produce a written audit note in BayesFilter `docs/plans` that states:
   - the common notation used for BayesFilter;
   - the mapping from DSGE notation into BayesFilter notation;
   - the mapping from MacroFinance notation into BayesFilter notation;
   - which filtering equations are exact collapsed LGSSM equations;
   - which nonlinear equations require structural deterministic completion;
   - which equations are already validated by monograph derivation;
   - which equations still require local derivation in BayesFilter notation.

4. Use ResearchAssistant where helpful to retrieve equation labels, theorem
   labels, appendix references, and citation provenance.

5. Use MathDevMCP where helpful to audit local derivation steps, surface
   assumptions, and mark unsupported steps as blockers rather than smoothing
   them over.

6. Create a backend-classification table at the mathematical level:

   | Backend/path | Exact for LGSSM? | Exact for mixed nonlinear structural models? | Approximation? | Reuse candidate? | Notes |
   | --- | --- | --- | --- | --- | --- |

Pass gate:

- The audit note exists.
- The note states explicitly what is exact, what is approximate, and what is
  still unresolved.
- The note is sufficient for an implementation agent to know which backend is
  mathematically allowed for each model class.
- No BayesFilter code implementation starts before this gate passes.

### Phase 2: code audit and migration decision

Actions:

1. Audit existing BayesFilter-adjacent code in `/home/chakwong/python` and
   `/home/chakwong/MacroFinance` against the mathematical audit.
2. For each candidate code path, classify it as:
   - extract/wrap as-is after tests;
   - extract/wrap with localized fixes;
   - keep only as labeled approximation;
   - do not reuse.
3. Record hidden assumptions, client-specific entanglements, and suspected bug
   risks.
4. Record which tests or regression fixtures are needed to preserve validated
   behavior during migration.

Pass gate:

- A written code-audit section or note exists.
- Every planned migration path has an explicit reuse decision.
- Greenfield reimplementation is justified in writing wherever chosen.

### Phase 3: write structural contracts and toy examples in docs

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
4. State explicitly in the docs that a shared BayesFilter API does not imply a
   single internal propagation algorithm for every model class.

Pass gate:

- The monograph states that structural deterministic completion is a generic
  BayesFilter contract, not a DSGE-only patch.
- The monograph distinguishes shared API from backend-specific propagation
  semantics.

### Phase 4: create BayesFilter core interfaces

Actions:

1. Add a BayesFilter package skeleton if one does not yet exist.
2. Add structural metadata dataclasses.
3. Add validation helpers.
4. Add minimal result metadata.
5. Encode failure-closed behavior for mixed models lacking structural metadata.

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
- approximation label required for mixed full-state integration;
- failure when mixed nonlinear models request structural filtering without
  structural metadata.

### Phase 5: AR(p) and toy structural fixtures

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

### Phase 6: implement structural SVD sigma-point backend

Actions:

1. Extract, wrap, or reimplement generic SVD sigma-point primitives strictly
   according to the Phase 2 reuse decision.
2. Do not rewrite audited reusable primitives merely for style consistency.
3. Generate sigma points in `innovation` or `stochastic_state` space.
4. Complete deterministic states pointwise.
5. Evaluate observations on completed states.
6. Return run metadata.
7. Preserve explicit approximation labels where legacy behavior is retained only
   as approximation.

Tests:

- AR(2) linear recovery.
- Nonlinear toy reference.
- deterministic manifold preservation.
- finite gradients for small differentiable examples.
- no unlabeled full-state approximation.

### Phase 7: exact/square-root Kalman with degenerate Q

Actions:

1. Wrap, extract, or reimplement exact Kalman code according to the Phase 2
   audit decision.
2. Preserve support for singular `Q` where mathematically valid.
3. Add square-root/SVD factor variants.
4. Add analytic derivative hooks where available and audited.

Tests:

- LGSSM reference;
- AR(p) companion-form reference;
- singular `Q` smoke;
- analytic derivative vs finite difference on small cases.

### Phase 8: MacroFinance adapter pilot

Actions:

1. Inspect `/home/chakwong/MacroFinance` analytic Kalman derivative code and
   monograph support.
2. Do not port wholesale.
3. Extract or wrap only the audited reusable generic pieces.
4. Keep MacroFinance-specific financial model construction in MacroFinance.
5. Build a minimal adapter around one model or fixture.
6. Reuse BayesFilter Kalman/derivative contracts.

Tests:

- existing MacroFinance derivative fixture reproduces prior result;
- BayesFilter wrapper matches MacroFinance reference likelihood/gradient;
- no duplicate filter implementation is created in MacroFinance.

### Phase 9: DSGE adapter pilot

Actions:

1. In `/home/chakwong/python`, adapt SmallNK, Rotemberg, and SGU metadata into
   BayesFilter structural contracts.
2. Keep DSGE economics and perturbation solution in `dsge_hmc`.
3. Route filtering through BayesFilter when structural nonlinear filtering is
   requested.
4. Fail closed for mixed-state DSGE models that do not yet have structural
   maps.
5. Preserve old collapsed sigma behavior only if it is blocked or explicitly
   labeled as approximation.

Tests:

- SmallNK all-exogenous partition works.
- Rotemberg `dy` identity is preserved.
- SGU exogenous rows are `a,zeta,mu`.
- Existing exact Kalman tests are unchanged.
- Old collapsed sigma path is blocked or labeled.

### Phase 10: HMC readiness gates

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
- compiled/eager parity when a compiled backend exists;
- legacy reused paths agree with audited reference fixtures where applicable.

## Documentation deliverables

BayesFilter docs should include:

- structural state-space contract chapter;
- AR(p) example;
- DSGE endogenous/exogenous state chapter;
- MacroFinance adapter example;
- filter choice table;
- HMC target checklist;
- derivative validation checklist;
- a short explicit statement that shared API does not imply one propagation
  backend for all model classes.

The DSGE-side docs should point back to this plan and describe only:

- DSGE model metadata;
- perturbation solution integration;
- observation equation mapping;
- test fixtures for NK/Rotemberg/SGU/EZ.

## Stop rules

Stop and ask for direction if:

- the mathematical audit has not yet established whether a path is exact,
  approximate, or unresolved;
- BayesFilter package layout is not yet decided;
- MacroFinance derivative code cannot be inspected without changing that repo;
- DSGE structural completion maps require new model derivations not present in
  docs;
- a test can only pass by adding noise to deterministic coordinates;
- a full-state approximation is being used without a label;
- a greenfield rewrite is being considered even though an audited reusable path
  appears available;
- a gradient claim depends on unverified SVD/eigen derivative assumptions.

## Reset handoff

At the start of the next implementation session:

```bash
cd /home/chakwong/BayesFilter
git status --short
sed -n '1,260p' docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md
sed -n '1,220p' docs/chapters/ch18b_structural_deterministic_dynamics.tex
```

Then inspect client-side context:

```bash
cd /home/chakwong/python
sed -n '1,220p' docs/plans/svd-structural-dsge-state-partition-implementation-plan-2026-05-04.md
rg -n "SVDSigmaPointFilter|_build_dsge_sigma_filter_components|transition_points" src tests
```

And inspect MacroFinance source context:

```bash
cd /home/chakwong/MacroFinance
rg -n "Kalman|filter|derivative|state_space|state-space|sigma|particle" .
sed -n '1,220p' main.tex
```

Do not treat `/home/chakwong/python` as the source of truth for the generic
filter design.  It is a DSGE client and integration test site.

Do not treat `/home/chakwong/MacroFinance` as the source of truth for the
generic filter design.  It is a MacroFinance client and regression/reference
site.

## One-sentence principle

BayesFilter owns audited common filtering contracts and implementations; model
projects own structural maps, model logic, and client adapters.
