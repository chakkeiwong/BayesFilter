# Plan: dependency-aware closure roadmap for remaining original-plan gaps

## Date

2026-05-05

## Purpose

This plan closes the remaining gaps from
`bayesfilter-structural-state-partition-core-plan-2026-05-04.md` after the
completed BayesFilter core work, MacroFinance adapter pilot, metadata closure,
and executable readiness gates.

The main risk now is not missing a single helper function.  The main risk is
executing phases in the wrong order and accidentally promoting approximation,
fixture-only evidence, or target-readiness checks into stronger claims such as
structural filtering correctness, production readiness, or HMC convergence.

This roadmap orders the remaining work by dependency:

```text
workspace hygiene
  -> provenance/control docs
    -> DSGE structural adapters
      -> particle-filter semantics
    -> factor-backend and derivative audit
      -> SVD/eigen derivative certification
        -> HMC sampler readiness
    -> MacroFinance provider evidence
      -> expanded MacroFinance adapters
        -> HMC sampler readiness
  -> release-quality docs and literature gates
```

## Current Closure State

Already closed or substantially closed:

- BayesFilter structural partition metadata and fail-closed filter config.
- AR/toy structural fixtures and structural sigma-point metadata tests.
- Exact value-side Kalman path and singular-process-covariance smoke coverage.
- MacroFinance one-country value adapter, delegated derivative bridge, target
  readiness gate, metadata extractors, production blocker metadata, and
  executable readiness gates.

Still open:

- DSGE adapter pilot.
- Particle-filter backend semantics and implementation.
- Square-root/SVD Kalman variants and generic analytic derivative hooks.
- SVD/eigen derivative certification for HMC promotion.
- MacroFinance expanded-provider evidence after the pilot gates.
- Actual HMC sampler diagnostics and promotion labels.
- Release-quality documentation/literature consolidation.

## Dependency-Aware Gap Register

| Order | Gap | Why this order is required | Closure evidence | Blocks |
| --- | --- | --- | --- | --- |
| R0 | Workspace hygiene and scope separation | All later phases write near already-dirty files.  Without an explicit scope boundary, reset-memo provenance and commits become unreliable. | Cleanly classified dirty files; explicit commit/staging rules; no ambiguous file in a phase write set. | Every implementation phase. |
| R1 | Provenance and control-layer consolidation | The project now has multiple plans, audits, and readiness labels.  A control layer must prevent stronger claims than the evidence supports. | Source map, blocker register, reset memo, and claim labels agree on closed, blocked, approximate, and candidate work. | DSGE, MacroFinance, HMC, and release docs. |
| R2 | DSGE adapter pilot | The original plan's main unclosed structural use case is DSGE metadata.  Particle filtering and HMC tests need model-side structural maps first. | At least one DSGE model exports certified structural partition metadata through an adapter and mixed models fail closed when maps are missing. | Structural particles, DSGE HMC readiness, DSGE case-study docs. |
| R3 | Particle-filter semantics before implementation | Particle filters can silently change the model by injecting noise into deterministic coordinates.  The semantics must be fixed before code. | Written semantics audit plus tests that preserve deterministic completion and block unlabeled deterministic noise. | Particle backend promotion and PF/PMCMC documentation. |
| R4 | Factor-backend and derivative-hook audit | Derivative and HMC evidence depends on the numerical backend.  Certifying formulas before classifying backends certifies the wrong object. | Backend matrix covering value, derivative, singular covariance, compiled support, and approximation labels. | SVD/eigen derivative certification and HMC promotion. |
| R5 | SVD/eigen derivative certification | HMC gradients through spectral operations are valid only under explicit assumptions such as separated singular/eigen values. | Gap telemetry, finite-difference/JVP/VJP checks, MathDevMCP-labeled obligations where formulas are documented, and fail/warn behavior near small gaps. | HMC labels for spectral backends. |
| R6 | MacroFinance expanded-provider evidence | The pilot gates exist, but large-scale, cross-currency, and production providers still need provider-owned evidence. | Provider-owned masked derivative metadata, blockwise oracle checks, final-data identification evidence, and production exposure gates. | MacroFinance production adaptation and MacroFinance HMC runs. |
| R7 | HMC sampler readiness | Finite target evaluation is not convergence.  Sampler labels require real chain diagnostics after target and derivative gates pass. | Tiny then medium chain diagnostics with finite values, acceptance bounds, zero divergences, split-R-hat, ESS, and backend sensitivity checks. | Any `sampler-usable` or `converged` claim. |
| R8 | Release-quality docs and literature gates | Documentation should be synchronized only after implementation gates settle, otherwise it will either lag or overclaim. | Docs build, source-map parse, stale-claim search, primary-source/literature audit for all promoted claims. | Release or monograph-ready status. |

## Dependency Rules

1. R0 and R1 are hard prerequisites for all later implementation commits.
2. R2 precedes DSGE structural particles and DSGE HMC because it supplies the
   model-side structural maps.
3. R3 can begin after R1 for pure BayesFilter toy semantics, but client-facing
   particle claims wait for R2 adapters.
4. R4 precedes R5 because derivative certification must name the exact factor
   backend being certified.
5. R5 and R6 both feed R7.  HMC sampler work may start only for targets whose
   value, gradient, compiled/backend, and provider-evidence gates pass.
6. R8 is last for release language, but its stale-claim search should be run
   after every phase as a lightweight audit.

## Execution Discipline

Every phase below follows:

```text
plan -> audit -> execute -> test -> audit -> tidy -> update reset memo
```

Do not continue to a dependent phase when its input gate fails.  When a phase
finds that the next phase is not justified, stop and ask for direction.

Do not stage unrelated dirty files.  In the current workspace, there are known
dirty chapter/reset-memo/code files from other passes; each implementation pass
must stage by explicit path or hunk.

Each reset-memo update must include:

- phase objective and pre-audit result;
- files touched and intentional scope;
- tests run and exact pass/fail interpretation;
- audit result, including any new blocker;
- tidy-up actions;
- whether the next dependent phase is still justified.

When the reset memo already contains unrelated unstaged edits, stage only the
hunks created by the current phase.

## Phase R0: workspace hygiene and scope separation

### Motivation

The repo currently has unrelated dirty files from structural-filtering and
chapter-writing work.  Any remaining-gap closure pass that starts from this
state risks mixing commits, confusing reset-memo provenance, and making later
audits unreliable.  Hygiene is therefore a dependency for every other phase.

### Actions

1. Record `git status --short --branch`.
2. Classify dirty files as:
   - current pass;
   - pre-existing user/parallel-work edits;
   - generated/cache files;
   - unknown.
3. Commit, shelve, or explicitly leave unrelated work unstaged according to the
   user's scope.  Do not clean, revert, or stage pre-existing edits merely to
   make the worktree look tidy.
4. Establish commit-scope rules for the next phases.

### Tests

- `git status --short --branch`
- `git diff --name-only`
- `git diff --cached --name-only` before each commit

### Exit Gate

- No phase starts with ambiguous dirty files in its write set.
- Reset memo can be updated without accidentally staging unrelated sections.

## Phase R1: provenance and control-layer consolidation

### Motivation

The remaining work spans BayesFilter, DSGE, MacroFinance, documentation, and
HMC diagnostics.  Before adding behavior, the project needs a clean control
layer that says what is closed, what remains blocked, and which labels are
allowed.  Otherwise later tests can pass while the monograph still overclaims.

### Actions

1. Update or create a blocker register that maps each remaining gap to:
   - dependency;
   - current evidence;
   - required gate;
   - owner repo;
   - allowed claim label.
2. Reconcile `docs/source_map.yml` with all current plan/audit files.
3. Ensure reset memo has an explicit "closed vs blocked" summary.
4. Keep claims conservative:
   - `filter-correct` only after value/gradient tests;
   - `sampler-usable` only after finite chain diagnostics;
   - `converged` only after strict multi-chain diagnostics.
5. Add an explicit label policy for approximation-only paths:
   - collapsed linear-Gaussian moment filter;
   - structural nonlinear filter;
   - enlarged-state approximation;
   - blocked pending metadata.

### Tests

- YAML parse for `docs/source_map.yml`.
- `rg -n "converged|production-ready|Identified|not_claimed|blocked" docs`
  review for stale overclaims.
- `git diff --check`.

### Exit Gate

- Every remaining gap has a clear dependency and a clear promotion gate.

## Phase R2: DSGE adapter pilot

### Motivation

The original plan was motivated heavily by structural DSGE state partitions.
MacroFinance is now well-gated, but the DSGE client path remains the largest
unclosed core-use-case gap.  This should run before particle filtering or HMC
sampler work because it supplies the structural maps those phases need.

### Actions

1. In `/home/chakwong/python`, audit current SmallNK, Rotemberg, SGU, and EZ or
   NAWM-style state metadata.
2. Add client-side adapter contracts that map DSGE metadata into BayesFilter:
   - state names;
   - stochastic indices;
   - deterministic/completion indices;
   - innovation dimension;
   - observation mapping;
   - approximation label for legacy collapsed/full-state paths.
3. Start with SmallNK if it is all-exogenous or easiest to certify.
4. Then add Rotemberg identity preservation, especially the `dy` identity.
5. Then add SGU exogenous-row contract: `a`, `zeta`, `mu`.
6. Fail closed for any model lacking deterministic completion maps.

### Tests

BayesFilter side:

- Existing structural tests still pass.
- No unlabeled mixed full-state integration.

DSGE side:

- `tests/contracts/test_bayesfilter_dsge_adapter.py`
- SmallNK partition test.
- Rotemberg `dy` identity test.
- SGU exogenous-row test.
- Legacy collapsed path blocked or labeled.

### Exit Gate

- DSGE structural metadata is adapter-ready for at least one simple model.
- Mixed models without structural maps fail closed.
- No DSGE economics are moved into BayesFilter.

## Phase R3: particle-filter semantics before implementation

### Motivation

Particle filters depend on the same structural transition semantics as DSGE
adapters.  Implementing particles before structural maps are validated would
invite the exact error the original plan warns against: adding artificial noise
to deterministic coordinates just to make a filter run.

### Actions

1. Write a short particle semantics audit:
   - shock-space proposals;
   - stochastic-state proposals;
   - deterministic completion;
   - proposal/target correction;
   - approximation labels for enlarged proposal spaces.
2. Add `ParticleFilterConfig` and metadata only after the semantics audit.
3. Implement a minimal bootstrap particle filter for structural toy models.
4. Reject artificial deterministic-coordinate noise unless an approximation
   label and correction policy are supplied.

### Tests

- AR/linear toy particle likelihood smoke.
- Deterministic completion preserved pointwise.
- Artificial deterministic noise blocked without approximation label.
- Optional dense-reference comparison on a small nonlinear toy.

### Exit Gate

- Particle filtering is available only for declared structural proposal spaces.
- No model-changing deterministic noise is silently accepted.

## Phase R4: factor-backend and derivative-hook audit

### Motivation

SVD/eigen derivative certification and HMC promotion depend on knowing which
factor backends are trusted.  This phase must happen before derivative
certification because otherwise tests may be certifying the wrong numerical
object.

### Actions

1. Audit exact, square-root, QR, and SVD Kalman candidates.
2. Classify each path:
   - exact reusable;
   - reusable with fixes;
   - approximation only;
   - not reusable.
3. Decide which derivative hooks belong in BayesFilter versus client repos.
4. Add or refine result metadata:
   - factor backend;
   - differentiability status;
   - compiled status;
   - singular/eigen-gap diagnostics if relevant.

### Tests

- LGSSM reference parity across accepted factor backends.
- Singular `Q` smoke.
- Small finite-difference derivative check where derivative hooks exist.
- Eager/compiled parity only for compiled paths that are actually supported.

### Exit Gate

- The project knows which backends are trusted for value, derivative, and HMC
  target use.

## Phase R5: SVD/eigen derivative certification

### Motivation

The original plan explicitly blocks gradient claims based on unverified
SVD/eigen assumptions.  This phase converts that caution into proof-carrying
and numerical evidence.

### Actions

1. Add spectral-gap telemetry to SVD/sigma-point derivative paths.
2. Use MathDevMCP for small labeled derivative obligations where formulas are
   used in docs or code.
3. Add finite-difference, JVP/VJP, eager/compiled, and stress tests.
4. Require failure or warning labels near repeated singular values or small
   gaps.

### Tests

- SVD derivative finite-difference checks away from small gaps.
- Stress tests near small singular/eigen gaps.
- JVP/VJP parity where available.
- HMC target gate remains blocked when derivative assumptions fail.

### Exit Gate

- SVD/eigen derivative paths have explicit validity regions and telemetry.
- HMC promotion cannot proceed through an uncertified spectral derivative path.

## Phase R6: MacroFinance expanded-provider evidence

### Motivation

The MacroFinance pilot is closed at the metadata/gate level, but expanded
providers still need stronger evidence before adapter promotion.  This phase
uses the gates already added to BayesFilter rather than inventing new
readiness language.

### Actions

1. Add provider-level `masked_derivative_order_supported` metadata for
   large-scale providers, or keep caller override as non-production-only.
2. Expand cross-currency oracle checks to deterministic blockwise samples:
   - dynamics;
   - transition covariance;
   - observation loadings;
   - measurement error.
3. Keep production exposure blocked until a real final calibrated ten-country
   provider reports:
   - `final_ready=True`;
   - no blockers;
   - final-data identification evidence;
   - sparse backend policy.

### Tests

- Large-scale dense and sparse mask gate tests.
- Cross-currency blockwise oracle discrepancy tests.
- Production scaffold remains blocked.
- Final-ready fake or fixture passes only when all final evidence exists.

### Exit Gate

- Expanded MacroFinance adapters are promoted only when BayesFilter gates pass
  using provider-owned evidence.

## Phase R7: HMC sampler readiness

### Motivation

Target readiness is not sampler convergence.  BayesFilter now has target and
diagnostic gates, but the original Phase 10 requires real HMC smoke/medium runs
before stronger labels are allowed.

### Actions

1. Run target gates on candidate adapters.
2. Run tiny HMC smoke only after value, gradient, and compiled parity pass.
3. Run medium HMC only after tiny smoke passes.
4. Feed real chain diagnostics into `evaluate_macrofinance_hmc_diagnostic_gate`
   or equivalent DSGE gates.
5. Promote labels conservatively:
   - `filter-correct`;
   - `sampler-usable`;
   - `converged`.

### Tests

- Finite chains.
- Acceptance in target interval.
- Zero divergences.
- ESS threshold.
- Split R-hat threshold.
- Backend sensitivity comparison.

### Exit Gate

- No target can be called `sampler-usable` without real chain diagnostics.
- No target can be called `converged` without strict multi-chain diagnostics.

## Phase R8: release-quality documentation and literature gates

### Motivation

After implementation, the docs must not lag behind the code.  The final release
risk is stale language: claims that were once blocked but are now closed, or
claims that remain blocked but sound complete.

### Actions

1. Update:
   - structural state-space contract chapter;
   - AR(p) example;
   - DSGE adapter chapter/section;
   - MacroFinance adapter example;
   - filter choice table;
   - HMC target checklist;
   - derivative validation checklist;
   - source map and reset memo.
2. Run literature/provenance audit on any claim involving:
   - particle filtering;
   - PMCMC;
   - SVD/eigen derivatives;
   - transport/surrogate methods;
   - production macro-finance readiness.
3. Remove stale blocked language only after the corresponding gate passes.

### Tests

- `latexmk` or project doc build target if available.
- YAML source-map parse.
- Placeholder/stale-claim search.
- `git diff --check`.

### Exit Gate

- Docs and code agree on what is implemented, approximate, blocked, and
  production-ready.

## Optimal Commit Strategy

Use one commit per dependency layer:

1. Hygiene/provenance.
2. DSGE adapter pilot.
3. Particle semantics/backend.
4. Factor/derivative audit and metadata.
5. SVD/eigen derivative certification.
6. MacroFinance expanded-provider evidence.
7. HMC sampler readiness.
8. Release docs.

Do not combine client repo changes and BayesFilter changes unless the tests
require an atomic cross-repo handoff; even then, record both commit hashes in
the reset memo.

## Final Success Criteria

The original plan is fully closed when:

- DSGE and MacroFinance clients both route appropriate workloads through
  BayesFilter contracts.
- Mixed structural models preserve deterministic identities pointwise.
- Full-state approximations are blocked or labeled.
- Particle filters respect structural completion semantics.
- Factor and derivative backends have explicit validity labels.
- SVD/eigen derivative claims have telemetry and numerical/proof evidence.
- HMC labels are backed by real target and chain diagnostics.
- Documentation, source map, and reset memo agree with implementation state.
