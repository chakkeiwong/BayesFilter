# Plan: BayesFilter structural SVD code implementation handoff

## Date

2026-05-05

## Purpose

This is a code-implementation handoff for another agent.  The current session
is reserved for documentation and planning only.  Do not implement code from
this plan in this session.

The implementation target is `/home/chakwong/BayesFilter`, not
`/home/chakwong/python`.  The DSGE repository already exposes metadata and
fail-closed guards; BayesFilter should own the reusable filtering machinery.

## Current Answer

Yes, there is still code implementation to do.

No, the remaining implementation should not be done directly in this
documentation-only session.  The next coding agent should start with the gates
below and should not jump directly to HMC.

## Current Repository Status

Observed `/home/chakwong/python`:

```text
## main...origin/main
?? .codex
?? .serena/
```

Observed `/home/chakwong/BayesFilter`:

```text
## main...origin/main
?? "docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf"
?? docs/plans/templates/
```

BayesFilter is committed and pushed relative to `origin/main`.  The untracked
Julier PDF and templates should be handled deliberately before a coding agent
starts implementation.

## Source Plans

The coding agent must read these before editing code:

- `docs/plans/svd-structural-filter-final-gap-closure-plan-2026-05-05.md`
- `docs/plans/svd-structural-filter-final-gap-closure-audit-2026-05-05.md`
- `docs/plans/svd-structural-filter-final-gap-closure-reset-memo-2026-05-05.md`
- `/home/chakwong/BayesFilter/docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
- `/home/chakwong/BayesFilter/docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
- `/home/chakwong/MathDevMCP/docs/proof-carrying-derivation-agent-guide.md`

## Implementation Objective

Implement and validate BayesFilter structural filtering code so that:

1. exact collapsed LGSSM filtering remains exact and separately labeled;
2. nonlinear structural sigma-point filtering integrates over declared
   stochastic or innovation coordinates;
3. deterministic/endogenous coordinates are completed pointwise through model
   maps;
4. mixed full-state nonlinear filtering remains fail-closed unless explicitly
   labeled as an approximation;
5. result metadata records integration space, completion policy, approximation
   label, differentiability status, and compiled status;
6. no HMC promotion occurs until value, derivative, and JIT gates pass.

## Non-Goals

- Do not modify `/home/chakwong/python` filtering code unless an adapter test
  explicitly requires a small DSGE-side change.
- Do not port DSGE economics into BayesFilter.
- Do not port MacroFinance model economics into BayesFilter.
- Do not call Rotemberg or SGU structurally fixed based on first-order bridge
  tests.
- Do not treat TFP NUTS as the default fix for HMC issues.
- Do not commit generated PDFs unless explicitly requested.

## Phase 0: Repo Hygiene and Inputs

Plan:
- Confirm BayesFilter is clean except known untracked documentation inputs.
- Decide whether to commit, ignore, or leave untracked:
  - `docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf`;
  - `docs/plans/templates/`.

Execute:
- Run:

```bash
cd /home/chakwong/BayesFilter
git status --short --branch
git log -5 --oneline --decorate
```

Test:
- No code changes.

Audit:
- If there are tracked dirty code files, stop and ask for direction.

Exit gate:
- Coding starts only from a known clean or deliberately documented BayesFilter
  state.

## Phase 1: Mathematical Source Audit

Plan:
- Write a BayesFilter `docs/plans` audit note before backend implementation.
- Reconcile DSGE, MacroFinance, and BayesFilter notation.
- Classify exact, structural nonlinear, and labeled-approximation paths.

Execute:
- Use ResearchAssistant for source retrieval and citation/equation provenance
  when useful.
- Use MathDevMCP for obligation-level derivation checks when useful.
- Produce a table:

| Path | Exact LGSSM | Structural nonlinear | Labeled approximation | Derivative status | Reuse decision |
| --- | --- | --- | --- | --- | --- |

Test:
- The audit note must name exact source files, sections, labels, or explicit
  derivations.

Audit:
- Do not accept prose-only mathematical claims for derivative/Hessian logic.

Exit gate:
- No backend implementation until this written audit exists.

## Phase 2: Code Reuse and Migration Audit

Plan:
- Audit existing candidate code in:
  - `/home/chakwong/BayesFilter/bayesfilter`;
  - `/home/chakwong/python/src/dsge_hmc`;
  - `/home/chakwong/MacroFinance`.

Execute:
- Classify each candidate code path as:
  - reuse as-is;
  - reuse with localized fixes;
  - keep as labeled approximation;
  - reject and reimplement.

Test:
- The audit must identify regression tests required before migration.

Audit:
- Prefer audited extraction/wrapping over greenfield rewrites, but do not
  preserve code that violates the structural contract.

Exit gate:
- Implementation plan names exact files and tests.

## Phase 3: BayesFilter Structural Sigma-Point Backend

Plan:
- Harden or extend `/home/chakwong/BayesFilter/bayesfilter/filters/sigma_points.py`.
- Preserve the existing eager NumPy reference behavior unless the audit says
  otherwise.

Implementation instructions:
- Support declared integration spaces:
  - `innovation`;
  - later `stochastic_state`;
  - `full_state` only as explicit labeled approximation for mixed models.
- Ensure mixed models call deterministic completion pointwise.
- Preserve deterministic residual diagnostics.
- Return `FilterRunMetadata` with honest fields:
  - `filter_name`;
  - `partition`;
  - `integration_space`;
  - `deterministic_completion`;
  - `approximation_label`;
  - `differentiability_status`;
  - `compiled_status`.

Tests:

```bash
cd /home/chakwong/BayesFilter
pytest -q \
  tests/test_structural_partition.py \
  tests/test_structural_sigma_points.py \
  tests/test_structural_ar_p.py
```

Audit:
- If toy structural residuals fail, stop before DSGE adapter work.

Exit gate:
- Toy structural filters pass value, residual, and metadata assertions.

## Phase 4: Exact and Degenerate Kalman Spine

Plan:
- Keep exact collapsed LGSSM filtering separate from nonlinear structural
  sigma-point filtering.

Implementation instructions:
- Accept singular/rank-deficient process covariance when mathematically valid.
- Preserve exact linear Gaussian likelihood semantics.
- Add or preserve metadata so exact Kalman results cannot be confused with
  structural nonlinear approximations.

Tests:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
```

Audit:
- Do not force deterministic-completion machinery onto exact linear Kalman
  paths when collapsed moments are exact.

Exit gate:
- Exact linear tests pass independently of nonlinear structural tests.

## Phase 5: DSGE Adapter Integration

Plan:
- Consume DSGE metadata from `/home/chakwong/python` without making BayesFilter
  own DSGE economics.

Implementation instructions:
- Use explicit metadata exposed by DSGE models:
  - `bayesfilter_state_names`;
  - `bayesfilter_stochastic_indices`;
  - `bayesfilter_deterministic_indices`;
  - `bayesfilter_innovation_dim`;
  - `bayesfilter_deterministic_completion`.
- SmallNK may be all-stochastic.
- Rotemberg and SGU must be mixed structural or blocked.
- EZ remains blocked until timing audit.

Tests in BayesFilter:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

Tests in DSGE repo if an adapter fixture is added there:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

Audit:
- Legacy mixed full-state SVD probes must remain explicitly labeled
  approximations.

Exit gate:
- Adapter gates pass without enabling unsafe mixed full-state nonlinear
  filtering.

## Phase 6: Model-Specific Completion Evidence

Plan:
- Add model-specific residual tests before promoting Rotemberg, SGU, or EZ.

Implementation instructions:
- Rotemberg:
  - test `dy_next = y_next - y_current` pointwise;
  - test second-order/pruned identity preservation before nonlinear promotion.
- SGU:
  - derive and test residuals for `d,k,r,riskprem`;
  - keep nonlinear structural filtering blocked until residuals pass.
- EZ:
  - perform timing/partition audit;
  - expose metadata only after classification is clear.

Tests:
- Add focused tests in the owning repo once derivations are written.

Audit:
- Do not infer model classification solely from zero rows in `eta`.

Exit gate:
- Each model has explicit residual evidence or an explicit blocker.

## Phase 7: Derivative, Hessian, and JIT Gate

Plan:
- Do not promote to HMC until derivative and compiled-target gates pass.

Implementation instructions:
- Split derivative obligations into small components:
  - state transition;
  - deterministic completion;
  - observation map;
  - covariance prediction/update;
  - factorization;
  - log-likelihood update;
  - parameter transform.
- Use proof-carrying derivation workflow from MathDevMCP where useful.
- Add spectral-gap stress tests for SVD/eigen differentiation.
- Add eager versus compiled value and gradient parity tests.

Tests:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_derivative_validation_smoke.py
```

Audit:
- If SVD/eigen gradients are unstable near repeated eigenvalues, do not
  promote the backend to HMC.  Consider custom derivatives or a different
  factor backend.

Exit gate:
- Gradient and compiled-target tests pass for the intended model class.

## Phase 8: HMC Ladder

Plan:
- Run HMC only after value, residual, derivative, and JIT gates pass.

Implementation instructions:
- Start with LGSSM.
- Then nonlinear toy SSM.
- Then SmallNK.
- Then Rotemberg only after structural completion gate.
- Then SGU only after equilibrium residual gate.
- EZ only after timing audit.

Tests:
- Use existing DSGE HMC extended tests only as ladder stages, not as proof of
  convergence unless diagnostics pass.

Audit:
- Report labels precisely:
  - finite smoke;
  - stable gradient;
  - compiled target;
  - converged posterior.

Exit gate:
- No convergence claim without multi-chain R-hat, ESS, divergence, and recovery
  criteria.

## Final Commit Instructions for Coding Agent

- Commit BayesFilter implementation and tests in `/home/chakwong/BayesFilter`.
- If DSGE adapter tests require small `/home/chakwong/python` changes, commit
  those separately in `/home/chakwong/python`.
- Do not stage `.codex/`, `.serena/`, generated PDFs, or unrelated dirty docs.
- Update the relevant reset memo after every phase.

## Minimal First Coding Task

The smallest justified coding task is not HMC.  It is:

1. write the mathematical source audit in BayesFilter;
2. write the code reuse/migration audit;
3. add or harden BayesFilter structural sigma-point tests for toy structural
   models and metadata correctness.

Only after those pass should a coding agent touch DSGE structural adapter
integration or derivative/JIT/HMC work.
