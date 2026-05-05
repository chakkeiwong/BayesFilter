# Plan: structural SVD final execution path with research and derivation gates

## Date

2026-05-06

## Purpose

This plan supersedes the earlier implementation roadmap as the execution plan
for the structural SVD filtering phase.  It keeps the same twelve substantive
phases, but makes the dependency path and local tool gates explicit.

The main correction is operational: ResearchAssistant and MathDevMCP are not
optional conveniences for this phase.  They are evidence gates for source
support, derivation routing, document-code consistency, and release claim
discipline.

## Final goal

BayesFilter should become the shared filtering layer for structural
state-space models used by generic nonlinear SSMs, DSGE, MacroFinance, and
future NAWM-scale systems.  The final implementation must preserve three
separate paths:

1. exact collapsed linear-Gaussian/Kalman filtering;
2. nonlinear structural filtering over innovation or stochastic coordinates
   with pointwise deterministic completion;
3. explicitly labeled approximation paths, including mixed full-state
   nonlinear sigma-point approximations.

No HMC target may be promoted until value, residual, derivative, and JIT gates
all pass.

## Evidence policy

Every phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

The reset memo entry for each phase must record:

- exact files reviewed or changed;
- exact commands and MCP calls used;
- test results;
- ResearchAssistant source-support status, when relevant;
- MathDevMCP derivation or doc-code status, when relevant;
- interpretation;
- whether the next phase remains justified.

Allowed evidence labels:

| Label | Meaning |
| --- | --- |
| `source_supported` | ResearchAssistant has a local source/review record that supports the claim, or the phase records an explicit reviewed local source artifact. |
| `source_missing` | ResearchAssistant/local source search did not find support; the claim must remain a blocker or be supported by a new reviewed source artifact. |
| `derivation_checked` | MathDevMCP or an explicit proof/test obligation checks the bounded claim. |
| `human_review_required` | MathDevMCP can locate/route the obligation but cannot certify it with the available backend. |
| `doc_code_consistent` | MathDevMCP or tests show the code implements the documented contract. |
| `value_only` | Value-side tests pass, but derivative/JIT/HMC promotion remains blocked. |
| `approximation_only` | The backend targets a labeled approximation, not the exact structural law. |
| `not_claimed` | The evidence does not support the stronger claim. |

## Hard stop rules

Stop and ask for direction if a phase would:

- change DSGE or MacroFinance economic semantics;
- add artificial process noise to deterministic coordinates;
- infer state roles only from zero rows of a shock-impact matrix;
- promote Rotemberg, SGU, EZ, or NAWM-style filtering without model-specific
  residual evidence;
- certify SVD/eigen derivatives by prose rather than proof/test evidence;
- claim HMC convergence from smoke tests;
- proceed from `source_missing` to a literature claim;
- proceed from `human_review_required` to a theorem-level claim without a
  written human derivation and test obligation;
- stage unrelated dirty files, generated PDFs, or another agent's changes.

## Tool readiness observed

ResearchAssistant is available in read-only/offline mode.  Its local lifecycle,
metadata-only ingest, PDF-text ingest, structured source inspection, and parser
smoke workflows report ready.  Broad local searches for Julier/Uhlmann
unscented-transform, SVD matrix-backprop, and NUTS support returned no matching
paper summaries in the current local index.  Therefore Phase 1 must create a
source-support table rather than assuming the literature gate is already
closed.

MathDevMCP can search labeled LaTeX blocks and route typed obligations.  It
located the structural pushforward proposition
`prop:bf-structural-ukf-pushforward` and related structural contract labels.
For the proposition it routed the obligation to human review because the
available backend cannot certify the measure-theoretic pushforward claim from
the local notation alone.  Therefore Phase 1 must pair MathDevMCP extraction
with explicit human-readable derivations and acceptance tests.

## Dependency path

```text
Preflight hygiene
  -> Phase 1 source and derivation audit
  -> Phase 2 code reuse and doc-code audit
  -> Phase 3 structural sigma-point core
  -> Phase 4 exact Kalman spine
  -> Phase 5 generic structural fixtures
  -> Phase 6 MacroFinance adapter classification
  -> Phase 7 DSGE adapter integration
  -> Phase 8 model-specific DSGE residual evidence
  -> Phase 9 derivative and Hessian safety
  -> Phase 10 JIT/static-shape production gate
  -> Phase 11 HMC ladder
  -> Phase 12 documentation/provenance/release gate
```

Phases 3--5 may be executed as one BayesFilter foundation block after Phases 1
and 2 pass, but their results and labels must remain separate.  Phase 8 may be
split by model: Rotemberg, SGU, and EZ have different blockers.  Phase 11 must
not begin for a model until that model has passed Phases 8--10.

## Preflight: workspace and baseline

### Motivation

The workspace has pre-existing untracked and dirty documentation files.  The
execution pass must not stage unrelated assets or another agent's changes.

### Implementation instructions

1. Run:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

2. Record current status in the reset memo.
3. Stage only files owned by the current phase.
4. Do not stage generated PDFs, `Zone.Identifier` files, `.codex/`, `.serena/`,
   or unrelated plan/references changes.

### Exit gate

Continue only if the write set is scoped and no tracked BayesFilter code file
is unexpectedly dirty.

## Phase 1: source and derivation audit

### Motivation

The structural/full-state distinction is mathematical before it is
computational.  Code changes are unsafe until the exact, structural nonlinear,
and approximation-only claims are classified with source and derivation
provenance.

### Implementation instructions

1. Create a source/derivation audit note under `docs/plans`.
2. Reconcile notation from:
   - BayesFilter Chapter 2 structural contracts;
   - BayesFilter Chapter 4 API contract;
   - BayesFilter Chapter 18b structural deterministic dynamics;
   - `/home/chakwong/python/docs/monograph.tex`;
   - `/home/chakwong/latex/CIP_monograph/main.tex`;
   - `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`.
3. Use ResearchAssistant to inspect local support for:
   - unscented transform and sigma-point approximation;
   - additive-noise augmentation and approximation order;
   - Matrix Backprop/SVD-eigen derivative risk;
   - HMC/NUTS diagnostics and convergence claims.
4. If local ResearchAssistant searches are empty, record `source_missing` and
   either ingest/review sources in a separate approved workflow or keep claims
   as blockers.
5. Use MathDevMCP to extract and route obligations for:
   - `def:bf-structural-state-partition`;
   - `asm:bf-approximation-labeling`;
   - `prop:bf-structural-ukf-pushforward`;
   - `eq:bf-structural-ukf-prop-deterministic-completion`.
6. For obligations routed to human review, write explicit bounded derivations
   and acceptance tests rather than treating the MCP diagnostic as proof.
7. Produce a table:

| Claim path | Exact LGSSM | Structural nonlinear | Labeled approximation | Source status | Derivation status | Promotion status |
| --- | --- | --- | --- | --- | --- | --- |

### Tests and tool gates

```bash
rg -n "exact|structural nonlinear|labeled approximation|deterministic completion" docs/plans docs/chapters
```

Required MCP checks:

- ResearchAssistant local source searches or source summaries for each
  literature claim.
- MathDevMCP label lookup/search for the structural contract labels above.

### Audit

- Confirm SmallNK is not used as evidence for Rotemberg, SGU, EZ, or
  NAWM-style models.
- Confirm zero shock-impact rows are consistency evidence only, not role
  metadata.
- Confirm `source_missing` and `human_review_required` claims remain blockers.

### Exit gate

No backend implementation begins until this written audit exists.

## Phase 2: code reuse and document-code audit

### Motivation

The codebase already contains structural filtering, exact Kalman, particle,
adapter, and backend metadata machinery.  The goal is to reuse passing code and
fix only proven gaps.

### Implementation instructions

1. Audit candidate code in:
   - `bayesfilter/structural.py`;
   - `bayesfilter/filters/sigma_points.py`;
   - `bayesfilter/filters/kalman.py`;
   - `bayesfilter/filters/particles.py`;
   - `bayesfilter/adapters/dsge.py`;
   - `bayesfilter/adapters/macrofinance.py`;
   - `bayesfilter/backends.py`;
   - `/home/chakwong/python/src/dsge_hmc`;
   - `/home/chakwong/MacroFinance`.
2. Classify each code path as:
   - reuse as-is;
   - reuse with localized fixes;
   - keep as labeled approximation;
   - reject and reimplement;
   - client-owned, do not migrate.
3. Use MathDevMCP `compare_label_code` or `implementation_brief` where a
   documented contract maps directly to code.
4. Write the regression test list before editing implementation code.

### Tests and tool gates

```bash
rg -n "Kalman|SVD|sigma|particle|gradient|Hessian|transition_points|completion|approximation_label" \
  bayesfilter tests /home/chakwong/python/src/dsge_hmc /home/chakwong/MacroFinance
```

Required audit output:

| File or module | Reuse decision | Contract source | Required tests | Owner |
| --- | --- | --- | --- | --- |

### Exit gate

Every planned implementation file has a reuse decision and a test list.

## Phase 3: BayesFilter structural sigma-point core

### Motivation

BayesFilter's structural sigma-point reference backend should be the local
value-side proof of concept for innovation integration, deterministic
completion, and honest approximation metadata.

### Implementation instructions

1. Harden `bayesfilter/filters/sigma_points.py` only if Phase 2 finds a real
   gap.
2. Mixed structural models must default to innovation or stochastic-state
   integration.
3. Mixed full-state integration must require explicit opt-in and a nonempty
   approximation label.
4. Deterministic coordinates must be completed pointwise.
5. Metadata must record integration space, completion policy, approximation
   label, differentiability status, and compiled status.

### Tests

```bash
pytest -q \
  tests/test_structural_partition.py \
  tests/test_structural_sigma_points.py \
  tests/test_structural_ar_p.py \
  tests/test_filter_metadata.py
```

### Audit

- Value pass gives `value_only` evidence unless derivative/JIT gates also pass.
- Do not rewrite passing code for style.

### Exit gate

Toy structural likelihood, residual, and metadata tests pass.

## Phase 4: exact Kalman and degenerate linear spine

### Motivation

Exact collapsed LGSSM/Kalman filtering is a distinct exact path.  It must not
be mixed with nonlinear structural sigma-point approximation.

### Implementation instructions

1. Preserve covariance Kalman and any square-root/factor paths as exact linear
   Gaussian value paths.
2. Accept rank-deficient process covariance where the filter equations are
   mathematically valid.
3. Keep exact Kalman metadata distinct from nonlinear approximation metadata.
4. Add tests only if Phase 2 identifies an uncovered exactness boundary.

### Tests

```bash
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
```

### Audit

- Do not force exact linear models through nonlinear structural completion.
- Do not add artificial nuggets to deterministic coordinates unless explicitly
  labeled.

### Exit gate

Exact linear tests pass independently of nonlinear structural tests.

## Phase 5: generic structural fixtures

### Motivation

The structural state issue is not DSGE-specific.  Generic fixtures prevent the
BayesFilter core from becoming a case-study workaround.

### Implementation instructions

1. Add or harden fixtures for:
   - AR(2) lag shift;
   - deterministic accumulator;
   - nonlinear stochastic plus deterministic completion toy model;
   - missing-data observation fixture.
2. Assert pointwise residuals.
3. Compare one-step likelihoods to analytic, dense-quadrature, or exact Kalman
   references where available.

### Tests

```bash
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
```

### Audit

Fixture failures block DSGE and MacroFinance promotion.

### Exit gate

Generic fixtures demonstrate that the BayesFilter contract solves a structural
SSM problem, not only a DSGE case.

## Phase 6: MacroFinance adapter and analytic derivative classification

### Motivation

MacroFinance has valuable state-space and derivative-provider work, but its
economic logic should remain in MacroFinance.  BayesFilter should consume
contracts and evidence, not copy model semantics.

### Implementation instructions

1. Audit MacroFinance state-space and derivative code.
2. Classify:
   - exact collapsed LGSSM;
   - affine or lag-stack structure;
   - nonlinear structural cases;
   - derivative-provider availability;
   - final-provider blocker/readiness metadata.
3. Use ResearchAssistant only for derivative/HMC literature claims, not for
   provider-owned data claims.
4. Use MathDevMCP or finite-difference tests before migrating any analytic
   derivative formula.

### Tests

```bash
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
```

### Audit

- Do not migrate derivative formulas without source, notation, and numerical
  checks.
- Do not force nonlinear structural filtering onto exact affine LGSSM models.

### Exit gate

MacroFinance adapter readiness is classified without changing MacroFinance
model semantics.

## Phase 7: DSGE adapter integration

### Motivation

The DSGE repo exposes BayesFilter metadata and fail-closed guards.  BayesFilter
should consume that metadata without owning DSGE economics.

### Implementation instructions

1. Consume:
   - `bayesfilter_state_names`;
   - `bayesfilter_stochastic_indices`;
   - `bayesfilter_deterministic_indices`;
   - `bayesfilter_innovation_dim`;
   - `bayesfilter_deterministic_completion`.
2. Keep SmallNK all-stochastic.
3. Keep Rotemberg and SGU mixed structural but not structurally promoted until
   residual tests pass.
4. Keep EZ fail-closed until timing audit.
5. Preserve legacy full-state SVD labels for mixed DSGE smoke probes.

### Tests

BayesFilter:

```bash
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

DSGE client:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

### Audit

Passing adapter metadata is adapter readiness only, not nonlinear filtering
correctness.

### Exit gate

SmallNK, Rotemberg, and SGU metadata gates pass; EZ remains explicitly blocked.

## Phase 8: model-specific DSGE completion evidence

### Motivation

Rotemberg and SGU need model-specific residual evidence before nonlinear
structural filtering is safe.  EZ needs timing classification before metadata
exposure.

### Implementation instructions

1. Rotemberg:
   - test `dy_next = y_next - y_current` pointwise;
   - add second-order/pruned residual tests before promotion.
2. SGU:
   - derive residuals for `d,k,r,riskprem`;
   - test completion over sigma-point grids and near-boundary parameters.
3. EZ:
   - audit timing and state partition;
   - expose metadata only after classification.
4. Use MathDevMCP for any labeled derivation that exists in local TeX.
5. If the model equations are only in code, create a written derivation note
   and executable residual tests before promotion.

### Tests

Focused tests belong primarily in `/home/chakwong/python` because the model
semantics are client-owned.  BayesFilter may add adapter tests that assert
fail-closed or metadata-consumption behavior.

### Audit

- Do not infer roles solely from `eta`.
- Do not promote first-order bridges to second-order/pruned correctness.

### Exit gate

Each DSGE model is either residual-tested for the intended approximation order
or explicitly blocked.

## Phase 9: derivative and Hessian safety gate

### Motivation

HMC needs gradients.  SVD/eigen derivatives are unsafe near repeated singular
values/eigenvalues unless validity regions and stress behavior are tested.

### Implementation instructions

1. Split obligations into:
   - transition;
   - deterministic completion;
   - observation;
   - covariance prediction;
   - factorization;
   - likelihood update;
   - parameter transform.
2. Use ResearchAssistant to support literature claims about matrix
   backpropagation, SVD/eigen derivative assumptions, and HMC requirements.
3. Use MathDevMCP to route labeled derivation obligations and record when human
   review is required.
4. Add finite-difference, JVP/VJP, Hessian symmetry, and spectral-gap stress
   tests.
5. Record derivative status in result metadata.

### Tests

```bash
pytest -q tests/test_derivative_validation_smoke.py
```

Additional backend-specific tests must be added before promotion.

### Audit

If eigen/SVD gradients are unstable near small gaps, do not promote to HMC.
Consider custom derivatives, different factors, or stricter validity labels.

### Exit gate

Gradient and Hessian tests pass for the specific backend/model class being
promoted.

## Phase 10: JIT and static-shape production gate

### Motivation

Mathematical correctness is not production readiness.  HMC targets need
compiled value/gradient parity and static-shape behavior.

### Implementation instructions

1. Identify static and dynamic dimensions.
2. Add eager-versus-compiled value and gradient parity tests.
3. Assert no Python callbacks in production HMC target paths.
4. Record compile time and steady-state runtime.
5. Keep eager-only backends labeled as eager-only.

### Tests

```bash
pytest -q tests/test_backend_readiness.py
```

If this file does not exist yet, Phase 10 must create the backend-specific
tests before promotion.

### Audit

- Do not treat TFP NUTS as a default fix.
- Do not call eager reference code HMC-ready.

### Exit gate

Compiled target value and gradient match eager/reference results.

## Phase 11: HMC validation ladder

### Motivation

HMC should be a late-stage validation ladder, not the first debugging tool.

### Implementation instructions

Run in order:

1. LGSSM recovery;
2. nonlinear toy SSM;
3. generic structural AR/lag model;
4. SmallNK;
5. Rotemberg after second-order completion evidence;
6. SGU after equilibrium residual evidence;
7. EZ after timing audit;
8. NAWM-scale models only after smaller DSGE gates.

Record:

- finite target and gradient;
- acceptance rate;
- divergences;
- split R-hat;
- ESS;
- posterior recovery;
- compile and runtime.

### Audit

- Label smoke tests as smoke tests.
- No convergence claim without multi-chain diagnostics.

### Exit gate

Each model moves to the next ladder stage only after diagnostics pass.

## Phase 12: documentation, provenance, and release gate

### Motivation

The release must make exact, structural nonlinear, and approximation-only
claims hard to confuse.

### Implementation instructions

1. Update BayesFilter monograph chapters and source map only for committed or
   intentionally staged artifacts.
2. Update reset memos after every phase.
3. Register plan, audit, and result artifacts.
4. Use exact labels:
   - exact LGSSM;
   - structural nonlinear;
   - labeled approximation;
   - finite smoke;
   - converged posterior.
5. Do not commit generated PDFs unless explicitly requested.
6. Use ResearchAssistant claim-support audit for literature-backed release
   claims.
7. Use MathDevMCP release/document-code checks for mathematical and
   implementation claims.

### Tests

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ..
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
rg -n "converged|production-ready|structurally fixed|HMC-ready" docs bayesfilter tests
```

### Exit gate

Docs, source map, tests, reset memo, ResearchAssistant support status, and
MathDevMCP derivation/doc-code status agree with the evidence.

## Immediate next action

The next execution pass should run Preflight and Phase 1 only:

1. create the source/derivation audit note;
2. build the source-support table;
3. run MathDevMCP label extraction and obligation routing;
4. write explicit human derivations and acceptance-test obligations for any
   `human_review_required` claims;
5. update the reset memo with whether Phase 2 is justified.

Do not edit backend code until Phase 1 passes.
