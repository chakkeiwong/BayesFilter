# Plan: exact linear Gaussian likelihood spine writing pass

## Date

2026-05-04

## Scope

This is the next bounded documentation pass for the BayesFilter monograph after
the structural state-space reframing commit.  The goal is to strengthen the
exact linear Gaussian likelihood spine in Part II:

- `docs/chapters/ch05_prediction_error_decomposition.tex`
- `docs/chapters/ch06_stable_linear_filtering.tex`
- `docs/chapters/ch07_missing_data_mixed_frequency.tex`
- `docs/chapters/ch08_large_scale_lgssm.tex`

This pass should not finalize Kalman score/Hessian formulas.  Those belong to
the next analytic-derivative pass.  The current pass writes the value-side
likelihood contract, stable numerical backend policy, missing-data contract,
and large-scale validation/readiness criteria.

## Motivation

The monograph now treats BayesFilter as infrastructure for Bayesian estimation
of structural state-space models.  The first technical backbone must therefore
be the exact linear Gaussian state-space likelihood.  It is the regression
oracle for:

- structural partition tests such as AR(p) lag stacks with singular process
  covariance;
- nonlinear-filter value tests in linear special cases;
- analytic Kalman derivative recursions;
- custom-gradient wrappers;
- large-scale MacroFinance validation;
- downstream DSGE and NAWM readiness gates.

Writing this spine before nonlinear filters avoids a common failure mode:
letting SVD sigma-point, particle-filter, or HMC debugging language define the
target before the exact reference target is clear.

## Sources

Use these source inputs conservatively:

- `docs/source_map.yml`
- `docs/plans/bayesfilter-monograph-writing-continuation-plan-2026-05-03.md`
- `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
- `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`
- `/home/chakwong/python/docs/chapters/ch08_kalman_filter.tex`
- `/home/chakwong/latex/CIP_monograph/chapters/ch16_kalman_filter.tex`
- `/home/chakwong/latex/CIP_monograph/chapters/ch18_mixed_frequency.tex`
- MacroFinance implementation/test references listed in `docs/source_map.yml`

Original source projects are read-only.  Work only inside
`/home/chakwong/BayesFilter/docs`.

## Non-goals

- Do not write final Kalman score/Hessian recursions.
- Do not claim that analytic derivatives are audited in this pass.
- Do not implement BayesFilter code.
- Do not claim industrial readiness for a backend family.
- Do not copy source chapters wholesale.
- Do not commit generated PDFs or LaTeX byproducts.

## Execution cycle

Every phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

The reset memo is:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
```

## Phase L0: hygiene and source inventory

Objective:

Record the starting state and identify the exact source material for this pass.

Actions:

1. Record `git status --short`.
2. Parse `docs/source_map.yml`.
3. Run `git diff --check`.
4. Inspect the four Part II target chapters.
5. Inspect the relevant Kalman/mixed-frequency source chapters enough to avoid
   unsupported formulas.

Pass gate:

- Dirty state is understood.
- Source map parses.
- No whitespace errors.
- Relevant sources are identified.

## Phase L1: prediction-error decomposition chapter

Objective:

Make Chapter 5 the exact value-side likelihood reference.

Actions:

1. Clarify timing and indexing for the linear Gaussian model.
2. State covariance assumptions, including singular process covariance and
   positive definite innovation covariance.
3. Present prediction, innovation, update, and likelihood equations.
4. Explain all-missing and masked observation contributions at a high level
   without duplicating Chapter 7.
5. State initialization policies and why they define different likelihoods.
6. State that derivative recursions are deferred.

Pass gate:

- Chapter 5 can serve as the reference value contract for later derivative and
  nonlinear chapters.
- No Hessian/score formulas are introduced.

## Phase L2: stable linear filtering chapter

Objective:

Make Chapter 6 a numerical backend policy for exact linear Gaussian
likelihoods.

Actions:

1. Distinguish covariance-form clarity from production solve/square-root
   backends.
2. Add Joseph-form or symmetry-preserving covariance update discussion without
   overclaiming.
3. Expand solve-form likelihood policy.
4. Explain Cholesky, QR, and SVD/spectral fallback roles.
5. Define diagnostics for pivots, solves, reconstruction, finite values, and
   backend deltas.
6. Keep spectral fallback separate from nonlinear SVD sigma-point HMC.

Pass gate:

- Chapter 6 explains how exact likelihood semantics are preserved across
  numerical backends.
- It does not imply that spectral/SVD fallback certifies gradient safety.

## Phase L3: missing data and mixed frequency chapter

Objective:

Make Chapter 7 a precise contract for masks, selection matrices, all-missing
steps, and compiled static-shape policy.

Actions:

1. Expand selection-form likelihood contribution.
2. Add all-missing time step behavior.
3. Add mixed-frequency aggregation/auxiliary-state policy at the contract
   level.
4. State static-shape choices for compiled implementations.
5. Keep derivative formulas deferred.

Pass gate:

- Sparse/mixed-frequency handling is described as part of the likelihood
  contract, not ad hoc preprocessing.

## Phase L4: large-scale LGSSM chapter

Objective:

Make Chapter 8 the bridge from exact textbook filtering to industrial-scale
validation.

Actions:

1. Define scale metadata: dimensions, time length, parameter dimension, masks,
   backend ladder, memory layout, and compilation policy.
2. Add validation ladder for value, gradients, Hessians, stress scenarios, and
   backend deltas.
3. State hypotheses for later MacroFinance analytic derivative consolidation.
4. Avoid claiming that current evidence solves every NAWM-scale model.

Pass gate:

- Chapter 8 clearly states what must be tested before large-scale HMC
  experiments.

## Phase L5: provenance, reset, audit, and commit

Objective:

Record results, run mechanical checks, and commit the bounded documentation
pass.

Actions:

1. Update `docs/source_map.yml` with this plan and changed chapter status.
2. Update reset memo with per-phase results.
3. Run:

   ```bash
   python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
   git diff --check
   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
   ```

4. Search touched chapters for unsupported risky claims:

   ```bash
   rg -n "guarantee|converged|production-ready|industrial-ready|certified|unbiased|must always|will always" docs/chapters/ch05_*.tex docs/chapters/ch06_*.tex docs/chapters/ch07_*.tex docs/chapters/ch08_*.tex
   ```

5. Stage only relevant docs/plans/chapters/source-map files.
6. Commit with message:

   ```text
   Strengthen BayesFilter linear Gaussian likelihood spine
   ```

Pass gate:

- Build passes.
- Source map parses.
- No unsupported high-stakes claim is introduced.
- Generated PDF/byproducts are not staged.

## Independent audit before execution

Audit stance:

This plan is sensible if it remains value-side and contract-focused.  It would
be unsafe if it tried to finalize analytic score/Hessian formulas without the
MacroFinance derivation/code audit.

Required constraints:

1. Keep all derivative formulas deferred to the next pass.
2. Treat exact Kalman likelihood as the value oracle, not as evidence that all
   derivative or HMC backends are production-ready.
3. Keep missing-data/mixed-frequency prose at the contract level unless source
   and code audits support implementation-specific claims.
4. Explicitly preserve the distinction between exact linear SVD/square-root
   factorization and nonlinear SVD sigma-point filtering.
5. Include singular `Q`/degenerate transition policy because structural
   partitions and AR(p) lag stacks require it.

Audit result:

Proceed through L0--L5 as a bounded first writing pass.  Stop only if the LaTeX
build fails structurally, `docs/source_map.yml` becomes invalid, or the prose
would require unsupported derivative or production-readiness claims.

## Next hypotheses after this pass

1. The exact linear Gaussian likelihood spine can serve as the value oracle for
   BayesFilter code extraction and AR(p) structural tests.
2. MacroFinance analytic Kalman derivative recursions can be consolidated next
   without changing the Chapter 5 value contract.
3. Singular process covariance can be handled as structural information in
   exact Kalman backends without forcing nonlinear deterministic-completion
   machinery.
4. Stable solve/square-root backend diagnostics can become the production
   checklist for large-scale LGSSM HMC experiments.
