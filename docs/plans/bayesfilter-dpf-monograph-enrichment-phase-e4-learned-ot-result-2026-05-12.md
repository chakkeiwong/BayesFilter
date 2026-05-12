# Phase E4 result: learned/amortized OT enrichment

## Date

2026-05-12

## Purpose

This note records the learned/amortized OT enrichment phase of the DPF monograph
round.

## Plan tightening before execution

Pretending to be another developer, the E4 subplan was audited before
execution.  The audit concluded that the plan should more explicitly prevent the
chapter from overreading residual metrics as posterior guarantees.

That tightening was added to the E4 plan in the form of an explicit guardrail
table distinguishing:
- what a residual quantity measures directly,
- what it does not by itself prove,
- and what further evidence would still be needed.

Plan audit note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-plan-audit-2026-05-12.md`

## Execution

### 1. Target chapter deepened

Target chapter:
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

Direct enrichment performed:
- added a source-spine paragraph tying the teacher map to EOT/Sinkhorn sources
  and the learned map to permutation-equivariant set operators;
- added an explicit approximation-hierarchy table;
- added a permutation-equivariance equation for learned set maps;
- strengthened the residual/target-shift interpretation section so that map-level
  residuals are no longer left too close to posterior-level conclusions;
- added the residual and target-shift hierarchy table;
- added explicit guardrail language that a map-level residual or a transport-cost
  discrepancy does not by itself justify a quantitative posterior claim;
- added the residual interpretation guardrail table required by the E4 audit;
- strengthened the training-distribution section so that out-of-distribution
  generalization is more clearly separated from in-distribution residual quality.
- added a learned-OT implementation issue map covering permutation sensitivity,
  state-dimension transfer, weight-simplex extrapolation, regularization
  mismatch, and teacher drift.

### 2. Main enrichment gains

The chapter now distinguishes more clearly among:
- the learned map residual relative to the OT teacher map;
- the induced distribution-level discrepancy;
- and any downstream posterior-level consequences.

That makes the chapter much less likely to drift into the easy but misleading
reading that a small training residual automatically implies a negligible target
change.

The chapter also now contains the required E4 artifacts that were previously
only described in planning language:
- approximation hierarchy table;
- learned-OT implementation issue map;
- residual and target-shift hierarchy table;
- residual interpretation guardrail table.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  after the E4 text/table additions.
- `latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex` was run to
  force BibTeX/LaTeX convergence after adding new set-operator citations.
- `git diff --check` passed.
- focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors after the final
  E4 build.
- local text audit confirmed stronger coverage of residual interpretation,
  training-distribution dependence, implementation diagnostics, and explicit
  non-equivalence claims.

## Audit

### Primary criterion

Satisfied.

The chapter now makes it much harder to read learned OT as a free exact
acceleration or to treat residual metrics as if they immediately implied
posterior correctness.

### Veto diagnostics

- **teacher-versus-learned distinction too compressed**: cleared.
- **residuals treated only as empirical error bars**: cleared.
- **training-distribution dependence too casual**: cleared.

## Interpretation

This phase gives the learned-OT chapter the missing depth needed for the
monograph standard.  The chapter now functions more as a mathematical guide to
approximation layering than as a short interpretive note.  That is exactly what
is needed before the final HMC chapter can build on it responsibly.

## Next phase justified?

Yes.

Phase E5 is now justified because the DPF-specific HMC chapter can now compare
rungs against a clearer learned-OT approximation story rather than against a
vague surrogate layer.
