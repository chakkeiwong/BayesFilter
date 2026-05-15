# Phase P11 plan: mathematical derivation audit

## Date

2026-05-14

## Target scope

All load-bearing equations and theorem-like claims in the reviewer-grade DPF
block.

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, and passed P3-P10 results.
- P2 remains the source-grounding register unless superseded by a later
  reviewed source-intake artifact.
- Allowed write set: DPF reviewer-grade chapters for audit repairs and the P11
  audit artifact.  Do not touch student-baseline files.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Audit the mathematics after the chapter rewrites.  This phase is not a prose
review.  It checks whether central equations have definitions, assumptions,
derivations, source support, and implementation relevance.

## Required derivation obligations

At minimum, audit:

1. nonlinear filtering recursion;
2. SIS recursive weight formula;
3. bootstrap likelihood estimator status;
4. homotopy derivative and normalizer role;
5. continuity and log-density transport equations;
6. EDH covariance and affine coefficient derivation;
7. linear-Gaussian recovery of mean and covariance;
8. LEDH local information vector;
9. PF-PF change-of-variables density;
10. PF-PF corrected weight;
11. flow Jacobian ODE;
12. log-determinant trace identity;
13. soft-resampling mean preservation and nonlinear bias;
14. OT coupling constraints;
15. EOT first-order conditions and Sinkhorn scaling;
16. barycentric projection dimensions;
17. learned teacher/student residual hierarchy;
18. permutation-equivariance equation;
19. HMC value-gradient contract;
20. rung-by-rung target-status classification.

## Implementation instructions

1. Create a derivation-obligation register with columns:
   - obligation ID;
   - chapter;
   - equation/claim;
   - assumptions;
   - derivation location;
   - source support;
   - audit method;
   - result;
   - required repair.
2. Use MathDevMCP where the obligation is local and symbolic enough.
3. Use manual derivation checks where the obligation is conceptual or depends on
   probabilistic assumptions.
4. For each failed or uncertain obligation:
   - repair the chapter;
   - weaken the claim;
   - or record an explicit unresolved blocker.
5. Re-run targeted searches after repairs.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-{YYYY-MM-DD}.md`

## Audit rules

- A formula is not audit-ready if any symbol is undefined nearby.
- A derivation is not sufficient if it proves a different object than the claim.
- Source citation alone is not enough for a central formula unless the local text
  translates assumptions and notation.
- Conservative abstention is better than a false pass.

## Required local tests/checks

- Run overclaim searches after audit repairs.
- Re-run label and citation checks for any edited sections.
- Confirm every failed obligation has a recorded disposition.
- Record source-support status for each obligation: locally derived,
  bibliography-spine provenance, reviewed source evidence, or unresolved.
- Run the established LaTeX build if audit repairs edit chapter text.

## Veto diagnostics

The phase fails if:

- central equations remain unchecked;
- failed obligations are silently ignored;
- MathDevMCP limitations are not recorded;
- reviewer-critical claims are left as "obvious";
- repairs create notation inconsistency.
- the phase edits or stages student-baseline files;
- source citations are accepted as proof without local notation translation;
- MathDevMCP/manual/blocked status is omitted for any load-bearing obligation.

## Exit gate

Proceed to PDF/hostile-reader audit only after every load-bearing obligation has
passed, been repaired, been weakened, or been explicitly recorded as unresolved.
