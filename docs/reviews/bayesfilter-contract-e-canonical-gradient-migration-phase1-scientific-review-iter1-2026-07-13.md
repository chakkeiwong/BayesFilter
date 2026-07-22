# Contract E Canonical Migration Phase 1 Scientific Review, Iteration 1

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute after the platform blocked Claude
repository disclosure. Read-only scope was the Phase 1 normative specification
and numerical/statistical design freeze.

## Verdict Before Repair

`VERDICT: REVISE`

## Material Findings

1. Floating-point multipliers `8`, `32`, `64`, and `256` were not derived from
   the executed reductions/kernels and produced extremely loose nominal bounds
   at `N=10000`.
2. Strict positive row mass did not control quotient conditioning.
3. The raw-covariance gate confused deterministic ridge bias with roundoff, used
   `lambda*d` rather than `lambda*sqrt(d)` as the Frobenius ridge scale, and did
   not reproducibly freeze ridge adequacy.
4. The FD screen omitted an executable relative-error definition, `p`, and
   callable-level error bounds, and its constants had no confidence-interval
   interpretation.
5. The value-equivalence table incorrectly classified a boundary-crossing
   interval as non-equivalent instead of inconclusive.
6. Five-seed Bonferroni Student-t coverage omitted its iid-normal assumption and
   could not support an unconditional exact 95% claim.
7. The gradient sign/order-one veto language was not operationally defined.
8. The normative pullback omitted nontrivial primitive adjoints and contradicted
   itself about returning the prepared-ridge adjoint.

## Repair Applied

- Reclassified every unjustified adequacy multiplier as an explicit
  pre-promotion blocker; retained only finite-value, strict-positivity, fixed
  identity, and Cholesky-chart hard validity vetoes.
- Separated exact-identity backward error from scientific algorithm adequacy.
- Corrected the raw-ridge Frobenius scale and made ridge/domain adequacy a Phase
  3 blocker.
- Defined `p`, the relative-error denominator and inequality, and labeled
  `0.05*sqrt(p)` `heuristic_only`, not a confidence interval.
- Made the FD ladder conditional on predeclared callable-specific endpoint and
  score error bounds, which remain a Phase 5 blocker.
- Corrected equivalence decisions; disclosed the Student-t assumption and
  blocked Phase 8 equivalence promotion pending a justified model or stronger
  pre-result design.
- Demoted undefined qualitative gradient red flags until the near-zero margin is
  frozen.
- Added the affine, triangular-solve, Cholesky, uniform-moment, injection,
  weighted-moment, row-quotient, normalization, and ridge-adjoint formulas.
- Extended the diagnostic helper and tests to compare the ridge adjoint, every
  independent input against a directional-FD ladder, and the probability-to-
  logit pullback.

The repaired artifacts require a new bounded review; this record does not claim
convergence.

## Iteration 2 Follow-Up

The fresh iteration-2 reviewer returned `VERDICT: REVISE` on two remaining
details:

1. Bonferroni provides simultaneous coverage of at least, not exactly, 95% under
   the stated marginal Student-t assumptions.
2. The FD ladder did not define the coarsest estimate, representability-adjusted
   step ratios, or even-cardinality selection tie.

The design now uses only symmetric representable endpoint pairs, computes the
Richardson denominator from the actual adjacent-step ratio, makes the coarsest
estimate ineligible, enumerates consecutive common-intersection runs, and fixes
both run and midpoint tie rules. The coverage statement now says at least 95%.
An iteration-3 review is required.
