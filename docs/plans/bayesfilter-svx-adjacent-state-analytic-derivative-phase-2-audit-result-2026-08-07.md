# SVX adjacent-state analytic derivative phase-2 audit result

Date: 2026-08-07
Status: `PHASE_2_AUDIT_INCONCLUSIVE`

## Question

Can the active `SVX-ZC` frozen-core finite value program be repaired with a
same-program analytic adjacent-state derivative backend?

## What was audited

I audited the derivation and math-to-code mapping for the active SVX-ZC score
repair program using MathDevMCP CLI.

### Tools used

- `assumptions-for` on the active target phrase
- `debug-derivation` on the derivative chain
- `audit-math-to-code` on the LSQ derivative relation

## Findings

### 1) Assumption discovery was inconclusive

`assumptions-for` did not return route-required assumptions for the target.
It produced a human-review-required result rather than a certified assumption
set. That means the target still needs typed formalization or a more specific
route rule before the assumption set can be treated as complete.

### 2) The derivation chain was not certified at the first step

`debug-derivation` localized the first gap at the first encoded step in the
chain. The issue was not a mathematical refutation; it was a router/grammar
failure on the way the step was encoded.

### 3) Math-to-code audit reported a structural mismatch

`audit-math-to-code` reported that the equation `A dot c = dot b - dot A c`
was not structurally matched by the current code surface. The audit summary was
that the code is still missing required equation terms or has structural
conflicts.

## Interpretation

The audit does **not** show that the program is impossible.
It shows that the derivation is not yet packaged in a form that the local
math tooling can certify, and that the current code still does not implement the
required analytic derivative chain.

## Current blocker

The active SVX batch score backend is still the autodiff-based path in
`bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`, while the desired
analytic adjacent-state derivative is still absent.

## Next smallest useful action

1. formalize the target into a typed obligation or more specific derivation
   steps;
2. restate the LSQ / normalizer chain in tool-friendly form;
3. map each term to one code hook before changing the active backend.

## Non-claims

- No proof certificate was obtained.
- No backend replacement was completed.
- No tuning rerun was justified.
- No claim of HMC readiness or route equivalence is made.
