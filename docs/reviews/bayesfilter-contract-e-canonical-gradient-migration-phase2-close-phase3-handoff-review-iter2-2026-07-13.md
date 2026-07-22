# Phase 2 Close And Phase 3 Handoff Review, Iteration 2

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute.

## Findings

1. Defining-equation residuals for Cholesky and triangular solve did not provide
   an executable componentwise forward-error bound without an inverse-operator
   or conditioning argument.
2. The operation ledger undercounted composed matrix products/additions and used
   one dot-product length where input and output pairings differ.

The derivative API and scientific separation were otherwise unambiguous.

## Verdict

`VERDICT: REVISE`

## Repair

The unsupported propagated-bound gate was removed. Acceptance now uses a frozen
binary-exact certificate with bitwise equality and separately labels a
nontrivial chart as diagnostic/inconclusive absent a justified forward-error
bound. All scientific adequacy blockers remain intact.
