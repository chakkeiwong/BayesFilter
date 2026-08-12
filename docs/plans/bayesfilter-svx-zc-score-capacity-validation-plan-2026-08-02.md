# SVX-ZC Score Capacity Validation Plan

Date: 2026-08-02

## Question

Does the score produced by the nominated SVX-ZC finite likelihood program
behave as the derivative of that same program, and does it change materially
when the nominated value capacity is changed?

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Verify score implementation consistency and characterize score sensitivity to degree, rank, and quadrature near the value nominee. |
| Mechanism under test | TensorFlow total autodiff through `scalar_adjacent_state_fixed_tt_value`, checked by fixed-branch central finite differences. |
| Expected failure mode | Non-finite score, branch incompatibility, no stable finite-difference window, or score movement that is materially larger than the value-prefix evidence suggests. |
| Promotion criterion | Every tested cell has finite score, all finite-difference rows are valid, and each parameter has a stable decreasing/roundoff finite-difference window under the existing owner policy. |
| Promotion veto | Non-finite value/score, branch hash mismatch, invalid finite-difference row, or failure of the declared finite-difference policy. |
| Continuation veto | Corrupt harness/artifact, inability to execute the frozen cells, or exhaustion of the bounded score-validation budget. A capacity-dependent score difference is not by itself a continuation veto. |
| Repair trigger | A localized runner/serialization/resource failure under the unchanged target, cells, and budget. |
| Explanatory diagnostics | Score vectors, per-parameter finite differences, score deltas and norms across capacities, likelihood deltas, fit residuals, condition numbers, wall time, and branch identities. |
| Not concluded | Exact score correctness, score convergence beyond the tested neighborhood, exact likelihood accuracy, HMC readiness, posterior correctness, production readiness, GPU/XLA readiness, or transfer to another scope. |

## Frozen scope

- Model row: `actual_sv` / SVX-ZC.
- Route: `zhao_cui_fixed_adjacent_state_squared_tt_v1`.
- Route classification: `extension_or_invention`; `docs/main.tex` remains the
  mathematical authority.
- Dataset seed: `81101`; horizon `T=10`.
- Center parameter: `[0.2533471031357997, -0.916290731874155]` in
  `[gamma_unconstrained, log_beta]` coordinates.
- Coordinate half-width: `8.0`; density tau `0.0`; TensorFlow `float64`.
- Deliberate CPU-only, non-XLA diagnostic execution with
  `CUDA_VISIBLE_DEVICES=-1`.
- UKF initialization is the repository default and is rebuilt for every cell.
- Fixed finite-difference ladder: `h = 1e-2, 3e-3, 1e-3, 3e-4`.
- Existing owner policy: componentwise relative error at most
  `0.05 * sqrt(number_of_parameters)`, plus a stable adjacent finite-difference
  window. This is a derivative-consistency policy, not a confidence interval.

## Cells and budget

The value-only tuner nominated `(degree=10, rank=2, order=25)`. Test that cell,
its selected degree and rank neighbors, and two quadrature confirmations:

```text
(10, 2, 25)  center nominee
(12, 2, 25)  higher-degree neighbor
(10, 4, 25)  higher-rank neighbor
(10, 2, 29)  quadrature confirmation
(10, 2, 33)  quadrature confirmation
```

The budget is five score cells, each with one autodiff value plus two
parameters times four central-difference steps (17 value evaluations), for a
maximum of 85 value evaluations and 5 score evaluations. A fresh versioned
output root is required; prior evidence is never overwritten.

## Evidence contract

| Item | Frozen declaration |
|---|---|
| Scientific question | Is the score the derivative of the fitted finite likelihood, and how capacity-sensitive is it locally? |
| Comparator | The same fixed-branch scalar likelihood re-evaluated at central parameter perturbations. |
| Primary pass criterion | All five cells pass the existing finite-difference policy and fixed-branch compatibility checks. |
| Hard vetoes | Non-finite value/score, branch mismatch, invalid row, failed stable window, or missing artifact. |
| Descriptive only | Cross-capacity score/value deltas, score norms, fit residuals, conditioning, and runtime. They cannot rank or promote a capacity alone. |
| Nonclaims | No exact/reference score claim and no score convergence claim outside the tested cells. |
| Artifact | `docs/plans/artifacts/bayesfilter-svx-zc-score-capacity-validation-20260802/attempt01/` with per-cell JSON, result JSON, and run manifest. |

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Reuse value nominee | Active 2026-08-01 value artifact | Value stability may not imply score stability | Compare score vectors and FD at neighbors | Baseline hypothesis |
| Five local cells | Minimal ladder that distinguishes degree/rank/order effects | Important instability may occur elsewhere | Explicit scope and nonclaim | Bounded diagnostic |
| Existing FD policy | Repository owner-directed implementation policy | Threshold can admit a derivative with shared bias | Record it as deterministic consistency evidence only | Reviewed policy |
| CPU float64 non-XLA | Prior bounded value diagnostic scope | Not evidence for default GPU/XLA execution | Manifest and result labels | Explicit exception |

## Skeptical audit and pre-mortem

The plan does not use the three-significant-digit value rule as a score veto;
that would be an invalid proxy promotion. It also does not compare score cells
with a dense/reference score, because that would answer a different target and
would not establish exactness. A run can pass while all capacities share the
same derivative bias; this is recorded as a nonclaim. A run can fail because of
branch drift or finite-difference cancellation rather than the filtering idea;
the per-row hashes and full `h` ladder distinguish those causes. A fresh output
root and fixed cell list prevent accidental overwriting or post-hoc cell
selection.

## Execution and interpretation

Run the dedicated runner under the repository environment. Mark the score
implementation `DERIVATIVE_CONSISTENT_FOR_TESTED_CELLS` only if all hard
criteria pass. Report capacity-to-capacity changes descriptively. If those
changes are large, the next action is a separate score-capacity tuning plan;
if they are small, the result still does not establish exact score convergence.
