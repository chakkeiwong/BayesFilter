# Plan: Capacity Self-Convergence Tuning From Likelihood Values

Date: 2026-08-01

## Decision requested

Define a model-independent tuning procedure for filtering approximations when
an exact or dense-reference likelihood is unavailable. The procedure treats
polynomial degree, TT rank, and quadrature order as explicit capacity/design
parameters and uses adjacent-capacity likelihood self-convergence as the
tuning signal.

This plan applies first to the SVX-ZC fixed-branch route, but the protocol is
intended to be reusable by any deterministic filtering route that can emit a
scalar likelihood and its per-time likelihood increments.

## Research question

For a fixed model, data record, coordinate map, filtering route, and numerical
branch policy, does an explicitly declared capacity ladder produce a likelihood
that is stable to the requested three-significant-digit precision? If so, what
is the smallest capacity that reaches that stability without violating
numerical invariants?

This is a capacity-resolution question. It is not a claim that the selected
finite-capacity likelihood equals an unavailable exact likelihood.

## Target and scope

The operational target is the scalar likelihood returned by the selected
filter configuration:

\[
  \widehat{\ell}_{c}(\theta;y_{1:T}),
  \qquad
  c=(\text{degree},\text{rank},\text{quadrature order},\ldots).
\]

For HMC-facing use, the corresponding scalar remains the complete transformed
target built from this filter value, prior, and parameter-transform Jacobian.
This plan tunes only the filtering value. It does not tune or certify score
accuracy, HMC convergence, or posterior correctness.

The capacity identity must include at least:

- polynomial degree/basis identity;
- TT rank tuple;
- fit and propagation quadrature orders;
- coordinate-map bounds and measure convention;
- ridge, density floor, normalizer floor, and branch policy;
- initialization rule and deterministic seed policy;
- model, observations, horizon, parameter point, dtype, and backend.

Changing any of these creates a new tuning scope and invalidates direct
adjacent-capacity comparisons unless the change is explicitly classified as a
separate ladder.

## Evidence contract

### Comparator

The comparator is the immediately lower capacity under the same frozen route
and data. No exact likelihood is required for this phase. A dense or analytic
reference may be recorded when available, but it is explanatory only and is
not required for promotion.

### Primary criterion

For each adjacent pair `(c_low, c_high)`, compute

\[
  \Delta_v(\theta)=
  \left|\widehat{\ell}_{c_{high}}(\theta)-
  \widehat{\ell}_{c_{low}}(\theta)\right|.
\]

Use a leading-significant-digit prefix criterion. With the requested three
significant digits, the first two significant digits must agree; the third
significant digit is allowed to change. For example, `-20.447` and `-20.181`
share the first two significant digits `20` and therefore pass. Let

```text
scale = max(abs(value_low), abs(value_high))
place = 10 ** (floor(log10(scale)) - 2)
```

for `scale > 0`; use the declared near-zero fallback place `1e-3` when both
values are zero or numerically near zero. The pair is `value_stable_3sf` when

```text
absolute_delta_v <= place
```

The decision is `first_two_significant_digits_equal`, not an absolute delta
bound. The artifact must report the raw values, normalized delta, three-digit
display values, and the extracted sign/exponent/leading-digit prefixes. The
absolute place value remains descriptive metadata only. This avoids treating
the third digit as a veto when the declared rule explicitly allows it to
change.

Because accumulated likelihoods can hide cancellation, repeat the comparison
for every per-time likelihood increment. Increment stability is an explanatory
diagnostic and a warning trigger, not a promotion veto in this value-only
phase. The requested target is the total likelihood; a later plan may promote
increment stability only after justifying why the downstream use requires it.

### Promotion rule

A capacity is `self_converged_value` only if:

- it passes all mathematical and numerical invariants;
- its adjacent higher-capacity comparison is stable at three significant
  digits; and
- the same result is observed at the declared validation parameter points,
  not only at the center point.

Select the smallest capacity meeting the rule. If the largest tested capacity
does not stabilize, mark the scope `under_resolved` and do not silently select
it as an adequate default.

The rule is deliberately about value capacity. It does not promote score
accuracy; score testing remains a later phase.

## Capacity ladder

The first SVX-ZC ladder should hold all non-capacity choices fixed and vary one
capacity axis at a time, then test a combined candidate:

| Ladder | Fixed fields | Varied fields |
|---|---|---|
| Degree | rank, quadrature, ridge, map, seed | degree `d0 < d1 < d2` |
| Rank | degree, quadrature, ridge, map, seed | rank tuples `r0 < r1 < r2` |
| Quadrature | degree, rank, all fit settings | fit/propagation order |
| Combined | selected degree/rank/order candidates | next adjacent combined capacity |

The exact values must be frozen in the execution note before running. The
existing degree-8/rank-(1,2,4,6) run is historical context, not a default or a
prescribed ladder. The ladder must include at least one capacity above the
current rank-6/degree-8 candidate if resource budgets allow it; otherwise the
artifact must state that the upper-resolution boundary was not tested.

## Data and evaluation points

Use a fixed calibration point for the initial ladder and an untouched set of
validation points in the same declared parameter neighborhood. Keep model,
observations, horizon, coordinate map, branch seeds, and initialization policy
identical across capacities. Do not select capacity on the final HMC claim
data.

For each point, record:

- total likelihood;
- per-time likelihood increments;
- normalizers and mass closure;
- finite/nonnegative density checks;
- fit residual and holdout residual as explanatory fit diagnostics;
- condition numbers and branch identity hashes;
- wall time and resource status.

## Numerical vetoes and diagnostics

Hard vetoes:

- non-finite likelihood, increment, density, normalizer, or retained object;
- invalid mass/normalization closure;
- negative density beyond the declared arithmetic tolerance;
- condition-number or storage-budget failure;
- unauthorized branch drift across an adjacent comparison: branch hashes are
  expected to differ when capacity changes, but the structured manifest diff
  may contain only the predeclared capacity fields and their derived budgets;
- failed serialization or incomplete artifact;

Explanatory diagnostics:

- square-root density fit residual;
- rank saturation and basis projection diagnostics;
- initializer identity and UKF warm-start moments;
- runtime and memory;
- any available dense/analytic reference gap.

The old `fit residual <= 1e-8` gate must not be copied into this protocol as a
scientific promotion criterion. It remains useful for diagnosing basis
truncation, rank saturation, and failed fitting. A fit residual that is finite
but materially nonzero must be reported as capacity information, not relabeled
as a numerical failure solely because it misses machine-scale accuracy.

## Implementation sequence

1. Add a small model-independent value self-convergence utility that accepts
   scalar values and optional per-time increments, compares the leading
   significant-digit prefix, and returns structured audit metadata.
2. Add tests for positive, negative, near-zero, and cancellation-prone
likelihood values. Near zero must use the declared fallback place and the
normalized diagnostic denominator `max(scale, 1e-300)`; it must not produce
NaN or an accidental pass from division by zero.
3. Extend the SVX-ZC benchmark runner (or add a separate tuning runner) to
   execute the frozen degree/rank/quadrature ladders and emit one record per
   capacity and adjacent comparison.
4. Preserve the existing route, coordinate map, UKF initialization identity,
   branch hashes, and fixed-seed policy. Do not alter the prior terminal
   admission artifacts.
5. Add validation-point evaluation and require capacity selection to be based
   on all declared points.
6. Write a result note that separates `self_converged_value`, `under_resolved`,
   `numerically_invalid`, and `fit_failed` outcomes.
7. Only after the value ladder is reviewed should a later plan add score-delta
   tuning or HMC-specific capacity checks.

## Budget and stop conditions

- Use a bounded number of capacity cells and validation points declared before
  launch.
- Use a fresh versioned output root; never overwrite attempts 01-07 or any
  earlier admission artifact.
- Stop a ladder branch after a hard numerical veto, a resource-budget veto, or
  successful three-significant-digit stability at the smallest candidate.
- Do not increase degree/rank after observing a failure unless that candidate
  was already part of the predeclared ladder and budget.
- Do not spend compute on empirical score deltas in this phase.

## Skeptical audit before execution

The plan passes the pre-execution audit with these explicit limitations:

- It does not use a proxy fit residual as the likelihood criterion.
- It does not require an unavailable exact likelihood reference.
- It reports final-sum cancellation risk through per-time increments without
  silently turning that explanatory diagnostic into a promotion veto.
- It freezes route, data, branches, and seeds so adjacent differences mean
  capacity changes rather than implementation drift.
- It distinguishes numerical invalidity from an under-resolved but finite
  approximation.
- It does not claim score, HMC, posterior, production, or scientific validity.
- The three-significant-digit rule is made operational by requiring equality of
  the first two significant digits and allowing the third to change. If project
  policy later chooses a different prefix or rounding convention, that is a new
  reviewed tuning scope rather than an in-place reinterpretation.

## Planned artifacts

- `docs/plans/artifacts/bayesfilter-capacity-self-convergence-20260801/<attempt>/run_manifest.json`
- `docs/plans/artifacts/bayesfilter-capacity-self-convergence-20260801/<attempt>/capacity_*.json`
- `docs/plans/artifacts/bayesfilter-capacity-self-convergence-20260801/<attempt>/result.json`
- `docs/plans/bayesfilter-capacity-self-convergence-likelihood-tuning-result-2026-08-01.md`

## Nonclaims

This plan will not establish that the selected approximation equals the exact
nonlinear filtering likelihood, that the score is accurate, that HMC targets
the exact posterior, that the model is scientifically validated, or that the
selected degree/rank is transferable to another model, dataset, horizon,
parameter neighborhood, backend, or dtype.
