# SVX-ZC Capacity Self-Convergence Tuning Result

Date: 2026-08-01

Plan:
`docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-execution-plan-2026-08-01.md`

Active derived artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/reinterpretation-attempt01/result.json`

Source value artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/attempt03/result.json`

## Outcome

The user's intended rule is now implemented: with a three-significant-digit
request, the first two significant digits must agree; the third digit may
change. The active status is:

```text
CENTER_NOMINATION_ONLY
```

The reinterpretation was read-only. It made zero filter calls, zero score
calls, and zero dense-reference calls. It reused the 17 valid value cells from
the preserved source artifact.

The smallest center-point nominee is:

```text
degree = 10
rank = 2
quadrature order = 25
```

Two larger center candidates also pass the same prefix rule: `(degree=10,
rank=4)` and `(degree=10, rank=6)`. They are not selected because the tuner
minimizes the stored coefficient count after the hard checks.

This is a nomination, not final promotion. The planned quadrature confirmation
and four untouched validation points have not been run.

## Corrected value rule

For a likelihood value, extract the sign, decimal exponent, and first two
significant digits. A comparison passes when these prefixes agree. Therefore:

```text
-20.4475727 -> prefix (-, 20, 20)
-20.1809302 -> prefix (-, 20, 20)
```

The third digit changes from `4` to `1`, but that is allowed. The raw delta is
`0.2666426`; it is descriptive metadata and is not a veto under the requested
rule.

By contrast:

```text
-20.447 -> prefix (-, 20, 20)
-21.180 -> prefix (-, 21, 21)
```

would fail because the second significant digit changes.

## Calibration values

| Degree | Rank 2 | Rank 4 | Rank 6 | Rank 8 |
|---:|---:|---:|---:|---:|
| 4 | `-24.373244735` | `-24.349383973` | N/A | N/A |
| 6 | `-22.185146908` | `-22.157272779` | `-22.157106666` | N/A |
| 8 | `-21.034549654` | `-21.013697548` | `-21.013462231` | `-21.014142323` |
| 10 | `-20.463845484` | `-20.447572749` | `-20.447366129` | `-20.447364033` |
| 12 | `-20.196309526` | `-20.180930157` | `-20.180712678` | `-20.180709028` |

At degree 10, the rank refinements are prefix-stable. At degree 10 to 12,
the first two significant digits remain `20` for all ranks, so those degree
comparisons are also prefix-stable under the intended rule.

## Numerical validity

All 17 source cells passed:

- finite total likelihood and increments;
- marginal mass error `<=1e-10`;
- fixed order-9 density finiteness and nonnegativity checks;
- solved-system condition number `<=1e10`;
- common frozen-scope identity across capacities.

Fit residuals remain explanatory capacity diagnostics. The degree-12 rank-4
fit residual is about `0.0231`; it is not a promotion criterion and does not
contradict the value-prefix nomination.

## Decision table

| Decision | Criterion | Veto status | Next action | Nonclaim |
|---|---|---|---|---|
| Nominate `(10,2)` for follow-up value checks | First two of three significant digits agree with degree and rank neighbors at the center | No numerical veto; quadrature and validation deferred | Run order confirmation and four frozen validation points for `(10,2)`, `(12,2)`, and `(10,4)` | Not final capacity promotion, exact likelihood accuracy, score accuracy, HMC readiness, posterior correctness, GPU/XLA readiness, production readiness, or cross-scope transfer |

## Inference-status table

| Status | Verdict |
|---|---|
| Hard numerical screen | Passed for all 17 preserved cells. |
| Statistical ranking | Not applicable; this is deterministic frozen-input capacity comparison. |
| Descriptive differences | Raw likelihood deltas, fit residuals, condition numbers, runtime, and increment warnings. |
| Default readiness | Not established; only a center nomination exists in CPU/non-XLA scope. |
| Evidence needed next | Quadrature refinement and untouched validation-point checks. |

## Provenance and budget

The source grid was run three times during implementation repair, with identical
values each time. The active reinterpretation did not rerun the filter, so it
consumed zero new scientific cells. The historical source attempts and their
cell-count overrun remain documented in their original artifacts; they are not
silently reclassified as budget-compliant.

## Nonclaims

This result does not claim exact nonlinear filtering accuracy, score accuracy,
HMC validity or convergence, posterior agreement, statistical superiority,
GPU/XLA readiness, production readiness, or transfer to another model, dataset,
horizon, neighborhood, backend, or dtype.
