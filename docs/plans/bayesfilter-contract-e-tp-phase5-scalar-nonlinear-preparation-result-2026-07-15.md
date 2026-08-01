# Contract E--TP Phase 5 Scalar Nonlinear Preparation Result

metadata_date: 2026-07-15
status: PARTIAL_PASS_ACTUAL_KSC_GENERALIZED_PROGRESSIVE_REPAIR_NEGATIVE
entry_plan: `docs/plans/bayesfilter-contract-e-tp-phase8b-nonoracle-continuation-and-nonlinear-chart-handoff-2026-07-15.md`

## Scientific Question

Can one fixed nonlinear target-continuation feature, together with mass, state,
and squared state, preserve both the value and total score of the recursive
Contract E--TP scalar for actual SV, KSC-SV, and generalized SV at `T=10`?

The continuation feature is computed from the target transition and target
likelihood.  It is not computed from the Gaussianized LEDH proposal surface.

## Implementation And Mathematical Checks

The new scalar route is
`bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py`.  It implements a
fixed-grid backward target-continuation operator, the established affine LEDH
proposal surfaces, target importance corrections, fixed positive four-point
Contract E--TP charts, total autodiff of the same executed scalar, and an
independent dense target filter.

The primitive and adapter suites pass (`24 passed` before recursive execution;
the final focused scalar suite is `10 passed`).  Direct one-step continuation
equals direct quadrature, and its total derivative agrees with central finite
difference.  Every controlling recursive scalar below passes the owner FD-only
screen and every prepared chart is positive with residual at roundoff scale.

## Root-Cause Repair

The first generalized-SV implementation was wrong relative to the established
row time ordering.  The row stores stationary **previous** states and applies a
transition before every recorded observation, including observation zero.
The new path initially treated its stationary rule as the state corrected by
observation zero.  Continuous stationarity can conceal the value mismatch, but
the two finite quadrature programs and their derivatives are different.

The repair adds an immutable `transition_before_first_observation` model field
and applies the same convention in both Contract E--TP and the dense reference.
All pre-repair generalized artifacts are historical diagnostics only.

## Results

All relative score differences below are descriptive comparisons with the
finest dense target reference.  No cross-method equivalence margin is available.

| Row / rung | Value difference | Worst score relative difference | Same-scalar FD | Chart | Classification |
| --- | ---: | ---: | --- | --- | --- |
| actual SV, `T=1`, order 25 | `-3.34e-8` | `4.87e-6` | pass | no reset | primitive pass |
| KSC-SV, `T=1`, order 25 | `4.05e-5` | `1.79e-3` | pass | no reset | primitive pass |
| generalized SV, repaired `T=1`, order 25 | `1.62e-9` | `1.43e-5` | pass | no reset | primitive pass |
| actual SV, `T=10`, look-ahead 1 | `1.78e-2` | `2.79e-1` | pass | pass | feature rejected |
| actual SV, `T=10`, look-ahead 4 | `2.25e-3` | `1.23e-2` | pass | pass | viable diagnostic |
| actual SV, `T=10`, look-ahead 8 | `-3.13e-5` | `2.89e-4` | pass | pass | viable diagnostic |
| KSC-SV, `T=10`, look-ahead 1 | `1.66e-2` | `6.93e-2` | pass | pass | feature rejected |
| KSC-SV, `T=10`, look-ahead 4 | `3.32e-3` | `2.40e-2` | pass | pass | viable diagnostic |
| KSC-SV, `T=10`, look-ahead 8 | `1.54e-5` | `4.01e-4` | pass | pass | viable diagnostic |
| actual SV, `T=10`, look-ahead 8, order 41 | `-2.61e-5` | `5.21e-4` | pass | pass | adjacent refinement viable |
| KSC-SV, `T=10`, look-ahead 8, order 41 | `-3.11e-5` | `3.07e-4` | pass | pass | adjacent refinement viable |
| generalized SV, repaired `T=10`, full prefix 9, order 25 | `1.04e-2` | `3.23e-2` | pass | pass | viable full-prefix diagnostic only |
| generalized SV, repaired `T=10`, look-ahead 8, order 25 | `1.51e-2` | `8.40e-1` | pass | pass | bounded feature rejected |
| generalized SV, repaired `T=10`, full prefix 9, order 41 | `1.74e-2` | `5.46e-1` | pass | pass | refinement instability |
| actual SV, `T=100`, look-ahead 8, order 25 | `9.72e-3` | `8.94e-2` | pass | pass | viable but score drift requires repair |
| KSC-SV, `T=100`, look-ahead 8, order 25 | `-1.48e-3` | `7.78e-3` | pass | pass | viable prefix diagnostic |
| actual SV, `T=100`, look-ahead 8, order 41 | `6.37e-3` | `6.17e-2` | pass | pass | improved but repair remains |
| KSC-SV, `T=100`, look-ahead 8, order 41 | `1.86e-5` | `1.16e-2` | pass | pass | adjacent refinement viable |
| actual SV, `T=100`, look-ahead 16, order 25 | `8.10e-5` | `6.88e-4` | pass | pass | repaired viable prefix diagnostic |

Controlling actual/KSC artifacts are:

- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_actual_sv_t10_order25_lookahead8_result_20260715.json`;
- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_ksc_sv_t10_order25_lookahead8_result_20260715.json`.

Controlling repaired generalized artifacts are:

- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order25_lookahead9_timeorderfix_result_20260715.json`;
- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order25_lookahead8_timeorderfix_result_20260715.json`;
- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order41_lookahead9_timeorderfix_result_20260715.json`.

## Mathematical Interpretation

The actual/KSC ladder supports the mechanism that missing continuation
information, rather than derivative wiring, caused the one-step score error.
It does not prove an eight-step default or full-horizon accuracy.

Generalized SV is different.  Its high persistence makes the ninth future
observation consequential for the `gamma` derivative at this prefix.  Moreover,
changing teacher order changes the selected square chart and produces a much
larger score gap.  The favorable order-25 full-prefix result therefore cannot
be selected as a refined winner.  A stable generalized route needs either an
overcomplete positive chart, multiple continuation features, or a preparation-
region design whose active set is stable under refinement.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| retain KSC look-ahead 8 as a hypothesis | `T=100` worst score gaps `0.00778/0.0116` at orders 25/41 | FD/chart/reference checks pass | no margin and one center | later full-row feasibility/refinement | no equivalence/default claim |
| retain actual-SV look-ahead 16 as a hypothesis | `T=100` worst score gap falls to `0.000688` | engineering gates pass | no adjacent order-41 check at window 16, one center | later refinement/GPU feasibility | no full-horizon/default claim |
| reject generalized look-ahead 8 | `gamma` score gap `0.84` | no engineering veto | feature/refinement instability | repair chart/features at `T=10` | not rejection of Contract E--TP direction |
| retain generalized full-prefix result as diagnostic | worst score gap `0.0323` at order 25 | order-41 sensitivity vetoes promotion | active-chart instability | overcomplete/multi-feature repair | no transferable window claim |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | candidate scalars, FD, charts, and dense references valid |
| Statistically supported ranking | none |
| Descriptive-only differences | all target-reference differences in this note |
| Default readiness | false |
| Next evidence | generalized chart/feature repair, actual/KSC refinement and `T=100` prefixes |

## Post-Run Red Team

The strongest alternative explanation for actual/KSC agreement is favorable
center-only active-chart selection at order 25.  An adjacent teacher-order or
held-out parameter failure would overturn promotion.  For generalized SV, the
weakest evidence is precisely the non-stable active square chart; selecting the
best observed order would be scientifically invalid.

## Generalized-SV Progressive-Feature Repair

The follow-up repair tested multiple bounded target-continuation marks with
requested horizons `(1,4,9)` and fixed KKT charts. All accepted runtime results
pass same-scalar FD, strict positivity, full row rank, and exact feature
matching. The following center comparisons are descriptive only.

| Teacher/chart design | Value gap | Worst score gap | Classification |
| --- | ---: | ---: | --- |
| order 25, progressive square-equivalent | `0.00561` | `24.0%` | added marks insufficient |
| order 25, full-prefix only, max-min KKT | `0.0184` | `82.7%` | overcomplete reference rejected |
| order 25, progressive max-min KKT | `0.00787` | `20.4%` | combined design rejected |
| order 25, progressive analytic-center KKT, 8 quantiles | `0.00228` | `13.7%` | engineering pass, not stable enough |
| order 41, progressive analytic-center KKT, 8 quantiles | `0.00250` | `9.53%` | improved but refinement-sensitive |
| order 41, progressive analytic-center KKT, 12 quantiles | `0.00519` | `32.9%` | capacity refinement worsens |

The per-time order-41, eight-quantile diagnostic localizes the first sustained
`gamma` score drift to increments 3--5, followed by partial cancellation. The
12-quantile arm produces larger alternating errors from increment 3 onward.
This is not a terminal-step artifact, finite-difference failure, or missing
feature-equality term. It is evidence that these finitely many continuation
marks and this chart family do not determine the downstream recursive
functionals needed by generalized SV.

Controlling artifacts include:

- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order41_progressive1_4_9_basis_quantile8_analytic_fill_localized_result_20260715.json`;
- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_generalized_sv_t10_order41_progressive1_4_9_basis_quantile12_analytic_fill_localized_result_20260715.json`.

Decision: close this generalized-SV feature family as a row-specific negative
result. Do not run `T=100`. This does not invalidate the scalar core, actual SV,
KSC-SV, or Contract E--TP as a research direction. A future generalized-SV
repair must introduce a materially different state summary, such as a
distributional basis with an independently justified approximation guarantee,
rather than another ad hoc anchor-count increase.
