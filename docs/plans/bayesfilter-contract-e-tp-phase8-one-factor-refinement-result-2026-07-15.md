# Contract E--TP Phase 8 One-Factor Refinement Result

metadata_date: 2026-07-15
status: PHASE8_COMPLETE_PARTIAL_REPAIR_EXTENSION_NOT_PROMOTED
plan: `docs/plans/bayesfilter-contract-e-tp-phase8-one-factor-refinement-plan-2026-07-15.md`
ledger: `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8_refinement_20260715/refinement_ledger.json`

## Result

Twenty-five CPU-hidden TensorFlow float64 arms passed finite-value, first-step
time-order, carried-mass, fit, and own-scalar FD gates. Every arm consumed an
embedded target prefix; no seed-only regeneration entered the ladder. All
comparisons remain descriptive.

| Row/factor | Evidence | Decision |
| --- | --- | --- |
| actual degree `8 -> 12` | `T=1` worst score gap `16.05% -> 0.938%`; `T=2` `13.70% -> 2.16%`; value improves at both prefixes | partial repair; retain as hypothesis only |
| actual width `8 -> 6` | `T=1` worst score gap `16.05% -> 1.96%`; width 10 worsens to `50.26%` | support truncation matters, but degree 12 is descriptively better at `T=1` |
| KSC width `8 -> 6` | `T=1` worst score gap `26.44% -> 10.99%`; `T=2` `37.53% -> 13.40%`; width 10 worsens | partial repair; retain as hypothesis only |
| quadrature `17 -> 25 -> 33` | essentially flat at fixed degree/rank | not the limiting factor at tested rungs |
| generalized rank `2 -> 3 -> 4` | value, fit, and score improve through rank 4 | promising local trend only |
| generalized rank `4 -> 6 -> 8` | value/fit continue improving but `gamma` score overshoots and worsens | nonmonotone score; no candidate advances |

Degree 16 at order 17 nearly interpolates the one-dimensional fit grid. Its
small fit residual is not independent accuracy evidence, and it was not
selected. Across generalized ranks, decreasing fit residual coexists with a
worsening score. This directly demonstrates that fit residual is explanatory
only and cannot be used as the primary selection metric.

## Scientific Verdict

The fixed-parameter adjacent-state squared-TT extension's Phase 7 error is not
a derivative-wiring defect. It is sensitive to polynomial capacity,
coordinate truncation, and TT rank. One-factor changes can materially repair
actual/KSC at short prefixes, but the repaired configurations remain
inaccurate and do not support equivalence, promotion, or longer-horizon use.

Generalized rank is nonmonotone in the primary score diagnostic. That route
closes as a capacity negative at the tested rungs. The separate generalized
Contract E--TP feature-family negative result is unchanged.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | all 25 arms pass |
| Viable hypotheses | actual degree 12; KSC width 6 |
| Statistically supported ranking | none |
| Descriptive-only differences | all value/score gaps |
| Extension/default readiness | false |
| Next action | GPU/XLA scaling only for correctness-eligible Contract E--TP rows |

## Post-Run Red Team

The strongest alternative explanation for the short-prefix improvements is
favorable center truncation or interpolation rather than a converged density
approximation. A defensible extension would require a joint support/capacity
convergence study and longer prefixes. That evidence is not needed to answer
the present causal diagnostic and is not smuggled into Phase 9.
