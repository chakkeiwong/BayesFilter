# Contract E--TP Phase 3 Recursive LGSSM Result

metadata_date: 2026-07-15
phase: 3
status: ENGINEERING_PASS_SCIENTIFIC_CANDIDATE_FAIL_REPAIR_REQUIRED
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

The corrected-LEDH recursive LGSSM adapter is implemented in
`bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py`. At each time step it:

1. starts from carried nonuniform parent weights and a fixed innovation rule;
2. executes the same LGSSM LEDH flow as the canonical graph;
3. includes transition density, observation density, proposal density, and
   flow Jacobian in the teacher correction;
4. adds the current likelihood increment before reset;
5. projects mass, first moments, second moments, and the exact next corrected
   LEDH finite predictive contribution; and
6. carries the projected nonuniform weights into the next step.

The adapter passes target, chart, feature, carried-weight, autodiff, and
same-scalar finite-difference checks. The current feature/chart candidates do
not pass recursive Kalman value and score accuracy. This is a scientific
candidate failure and a repair trigger, not a derivative-wiring failure.

## Frozen Scope

Prior owner policy approved only a center-scoped LGSSM criterion and deferred
full-box HMC readiness. Phase 3 therefore prepared and evaluated charts only at
the physical center `(0.72,0.55,0.35,0.35,0.45)`. No off-center parameter
region is claimed.

The order-3 `T=50` preparation contains 49 fixed positive square charts. The
smallest center weight is `9.073044384197426e-05`, the largest scaled condition
number is `1356.2465764049832`, and the largest feature residual is
`4.440892098500626e-16`.

## Evidence Ladder

### Quadrature check at T=1

The corrected-LEDH finite teacher converges rapidly to Kalman as one-dimensional
Gauss--Hermite order increases:

| Order | Absolute value error | Score-vector Euclidean error |
| ---: | ---: | ---: |
| 3 | `1.32415e-02` | `2.21583e-01` |
| 5 | `1.16584e-04` | `5.69233e-03` |
| 7 | `8.54795e-06` | `3.47268e-04` |

This supports quadrature error, rather than target or score wiring, as the
order-3 one-step discrepancy.

### Exact local proposition at T=2

At order 3, the compressed recursion and its uncompressed 729-point teacher
have identical two-step value and five-component total score to float64
roundoff. The feature residual is `2.220446049250313e-16`, and the carried
nonuniform weights sum to one.

At order 5, the center result passes the prior value and gradient screens:

| Quantity | Result |
| --- | --- |
| Value difference to Kalman | `0.00020462890126538014` |
| Largest componentwise score relative error | `0.002502421488769986` |
| Same-scalar FD maximum relative error | about `1.28e-09` |
| Sign reversal | none |

### Recursive failure

At order 3, `T=50` is an engineering pass but fails the frozen center accuracy
screens:

| Quantity | Result |
| --- | --- |
| Value difference to Kalman | `-0.07506760441708593` |
| Score relative errors | `(0.00445, 0.12170, 1.31161, 0.04642, 0.05869)` |
| Sign reversal | `phi3` |
| Same-scalar FD maximum relative error | about `1.59e-08`, pass |

At order 5, `T=5` also passes engineering/FD but fails recursive accuracy:

| Quantity | Result |
| --- | --- |
| Value difference to Kalman | `0.05458909929326694` |
| Score relative errors | `(0.00983, 1.20819, 2.63509, 0.00661, 0.05212)` |
| Sign reversal | none |
| Same-scalar FD maximum relative error | about `1.67e-09`, pass |

The transition from an order-5 `T=2` pass to a `T=5` failure localizes the
problem after the guaranteed one-step look-ahead horizon.

## Capacity Diagnosis

A direct KKT projection using top-teacher-weight anchors plus one basic feasible
support produced negative weights for every tested anchor cap from 24 through
128. It was rejected; no clipping was used.

A strictly positive overcomplete alternative mixed multiple independently
prepared positive basic feasible teachers. This changed only chart capacity:

| Prepared bases | Approximate anchors per reset | T=5 value difference to Kalman |
| ---: | ---: | ---: |
| 2 | 22 | `0.08982` |
| 4 | 44 | `0.03827` |
| 8 | 86--88 | `-0.00308` |
| 16 | 173--176 | `-0.00567` |
| 32 | 346--351 | `-0.00567` |

The 8-basis arm passed same-scalar FD and removed sign reversals, but `phi2` and
`phi3` score relative errors remained approximately `47.96%` and `26.58%`.
Thus no capacity arm passes the joint value-and-score criterion, and no arm is
selected for `T=50` promotion.

## Interpretation

Claimed target: total derivative of the executed finite corrected-LEDH
Contract E--TP observed-data scalar.

Quantity computed: TensorFlow autodiff of that same finite recursive scalar
with fixed charts and fixed quadrature.

Verdict: the derivative is correct relative to the finite scalar, supported by
AD/FD agreement and exact `T=2` teacher/student preservation. The current
feature system is wrong for the stronger claim that preserving one-step value
and score is sufficient to preserve multi-step Kalman score accuracy. It loses
information after the look-ahead horizon.

The next repair should add multi-step/backward-informed score features or a
recursively enriched tangent statistic, then repeat the smallest failed prefix.
It must not loosen the center criterion or select a chart from value alone.

## Checks And Artifacts

- focused Phase 1--3 tests: `24 passed`;
- Python compilation and scoped diff hygiene passed;
- all runs were explicit float64 CPU-hidden reference/diagnostic exceptions;
- controlling artifacts are under
  `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase3_*`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept recursive implementation as experimental engineering evidence | same-scalar AD/FD, chart, feature, carried-weight checks pass | no harness/target/derivative veto | off-center chart validity absent | retain for repair | scientific accuracy or HMC readiness |
| Reject current recursive feature/chart candidate | joint center value and score screens fail beyond T=2 | candidate promotion veto fired | which future-score statistic is sufficient | enrich features and rerun smallest prefix | Contract E--TP research direction rejected |
| Continue master program | failure is LGSSM candidate-specific; Phase 4 tests structural support independently | no continuation veto | later adapters depend on structural fixture | execute Phase 4 | all-model validity |

Phase 3 gate: `ENGINEERING_PASS_SCIENTIFIC_CANDIDATE_FAIL`. The master program
continues to Phase 4, while Phase 8 retains the LGSSM feature-enrichment repair.
