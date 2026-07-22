# Contract E--TP Phase 6 Zhao--Cui Comparator Certification Result

metadata_date: 2026-07-15
status: PHASE6_COMPLETE_NO_SOURCE_ROUTE_PARAMETER_LEARNING_COMPARATOR
plan: `docs/plans/bayesfilter-contract-e-tp-phase6-zhao-cui-comparator-certification-plan-2026-07-15.md`
controlling_ledger: `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase6_zhao_cui_comparators/phase6_comparator_eligibility_ledger_v2.json`

## Decision

Phase 6 repaired and certified a TensorFlow fixed-parameter adjacent-state
squared-TT extension, including the total derivative through the fitted
previous marginal.  It did **not** produce a Zhao--Cui source-route parameter
learning comparator.

The distinction is mathematical.  Zhao--Cui equations (15)--(16) fit a joint
object over `(x_t, theta, x_{t-1})`.  The repaired BayesFilter route fits
`(x_t,x_{t-1})` at an externally supplied `theta`.  It preserves the
adjacent-state squared fit and Proposition-2 previous-state marginalization,
but it removes the parameter TT coordinate.  Under repository policy this is
`extension_or_invention`, not `fixed_hmc_adaptation` or `source_faithful`.

## Controlling Scalar Results

All entries below use explicit CPU-hidden TensorFlow float64, degree 8,
17-point quadrature per axis, adjacent rank 2, ridge `1e-10`, two fixed ALS
sweeps, and the norm-balanced orthonormal-mode initializer.  Those settings are
a diagnostic warm start, not a promoted default.

| Row | T | Status | Value | Score | Worst own-scalar FD error | FD-only threshold | Max fit residual | Max scaled condition |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Actual transformed SV | 1 | extension certified | `-1.7583212548` | `[-0.0891675,-0.2677393]` | `6.01e-8` | `0.07071` | `0.05614` | `1.00` |
| Actual transformed SV | 2 | extension certified | `-3.9080698126` | `[-0.1900795,0.5789780]` | `7.58e-8` | `0.07071` | `0.05614` | `1.75` |
| Actual transformed SV | 10 | extension certified | `-21.0267255123` | `[-1.0334798,0.9493837]` | `5.36e-8` | `0.07071` | `0.05667` | `4.28` |
| KSC-SV | 1 | extension certified | `-1.7389563337` | `[-0.0887250,-0.2155320]` | `6.06e-8` | `0.07071` | `0.05758` | `1.00` |
| KSC-SV | 2 | extension certified | `-3.9265896228` | `[-0.1933754,0.8420539]` | `3.34e-7` | `0.07071` | `0.05758` | `1.72` |
| KSC-SV | 10 | extension certified | `-20.9891034677` | `[-1.0372121,1.0432394]` | `7.41e-8` | `0.07071` | `0.05758` | `4.34` |
| Generalized SV | 1 | extension certified | `-1.0896993890` | `[-0.0035670,-0.0043070,-0.0050637]` | `3.60e-7` | `0.08660` | `0.00222` | `1.00` |
| Generalized SV | 2 | extension certified | `-2.3453823367` | `[-0.2026385,0.0011530,-0.0097079]` | `3.00e-7` | `0.08660` | `0.07425` | `1.17` |
| Generalized SV | 10 | extension certified | `-16.8900149134` | `[-0.9134747,-0.0659365,0.0348357]` | `8.91e-8` | `0.08660` | `0.07425` | `1.36` |

Every controlling FD table retains all declared steps
`(1e-2,3e-3,1e-3,3e-4)`, independently derives plus/minus/base compatibility
identities, and has a passing adjacent stable window in every parameter.
Marginal mass errors are at most `9.99e-16`.

## Row Eligibility

| Row | Phase 6 classification | Phase 7 use |
| --- | --- | --- |
| Actual transformed SV | certified fixed-parameter adjacent-state extension | same-target extension diagnostic; not Zhao--Cui source column |
| KSC-SV | certified extension | same-target extension diagnostic; not Zhao--Cui source column |
| Generalized SV | certified extension, but Contract E--TP feature family is already a row-specific negative result | diagnostic comparison only; no `T=100` promotion |
| LGSSM `d=3,p=5` | `zhaocui_comparator_unavailable` | compare Kalman/Contract E routes only; never count Kalman twice |
| Predator--prey | `zhaocui_comparator_unavailable` | compare valid non-Zhao routes only; retained-grid substitution forbidden |
| Austria SIR observed-data | `blocked_target_measure_mismatch` | no observed-data comparison; retain P90/P91 component evidence separately |

## Root Causes And Repairs

### 1. Legacy scalar route was the wrong comparator program

The pre-existing scalar route fitted the already marginalized current-state
density and explicitly disclaimed integrated-axis marginalization.  Projection
and marginalization do not generally commute, so this was wrong relative to a
Zhao--Cui Algorithm-2 comparator claim.

Repair: added a two-axis `(x_t,x_{t-1})` squared-TT update and carried the
normalized marginal obtained by integrating axis 1 of the fitted object.

### 2. Scalar FD identity self-attested compatibility

The legacy score path assigned the base compatibility hash to its plus and
minus runs instead of deriving each realized identity.

Repair: added independently derived scalar compatibility hashes that exclude
`theta` and target values but include dimensions, basis, ranks, quadrature,
sweeps, ridge, coordinate maps, seeds, target ids, shift indices, and realized
update structure.

### 3. Stable ALS solver lacked a TensorFlow reverse derivative

TensorFlow does not define a gradient for `tf.linalg.lstsq(..., fast=False)`.

Repair: preserved the stable forward solve and supplied its exact
overdetermined least-squares pullback.  The adjoint solve uses thin QR rather
than normal equations.  A standalone matrix/rhs directional test passes.

### 4. Higher-rank initializer was rank deficient

The initial rank-2 cores left higher-rank channels zero or proportional.  The
first generalized-SV adjacent solve had scaled condition number `4.85e4` and
the pseudoinverse branch was not differentiable under the assumed full-rank
contract.  Its worst FD error was `0.513`.

Repair: initialized channel `k` with orthonormal polynomial mode `k` in both
cores and coefficient `1/sqrt(r)`.  This is norm-balanced by construction and
contains no fitted amplitude.  The repaired generalized-SV `T=2` scaled
condition maximum is `1.17`, and worst FD error is `3.00e-7`.

### 5. Final source review found a classification error

The first repaired artifacts labeled actual SV `fixed_hmc_adaptation`.  That
was wrong because `theta` is external rather than a fitted TT coordinate.

Repair: reclassified the route as
`fixed_parameter_adjacent_state_squared_tt_extension`, regenerated all nine
controlling artifacts, and rebound every branch identity.  Values, scores, and
numeric FD rows are exactly unchanged; compatibility hashes correctly change
because route classification is branch metadata.

## Required Checks

The first attempted broad command referenced nonexistent test files and ran no
tests.  The corrected exact-path command passed:

```text
131 passed, 2 warnings in 41.05s
```

It covered the new adjacent-state route, stable ALS, fixed-branch derivatives,
Contract E--TP primitives/streaming/structural code, LGSSM recursive and
progressive tests, scalar-SV adapters, and predator--prey.  The warnings are
existing TensorFlow Probability `distutils` deprecations.

## Mathematical Execution Review

| Review item | Verdict |
| --- | --- |
| Same scalar differentiated | correct for the finite fixed-parameter extension; total TensorFlow derivative includes the previous fitted marginal |
| Adjacent-state variables and marginal axis | correct: `(x_t,x_{t-1})`, integrate axis 1 |
| Change of measure | correct: previous marginal is already in reference measure; only current-state Jacobian/reference factor is added |
| Max-log shift | differentiated; realized argmax index is part of FD compatibility identity |
| Stable least-squares derivative | algebraically correct and independently FD-tested; QR avoids squared-condition normal equations |
| Higher-rank differentiability | repaired by independent norm-balanced modes; realized condition telemetry is small on controlling runs |
| Zhao--Cui source classification | extension only; no source-route parameter-learning comparator exists |
| FD policy | `0.05*sqrt(p)` used only for individual-direction own-scalar FD, never cross-method agreement |
| Cross-method accuracy | not established; fit residuals up to `0.07425` require Phase 7/8 resolution checks |

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Hard engineering veto screen | passes for all nine controlling extension artifacts |
| Zhao--Cui parameter-learning comparator availability | unavailable for all primary rows |
| Statistically supported ranking | none |
| Descriptive differences | Phase 7 only; none are promoted here |
| Default/HMC/GPU/full-horizon readiness | false |
| Next justified action | Phase 7 same-target comparisons with extension labels preserved, followed by Phase 8 one-factor resolution/rank refinement |

## Post-Run Red Team

The strongest alternative explanation for future cross-method disagreement is
approximation resolution, not derivative wiring: own-scalar derivatives are
now precise, but fitted square-root residuals reach `0.07425`.  A Phase 7 gap
therefore cannot be interpreted as scientific method disagreement until Phase
8 varies quadrature, degree, and rank one factor at a time.  Conversely, a
small center gap cannot establish broad equivalence because these are one-point,
short-prefix deterministic diagnostics.

## Nonclaims

No adaptive TT-cross/TTSIRT reproduction, Zhao--Cui source-faithfulness,
parameter-learning comparator, exact filtering, cross-method equivalence,
statistical superiority, HMC readiness, default readiness, GPU/XLA readiness,
full-horizon readiness, or complete leaderboard is concluded.
