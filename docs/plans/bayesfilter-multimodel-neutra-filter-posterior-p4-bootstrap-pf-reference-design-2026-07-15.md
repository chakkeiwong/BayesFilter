# P4 Bootstrap-PF Reference Design

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `FROZEN_BEFORE_PF_RESULTS`

## Question And Role

Can a bounded TensorFlow bootstrap particle filter produce a stable enough
likelihood diagnostic on the frozen seed-81104 T=20 predator-prey dataset to
screen `PP-UKF` and `PP-SGQF` values at fixed parameter points?

The PF is a stochastic filter diagnostic. It is not exact, is not a
differentiable HMC target, does not issue a posterior identity, and provides no
score authority. Filter score admission remains analytic implementation plus
centered-FD evidence for the same deterministic filter program.

## Frozen Target

- observations SHA-256:
  `dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387`;
- T=20, `y0` observes the initial state before any transition;
- initial `N((50,5), I2)`;
- RK4 delta 2.0, internal step 0.1;
- process and observation covariance `4 I2`;
- no positivity projection;
- six physical parameters in the model-declared box.

Fixed audit points in six-probit source coordinates:

1. transformed physical truth;
2. source origin;
3. `(-0.35,0.20,-0.20,0.25,-0.25,0.20)`;
4. `(0.30,-0.30,0.25,-0.20,0.20,-0.25)`.

These points are fixed before PF results. They screen a bounded region and do
not establish global approximation quality.

## PF Algorithm

Use TensorFlow `float64`, stateless random draws, vectorized particle and seed
axes, and `tf.while_loop` over time and the 20 RK4 substeps. At `t=0`, sample
from the fixed initial Gaussian, weight by `p(y0|x0)`, and resample. At positive
time, propagate with the RK4 mean plus process noise, weight by the Gaussian
observation density, and resample. Use stateless multinomial resampling after
every observation. Record ESS before resampling, nonfinite counts, state minima,
and resampling count.

No NumPy, callback, Python particle loop, or Python time loop is allowed in the
active PF. The outer fixed audit-point and rung orchestration may be Python
because it is diagnostic/reporting work, not a target or training path.

## Budget Ladder

| Rung | Particles | Independent seeds | Role |
| --- | ---: | ---: | --- |
| PF0 | 4,096 | 8 | feasibility and gross degeneracy screen |
| PF1 | 16,384 | 12 | required stability rung |
| PF2 | 65,536 | 16 | conditional repair rung only if PF1 is finite but inconclusive |

Seeds are `81400 + rung_offset + index`, where offsets are 0, 100, and 200.
Every rung is preserved. Stop before PF2 if PF1 stabilizes; run PF2 only under
the rule below. Total PF budget is below 20 CPU/GPU wall-minutes per filter
phase on the local hardware and below the P4 admission bucket.

## Evidence Contract

For each audit point and rung, calculate the mean log-likelihood, sample standard
deviation, standard error, and a two-sided 95% Student-t interval over seeds.

PF1 reference stabilization passes when, simultaneously at all four points:

- every run is finite;
- minimum ESS is at least 16 particles;
- absolute PF0-to-PF1 mean shift is at most
  `max(0.5, 2*sqrt(SE0^2+SE1^2))` log units; and
- PF1 95% interval half-width is at most 0.5 log units.

If PF1 is finite but any stabilization rule fails, run PF2. PF2 stabilization
uses the same rules comparing PF1 to PF2 and a PF2 half-width at most 0.35.
If PF2 fails, the reference is `INCONCLUSIVE` and neither deterministic filter
may be admitted from this PF artifact.

Once stabilized, a deterministic filter value passes the PF screen when, at all
four points, it lies inside the stabilized 95% PF interval expanded by a fixed
practical margin of 1.0 log unit. This is a viability screen, not evidence that
the filter is exact, unbiased, superior, or statistically indistinguishable
from the PF.

## Diagnostic Roles

| Diagnostic | Role |
| --- | --- |
| finite PF likelihood and valid artifact | continuation veto for the PF reference |
| ESS `<16` | PF stabilization veto/repair trigger |
| PF rung mean shift and interval half-width | PF reference promotion criterion |
| deterministic-filter gap to expanded PF interval | filter target-admission veto |
| state minima and resampling count | explanatory diagnostics |
| truth-point closeness | explanatory only |

## Stop And Repair

- PF harness/data/time-order failure: repair and rerun in a fresh root.
- PF1 finite but inconclusive: run the predeclared PF2 rung.
- PF2 inconclusive: close `PP-UKF` and `PP-SGQF` as
  `TARGET_BLOCKED_REFERENCE_INCONCLUSIVE`; do not expand particles or margins
  without a refreshed plan.
- Stabilized PF but deterministic filter fails: block only that filter cell and
  continue the other.
- A passed filter still requires typed posterior recomposition and all later
  HMC/training gates.

## Nonclaims

No exact likelihood, unbiased log-likelihood, score correctness, posterior
calibration, filter ranking, HMC readiness, NeuTra readiness, or cross-dataset
validity is concluded from this reference.
