# P5 Subplan: Structural Target Design And Route Freeze

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_TARGET_DESIGN_EXECUTION`

## Phase Objective And Entry Conditions

Turn the Chapter 18b quadratic structural example into a prospectively defined
TensorFlow parameter-estimation target before implementing or running serious
HMC/NeuTra. Freeze the graph-native dataset, inferred parameter hypothesis,
prior and chart, structural UKF route, deliberately wrong artificial-noise
negative control, reference checks, and target-admission criteria.

Entry requires:

- P4 has a written phase result and all predator-prey cells have terminal
  states;
- P0/P1 identity, recomposition, GPU/XLA, batching, archival, and controller
  harnesses remain valid;
- the Chapter 18b mathematical equations and current NumPy worked fixture are
  available as source/reference evidence only; and
- the shared GPU is not occupied by another serious campaign.

P0 did not freeze a structural posterior. Its blockers
`MISSING_FROZEN_PARAMETER_PRIOR`, `MISSING_FROZEN_DATA_IDENTITY`,
`MISSING_GRAPH_NATIVE_MODEL_AND_DATA`, `MISSING_PARAMETER_SUBSET_AND_CHART`,
`MISSING_POSTERIOR_RECOMPOSITION`, and
`MISSING_STRUCTURAL_NEGATIVE_CONTROL` are inherited and must be closed rather
than assumed away.

## Mathematical Target And Candidate Parameter Hypothesis

The declared state-space law is

```text
m_t = rho m_(t-1) + sigma epsilon_t,  epsilon_t ~ N(0,1)
k_t = phi k_(t-1) + gamma m_t^2
y_t = m_t + k_t + e_t,               e_t ~ N(0,R)
```

with initial `N((0,0), diag(0.04,0.09))` and worked calibration
`(rho,sigma,phi,gamma,R)=(0.8,0.5,0.7,0.4,0.25)`. Those values are synthetic
truth, not posterior defaults.

The first candidate infers all five physical parameters. This is a hypothesis,
not an already justified target. Prospective support boxes are:

| Parameter | Candidate support | Design reason | Risk |
| --- | --- | --- | --- |
| `rho` | `(0.05,0.98)` | stable positive AR coefficient containing chapter value | excludes negative persistence and near-unit-root tail |
| `sigma` | `(0.05,1.25)` | positive shock scale bracketing chapter value | upper bound is convenience, not empirical knowledge |
| `phi` | `(0.05,0.98)` | stable deterministic accumulation coefficient | excludes negative persistence and near-unit-root tail |
| `gamma` | `(0.02,1.00)` | positive quadratic effect bracketing chapter value | upper bound may induce explosive prior-predictive paths |
| `R` | `(0.02,1.00)` | positive observation variance bracketing `0.25` | variance-scale upper bound is a design hypothesis |

Use independent physical Uniform distributions over these boxes and a
five-probit chart `lower_i + width_i Phi(u_i)` with the complete log-Jacobian.
This makes the source prior standard normal and mirrors an admitted repository
chart pattern, but it remains a convenience hypothesis. It may be admitted only
if the prior-predictive and local-information checks below pass. A failed
five-parameter audit does not authorize silently fixing a parameter or narrowing
the prior after seeing HMC; it yields a target-design blocker and a prospective
replan.

## Dataset Design

- Treat `T=100` as a prospective hypothesis, not a frozen answer. Generate the
  final graph-native synthetic trajectory with TensorFlow stateless root seed
  `(20260716,15001)` only if the horizon audit below admits `T=100`.
- Draw `x_0` from the declared initial Gaussian, then for `t=1..99` draw one
  scalar `epsilon_t`, compute `m_t`, compute `k_t` deterministically, and draw
  scalar observation noise. No independent noise is ever drawn for `k_t`.
- Record state and observation tensor hashes, generator source closure, seed,
  dtype, exact time convention, and maximum deterministic residual.
- The seed and horizon are a bounded synthetic-fixture choice, not population
  evidence. Truth recovery remains explanatory only.
- Before target issuance, generate three T=200 design trajectories with roots
  `(20260716,15101)`, `(20260716,15102)`, and `(20260716,15103)` and evaluate
  their exact prefixes at horizons `50`, `100`, and `200`. These design rows are
  disjoint from the final dataset and cannot be pooled into it. Admit `T=100`
  only if every T=100 row passes the frozen likelihood-information gate below
  and the accumulated information matrices are positive-semidefinite
  nondecreasing from T=50 to T=100 to T=200 within `1e-8` relative numerical
  tolerance. If T=100 fails but T=200 passes, stop with
  `TARGET_DESIGN_HORIZON_REPLAN_REQUIRED`; do not silently promote T=200.

## Route Freeze

### Structural UKF Candidate

Use augmented sigma points on `(m_(t-1),k_(t-1),epsilon_t)` and compute every
`k_t` point from `phi k_(t-1)+gamma m_t^2`. The production candidate must be a
batch-native `[B,5]` TensorFlow manual value/score path using the existing
principal-square-root batched structural engine and `tf.while_loop` over time.
No full-state artificial process covariance is allowed.

### Artificial-Noise Negative Control

Create a diagnostic-only route with an explicit independent
`eta_k ~ N(0,0.04)` innovation and
`k_t = phi*k_(t-1) + gamma*m_t^2 + eta_k`. This is the precise additive-noise
law whose sigma points leave the declared structural support. It must have a
distinct diagnostic signature, a two-dimensional innovation contract, nonzero
pointwise deterministic residuals with the residual equal to `eta_k`, and a
declared `k` process-covariance increment of exactly `0.04`. On the chapter
one-step fixture it must leave the predictive mean unchanged, change
`S=0.6121674304` to `0.6521674304`, and change the log likelihood from
`-0.7029747609` to `-0.7328186210` within `5e-6`. It is permanently ineligible
for posterior identity, HMC, training, or fallback. Merely inflating a reported
covariance after otherwise structural point propagation is not an adequate
negative-control implementation because that construction can retain zero
pointwise residuals.

### Zhao-Cui Extension

`STR-ZC` remains `extension_or_invention`. P5 target design may specify a
fixed-route tensor/SIRT-inspired structural filter only after the structural
UKF target and negative-control detector are admitted. It cannot be called
source-faithful, cannot reuse the generic retained-grid production-ineligible
route, and receives its own later design/review subplan and target signature.

## Research Intent And Evidence Contract

| Field | Frozen target-design contract |
| --- | --- |
| Question | Is a five-parameter graph-native structural UKF posterior scientifically and numerically defined well enough to enter value/score admission? |
| Baseline | Chapter equations and worked one-step numbers; NumPy fixture is reference-only |
| Primary pass | Dataset replay/hash, exact structural identity, prior-predictive stability, likelihood-only source-coordinate innovation-information admission at T=100, structural/naive signature separation, one-step reference reproduction, and complete prospective posterior specification |
| Hard vetoes | Any independent `k_t` noise in intended route; deterministic residual above tolerance; nonfinite/explosive prior-predictive majority; materially rank-deficient five-parameter local information; negative control not detected; active-path NumPy/host callback/Python time loop; post-result target change |
| Repair triggers | TensorFlow/reference mismatch, derivative error, branch/status failure, generator drift, or harness serialization defect |
| Explanatory only | Truth distance, one-seed trajectory appearance, likelihood curvature magnitude, negative-control numerical gap, runtime |
| Not concluded | Posterior correctness, filter exactness, parameter identifiability beyond the designed local checks, HMC convergence, NeuTra quality, Zhao-Cui extension validity, calibration, or readiness |

## Required Artifacts And Checks

1. Write a mathematical target note in project notation distinguishing the
   structural predictive law from the artificial-noise law.
2. Implement the batched TensorFlow simulator, five-probit map, prior,
   Jacobian, structural transition, manual derivatives, UKF likelihood/status,
   and independent prior/likelihood/Jacobian recomposer.
3. Replay the chapter one-step structural and artificial-noise calculations
   against the NumPy reference and checked chapter values.
4. Require pointwise
   `k_t-phi*k_(t-1)-gamma*m_t^2=0` for all intended-route generator and sigma
   points. Require the negative control to fail this identity and have a
   distinct dependency/signature closure.
5. Run one compiled prior-predictive batch of 4,096 independent physical-prior
   draws to T=200 with root `(20260716,15201)`, and inspect the exact T=50,
   T=100, and T=200 prefixes. A trajectory is numerically valid only if every
   state and observation is finite and has absolute magnitude at most `1e6`.
   Require at least 99% valid trajectories at every prefix. Record per-prefix
   valid fractions and q50/q95/q99/max of the per-trajectory maximum magnitude.
   The fraction is the numerical/domain gate; the quantiles are explanatory and
   cannot calibrate or narrow the prior after inspection.
6. Compute a five-by-five *likelihood-only* local Gaussian-innovation Fisher
   surrogate in source coordinates. For each time step, differentiate the
   structural UKF predictive observation mean `mu_t` and log innovation
   variance `log S_t` and accumulate

   ```text
   I = sum_t [(d mu_t)(d mu_t)^T / S_t
              + 0.5 (d log S_t)(d log S_t)^T].
   ```

   Do not add the standard-normal source prior or chart Jacobian: either would
   make a data-unidentified parameter appear identified. Evaluate the matrix at
   the source-coordinate truth and at its ten fixed axis neighbors `u +/- 0.5
   e_i` for each of the three design trajectories and each horizon. Require an
   a fully batched centered source-coordinate FD route with fine step `5e-5`
   and coarse step `1e-4`. Require the maximum scale-normalized difference
   between fine and coarse derivatives of both `mu_t` and `log S_t` to be at
   most `5e-3`. Also require finite symmetric matrices, no
   eigenvalue below `-1e-8 * max(1,lambda_max)`, numerical rank five using
   `lambda_i > 1e-8 * lambda_max`, `lambda_min >= 0.10`, and condition number
   `<=1e6` for all eleven T=100 points on all three trajectories. The
   `lambda_min` threshold means that the local data likelihood contributes at
   least 0.10 precision in every source-coordinate direction; it is a gross
   weak-information veto, not proof of global or practical identifiability.
   T=50 and T=200 rows explain horizon sensitivity but cannot rescue a failed
   T=100 gate without a prospective replan.
7. Require batch permutation, source-coordinate FD, eager/CPU-XLA parity, trusted
   GPU/XLA, status/branch, spectral-gap, static-source no-NumPy/no-callback, and
   wrong-substitution tests.
8. Only after all gates pass, freeze data hashes, target contract, route identity,
   comparator estimands, and a target-specific agreement design; issue the typed
   identity in the next R1B subplan.

## Required Result And Handoff

Write a target-design result containing the decision table, engineering/
numerical/scientific ledgers, default audit, exact commands, environment, seeds,
wall time, source hashes, and all nonclaims.

If admitted, draft and audit the `STR-UKF` R1B/value-score subplan with the exact
target signature inputs and no HMC. If blocked, name whether the failure is
target design, prior-predictive domain, identifiability, structural semantics,
derivative implementation, or evidence coverage; do not reinterpret it as a
NeuTra failure. Refresh the separate `STR-ZC` extension design only after the
structural semantics and negative-control detector are valid.

## Forbidden Claims And Actions

- No serious HMC, NeuTra training, or target signature before target-design
  admission.
- No artificial `k_t` noise, covariance floor, or jitter reinterpreted as model
  process noise.
- No silent reduction of the inferred subset or post-result prior/support
  narrowing.
- No NumPy, host callbacks, scalar row mapping, or Python time/sample loops in
  the active model/likelihood/score path.
- No labeling `STR-ZC` source-faithful and no use of the generic retained-grid
  route as a production evaluator.

## Stop Conditions And Budget

Stop with a target-design blocker for irreconcilable chapter/code mismatch,
failure to detect artificial noise, any failed T=100 five-parameter information
gate, a required horizon replan, prior-predictive invalidity after two
prospective implementation repairs,
unavailable trusted GPU for the required XLA canary, or more than 8 CPU-hours
plus 2 trusted GPU-hours in this design rung. Local serialization/reporting
repairs use fresh attempt roots under the unchanged design.

## Skeptical Pre-Execution Audit

Decision: `PASS_FOR_TARGET_DESIGN_EXECUTION`.

P4 is closed with two mean-level confirmations and one source-route block; none
of that evidence is transferred into the structural target. The plan does not
treat worked constants as inferred defaults or let the source prior manufacture
an identifiability pass. It tests T=100 prospectively on disjoint prefixes,
freezes exact prior-predictive counts and seeds, and detects the wrong structural
route through both off-manifold point residuals and the artificial covariance
increment. The information gate remains local and therefore cannot establish
global identifiability; that limitation is an explicit nonclaim rather than a
proxy promotion. The support boxes remain convenience hypotheses and may pass
only the frozen numerical and information screens; a failure requires a new
prospective target design, not post-result narrowing.

The bounded external Claude review was attempted and denied by the managed
platform before launch because external disclosure of the private workspace
plan was not allowed. No Claude verdict exists. A second local skeptical review
is recorded in
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-structural-target-design-review-record-2026-07-16.md`.
Reviewer unavailability is advisory under the current policy and does not add a
scientific or execution veto.
