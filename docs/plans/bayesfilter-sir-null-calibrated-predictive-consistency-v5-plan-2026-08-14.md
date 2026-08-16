# V5 Null-Calibrated Predictive Consistency Plan

Date: 2026-08-14  
Status: `PRE_EXECUTION_REVIEW`  
Supersedes as primary gate: V4 heuristic all-cell precision/exact gate

## Research Question

For a frozen observation-only classifier-ratio estimator trained at a fixed
parameter `theta`, does an independent observation path generated at the same
`theta` fall inside a predeclared joint null-predictive region at least 95% of
the time?

This is a filter-independent **same-parameter predictive consistency** test.
It is not an exact test of `hat{s}(Y) = s_theta(Y)`. The distinction is part of
the result contract and cannot be removed by a passing run.

## Target And Scope

The output vector is the nine-cell score estimate

`u(Y) = (hat{s}_{20,0}, hat{s}_{20,1}, ..., hat{s}_{50,2})`.

One frozen estimator bundle supplies all nine coordinates. The bundle includes
the perturbation grid, anchored basis, architecture, regularization, trained
weights, calibration temperature, and fixed observed-path evaluation rule.
Controls are selected on a disjoint selection domain before the bundle is
frozen. No null-calibration or audit path may influence controls or weights.

Horizon outputs are computed from the same simulated `T=50` path prefixes, so
their dependence is preserved. Each outer null replicate uses fresh simulator
noise and is independent of every other outer replicate.

## Null-Predictive Region

Use three independent partitions after the frozen bundle is selected:

1. `null_fit`: estimate the output center and covariance geometry only;
2. `null_calibration`: determine the rejection threshold only;
3. `null_audit`: estimate fresh-copy rejection frequency only.

Let `mu` be the mean of `u(Y)` on `null_fit`. Compute the covariance SVD

`C = U diag(lambda) U^T`.

Retain every numerically positive singular value using the declared numerical
rank rule `lambda > eps * max(lambda) * 9`, where `eps` is machine epsilon for
the stored dtype. Define the squared whitened nonconformity score

`D(Y)^2 = (u(Y)-mu)^T C^+ (u(Y)-mu)`.

The SVD is a geometry choice, not an error margin: it accounts for different
score scales and cross-cell dependence. A rank-deficient covariance fails
closed unless the retained rank is at least 2 and the omitted variance is
reported. No ridge is silently inserted.

For `B_cal=200`, sort the calibration scores and set the default marginal
coverage threshold

`q = D_(ceil(0.95*(B_cal+1))) = D_(191)`.

The future same-parameter path is accepted iff `D(Y_new) <= q`. Conditional on
the frozen `mu,C` and exchangeability, this split-conformal order statistic
gives marginal coverage at least `ceil(0.95*(B_cal+1))/(B_cal+1)`, which is
at least 95%. The threshold is therefore a null quantile, not a hand-picked
score margin. If the claim is strengthened to “with 95% confidence, the
population coverage is at least 95%,” use the more conservative tolerance-bound
rank `D_(196)`; this is a separate declared mode, not a post-hoc choice.

## Audit And Decision Rules

Use `B_audit=500` fresh same-parameter paths. Let `F` be the number with
`D(Y)>q`. The audit is a falsification check, not a second overly-stringent
promotion gate. Reject the frozen null-consistency claim only when the
one-sided 95% Clopper-Pearson **lower** bound for the failure probability
exceeds `0.05`. For `B_audit=500`, this first occurs at `F=34` failures. A pass
therefore means “the audit did not falsify a 5% failure rate”; the formal
coverage claim comes from the predeclared calibration order statistic. The
audit is never used to retune `q`, `mu`, `C`, controls, or weights. An audit
veto triggers a new plan or termination; it is not repaired by changing the
threshold after looking at audit results.

The primary V5 result is one joint coverage statement, not nine independent
cell pass/fail statements. Per-cell coverage, conformal scores, and covariance
rank are explanatory diagnostics.

## Centering And Bias Diagnostic

The predictive region is centered at the simulated estimator mean `mu`, so it
tests repeatability around the estimator's own null distribution. To expose a
constant score bias, also report the simultaneous 95% bootstrap confidence
region for `mu` around the known score-identity center `0`:

`E_theta[s_theta(Y)] = 0`.

This zero-mean check is a filter-independent mathematical diagnostic, not a
replacement for pathwise score validation. A zero-mean failure is a bias veto
for the optional score-consistency interpretation, while a passing predictive
coverage result alone must not be described as score correctness.

## Gaussian Calibration And SIR Use

Run the same V5 harness first on the exact Gaussian observation simulator.
Use the exact Gaussian score only as a secondary diagnostic:

- report the true-score error vector and its joint max statistic;
- compare its distribution with the null-predictive `D` distribution;
- do not tune `q` to the exact score;
- do not require all nine exact cells to pass before continuing.

The Gaussian run validates implementation, exchangeability, partitioning,
coverage accounting, and the distinction between predictive consistency and
score accuracy. It cannot prove transfer of score accuracy to SIR.

Run SIR only under the explicitly weaker claim that the frozen SIR estimator
has calibrated same-parameter predictive consistency. The SIR artifact must
say that no pathwise exact score is available and must not claim an exact SIR
score, filter correctness, ranking, HMC readiness, or default readiness.

## Data And Compute Contract

- Parameter: `theta=[0,0,0]`.
- Perturbations: `{0.005,0.010,0.015,0.020,0.030,0.040}`.
- Anchored basis: V4 `phi0=r`, `phi1=r^3-alpha*r^5`.
- Selection domain: `50`.
- Frozen-bundle training domain: `60`.
- Null-fit domain: `70`, `B_fit=500` paths.
- Null-calibration domain: `80`, `B_cal=200` paths.
- Null-audit domain: `90`, `B_audit=500` paths.
- All null paths are generated in paired `T=50` form and sliced to `T=20,40,50`.
- `B_fit`, `B_cal`, and `B_audit` are disjoint and use independent seeds.
- GPU target: `tftwogpu`, TensorFlow/TFP, XLA enabled, TF32 disabled, memory
  growth configured before logical-device initialization.
- Every artifact records git commit, source hashes, environment, GPU state,
  seeds, partitions, covariance rank, threshold, audit failures, and wall time.
- Campaign budget after smoke: one full Gaussian attempt up to 60 GPU minutes;
  if it produces a valid non-falsified harness artifact, one full SIR attempt
  up to 90 GPU minutes. Localized infrastructure repairs may use one fresh
  retry within the same total 150-minute budget. Scientific gate changes,
  extra retraining campaigns, or increased sample counts require a new plan.

The frozen-bundle fit is one final estimator per cell. A separate five-bundle
retraining stability diagnostic may be run only after the primary coverage
test and cannot change the primary claim.

## Admission And Nonclaims

| Item | Role | Rule |
|---|---|---|
| Source/runtime/GPU/artifact checks | hard veto | fail closed |
| Conditional class balance | hard veto | every perturbation and split balanced |
| Head finite/signal/calibration checks | hard veto | existing V4 head screens retained |
| Joint null audit | primary promotion criterion | CP upper failure probability <= 0.05 |
| Zero-mean simultaneous check | bias veto/diagnostic | report and classify separately |
| Per-cell coverage and AUC/ECE | explanatory | never rank or promote alone |
| Gaussian exact-score error | explanatory | no all-cell exact gate |

Passing V5 establishes only the stated same-parameter predictive coverage for
the frozen bundle. It does not establish that the output equals the true
observed-data score, nor does it justify transfer of an error margin from
Gaussian to SIR.

## Pre-Mortem And Stop Conditions

| Failure mode | Early diagnostic | Required response |
|---|---|---|
| Null region accepts a constant or biased output | zero-mean simultaneous check; head signal veto | classify as predictive-only or stop score interpretation |
| Calibration/audit leakage | seed/domain and hash audit | invalidate run; no repair on same audit data |
| Covariance rank collapse | SVD rank and omitted variance | fail closed; investigate estimator degeneracy |
| Audit failure above 5% | exact CP upper bound | stop or write a new repair plan; do not retune q |
| Coverage passes but true Gaussian error is poor | exact-score diagnostic comparison | retain only predictive-consistency claim |
| Training randomness dominates | five-bundle optional diagnostic | report conditional-bundle scope; do not generalize |
| Retracing/resource failure | run manifest and focused smoke | localized repair under same budget, fresh artifact root |

The plan must not proceed to SIR if the harness itself fails, if the null audit
fails, or if required diagnostics are missing. It may proceed to SIR after a
valid Gaussian harness run even when Gaussian true-score error remains poor,
but only with the weaker predictive-consistency claim explicitly stated.

## Planned Artifacts

- `docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-2026-08-14.md`
- `docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-review-2026-08-14.md`
- `docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814/`
- a result/reset memo after execution

No V5 code or experiment is executed by this planning step.
