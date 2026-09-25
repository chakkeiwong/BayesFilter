# A09: standalone SGQF-initialized pair TT

Status: complete. Degree-4 nominees improve one-dimensional filtering under
the predeclared exploratory comparison. Four-dimensional promotion fails:
one valid-reference tail case exposes guide covariance collapse, and another
case fails both guide validity and reference precision. These failures require
guide repair; they do not reject the TT research direction.
Plan: [A09 amendment](observation-aware-tt-master-amendment-09-standalone-warm-tt-20260916.md).
The [plan review](artifacts/observation-tt-warm-improvement-20260916-01/plan-review.md)
and [terminal self-review](artifacts/observation-tt-warm-improvement-20260916-01/result-review.md)
are executor reviews, not independent reviews.

## Question and implementation

The question is whether improving the standalone warm TT improves the
exact-importance-corrected particle filter. Each post-initial step uses a
pair TT initialized from the SGQF Gaussian joint and fitted to the target
containing its own previous retained TT. There is no per-time-step choice
between TT and an exact Gaussian. The initial step uses the Gaussian guide.

The optional new regularizer penalizes deviation from the initial core:
`lambda * ||c - c_initial||_1`. Its proximal update, objective and reported
KKT use the same anchor. Existing zero-centered behavior remains the default.
Degree, row-count and regularization nominees were chosen offline; no
confirmation data changed them. Rank remains 3, with four sweeps and 128
proximal iterations, defensive mass 1e-5 and N=512 particles.

| d | Warm baseline | Capacity/sample nominee | Initial-centered nominee |
|---|---|---|---|
| 1 | degree 3, 1024 rows, zero-centered L1 .001 | degree 4, 4096 rows, zero-centered L1 .001 | degree 4, 4096 rows, initial-centered L1 .001 |
| 4 | degree 3, 1024 rows, zero-centered L1 .001 | degree 4, 1024 rows, zero-centered L1 .001 | degree 4, 1024 rows, initial-centered L1 .00001 |

The older generic-start pair TT is rerun as a comparator. Its defensive
mixture and row design also differ, so that comparison does not isolate the
effect of initialization. The four simple proposals are the transition,
stationary prior, SGQF marginal and SGQF joint, all used by the same exact-weight
particle consumer.

## Regression evidence

At times 1, 10 and 19 of each of three calibration sequences per dimension,
all nominees fit the same incoming baseline TT and use a separate 8192-row
audit panel. All 54 fits have smaller defended Hellinger discrepancy than
their own projected/scaled initializer. These are descriptive observations;
they do not compare TT against the exact analytic SGQF joint or establish a
filtering ranking.

| d | Mean fitted H2, baseline | Capacity | Initial-centered | Capacity reduction from baseline |
|---|---:|---:|---:|---:|
| 1 | .00107757 | .00030296 | .00029162 | 71.9% |
| 4 | .00326018 | .00153241 | .00208699 | 53.0% |

The four-dimensional degree-4 fits reach a capped core Gram condition estimate
of 5.24e10 and KKT residual .0137. Four sweeps/128 proximal steps therefore do
not establish solver convergence. Larger polynomial bases reduce the observed
joint fitting error but can also make fitting harder numerically. The joint
regression weights also differ from the actual particle ancestor distribution;
a smaller joint discrepancy alone cannot establish smaller conditional
importance-weight variance.

## Confirmation and uncertainty

Replacement confirmation uses twelve fresh T=20 sequences per dimension and
four particle repetitions. For each sequence, MSE averages squared differences
between the particle filtering mean and an independent reference, divided by
the coordinate stationary variance. It averages repetitions, times and state
coordinates. Independent sequences are the uncertainty units.

For each of the two nominees and each dimension, the primary contrast is
candidate MSE minus standalone warm-baseline MSE. The frozen 9999-resample
sequence bootstrap supplies simultaneous 95% intervals for eligible members
of these four contrasts. Incomplete contrasts are ineligible, leaving two
scalar intervals in this run. An upper endpoint below zero supports that improvement only after
reference and validity checks pass. With twelve sequences, these are
exploratory intervals, not a guarantee of nominal coverage. ESS is the actual
pre-resampling particle quantity `1/sum(normalized_weight**2)`, out of 512.

| d | Method | Matched sequences | Normalized MSE | Mean/min ESS | Build seconds | Particle seconds/repetition |
|---|---|---:|---:|---:|---:|---:|
| 1 | Standalone warm baseline | 12 | .00182035 | 381.14 / 55.63 | 1.923 | .392 |
| 1 | Degree-4 capacity/sample | 12 | .00168385 | 385.10 / 162.59 | 1.923 | .355 |
| 1 | Degree-4 initial-centered | 12 | .00167275 | 384.39 / 157.15 | 1.965 | .319 |
| 1 | Generic-start pair TT | 12 | .00186542 | 381.59 / 110.65 | 2.273 | .346 |
| 1 | SGQF joint | 12 | .00210320 | 375.16 / 93.05 | .166 | .183 |
| 4 | Standalone warm baseline | 11 | .00438245 | 322.20 / 3.41 | 9.385 | .882 |
| 4 | Degree-4 capacity/sample | 11 | .00427656 | 325.17 / 3.66 | 15.792 | 1.235 |
| 4 | Degree-4 initial-centered | 11 | .00432813 | 322.39 / 3.36 | 15.736 | 1.070 |
| 4 | Generic-start pair TT | 11 | .00462118 | 315.04 / 1.29 | 28.513 | .748 |
| 4 | SGQF joint | 11 | .00457563 | 316.85 / 4.67 | .196 | .198 |

The scalar reductions are 7.50% and 8.11%. Candidate-minus-baseline intervals
are [-.00024803, -.00002497] and [-.00025589, -.00003931], respectively.
Neither interval compares the two degree-4 variants with each other; their
ordering is not established. Four-dimensional intervals are withheld because
of incomplete coverage and failed evidence agreement. Its averages above
include the severe tail failure and are descriptive, not a favorable ranking.

In d4 sequence 4, the largest |y|/beta is 16.1225. At t=8 the accepted
SGQF level-2 guide has Cholesky diagonal
[4.09e-16, 3.54e-16, 2.22e-16, 3.19e-16] and log determinant -142.735.
Its chart is positive definite but collapsed in physical scale. The warm
TT mean in the affected coordinate is 2.36265 versus reference 3.83476.
All six guide-dependent methods fail the log-evidence agreement screen;
warm-baseline bias is -143.153 against allowance .707, and the degree-4
capacity bias is -142.464 against .809. The independent reference passes at
131072 particles. This is a demonstrated guide/proposal failure, not a small
TT-versus-SGQF loss difference. Both TT and analytic SGQF suffer it.

Across large-observation regimes, every warm TT has observed MSE losses to
transition and stationary-prior proposals; eight conditional heuristic losses
in total trigger the stated default-promotion screen. These differences are
descriptive falsification evidence, not statistically supported heuristic
rankings. No heuristic was used to tune the TT.

In d4 sequence 10, all SGQF levels fail the covariance-SPD check. The reference
also misses its precision screen: normalized mean MCSE .02558 exceeds .02 at
the maximum 131072 particles. Six guide-dependent methods do not run there;
the case is preserved without substitution. These two distinct d4 failures
prevent an unconditional four-dimensional success claim.

Timings cover a full T=20 sequence and include setup/compilation. Shared guide
cost is separate in the table. Including guide/build and one particle run,
d4 warm baseline/capacity/initial-centered/generic/SGQF joint cost about
11.34/18.10/17.88/30.33/1.46 seconds. No statistical speed ordering is claimed.
The [full tables](artifacts/observation-tt-warm-improvement-20260916-01/tables.md)
and report.json preserve all eight methods, conditional comparisons,
log-evidence screens, resampling and amortized timing.

## Repairs and provenance

Calibration-01 duplicated an A07 data-seed root and was stopped/ineligible.
Calibration-02 used new data and nominated the frozen settings. Final
call-chain review then found time-shift reuse of scalar random seeds between
sequences/repetitions, including independent-reference repetitions.
Confirmation-01 is retained as descriptive evidence with an explicit
invalidation record; its intervals cannot support the intended independence
claim. Calibration-02 also remains nomination/diagnostic evidence only, with
no uncertainty or optimality claim. Dependence can make selection noisier;
fresh confirmation can still evaluate the frozen hypotheses.

Revision 2 reserves ten million scalar seeds per sequence, separates fitting,
particle and reference roles, and uses repetition stride 1000 for T=20.
Confirmation-02 uses fresh data partition 2. Schedule and actual reference
endpoint tests pass. No TT setting or scientific criterion changed in this
repair. Historical campaign MCSE claims that inherited stride 10 require
separate independence audit; their descriptive measurements are preserved.

Numerical evidence is under
`../benchmarks/artifacts/observation_tt_warm_improvement_20260916/`:
`attempt-calibration-02/` and `attempt-confirmation-02/` are the final source
directories. Each manifest records commit, dirty-state source hashes, command,
environment, seeds, device/memory policy, wall time and output paths. Exact
executed source copies accompany the manifests. RTX 5080 and TensorFlow float64
XLA numerical kernels were used with verified memory growth. Host setup, dense
Gaussian coefficient conversion/CPU TT-SVD, references and reporting remain
explicit diagnostic exceptions; no end-to-end GPU/XLA claim is made.

Twenty-seven distinct relevant tests pass across the focused batches, including
the regularizer formula, zero-penalty parity, degree-4 Gaussian projection,
standalone retained recursion, particle call chain and repaired seed/reference
wiring. The GPU smoke also passes CPU/GPU fitter parity to 2.23e-16.

## Decision and remaining uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep degree-4 nominees as scalar improvements | Both eligible intervals below zero | Scalar reference/evidence/consumer screens pass | Twelve sequences; nomination noise | Broader fresh validation after method freezes | Best penalty, broad default or HMC readiness |
| Continue d4 repair, withhold promotion/ranking | Incomplete coverage; no eligible interval | Guide collapse/evidence disagreement in s4; guide/reference failure in s10; conditional heuristic losses | Guide accuracy versus regression/solver error | Diagnose and repair guide covariance scale and tail coverage first | TT direction rejected or SGQF superiority |
| Preserve observed regression gains | All 54 fit audits below their own initializer | No fit validity failure; conditioning unresolved | Three calibration sequences, reused algorithm streams | Controlled solver-convergence check after guide repair | Filtering gain from joint fit loss alone |

| Inference status | Finding |
|---|---|
| Hard veto screen | d4 guide/reference/evidence failures above; no stored nonfinite values or TT CDF-bracket failures in completed consumers |
| Statistically supported ranking | Exploratory scalar capacity and initial-centered nominees versus scalar warm baseline only |
| Descriptive-only differences | d4 averages, degree-4-versus-degree-4 ordering, generic/SGQF comparisons, fit losses, ESS/tails and speed |
| Default-readiness | Not established; d4 fails declared promotion screens |
| Next evidence needed | Guide scale/tail repair, adequate reference on the failed case, controlled solver check and fresh complete confirmation |

[Artifact audit](artifacts/observation-tt-warm-improvement-20260916-01/audit.json)
passes evidence integrity: 14880 particle-time records, 6992 TT CDF checks,
1077122 finite numeric values, 1311 frozen fit configurations/seed checks,
19 exact source snapshots and all 54 same-target audits. Integrity passing
does not override the scientific failures. The final confirmation took
1287.928 seconds; peak recorded TensorFlow allocation was 367403776 bytes.

The strongest alternative explanation for an apparent fit gain without a
filtering gain is that extra basis capacity helps the bulk joint approximation
while solver error or conditional weight tails dominate particle error. Three
calibration sequences and twelve confirmation sequences cannot distinguish
all of these mechanisms. More independent confirmation could overturn a
close filtering comparison; a controlled solver-convergence experiment would
test the numerical explanation directly.

The next proposed master amendment should first repair guide covariance-scale
validation and tail coverage using the preserved failed cases as diagnostics,
then test convergence/conditioning on fixed targets and conditional errors on
particle ancestor clouds. Current failed cases cannot serve as fresh final
confirmation after those repairs. Another rank expansion has lower priority.
The repair needs predeclared settings and fresh downstream confirmation;
no additional sweep is implied by this result.
Guide robustness, scalable initialization and analytical total derivatives
remain separate open work. This run establishes neither source faithfulness,
raw retained-TT normalizer/moment accuracy, default readiness nor HMC readiness.
