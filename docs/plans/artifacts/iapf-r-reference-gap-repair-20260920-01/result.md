# Independent R iAPF: bounded repair result

2026-09-20. The modified Gaussian fitter passes the planned likelihood screen
on 32 repetitions each at dimensions 5 and 10. Fitted dimension-20 runs still
fail the optimizer-convergence check. Strict reproduction of the paper remains
open because the successful fitter changes equation (15), and the authors'
solver, floor, early-controller choices and original observations are unknown.

The owner requested remaining gaps, a reviewed plan and execution. The
[plan](../../iapf-r-reference-gap-repair-2026-09-20.md) received skeptical
self-review before execution and each material solver repair. No independent
review or subagent was used. The model is the paper's first linear-Gaussian
experiment, with T=100 and newly simulated observations. This R reference does
not change the BayesFilter TensorFlow implementation or establish score/HMC
accuracy.

## Implementation and mathematical findings

For Gaussian density values p and backward targets y, equation (15) profiles to
`L = p'p - (p'y)^2/(y'y)`. Diffusing the Gaussian makes p tend to zero and
therefore L tend to zero, even without matching the target shape. This is a
property of that objective, not a claim that the authors' numerical solver
followed this direction. The default `paper_eq15` path is preserved.

The explicitly different `relative_l2` fit minimizes
`R = L/(p'p) = 1-(p'y)^2/((p'p)(y'y))`. Positive amplitude scaling cancels. On a
diffuse Gaussian with nonconstant targets, the limit is
`1-(sum(y))^2/(N*sum(y^2)) > 0`. Thus the demonstrated amplitude escape is
removed. Both fit modes use the same exact Gaussian-plus-floor proposal and
importance correction; a good on-cloud residual still does not certify a good
proposal away from the fitting cloud.

Saved-input calibration identified inadequate optimizer allowances. Increasing
L-BFGS-B memory to max(5,2*d), with maxit5000, completes six preserved old/new
failures. The completed d5/d10 comparisons use that frozen setting, floor power2
and the first-full-window controller. Eight calibration sensitivity arms
(d5/d10, floor powers1/2/3 and delayed initial doubling) all complete. They show
that floor probabilities and final particle counts depend on those choices;
one run per arm cannot rank settings. No choice was selected on heldout results.

The new `relative_l2_nlminb` option uses the same R objective and analytical
gradient with a different, explicit solver. It passes all seven saved fit cases,
including the first failed d20 fit. It is not an automatic fallback and is not
the solver used in the completed 32-repeat comparisons.

The original 68 R checks and 22 new relative-fit/solver checks pass. The six
pytest cases also include three executed-source mutations and the source
snapshot regression. Final result: six passed in 1.35s, recorded in
`final-pytest.log`. Failure records retain the actual fitting inputs, parameters,
gradient, residual, solver code and, in the latest version, message/evaluation
and iteration counts. Tests exercise the actual controller-to-fitter call chain.

## Completed likelihood comparisons

Each method uses the same observations at its dimension. Repetitions 401:432
use independent method seeds. The comparisons are BPF with 10000 particles,
fully adapted APF with 5000, SIS with 10000 and exact Kalman. Confidence intervals
use 2000 bootstrap draws over whole paired repetitions, conditional on the
single observed dataset at each dimension. The predeclared accuracy screen
requires the mean likelihood-ratio interval to lie inside [0.9,1.1].

| Dimension | Complete repetitions | Mean Zhat/Z, 95% interval | SD | iAPF/fully-adapted variance ratio, 95% interval |
|---|---:|---|---:|---|
| 5 | 32 | 1.0011 [0.9897,1.0131] | 0.03289 | [0.1212,0.4587] |
| 10 | 32 | 1.0044 [0.9886,1.0204] | 0.04733 | [0.0575,0.1963] |

The iAPF and fully adapted filter pass the mean-accuracy screen, and the
intervals support lower iAPF variance for these datasets and unequal computing
budgets. BPF and SIS fail the accuracy screen, so they are ineligible for a
variance ranking. SIS illustrates why tiny empirical variance is meaningless
when essentially every likelihood estimate misses the target. No heuristic has
lower observed prefix log-likelihood MSE in either ordinary or large-innovation
observations; the conditional bootstrap heuristic veto does not fire.

The d5 batch timed out after 30 full comparisons and the iAPF member of repeat431.
The unchanged resume recomputed 431 and completed 432. Source hashes, data hashes,
settings and repeated likelihood/fit values agree exactly. The report removes
duplicate fit rows and includes all 32 requested repetitions. The d10 batch is
two complete 16-repeat pieces with identical frozen identities. There is no
successful-subset selection. Each dimension has 19200 unique fitting records.
Full tables, conditional intervals and attempts are in `tables.md` and
`summary.json`.

These are bounded comparisons of the alternative, not reproduction of the
paper's 1000-repeat tables. The original data differ; the mean final N is 2000
at d5 and 1968.75 at d10, whereas the paper reports 1000 at these dimensions.
The published standard deviations are descriptive context only. Reported wall
times include reference diagnostics and serialization, and do not support an
equal-cost efficiency claim.

## Dimension20 remains unresolved

Attempt08 hits the 5000-iteration L-BFGS-B limit at backward time92. Its squared
target weights have effective size 1.272 out of 1000. Let u=p/||p||, v=y/||y||,
and e=(I-vv')u, so R=e'e. The independently reconstructed residual Jacobian
J has singular values ranging from 4.292e-5 to 2.079, a condition ratio 48,446.
The Gauss-Newton approximation 2J'J consequently has a condition ratio about
2.35e9. This is local ill-conditioning, not rank deficiency or a proof of
insufficient cloud coverage. The gradient agrees with independent finite
differences to 2.61e-10. `mechanics04/conditioning.csv` preserves this evidence.

The nlminb option converges on that saved input, but fresh attempt09 fails at
time68 with squared-target ESS 1.409 and relative residual 3.006e-10. Replaying
with 10000 instead of 5000 iterations still reaches the iteration limit
(residual 2.068e-10, gradient 1.127e-7). The cap increase is rejected as a complete
repair; a tiny residual does not override nonconvergence. See
`mechanics05/solver-replay.csv` and `mechanics06/solver-budget.csv`.

This rejects the present high-dimensional numerical configuration, not iAPF as
a research direction. The next discriminating work is to compare local fit
curvature, independent-cloud target residuals and coverage on the saved d20
inputs, then evaluate an explicitly conditioned parameterization/solver or
larger fitting cloud. Any repaired method needs fresh pilot observations and
complete repeats before d40/d80 scaling. Additional brute-force iterations alone
are not supported by the replay.

## Decisions and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain relative fit as optional R reference at d5/d10 | Saved failures and both 32-repeat accuracy screens pass | No numerical or conditional heuristic veto in completed comparisons | One dataset per dimension; 32 repeats | Additional datasets and larger repeated study after resolving source choices | Paper replication or general superiority |
| Do not promote current d20 configuration | Both fresh pilots fail convergence | Numerical continuation check rejects each attempt | Local flatness, coverage and solver behavior are not causally separated | Fit-geometry/coverage study on preserved inputs, then fresh validation | Rejection of iAPF or proof that N1000 is insufficient |
| Keep strict paper-replication gap open | Alternative objective differs from eq15 | Source-fidelity requirement not met | Author numerical recipe and data unavailable | Recover recipe or declare a reconstructed-method target explicitly | Reproduction of published results |

| Inference status | Result |
|---|---|
| Hard veto screen | d5/d10 complete without fit failures; fresh d20 attempts fail optimizer convergence; no nonfinite/boundary acceptance |
| Statistically supported ranking | Conditional bootstrap supports lower variance than the fully adapted comparator at d5/d10, at their specified unequal budgets |
| Descriptive-only differences | Paper-table agreement, sensitivity-arm differences, runtime, particle-count averages and d20 fitting residuals |
| Default readiness | None: optional independent R reference; no TensorFlow, score, nonlinear, LEDH or HMC admission |
| Next evidence needed | Author numerical specification; repaired d20 solver/coverage; fresh datasets, 32 then 1000 repeats across the published dimensions; later matched-input TensorFlow comparison |

## Reproducibility and terminal review

All runs use base R4.1.2, CPU only, with CUDA_VISIBLE_DEVICES=-1 and one
BLAS/OpenMP thread. Each attempt retains exact commands, git provenance, source
snapshots, environment, seeds, data hashes, status and elapsed time in its
manifest. Both source/data verification and duplicate-replay checks pass.
`mechanics-manifest.json` records focused checks and the final source hashes.

Nine repair launches used 1291.196979/1550 worker-seconds. The earlier stage
used 248.382526, for 1539.579505 seconds combined, below the original 1800-second
authorization. One repair launch and 258.803021 worker-seconds remain; the stage
closes because the final saved-input solver qualification failed, as specified
in the plan, rather than spending the remaining allocation on another
unqualified pilot. Focused mechanics are conservatively charged 40/120 seconds.
No experiment is running.

Terminal self-review: the strongest alternative explanation for favorable
d5/d10 results is their particular observed datasets and the extra adaptation
cost. A multi-dataset, equal-cost experiment could reverse that ordering.
For d20, the code could be rejecting a practically adequate fit because the
optimizer's termination test is hard to satisfy in nearly flat directions;
independent-cloud validation plus a principled convergence criterion would
test this without silently dropping the guard. The weakest evidence is higher
dimension generalization and recovery of the authors' exact numerical method.
The source/objective distinction and all failed attempts remain explicit.
