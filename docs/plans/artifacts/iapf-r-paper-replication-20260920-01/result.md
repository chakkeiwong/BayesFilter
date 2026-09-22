# Independent R iAPF: implemented, but fitted replication failed

2026-09-20. The owner selected the first linear-Gaussian study, beginning with
bounded runs and then scaling toward the published settings. The independent
R reference is implemented. Its mathematical filter controls pass, but the
repeated fitted-iAPF comparisons fail during Gaussian fitting. The paper's
empirical results have **not** been replicated, and the fitted reference is
not yet suitable as an authority for debugging BayesFilter performance.

## What the experiments established

The reference implements Gaussian-plus-positive-constant twists, exact mixture
proposals and normalizers, backward regression, adaptive ESS resampling with
retained weights, the outer iAPF controller and a fresh final likelihood run.
It also includes independent Kalman and fully adapted APF comparators.

The final focused suite passes: 68 R identity/behavior checks, three
executed-source mutation checks, and the snapshot-path regression, organized
as five pytest tests (1.10 s). Evidence is in `mechanics03/`. The tests exercise
the actual reference implementation. They cover both resampling branches,
positive-floor sampling, fitting derivatives, stopping, particle doubling,
T=1, failed-fit preservation and independently reconstructed likelihoods.

At T=100 and d=5,10,20,40,80, the exact full-covariance backward-twist control
passes all 15 runs (three particle seeds per dimension, N=1000). Its maximum
absolute log-likelihood discrepancy from Kalman is 1.82e-12, and the maximum
spread of terminal log weights is 4.01e-11. The separately implemented fully
adapted APF agrees with the observation-twist route under matched seeds,
including resampling counts. These checks bypass estimated diagonal twists
and do not count as reproduction of fitted-iAPF performance.

The attempted 32-repeat fitted comparisons were interrupted:

| Data seed and dimension | Complete repeats | Stopping failure |
|---|---:|---|
| 55000005, d5 | 4/32 | Repeat105: underflow in a relative-fit diagnostic |
| 57000005, d5 | 1/32 | Repeat202: variance reached the numerical boundary |
| 57000010, d10 | 2/32 | Repeat203: variance reached the numerical boundary |

The first reporting failure was repaired without changing the objective,
gradient, optimizer or fitted parameters. The former failing seed then
completed, and a separate fresh-data pilot passed before the two renewed
batches. The later variance-boundary failures remain unresolved. No 1000-repeat
run was launched because the prerequisite fitting reliability check failed.

The successful d10 subset also has larger prefix log-likelihood error than the
fully adapted heuristic in both ordinary and large-innovation conditions.
Observed mean squared errors are 0.131 versus 0.0274 and 0.158 versus 0.0373,
respectively. These are descriptive results from two completed repeats;
selection caused by failed fits prevents a supported stochastic ranking.
The renewed d5 subset also loses descriptively to that heuristic in the
large-innovation condition. No heuristic or statistical screen is passed.

[`tables.md`](tables.md) and [`summary.json`](summary.json) preserve the
completed samples, published table comparisons, source identities and every
attempt. Confidence intervals are deliberately withheld for these incomplete
samples: resampling successful outcomes cannot correct the missing failures.
Observed SDs, runtimes and particle counts do not establish replication or
superiority. In particular, the reconstruction used more final particles than
the paper reports at d<=20; it did not reproduce the paper's efficiency claim.

## Why fitting failed

Equation (15) minimizes `sum_i (N(x_i;m,V)-lambda*y_i)^2`. Its profiled optimum
in lambda is `lambda=(p'y)/(y'y)`, so its loss is `p'p-(p'y)^2/(y'y)`.
For `V=s^2 I`, this loss is bounded above by
`N*(2*pi)^(-d)*s^(-2d)`, which tends to zero as `s` grows. A near-zero absolute
loss can therefore mean a vanishing Gaussian rather than a good relative
shape fit. This is a derivation for the written objective; it does not assert
that the authors' uninspected numerical procedure took this path.

The saved failures show this mechanism directly. At d5 backward time82 and
d10 time60, the largest fitted variance reached approximately 4.50e15 while
L-BFGS-B returned convergence code0. The relative residuals were 0.594 and
0.9998. Five focused alternative solver probes on the earlier failed fit did
not establish a reliable replacement. Four escaped toward vanishing density;
joint BFGS retained a small relative residual but exceeded its iteration limit.

The [reference note](../../../reference/iapf-independent-r-reference.md)
derives the proposal, importance correction and fitting degeneracy in detail.
`mechanics02/failed-fit.rds`, `boundary-d5.rds` and `boundary-d10.rds` preserve
the exact fitting inputs and parameters. The fail-closed variance guard remains;
removing it or silently replacing equation (15) by normalized regression would
not complete the requested paper replication.

## Reproducibility and source limits

Section 5.1 specifies neither the exact positive floor nor the optimizer.
The reconstruction documents its chosen chi-square-tail floor, initialization,
solver and first-full-window interpretation of the early doubling rule.
New simulated observations follow the published model, but original-author
observations and random seeds were not recovered.

A bounded public-source check inspected the
[Warwick paper record](https://wrap.warwick.ac.uk/79711/),
[Anthony Lee's publication list](https://awllee.github.io/publications.html),
his linked public repositories, and the comparator's repository/owner/commit
metadata. The inspected paper entry links the article and arXiv version, not
numerical replication code. The comparator metadata does not establish
original-author identity. GitHub repository/user searches did not recover an
author implementation. These checks are not an exhaustive proof of absence;
responses are saved in `source-search/`. General web search was unavailable
(HTTP502); direct retrieval of these sources succeeded.

The artifact audit found and repaired a source-snapshot basename collision:
the R algorithm and its test share a filename. Their pre-run hashes were
recorded separately. Three recovered algorithm versions now match those exact
hashes; all other saved sources and observation hashes also verify. Historical
snapshots remain intact. `source-recovery/` documents the recovery and the new
driver preserves repository-relative paths, with an executable regression.

All runs were explicit CPU-only independent-reference exceptions using base R
4.1.2, hidden GPU devices and one BLAS/OpenMP thread. Every serious attempt has
its command, source hashes, Git commit, environment, seeds, elapsed time, data
hashes and result location in `attempt*/manifest.json`. Seven of eight launch
slots consumed 248.38 of 1800 R-worker seconds. Remaining: 1551.62 seconds and
one launch slot. Compute exhaustion is not the stopping reason.

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept tested filter mechanics as an independent diagnostic | Exact controls pass through d80 | No control failure | Broader nonlinear models untested | Use identities and fixed-twist controls for narrowly scoped debugging | Canonical or score correctness |
| Reject current fitted reconstruction as a paper replication | No completed 32-repeat batch | Reproducible variance-boundary failures | Author fitting/floor/controller details | Recover those details or explicitly specify a different fitting method; test saved failures first | iAPF theory is invalid |
| Hold scaling to 1000 repeats | Fitting reliability prerequisite failed | Repeated failure, not cost | Whether another documented local solver suffices | Resume only after a validated repair and fresh pilot | Paper tables reproduced |

| Inference status | Finding |
|---|---|
| Hard veto screen | Current fitted implementation hits its variance boundary in both d5 and d10 renewed batches |
| Statistically supported ranking | None |
| Descriptive-only differences | Successful ratios, SDs, runtimes, particle counts and conditional heuristic errors in tables |
| Default readiness | Ineligible independent R reference; no TensorFlow, LEDH, nonlinear-score or HMC promotion |
| Next evidence needed | Reliable documented fitting on saved failures, fresh pilot, complete untouched repeat batches, then published-scale replication |

Post-run skeptical review: the strongest alternative explanation is the
unspecified local numerical procedure, rather than a defect in iAPF's exact
importance construction. A documented author solver/floor/controller that
survives these inputs would overturn the reconstruction's practical failure.
The weakest evidence is empirical performance: only successful subsets are
available, so their small SDs cannot support a claim. Mathematical controls
remain useful, while fitted replication remains open.
