# Frozen positive-floor R iAPF validation

2026-09-20. Owner instruction: refresh the program and continue execution.
This is the next bounded local CPU reference campaign after the completed
floor repair. Previous campaign evidence and budgets remain closed.

## Question, candidate and decision

Does the optional log-quadratic R iAPF, with the previously calibrated positive
floor power fixed at 8, retain conditional likelihood accuracy on two untouched
80-dimensional linear-Gaussian datasets? A fresh d40 dataset is a regression
control. All datasets have T=100. The exact comparator is the Kalman marginal
likelihood on the same observations, not training fit or a previous random run.

Each dataset has 32 independently seeded complete filter repetitions. The
primary screen requires its 2,000-resample percentile bootstrap 95% interval
for the mean likelihood ratio to lie wholly inside [0.9,1.1]. Require all three
datasets separately; never pool away a failure. This is a conditional numerical
accuracy screen, not a proof of unbiasedness, uniform accuracy or a joint 95%
confidence statement. The controller performs a fresh final filter draw after
stopping; the reported estimate is not the draw used to decide convergence.
Adaptation changes the learned proposal and its variance. A small empirical
interval cannot by itself establish unbiasedness or adequate tail coverage.

Promotion here means eligibility for the next independent-reference validation
stage only. The alternative squared-log fitting objective differs from paper
Eq.15. No published-result replication, original-author implementation claim,
default change, equal-cost ranking, nonlinear result, score/HMC/KDM/LEDH result
or TensorFlow production admission follows.

## Research intent and diagnostic roles

Expected failure: a small floor may still dominate on rare ancestors, or finite
cloud coverage/controller selection may leave inaccurate or variable likelihoods.
Primary criterion: the separate 32-repeat Kalman intervals above.
Promotion vetoes: nonconvergence/cap, invalid fit, nonfinite result, failed
Gaussian-limit tail check, or observed conditional heuristic underperformance.
An accuracy or heuristic failure rejects this candidate scope; it does not
invalidate the other untouched dataset or reject the research direction.
Continuation vetoes: source/data mismatch, corrupt or missing required output,
broken probability identities, or exhausted budget. A localized infrastructure
failure triggers repair and retry within the same budget, preserving the failure.
Explanatory diagnostics: particle counts, iterations, ESS/floor diagnostics
already emitted, fit conditioning, variance and runtime; none substitutes for
the Kalman screen. No tuning or replacement dataset is allowed after seeing
these results. A failure is followed by saved-output diagnosis, not retuning.

Constructed heuristic adversaries: bootstrap PF with 10,000 particles (plain
resampling), fully adapted PF with 5,000 (exact one-step Gaussian proposal),
and sequential importance sampling with 10,000 (no resampling). Compare mean
squared prefix log-likelihood errors separately at ordinary observations and
large innovations, defined in advance by the Kalman innovation exceeding the
dimension-specific chi-square 90th percentile. Any observed iAPF loss against
any of these is a promotion veto, with sampling uncertainty acknowledged.
Budgets are unequal, so no cost-efficiency claim is permitted. Exact Kalman is
the accuracy authority and is not a stochastic heuristic to be beaten.

Tail check on each final frozen guide at each time: the Gaussian-only limiting
importance second moment has quadratic matrix M=Q^-1+2H-V^-1, where
H=C'R^-1 C+A'(Q+V_next)^-1 A (omit the future term at T; use initial covariance
at time 1). Require lambda_min(M) divided by
||Q^-1||+2||H||+||V^-1|| > 100*d*machine_epsilon. This conservative veto tests
the checked Gaussian limit, not uniform particle-system variance. The exact
positive-floor model remains the executed algorithm.

## Frozen defaults and assumptions

| Choice | Provenance / justification | Failure mode / early check | Status |
|---|---|---|---|
| Diagonal log-quadratic QR fit | Previous independent R repair; solves the declared squared-log objective | Off-cloud error despite small residual; exact Kalman comparison and fit rank | Optional hypothesis, not Eq.15 |
| Positive floor power 8 | Prior fixed-guide mixture calibration; first grid value below 1e-8 disturbance | New data can change overlap; conditional errors and tail checks | Frozen candidate; no retuning |
| N0=1000, k=5, tau=.5, kappa=.5, first_full_window; 20-pass/16000-particle caps | Existing checked controller, held fixed to isolate the floor | Proposal variability or cap failure; inspect every repetition | Baseline reconstruction with known author ambiguity |
| T100, d80/d40, data model | Existing paper-study reconstruction | Only this model may suit a log-quadratic guide | Reference scope only |
| 32 repetitions and 95% bootstrap inside [.9,1.1] | Existing master accuracy screen; bounded precision | Rare failures missed / bootstrap coverage; report ranges and all failures | Screening criterion, not asymptotic guarantee |
| Fresh seeds listed below | Absent from inspected prior campaign manifests/settings | Accidental reuse; source/data hashes and method seed validation | Fixed convenience choices, no selection |
| Base R CPU, one BLAS/OpenMP thread | Explicit independent-reference exception, R4.1.2 | Runtime contention; bounded jobs and no timing ranking | Reference backend |

## Execution and budget

Versioned output root:
`docs/plans/artifacts/iapf-r-positive-floor-validation-20260920-01`.
Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, base R4.1.2;
`CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
At most two independent R workers concurrently. Total worker budget 4,000
seconds: replication allowance 3,800 and post-run tail/report allowance 200;
separate mechanics/test allowance 120 seconds. Three scientific launches and
at most one localized infrastructure retry within the same time budget.
Prior d80 measurements are about 36 seconds/repetition. These ceilings allow
bounded contention without changing particle or adaptation limits.

| Attempt | Dimension | Data seed | Repetition IDs | Timeout seconds |
|---|---:|---:|---|---:|
| attempt01-d80-a | 80 | 80000080 | 501..532 | 1500 |
| attempt02-d80-b | 80 | 81000080 | 601..632 | 1500 |
| attempt03-d40-control | 40 | 80000040 | 701..732 | 600 |

Use `docs/benchmarks/run_iapf_r_replication.py --campaign floor_validation
--mode replication --fit-mode log_quadratic --floor-power 8` with the table's
`--dimension --data-seed --first --repeats 32 --timeout --output` arguments.
Method seeds remain 53000000+dimension*100000+replication*10+method_id; IDs and
data are disjoint from prior calibration. Execute captured source dependencies,
preserve commands, commit, environment, seeds, CPU choice, hashes and wall times.
Post-run use the captured `diagnose_iapf_r_validation_tails.R` and the reporting
validator; retain per-time margins and machine-readable conditional verdicts.

## Skeptical review and pre-mortem

Self-review before implementation/execution: PASS with the stated limits.
The baseline is exact and on the same data; proxy metrics cannot promote.
Two fresh d80 datasets reduce easy-data dependence without assuming generality.
The d40 control also uses 32 repeats to avoid declaring accuracy from a pilot.
The frozen controller takes a fresh final draw after stopping; even so, finite
repetitions can miss rare weights. Heuristic costs differ and cannot support an efficiency ranking.
Captured-source execution, exact row/seed checks and mutation tests address
harness validity. The corrected phase logging and child-process timeout code
are already tested. No source/paper reinterpretation or default changes are
needed. A seemingly good result could miss a rare failure; a timeout could be
resource contention rather than a mathematical failure. Preserve these distinct
classifications. Review results only after each complete dataset; progress
counts do not permit early accuracy stopping or candidate selection.

## Closeout

Write `summary.json`, `result.md`, run/report manifests and a concise checkpoint;
refresh the master program. Include decisions, primary/veto status, inference
status, remaining uncertainty and next action. If all screens pass, the next
step is a source-grounded plan for the remaining paper-objective gap and the
TensorFlow comparison; this campaign alone does not close paper replication.
If a screen fails, first inspect the saved guides and separate floor recurrence,
controller/proposal variability, coverage failure and sampling uncertainty.
