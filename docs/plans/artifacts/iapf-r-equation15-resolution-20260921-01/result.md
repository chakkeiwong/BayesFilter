# Equation 15 resolution: completed bounded investigation

The amendment is complete. All four new combinations of Equation 15
parameterization and starting point fail the full-filter probes, including the
specified larger-iteration retries. The numerical fitting gap remains open.
The prefix investigation is resolved more precisely: a rare-error-dominated
prefix veto cannot establish failure of the paper's terminal-likelihood task.
The earlier QR reconstruction also retains its observed heuristic veto; a new
two-replica d=5 probe has greater ordinary-prefix error than the fully adapted
filter. None of these observations supports an overall statistical ranking.

## Execution and findings

The reviewed [plan](../../iapf-r-equation15-resolution-2026-09-21.md) ran on CPU
with GPU devices deliberately hidden, R 4.1.2, and at most two single-thread
workers. The paper model has T=100, with exact Kalman likelihoods. Source snapshots,
seeds, commands, hashes, attempts and numerical failures are preserved. No worker
or harness error occurred. Total charged worker time is 98.081 seconds over 59
numerical/test launches, including focused tests; the allowance was 10,000 seconds and
200 launches within the original September 21 20:04:26 UTC deadline. Previous
campaign use remains charged separately in `manifest.json`. No worker is running.

| Completed action | Evidence and outcome |
|---|---|
| Joint/profiled Eq15 algebra and actual endpoint checks | PASS: profile identity, analytic gradients versus finite differences, exact Gaussian, scaling, escape direction and controller wiring |
| Fixed-cloud fitting | 72 fits on two datasets at d=5/20/80 and t=1/50/100; 35 returned successfully and 37 failed; tiny residual sometimes accompanies a badly wrong guide shape |
| Full-filter fitting and repairs | 16 initial attempts plus 16 retries; 0 completed: 25 non-convergence, 5 numerical-boundary and 2 density-underflow outcomes |
| Same-data simple comparators | All 16 QR/BPF/fully adapted/SIS probe records completed; prefixes and required diagnostics are present |
| Untouched Eq15 validation | Correctly not triggered: no candidate completed the required probes; no failed arm was silently selected |
| Saved d=80 audit | All 1,000 parent hashes and previously reported conditional means reproduced; one replica supplies 84% of the six-estimate arm's ordinary-prefix error |
| Exact-guide diagnostic | 16 runs; terminal log error≤5.46e-12, 1,600 prefix identities accurate to 1.06e-11; analytic prefix variance can still be large |
| Timing repair | Reusable comparison runner separates algorithm time from extra diagnostics; 16 sequential timing records completed |

The full-filter fitting variants use joint or analytically profiled scale,
QR or target-weighted-moment starts, maxit 200 and one maxit 600 retry, the same
positive-floor hypothesis, and the paper's six-estimate controller. QR-start
attempts fail at backward time 99, immediately after fitting the final-time
Gaussian. Moment starts can fail even at time 100. These are rejected local
implementations of the unspecified numerical minimization, not a rejection of
the iAPF research direction or evidence that the authors had these failures.

[The mathematical explanation](mathematical-findings.md) derives both findings.
For Eq15, profiling the free scale gives
L*=||p||²−(pᵀb)²/(bᵀb). Letting the Gaussian spread out, or moving its mean away
from the cloud, sends all sampled density values toward zero and also sends
this loss toward zero. Correct derivatives therefore do not make a tiny loss
a useful-guide certificate. For exact future guidance, the final weights are
constant but a prefix requires p(filter)/p(smoother) correction weights, which
can have very large variance.

## Timing and comparison limits

Times below are means of two runs on fresh fixed data, in seconds. R workers
were sequential during this check; the machine was not certified exclusive.
These measurements remove the known outer diagnostic-writing imbalance. They
do not equalize statistical accuracy, optimize the R implementation, include
all process startup/final CSV writing, or establish a performance ranking.

| d | QR algorithm | QR extra diagnostics/I/O | BPF algorithm | Fully adapted algorithm | SIS algorithm |
|---|---:|---:|---:|---:|---:|
| 5 | 1.0980 | 0.4475 | 0.2740 | 0.1150 | 0.2095 |
| 20 | 3.0870 | 0.4085 | 0.9045 | 0.4195 | 0.8415 |

The known reporting overhead did not explain the whole observed time difference.
It remains wrong to infer the paper's matched-cost performance from these two
reference timings. The timer change leaves filter calls and random-seed recipes
unchanged; its diff against the previous frozen source was inspected.

## Decision and inference

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject the four new local Eq15 routes for scaling | No full-filter candidate reached terminal validation | Non-convergence, domain boundaries and underflow | Author optimizer, initialization, stopping and floor choices | Recover that numerical specification before attempting an author-procedure replication; any new bounded fit needs its own explicit hypothesis and early downstream screen | All possible Eq15 procedures fail, or original iAPF fails |
| Retain QR as a different independent reference | Earlier large study remains preserved; this amendment does not close objective identity | Existing d=80 particle/prefix vetoes; descriptive d=5 probe loss to fully adapted filter | Rare weights, data dependence and different fitting objective | Use only under its stated reference role; do not relabel as paper Eq15 | Author reproduction, default readiness or superiority |
| Keep terminal and prefix claims distinct | Exact-guide identity checked analytically and numerically | Existing frozen prefix veto unchanged | Learned-guide tails differ from exact-guide tails | State separate terminal-likelihood and filtering criteria in the next plan | Prefix evidence can be discarded or retroactively relabeled as passing |

| Inference status | Result |
|---|---|
| Hard veto screen | All new Eq15 variants fail validity before terminal validation; no missing evidence or worker crash |
| Statistically supported ranking | None established by this amendment; the old vetoing d=80 difference intervals contain zero |
| Descriptive-only differences | Training residual, guide KL, concentration by replica/time, small-probe errors and timings |
| Default-readiness | No algorithm, controller, numerical setting, or production default promoted |
| Next evidence needed | Recovered or explicitly newly specified fitting procedure, followed by fresh downstream validation; matched-accuracy timing only after a viable method exists |

## Terminal skeptical review

PASS for these bounded conclusions; FAIL for a full paper-replication claim.
The strongest alternative explanation for the fitting failures is our choice
of local solver, start and termination rules. The tests exclude several algebra
and call-chain errors but cannot establish author identity. A documented author
procedure that passes fresh full-filter probes would overturn the judgment that
the fitting specification remains unresolved. No amount of additional unchanged
QR replication resolves that difference.

The prefix derivation concerns an exact full-covariance, zero-floor guide.
Applying its exact numerical variance to the learned diagonal-plus-floor guide
would be wrong. The derivation instead demonstrates that terminal accuracy and
prefix accuracy are different demands. Ordinary/large-current-innovation
conditioning is also incomplete for a future-guided method; future information
helps explain the rare weights. Existing thresholds and results were not changed.

The unchanged first-study replication gaps are the original numerical settings,
the realized author data, remaining table-pattern discrepancies and a proper
matched-accuracy cost comparison. Later alpha-grid, PMMH and stochastic-volatility
experiments are outside the authorized first-study scope. Multidimensional full
R/TensorFlow parity remains an integration task; the current TF comparison
consumer accepts only one-dimensional state and observation. No LEDH, KDM,
gradient or HMC correctness/failure claim is made.

Machine-readable evidence: [summary](summary.json), [manifest](manifest.json),
[attempts](attempts.jsonl), [verification](result-verification.log),
[prefix audit](analysis-v1/prefix-concentration.json),
[fit failures](analysis-v1/fit-failures.csv), and [source snapshot](source-v1/sha256.json).

A final [bounded public-source recheck](public-source-recheck.md) recovered no
new verified author implementation; the relevant GitHub hit is the already
audited public reference with unverified original-author identity.
