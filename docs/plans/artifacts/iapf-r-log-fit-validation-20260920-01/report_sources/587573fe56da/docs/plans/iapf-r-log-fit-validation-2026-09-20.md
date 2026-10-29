# Frozen R log-fit validation

2026-09-20. Owner request: continue execution. This is a new bounded local
CPU allocation following the completed log-fit repair. Earlier results and
budgets remain closed. The independent R reference is explicitly authorized;
it is not the TensorFlow/GPU production implementation.

## Question and evidence contract

Does the frozen diagonal log-quadratic fitter continue to produce accurate
likelihood estimates on fresh linear-Gaussian datasets at d20, d40 and d80,
T100? The scientific target is the exact Kalman likelihood. Retain the
previously checked proposal, weights, backward recursion and controller.
The fitter minimizes unweighted squared log-target error using QR. It differs
from paper Eq.15; a successful run cannot establish reproduction of the
paper's fitting procedure or published 1,000-repetition results.

Primary d20 screen: for **each** fresh dataset, 32 complete independent filter
repetitions and a 2,000-resample percentile bootstrap 95% interval for the mean
likelihood ratio entirely within [0.9,1.1]. These are conditional-on-data screens,
not an unconditional theorem or a method ranking. Never pool datasets to hide
a failure. Higher-dimensional four-repetition pilots check feasibility and
obvious errors only; their likelihood ratios must be finite and within [0.1,10].
Their intervals and any optional follow-on intervals are descriptive until a
32-repetition validation is completed. No tuning on these datasets.

Hard validity/continuation vetoes: source/data mismatch, duplicate or missing
required rows, incomplete run, nonfinite log likelihood, rejected QR fit,
nonconvergence, variance boundary, or a violated probability identity. Preserve
partial output but do not interpret a successful subset as a complete sample.
Infrastructure failures trigger a localized repair/retry within this budget.
Scientific fit or accuracy failures reject that candidate in that scope and
trigger examination of the saved failing fit or prefix errors; they do not
reject iAPF generally or authorize changing its target silently.

Constructed heuristic adversaries, with fixed paper-study particle budgets:
BPF N10000 (prior proposals and resampling), fully adapted filter N5000
(exact one-step Gaussian observation conditioning), SIS N10000 (prior proposals
without resampling). Kalman is the exact certifying comparator. Compute each
method's mean squared log-prefix error separately for ordinary and large
innovations; large means Mahalanobis square above chi-square(d) 90th percentile.
iAPF losing to any particle heuristic in either situation vetoes advancement
in that scope. No runtime or variance ranking is asserted from unequal budgets
and descriptive means. ESS, fit condition numbers, training residuals and
runtime explain behavior; they cannot substitute for Kalman agreement.

## Frozen assumptions and smallest checks

| Choice / provenance | Status and justification | Risk / early check |
|---|---|---|
| Diagonal QR log fit, previous repaired code | Explicit alternative hypothesis; avoids density-amplitude loss degeneracy | Nonconcave/rank-deficient regression or poor off-cloud fit; strict guard and Kalman prefixes |
| No curvature clipping, ridge or optimizer fallback | Derived Gaussian admissibility requires negative quadratic coefficients; QR solves the declared objective without changing it | Reject and save context if inadmissible; exact Gaussian and affine-invariance tests |
| N0=1000, k=5, tau=.5, kappa=.5, T100 | Paper-study baseline retained, not retuned | Controller may exhaust resources; record iterations and final N |
| Floor power 2, first-full-window doubling | Frozen independently documented interpretation, not verified author settings | Changes floor mixture/controller; retain source tests and explicit non-replication qualification |
| New deterministic seeds below | Untouched validation convenience, fixed before execution | Dataset dependence; two d20 datasets, separate results |
| CPU R4.1.2, single BLAS/OpenMP thread | Independent-reference exception | Environment drift; manifest and captured-source execution |
| Bootstrap tolerance [0.9,1.1] | Existing bounded validation screen, not exact unbiasedness proof | Small-sample/heavy-tail uncertainty; report ranges, SD and repeated-data limitations |

## Execution and budget

Output root: `docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01`.
At most **1800 aggregate R-worker seconds and 8 launches**, including failed
attempts. At most two simultaneous workers; planned execution is sequential.
Separate mechanics/reporting allocation: 120 seconds. No GPU, paid compute,
package changes, external publication, or production-policy changes.

All runs use `--campaign validation --fit-mode log_quadratic`, floor power2,
first-full-window doubling and the existing T100 reference runner. Run using
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/run_iapf_r_replication.py`; the driver captures and executes its
R dependencies with CUDA_VISIBLE_DEVICES=-1 and single-thread BLAS/OpenMP.
Each output directory is new and includes command, source hashes, seeds,
environment, CPU status, wall time and all numerical results.

| Order | Mode | d | Data seed | Repetition IDs | Timeout ceiling (seconds) |
|---|---|---:|---:|---|---:|
| 1 | replication | 20 | 70000020 | 201:232 | 400 |
| 2 | replication | 20 | 71000020 | 301:332 | 400 |
| 3 | pilot | 40 | 72000040 | 1:4 | 300 |
| 4 | pilot | 80 | 72000080 | 1:4 | 600 |
| 5, if valid and budget fits | replication | 40 | 73000040 | 101:116 | 600 |
| 6, if valid and budget fits | replication | 80 | 74000080 | 101:108 | 800 |

Method seed = 53000000+d*100000+replication*10+method_id (1:4 in the order
iAPF/BPF/fully-adapted/SIS). Bootstrap seed = data_seed+900. Advance after
completeness, fit validity, the applicable accuracy screen and heuristic checks.
Optional follow-on timeout is ceil(1.5*pilot_seconds*new_repeats/4+20), capped
by its ceiling; skip if this reservation exceeds remaining total budget. Do not
lower a reservation merely to squeeze a run into the budget. Up to two unused
slots may repair infrastructure or inspect a saved scientific failure within
remaining time, without altering the frozen fitting method. Any actual
scientific repair needs a revised evidence contract before execution.

## Skeptical review before implementation/execution

The main baseline is exact Kalman, and neither training loss nor pilot success
is a promotion criterion. Data/method seeds are fresh. The prior 16-repetition
screen does not establish author replication or high-dimensional accuracy.
The four-repetition high-dimensional pilots cannot support ranking. All
source dependencies execute from their captured versions. The remaining
material harness defect is repeated rewriting of every prior fit/prefix row:
it adds quadratic reporting overhead and can cause a false timeout at 32
repetitions. Repair reporting to append each newly completed method once;
verify identical CSV content/schema and unchanged random-stream behavior with
an executed runner regression before launching. The numerical core stays
frozen. With that repair checked, this plan answers the stated question within
bounded compute. The strongest alternative explanation of a favorable result
is easy fresh data or log-fit suitability only for linear-Gaussian models;
neither nonlinear nor score/HMC/KDM/LEDH claims follow.

## Required closeout

Preserve `summary.json`, `result.md`, a report manifest and concise checkpoint
under the new root; refresh the master program. Report each dataset, all failed
attempts, conditional heuristics, hard vetoes, statistical versus descriptive
evidence, actual/remaining budget, next justified action and strongest
alternative explanation. Full paper-scale replication and author-setting
ambiguities remain separate gaps.

Preflight result: the saved previous runner and append runner executed two d5
T100 repetitions on seed75000005. All CSV rows and numerical fields matched
exactly (8 method rows, 1,200 fits, 800 prefixes; measured timing excluded).
This mechanics check took 12.052 seconds. The seven existing pytest cases also
pass. Source: `artifacts/iapf-r-log-fit-validation-20260920-01/mechanics01-append-parity`.
Skeptical audit passes for the frozen campaign; no numerical-core change.
