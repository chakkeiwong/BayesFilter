# Resolve the remaining first-study numerical specification

Owner instruction, 2026-09-21: refresh the master, review it, and execute.
This resumes the same first linear-Gaussian study within the existing 24-hour
allocation. It does not authorize the later PMMH/SV studies or change a default.

## Research intent and evidence contract

Question: can explicit local numerical solutions of the printed Equation15
produce a useful first-study reconstruction, and which remaining discrepancies
are fitting, controller, diagnostic-target, or timing issues?

Equation15 fits a normalized diagonal Gaussian density p(x;m,V) to a free
multiple lambda of backward targets b. Compare two local solvers for exactly
that residual: jointly optimize (m,log V,log lambda), or analytically eliminate
lambda and optimize (m,log V). Use the same base-R nlminb solver, fixed loss and
coordinate scaling, two explicit initializations, and the same positive floor.
These are plausible reconstructions of unspecified numerical choices, not
verified author implementations or guaranteed global minimizers. QR remains
the explicit different-objective comparator. The earlier failed box1 method
is preserved and will not be silently renamed or retuned to table entries.

Primary scientific endpoint: terminal likelihood relative to exact Kalman on
fresh data, with mean-ratio 95% interval inside [.8,1.2], SD upper interval <=
twice the paper SD, and mean particle count <=1.5 times the paper count. These
are the existing practical criteria, not formal equivalence to the author run.
Missing records, invalid factors, nonfinite states/weights or corrupted source/
data identity invalidate evidence. Optimizer nonconvergence or a failed arm
rejects that candidate and triggers the next specified reconstruction/repair.
It does not stop the entire investigation. A solver convergence code is not
evidence of a useful guide or of an attained global minimum.

Constructed simple comparators: BPF10,000 (unguided resampling), FA-APF5,000
(optimal one-step Gaussian proposal), SIS10,000 (no resampling); exact Kalman
and exact future twists provide tractable authorities. Evaluate ordinary and
large Kalman innovations separately, using the existing chi-square90% cutoff.
Observed conditional prefix loss remains a conservative project promotion
veto, not a paper-replication criterion or a statistically supported ranking.
Fit residual, guide shape, amplitude escape, conditioning, time-point/replica
concentration and runtime are explanatory; they cannot replace terminal checks.

No claim of author identity, global optimization, superiority, production,
LEDH/KDM/gradient/HMC validity, or matched cost follows from a small experiment.
Exact original data/settings and the final publisher technical text remain
unavailable; independent fresh datasets support a statistical reconstruction,
not exact numerical reproduction of the authors' realization.

## Ordered execution and repairs

1. Verify actual consumer wiring, Eq15 profile/joint algebra and gradients,
   Gaussian identities, and source anchors. Add focused tests for any new
   solver, including an exactly representable diagonal target and the existing
   correlated-target amplitude-escape counterexample. No new package required.
2. Analyze the saved d80 prefix errors: verify all records against the completed
   report, identify contributions by time and replica, and preserve all values.
   Derive why an exact future guide can have a deterministic terminal estimate
   yet noisy untwisted prefixes; demonstrate on a bounded independent fixture.
   This investigates the existing veto without changing it retrospectively.
3. On fresh paper-model fixtures (d=5,20,80; t=1,50,100; two fixed data seeds),
   compare joint/profiled nlminb from QR and target-weighted-moment starts.
   Use exact backward targets to isolate fitting from recursion. Report raw
   Eq15 loss, relative residual, Gaussian-component KL, log-determinant change,
   stationarity, boundary and convergence status. Preserve all candidates.
4. Use both QR-start solvers in fresh full-filter probes at d=5 and20, with
   paper six-estimate stopping, delayed doubling, N0=1000, T=100 and floor8.
   A returned iteration-limit code may receive one bounded maxit600 retry
   after the original maxit200 attempt is preserved; other candidate failures
   proceed to the alternative initialization, not indefinite retries.
   The moment-start versions are the prespecified repair if QR-start versions
   are invalid or fail the practical screen. All probes use separate data from
   stage3 and from previous campaigns. QR/BPF/FA/SIS run on the same data.
5. If an Equation15 candidate completes the d5 and20 probes, freeze it (joint
   before profiled; QR start before moment start, a predeclared deterministic
   order, not a superiority ranking) and run16 paired replications on each of
   two untouched d20 datasets. Include the four comparator methods. If no
   candidate completes, preserve that result and finish the mathematical and
   implementation diagnosis; do not spend remaining time repeating failures.
6. Fix timing boundaries in the reusable comparison runner. Record algorithm
   time (all fitting/controller/final filtering), diagnostic/I/O time, and total
   measured time separately. Run a sequential paired timing check on fresh d5
   and20 data, without concurrent R workers. No speed ranking from a few runs.
7. Assemble terminal result, uncertainty, decision and inference tables, review
   the strongest alternative explanations, and update master/checkpoint. A
   failed local reconstruction closes this bounded investigation, not the
   broader iAPF research direction. Do not silently expand into the whole paper.

## Defaults and assumption audit

| Choice | Provenance/status | Risk and earliest check |
|---|---|---|
| Normalized diagonal Gaussian + free scale | Paper Eq15 | Loss can escape to zero by losing amplitude; profile/joint identity and amplitude diagnostics |
| nlminb, maxit200 then one600 retry | Base-R plausible local solver; previous fit budget | Premature termination; code, normalized gradient and iteration trace |
| QR / target-weighted moment starts | Existing independent-reference initializations; hypotheses | Start-dependent minima; paired fixed-cloud comparison |
| Fixed scaling | Multiplication by a positive constant and invertible coordinates | Incorrect gradient chain rule; finite differences and direct raw residual equality |
| Floating-point variance limits | Existing +/-(-log double epsilon) numerical domain, not shape prior | Bound-selected answer; fail/flag, preserve candidate failure |
| Floor tail power8 | Existing frozen reconstruction hypothesis, not specified by paper | Tail/shape dependence; preserve floor and report its provenance; no silent floor tuning |
| Six likelihoods, k5, tau.5, kappa.5 | Printed Algorithm4 and implementation section | Early doubling ambiguity; retain explicit delayed convention, separate from window5 |
| Fresh data and fixed seeds | Defined by driver before launch | Data leakage; manifest partition identities and no tuning on validation |
| Small counts | Bounded diagnosis within remaining allocation | Rare weights and uncertain ranking; intervals and descriptive-only verdicts |

## Budget, environment and artifacts

Output: docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01.
Same campaign deadline: 2026-09-21 20:04:26 UTC. Previous use:107291.715764
of172800 worker-seconds; remaining65508.284236. This amendment allocates at
most10000 additional process-seconds (including failed attempts and focused
checks), at most two simultaneous single-thread R workers, at most200 launches.
It does not reset either prior usage or the elapsed deadline. Read-only source
and saved-result inspection does not launch numerical workers. Driver enforces
the earlier of this local cap and the existing deadline; failed candidates
consume budget. Stop if provenance cannot be repaired, the deadline/cap is
reached, or a necessary change crosses the scientific/hardware/privacy scope.

R4.1.2, CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1; CPU independent reference. Python reporting may use NumPy
only in explicitly diagnostic scripts. No TensorFlow production path changes.
Save git commit, source snapshots/hashes, exact commands, environment, seeds,
per-attempt wall time, failures, budget accounting, and report paths. Every
attempt has a new directory. The implementation/driver records exact commands
before running them; reports are versioned and historical artifacts preserved.

Skeptical pre-execution review: PASS after revising the proposed next steps.
The original suggestion risked treating a prefix veto as a paper criterion,
using unevenly instrumented runtimes, and selecting a fit because its training
loss fell. The revised sequence separates these questions, verifies both
parameterizations of the actual residual, retains failures, uses fresh full
filters and simple baselines, and reserves untouched validation. It explicitly
admits that local solver choices cannot recover unspecified author settings.
The most serious alternative explanation is structural amplitude escape rather
than a coding error; the earliest exact-target fixtures distinguish them.

## Executed outcome, 2026-09-21

Complete: tests, fixed-cloud comparisons, saved-prefix audit, exact-guide
identity/variance checks, full-filter repair ladder, sequential timing and
terminal skeptical review. All four Eq15 variants failed the probes; the
conditional validation phase correctly did not launch. No worker error occurred.
The amendment consumed 98.081 process-seconds in 59 launches.
[Result and decision](artifacts/iapf-r-equation15-resolution-20260921-01/result.md)
records the numerical failures and remaining source-specification dependency.
This is candidate rejection, not an iAPF failure theorem or a permission stop.
