# Marginal LEDH and original Zhao–Cui: matched horizon comparison

Active question: does replacing the generating-ancestor correction by the
all-ancestor mixture correction reduce likelihood and score errors for
predator–prey and SIR d=18 at T=10,20,40,50? The engineering question is whether
all consumers call the shared analytical implementation. The replication
question is whether the pinned author TT-cross/smoother can supply a matched
conditional reference at each horizon. These are separate questions.

## Evidence contract and research intent

Use the canonical x0-then-transition-observe targets, nominal parameter points,
one fresh T=50 dataset per model (seed 26100611), and exact saved prefixes.
The main diagnostic uses N=1008, iid_dual_cap, guarded_pairwise, four paired
design seeds 261006101–104, GPU FP64/XLA, Contract E, one annealing stage.
Compare ancestor and marginal_mixture with every other control held fixed.
Report each likelihood and every score component, differences, replication
standard errors, and distance to the matched independent references.

The primary criterion for proposing longer validation is reduced absolute
likelihood and componentwise score error against references stable across
particle/rank increases and independent fits. Four seeds and one dataset support
descriptive conclusions only; they do not establish superiority or a default.
Finite values, target/data/chart equality, analytical score consistency and
source parity are vetoes. A candidate numerical failure rejects that candidate;
it does not invalidate the other arm or stop a planned discriminating repair.
Only target mismatch, corrupted evidence, exhausted budget, or an unresolved
implementation defect stops the affected comparison.

Expected failure: SIR mixture tails may be unstable, reset/correction error may
dominate, or an inaccurate TT proposal may produce low importance ESS. Repair
triggers are invalid numerics, source callback mismatch, and insufficient
importance overlap. ESS, moment diagnostics, runtimes, and quadratic residuals
explain results; none alone establishes likelihood/score correctness.

Constructed heuristic adversaries: (1) bootstrap PF N=1008, the simplest valid
proposal for these Gaussian state equations; (2) ancestor LEDH with covariance
restoration only, testing whether higher moment fitting helps; (3) the original
ancestor LEDH with the same full corrections. Evaluate separately by model and
horizon, never pool SIR and predator–prey or short and long horizons. Underperformance
vetoes promotion. High-particle bootstrap/Fisher estimates and the original
TT-cross/smoother are approximate references, not exact oracles. No HMC,
production-readiness, paper-number replication, or cross-model tuning claim.

## Reference definitions and source audit

The original author tree is immutable. Zhao–Cui (JMLR 2024), equation (26) and
Algorithm 4 (pp.18), specifies backward path proposals and importance correction.
The pinned implementation is models/full_sol.m:139–206; lines191–200 form raw
log weights and save mean(log weight) as lml. For likelihood use
logsumexp(raw_log_weight)-log(number of paths), retaining zero weights in the
denominator. Preserve raw author lml as a separate diagnostic. Conditional
fixed-parameter callbacks use the author TT fitting/smoothing route; the changed
state-model callbacks are explicitly extension_or_invention_fixed_target; paper versus released-driver dynamics are not interchangeable (sections
6.3–6.4; eg3_sir/mainscript.m and eg4_predatorprey/mainscript.m).

Fit at T=50 and save smoothing samples at 10,20,40,50. Incremental smoothing
restores the fitter RNG, so each checkpoint uses only its prefix fit. This
observability extension is not a change to TT-cross. The score reader must check
the whole saved dataset hash, then use the exact prefix and horizon-specific
proposal. Verify this call chain with focused tests. Allow T>20 only for the
explicit current_target profile; retain published-replication bounds otherwise.

Use PP rank20 and SIR rank40 as previously diagnosed reference hypotheses,
fit seeds2,17,29 where the budget permits, 100000 smoothing paths. Quadratic
regression uses antithetic points, heldout points, and at least two shrinking
radii; select no radius using the LEDH answer. Compare conditional path-sampling
uncertainty and between-fit uncertainty separately. Repeat high-particle
bootstrap/Fisher at N=32768,131072 with four seeds if the pilot supports cost.
Neither a quadratic residual nor a small MCSE measures reference bias.

## Defaults, audit and bounded execution

Inherited LEDH controls come from run_ledh_nonlinear_master.py BASE; they are
untuned warm starts, not reviewed scientific defaults for these eight scopes.
Their purpose is to isolate the weight formula. Guards and caps remain enabled;
check finite trajectories and moment diagnostics in the N=144 T=10 pilot.
N=1008 obeys repository chunk policy K=N; N=144 is mechanics only. FP64 is a
reference exception to the production TF32 target, justified by matched arithmetic.
Nominal parameters are exact FP64 model constants. Data and random seeds are
convenience choices fixed before results; a single trajectory limits inference.
Rank20/40 and small regression radii inherit explicit prior overlap diagnostics,
remain hypotheses, and require target/horizon-specific overlap and residual checks.

Skeptical audit: corrected two invalid baselines before execution—raw author
mean log weights is not log likelihood, and old T=20 results use other data.
Matched prefix data and parameter charts are required. CPU-only author code
is an independent reference exception; GPU filters record memory growth and
XLA. No AD score, pfor, runtime retuning, or paper-profile relabeling is allowed.
The plan passes as a diagnostic comparison, not as scope-specific admission.

Authorized extension budget: 172800 aggregate job-seconds (48 hours), charged
for every subprocess attempt, including failures. Provisional allocations:
LEDH 21600, bootstrap 14400, author fits 86400, quadratic scores28800,
repairs21600 seconds. Reallocate within the total without expanding the target.
At most one GPU worker and two CPU author workers concurrently; thread limits2.
Every launch has a fresh numbered directory, timeout, full log, command,
Git/source hashes, environment, seed, hardware, wall time and status. Pilot
costs determine feasible replications; incomplete ranks/replications are reported
as gaps rather than fabricated references. Localized failures may be repaired
within the same budget. No package installation or paid compute.

Output root: docs/plans/artifacts/ledh-zhao-horizons-20261006-01/.
Result: docs/benchmarks/ledh-zhao-horizon-results-20261006.md.
Checkpoint: docs/reset-memos/ledh-marginal-merge-20261006.md.
Exact commands and attempt accounting are saved by the campaign supervisor.

Pre-mortem: finite outputs could conceal a wrong parameter chart, a different
initial-state convention, contaminated fitting RNG, or reference overlap bias.
Cheap parity/wiring tests precede expensive jobs. Post-run review must separate
implementation, numerical, and scientific conclusions, give a decision and
inference-status table, and state the strongest alternative explanation.

Before launch, regression design fixed to 128 antithetic fitting points and 64
heldout points per radius (PP has 28 full-quadratic coefficients, SIR10). This
is a bounded diagnostic design; rank, heldout residuals and radius stability
are required and a score is not promoted to oracle status from this design.

## Runtime repair, 2026-10-06 01:38 UTC

The pre-run runtime estimate was wrong for SIR. The saved prior rank40 T20
run took 37928.574 seconds, with roughly 1500--2300 seconds per later update
(sir-current-linear-r40-seed2-01/fit-progress.csv in the earlier campaign).
A 14200-second T50 limit would discard useful work before the first or second
checkpoint. Interrupt only the current SIR source attempt, charge its elapsed
time, and preserve its files. Continue the existing GPU and PP queues.

After those queues finish, replace the three planned SIR rank40 fits by one
rank40 fit2 T50 attempt capped at 115200 seconds and one rank20 fit17 T50
comparison capped at 14400 seconds. Run at most two CPU references, with the
same data, source, 100000 paths and exact checkpoints. Rank20 is a convergence
diagnostic and must not be pooled with rank40 as an independent replication.
Reserve each launch against the same 172800-second aggregate cap. Quadratic
references retain both declared radii and the fixed 128/64 design; allow up to
2400 seconds per horizon when the remaining budget covers the reservation.
If budget is insufficient, preserve the completed values and explicitly list
the missing conditional scores. No scientific target, promotion criterion or
parameter neighborhood changes.

Skeptical re-audit: actual prior runtime, rather than an optimistic launch cap,
now determines the expensive allocation. One rank40 fit gives conditional
path-sampling uncertainty only; it cannot estimate between-fit uncertainty.
Rank20/40 disagreement or poor overlap remains a reference limitation.

A bounded CPU FP64/XLA mechanics check uses N144, T10, the fixed data/design,
and both weight policies for each model. Compare analytical recursive scores
with centered differences at relative parameter steps 1e-5 and 5e-6. Report
normalized component errors (denominator 1+abs(score)), step stability, and
trace parity; 2e-4 is a debugging tolerance, not an accuracy promotion gate.
Persist cap/moment traces from the actual shared reset. Reserve 1200 seconds
and charge score-check-001/supervisor.json alongside attempts.json. This checks
the finite program only; it does not measure statistical score bias.

Completed prefix references may be read before the T50 fit finishes, or after a
later timeout. The existing author overlay atomically renames the saved MAT
file before closing a horizon-specific smoothing summary row. Require both for
a live/failed parent, verify the immutable source fingerprint (or completed
source-unchanged result), and retain the full target/data/shape/path-density
checks. Record parent status explicitly. A later failure cannot erase valid
earlier evidence, and an incomplete file cannot count as a completed prefix.

The first derivative diagnostic found SIR step-size sensitivity under ancestor
weights (normalized coordinate error 0.028 at h=5e-6) and a strict trace-parity
failure under mixture weights, although its finite difference error was 4.9e-6.
These are repair triggers, not evidence that the analytic formula is wrong.
Repeat SIR at h=1e-6,5e-7,1e-7,5e-8 on GPU FP64/XLA, saving actual trace
value/score differences, the traced program's own coordinate-zero derivative
check and moment-safety branch changes. Budget cap1200 seconds, charged by
score-check-002/supervisor.json. Do not relax a tolerance or change filter
controls to make this check pass. Distinguish rounding, discrete branch
crossing and a stable derivative mismatch from the resulting curves.

## Long-horizon quadratic runtime repair

The T40 quadratic reference reached its first complete radius at1371 seconds;
at1904 seconds only67 of198 points of the second radius were complete. Its
2300-second inner cap is therefore inadequate. Preserve any completed radius
from a timeout, and retry only missing predeclared radii in a fresh directory,
with the same paths, design seed,128 training points,64 heldout points, axes,
jackknife and target. Cap each repaired job at7200 seconds, and reserve a total
of14400 seconds for the rank20 T40/T50 repairs alongside every remaining
rank40 job. A second auxiliary ledger avoids simultaneous attempts.json writers.
It is included in all budget/report totals. No timeout is treated as a negative
scientific result. The report groups by actual fitted proposal, so completed
radii from one fit cannot masquerade as independent fits; the displayed score
still requires the smallest predeclared radius.

Skeptical audit: this fixes measured runtime underestimation, not a bad score,
and retains the mathematical object and all diagnostics. A completed partial
radius is accepted only from a recorded TimeoutError with a finished estimate
and verified proposal provenance. Other failed results remain rejected. No
extra fit, new neighborhood, changed path count, hardware or campaign-budget
increase is authorized by this repair.

The same bounded missing-radius repair also applies to the four rank40 score
jobs, if they time out. The auxiliary supervisor reserves the main supervisor's
remaining 2400-second score slots before every 7200-second repair, and includes
all completed, failed and active auxiliary jobs in its budget. Thus repairs
cannot use the 172800-second campaign cap twice. The already-running parent
uses its launch-time accounting; the auxiliary reservation protects its four
remaining slots. Future parent launches now use the shared accounting helper.
A budget denial preserves all completed evidence and lists the missing scores;
it does not change either the numerical design or the campaign allocation.

## Checkpoint return-format repair

A final call-chain inspection found that the generated checkpoint writer reads
the linear overlay's legacy_mean_log_weight field unconditionally. Original
pre_sol.m:270–343 returns samples, normalized weights and an optional proposal
history, with no lml diagnostic; its local observability overlay therefore has
no such field. The latest writer would fail in a new nonlinear paper/source
run, although the current linear comparison is unaffected. Preserve the missing
quantity as NaN with an explicit availability flag, rather than inventing an
absolute evidence value. Execute the actual generated Octave writer against
both documented return formats as the smallest regression. This is a routine
serialization repair, with no fitter, sampler, target or budget change. The
prior nonlinear results came from a writer predating this added field.

Validation: all 47 focused tests pass (61.21 seconds), including both generated
Octave writer cases. The two added cases verify the availability flag, NaN
for absent evidence, preserved linear values, weights and completion row.
No source solver or active fit was changed.
