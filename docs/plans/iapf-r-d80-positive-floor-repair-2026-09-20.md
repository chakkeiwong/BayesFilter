# Evaluate a smaller positive floor for the R reference

2026-09-20 owner-authorized continuation. Attempt07 reproduced all three saved
passes exactly. In the two bad passes, time93 floor probabilities averaged
.9999966004 and .9999942211: all4000/all8000 particles used the transition
component. Zeroing that floor only reduced log errors from -37.69775/-36.04595
to -1.20514/-1.89285. Removing all floors gave .65763/-.67641. These are selected
trajectory interventions, not statistical estimates of improvement.

## Frozen repair and its rationale

Keep log_quadratic fitting, diagonal guide covariance, positive Eq.16 mixture,
N0=1000, k=5, tau=.5, kappa=.5, iteration cap20 and particle cap16000. Change
only the explicit candidate floor-tail power from2 to8. The default remains2
and paper_eq15 remains the default objective. This optional R candidate is
not a reproduction of the authors' unspecified floor/solver choices.

Selection uses the analytic floor calibration, not likelihood performance.
On300 saved proposal clouds, powers2,3,4,8,16 have maximum floor probabilities
1,.999999999999893,.999999943685828,8.31552e-14,9.77018e-50. Eight is the first
of these candidates with maximum probability below1e-8. The diagnostic threshold
bounds expected transition draws at N<=16000,T100 by .016 on these clouds;
it is a finite-budget disturbance screen, not a uniform bound over unbounded
states or a universal numerical default. Power16 would weaken the tail component
further without a demonstrated need. Power8 is frozen before the fresh dataset.

For psi=N(m,V)+c and f=N(a,Q), J=N(m;a,Q+V), so
p_floor=c/(J+c). The previous rule controls c relative to the guide's peak,
not its overlap with f. Large observation innovations can make J much smaller
than c and switch off useful guidance. Positive c remains mathematically
required in this candidate; a zero-floor route is only a diagnostic.

## Safety and validity criteria

This is a Class C numerical change. Acceptance first requires a constructed
healthy centered Gaussian case to have exactly identical particles, weights,
prefixes and likelihood under powers2 and8 with the same RNG. On the three
saved frozen guides, report all changes, finite weights, ESS and floor
probabilities; do not optimize likelihood error. Check the Gaussian-limit
one-step squared-weight integrability matrix
M=Q^-1+2H-V^-1, H=C'R^-1 C + A'(Q+V_next)^-1 A
(omit the future term at T). Its scale-relative minimum eigenvalue must exceed
100*d*machine_epsilon for every saved/final guide. At t1 use initial covariance.
This follows by integrating f(x)[g(x)J_next(x)]^2/N(x;m,V). It proves finite
conditional Gaussian-limit second moments for each fixed ancestor, not uniform
variance bounds, controller convergence or the variance of the full estimator.
The positive floor itself remains a tail protection. Record all checked margins.

Repair feasibility requires completed controllers, admissible fits, finite
fresh likelihood ratios in[.1,10], no numerical/integrability veto and no
conditional heuristic loss to BPF10000, fully-adapted5000 or SIS10000. Compare
ordinary and large-innovation times separately (Kalman chi-square90% split).
These hard screens cannot establish accuracy or superiority with four repeats.
The exact Kalman likelihood is the primary accuracy reference; report all
ratios and bootstrap uncertainty descriptively. No tuning on the fresh data.

Healthy regression controls: rerun d20 datasets70000020/71000020, four repetitions
each at the original IDs201:204/301:304 and seeds. The unchanged baseline
results already exist. Reject promotion if either conditional log-prefix MSE
exceeds twice its power2 value or any final ratio leaves[.1,10]. The factor2 is
a conservative non-harm screen with limited power, not statistical equivalence.
Document even smaller observed changes. No new default may follow this test.

## Execution and budget

Seven launches used1101.086348/1800 seconds;698.913652 seconds remain. Reserve
at most600 seconds for the last launch, attempt08-positive-floor-repair.
CPU-only R, one BLAS/OpenMP thread, captured sources and data. Stages:
1. Exact healthy-case regression; saved-guide positive-floor replays and margins.
2. d20 four-repeat controls on each of the two existing datasets.
3. d80 two-repeat repair pilot on seed72000080, IDs1:2.
4. Only if stages1-3 pass: untouched d80 seed74000080, IDs101:104, four repeats.

Reuse the captured existing replication runner so the call chain, fitting,
comparators and CSV output remain identical except the explicit floor argument.
Preserve every stage and stop only on invalid evidence, numerical/non-harm
veto, incomplete controller, failed pilot feasibility, or total timeout.
An incomplete controller rejects this repair within this budget; it does not
reject iAPF. Keep later diagnostics already produced, with no subset inference.
Artifacts include full RDS, fits, method/prefix CSV, phase commands/statuses,
checks, integrability margins, source hashes and actual time. Results go in
the existing campaign root and its checkpoint/master program is refreshed.

Command: `python docs/benchmarks/run_iapf_r_replication.py --campaign validation
--output docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01/attempt08-positive-floor-repair
--dimension 80 --data-seed 72000080 --mode floor_repair --fit-mode log_quadratic
--floor-power 8 --timeout 600`.

## Skeptical review

Pass for a bounded optional repair evaluation. The confirmed mechanism concerns
the local floor, not an invalid iAPF identity or proof about author code. The
candidate is selected using mixture disturbance, with a separate tail-moment
check and healthy controls. No controller cap, data target, fit objective or
promotion threshold is relaxed after seeing fresh outcomes. The main risks
are heavy tails not visible in short runs, diagonal approximation error and
dataset dependence. Few repeats and unequal computing budgets prohibit ranking
and publication-scale claims. A pass justifies a new bounded multi-dataset
validation campaign; it cannot close the paper-replication gap.

## Localized infrastructure retry

Attempt08 used53.397834 seconds. Its healthy checks and300 tail checks passed,
and the d20-a phase produced all16 method rows, but `system2(stderr=TRUE)`
changed the return type from an exit code to captured text. The supervisor
misclassified this as a numerical failure and lost the captured phase log.
This is a harness failure, not a candidate rejection. Preserve attempt08.

Repair: direct both stdout and stderr to the same phase log and require one
numeric exit code. A focused smoke checks success/failure exit codes and both
log streams. Retry once in fresh attempt09-positive-floor-retry with timeout540
seconds. The original repair's combined allowance remains600 seconds (53.397834
plus540 reserved); the campaign remains1800 seconds. There are eight planned
scientific launches plus this one infrastructure retry, nine process launches
in total. The governing campaign-repair policy permits this localized retry
under unchanged scientific scope and compute; launch counting must not be
treated as a spent authority token. No further retry or new scientific arm is
authorized by this amendment. Re-audit passes: only log/status handling changes;
the frozen R numerical core, floor candidate, seeds and screens are unchanged.
