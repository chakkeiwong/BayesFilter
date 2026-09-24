# Bounded R log-quadratic fitting repair

2026-09-20, continuation explicitly requested by the owner after the paper/code
audit. The literal equation-(15) path remains separate. This is an optional
Algorithm-3 fitting choice, not a replication of Section 5.1's objective.

## Question and mechanism

Can a fit to log backward targets avoid the demonstrated concentration of the
relative-density objective and complete a fresh d20 filter without losing
agreement with Kalman? On the frozen audit cases, continued relative-L2
optimization reduced the training error but increased the exact continuous
error under the smoothing measure. Merely increasing the iteration budget is
therefore not the repair tested here.

Fit `log y_i = b0 + sum_j(b_j z_ij + c_j z_ij^2) + error_i` by ordinary least
squares using centered/scaled coordinates. If every c_j is strictly negative,
the fitted shape is a proper diagonal Gaussian with
`m_j = center_j - scale_j*b_j/(2*c_j)` and
`V_j = -scale_j^2/(2*c_j)`. The intercept absorbs its irrelevant amplitude.
This is already the starting fit used by the existing solvers; the repair
explicitly returns it instead of optimizing the concentrated relative-density
loss. Every particle contributes to squared log error. A full-rank QR solve
determines the coefficients without an iterative optimizer. A nonconcave or
rank-deficient regression is a rejection, never a weighted-moment fallback.

## Evidence contract, defaults and review

The comparison is the frozen relative-L2 implementation and its saved failures,
plus exact Kalman, BPF, fully adapted APF and SIS in a fresh d20 pilot. First
verify Gaussian recovery, normal equations, coordinate equivariance, failure
guards and the actual consumer call chain. Then run four complete repetitions
on one fresh dataset. This is a pilot: finite complete estimates and no fitting
guard failures allow longer validation, not statistical ranking or promotion.
The same floor power2, first_full_window controller, N0=1000, T100 and published
baseline particle counts remain labeled reconstructions. None is tuned on the
new pilot. Log-regression concavity/rank and factorization failures are hard
vetoes. Likelihood errors, fit residuals, runtime and small-sample variability
are descriptive; a ratio outside [.1,10] stops expansion as a debugging alarm,
not a published-accuracy criterion. Independent Gaussian-mixture moment fits
remain oracle diagnostics, not a model-specific runtime fallback.

| Choice | Provenance and purpose | Risk / earliest diagnostic | Status |
| --- | --- | --- | --- |
| Unweighted log loss | Algebra above; each particle contributes | Can bias approximation in tails; report original relative residual and final likelihood | New hypothesis |
| Center/scale and QR | Existing initializer; same linear subspace | Degenerate cloud/rank; reject before fitting | Numerical coordinate choice |
| Negative quadratic coefficients | Required for a normalizable Gaussian | Nonconcave target; reject, do not clip | Mathematical requirement |
| No ridge/trust radius | QR computes specified LS minimizer, no empirical-density descent | Ill-conditioned design; emit rank/condition and reject unresolved rank | Derived choice |
| Existing positive floor/controller | Frozen earlier reconstruction | High sensitivity; no claim of author settings | Baseline |
| Four repetitions, fresh data68000020 | Remaining compute, never tuning data | Inadequate uncertainty for a ranking | Pilot only |

Skeptical self-review: using analytic target moments directly as the filter
would exploit a linear-model oracle and would not fix the general R fitter.
This plan instead tests a callable-observation fit whose objective and exact
solution are explicit. Wrong-baseline and promotion-by-proxy risks are handled
by actual filtering checks and the no-ranking scope. Solver nonconvergence is
not suppressed. This does not close equation-(15), author-setting, d40/d80,
1000-repetition or TensorFlow comparison gaps. Verdict: proceed.

Budget: use at most200 of the remaining226.670573 worker seconds, one pilot
launch plus one localized repair/retry if needed, within the unchanged1550
repair-worker-second total. Owner continuation reallocates the old exhausted
10-launch count to12; it does not expand total compute. Remaining mechanics
allowance70 seconds. Unique output under the existing repair root, attempt11
onward. Captured-source execution, CPU-only R4.1.2, CUDA hidden, one BLAS/OpenMP
thread. Stop expanding if the pilot is incomplete, invalid or outside its alarm
bounds; record the reason and next discriminating repair. No external action.

## Continuation after the completed pilot

The four-repeat d20 pilot completed in31.882971 seconds. All likelihood ratios
were inside the debugging bounds (.981063 to1.066358); no fit failed. No fitting
choice is changed. Reallocate the remaining second launch from a possible retry
to16 complete repetitions on untouched data69000020, replication IDs101--116.
Timeout165 seconds; the total follow-on reservation31.882971+165 remains below
200, and total repair worker use remains below1550. This is the final available
launch. Keep captured sources and the same model/horizon/particles/floor/controller.

This smaller follow-on batch is a conditional accuracy screen, not completion
of the earlier32-repeat criterion or the paper's1000 repetitions. Report a
2000-resample percentile95% interval for mean likelihood ratio with fixed seed
69000920. The interval entirely inside[.9,1.1] allows a larger future validation;
it does not promote the method or support an efficiency ranking. All16 repetitions
and all four methods must complete. Report per-repetition mean squared log-prefix
errors separately for ordinary and large innovations, defined by the unchanged
chi-square90% threshold. Any higher iAPF mean error than a heuristic in either
situation vetoes advancement. These conditional checks are vetoes, never tuning
targets. Nonfinite log estimates, invalid fits, changed sources or incomplete
results invalidate the batch; report rather than select successful subsets.
Likelihood ratios may underflow for SIS: preserve and assess its finite log error.

Pre-run skeptical review: the new data are untouched; the log objective remains
frozen; a second dataset is more informative than reusing the pilot for another
setting. Four pilot repetitions are used only to permit continuation. Bootstrap
uncertainty with16 repetitions is limited and cannot support tail or cross-dataset
claims. The same local CPU/reference scope and total budget apply. Proceed.
