# Plausible numerical reconstructions of the first iAPF experiment

Owner instruction: try a few plausible unspecified choices and determine whether
any is close enough. This explicitly authorizes reconstructions; original-author
identity is not a prerequisite for this experiment. CPU-only independent R
reference work; no production, TensorFlow, KDM or LEDH default changes.

## Question and source

Can a disclosed numerical reconstruction reproduce the useful accuracy and
computational pattern of Section 5.2, without identifying the authors' actual
implementation? The cached Guarniero--Johansen--Lee paper, Section 5.1 equations
(15)--(16), Algorithm 4, and Section 5.2 Tables 1--2 are the primary anchors.
The recovered TeX is in the source-reconciliation artifact. The original data,
solver, floor function and early doubling convention are not available.

The model is unchanged: T=100, alpha=.42, A_ij=alpha^(|i-j|+1), initial
N(0,I), process and observation covariance I. Published iAPF SD(Zhat/Z) is
.09/.14/.19/.23/.35 at d=5/10/20/40/80; average final N is
1000/1000/1000/1033/1142; final resampling counts are
6.93/15.11/27.61/42.41/71.88. The paper uses 1000 replicates. This is a bounded
reconstruction screen, not that full replication or a runtime reproduction.

## Fixed choices and assumptions

All arms use N0=1000, k=5, tau=kappa=.5, diagonal Gaussian plus positive
constant guides, Algorithm 5 adaptive resampling, and a fresh final APF.
Caps remain 20 iterations/16000 particles; a cap is a candidate failure.

| Arm | Fitting | Positive floor | Doubling eligible |
|---|---|---|---|
| current | diagonal log-quadratic QR | Gaussian density at chi-square tail probability N^-8 | l>=k |
| delayed | same QR | same N^-8 floor | l>k |
| floor4 | same QR | density at tail probability N^-4 | l>k |
| local_eq15 | local bounded minimization of equation (15) | same N^-8 floor | l>k |

The QR objective differs from equation (15); these three arms are empirical
comparators, not implementations of that equation. Delaying doubling is a
plausible reading of Algorithm 4's explanation (oscillation AND high CV), and
avoids doubling before stopping is allowed. Both readings are disclosed.
Floor4 geometrically interpolates tested powers 2 and 8; power 2 already failed
at d80. These are assumptions about the unspecified c, not author settings.

The local Eq15 arm starts at the QR fit, bounds each variance to [v0/2,2v0]
and mean to m0 +/- sqrt(v0), and uses the existing profiled Eq15 objective,
analytic gradient, and L-BFGS-B (maxit=200, factr=10, pgtol=1e-8).
This compact neighborhood excludes variance escape and vanishing densities at
infinity. The factor 2 is an explicit reconstruction hypothesis, not a calibrated
default. Active bounds are reported; boundary dependence or optimizer failure
blocks nominating this arm. An exact diagonal Gaussian fixture must be
unchanged. No silent ridge, clipping, fallback, or accepting unconverged fits.

## Evidence contract and selection

Calibration: fresh seed 85000010, d10, eight repeats/arm, replication IDs
1101--1108; then d80 seed 85000080, two repeats of each surviving alternative,
IDs 1201--1202. These are nomination data only. Common method seeds pair arms;
divergent control flow means this is not perfect common random numbers.
Hard validity, tail-margin and boundary failures reject an arm, not the research
direction. A computational timeout is under-budgeted evidence, not mathematical
failure. Do not expand a numerical budget just to rescue a preferred arm.

Among eligible alternatives, select the bounded Eq15 reconstruction first if
it actually completes without boundary dependence, preserving the published
objective. Otherwise minimize on d10 calibration the predeclared discrepancy
abs(log(SD/.14))+abs(log(mean_N/1000))+.5*abs(log(mean_resamplings/15.11)).
This is empirical pattern matching, not evidence of the original implementation.
Do not tune to the heuristic adversaries. Freeze the selected arm before fresh
validation: d10 seed 86000010, IDs 1301--1332 (32); d80 seed 86000080,
IDs 1401--1416 (16). If budget permits, add d5,20,40 with seeds 86000000+d,
IDs 1501--1508 (8 each), in that order. No parameter changes after calibration.

Primary practical-reference screen on each validation dataset: mean likelihood
ratio bootstrap 95% CI contained in [.8,1.2], bootstrap SD 95% upper bound no
more than twice the published SD, and mean particle count <=1.5 times published.
These are broad, user-requested 'close enough for debugging' criteria, not the
previous stricter .9--1.1 screen and not a new algorithm default. Report whether
the older stricter screen also passes. Predeclare literal performance agreement
separately: SD CI contained in [.5,2] times published SD and mean resampling
count in [.5,2] times published. A materially lower SD can be useful but does
not replicate the reported variability. No pass from mere interval overlap.
Resampling and runtime are explanatory for practical-reference admission.
Bootstrap 4000 resamples, fixed seed 9102026; intervals condition on each newly
simulated dataset. Few repeats, rare tails and missing author data limit claims.

Vetoes: nonfinite states/weights, nonconverged fits, failed guide tail-margin
diagnostic, invalid artifacts, fresh-final-run violation, or observed heuristic
underperformance in either conditional situation. Constructed adversaries:
exact Kalman (oracle), BPF10000 (plain importance/resampling), FA-APF5000
(analytical one-step proposal), SIS10000 (no-resampling control). Compare mean
squared original-prefix likelihood-ratio error, separately for ordinary and
top-decile Kalman innovations, and terminal likelihood-ratio variance. Kalman
defines zero error, not an algorithm the Monte Carlo arm must beat; the other
three form the heuristic veto. These comparisons are descriptive unless paired
bootstrap intervals establish a difference; no stochastic ranking from means.

## Execution, budget, repairs

Transfer the existing remaining 868.02274921973 summed worker seconds, without
reusing prior allowances. Reserve 25 seconds for reporting. Calibration d10:
four launches <=45s each; d80 <=75s per eligible alternative. Validation uses
actual remaining time, with the required d10 and d80 cells before optional
dimensions. At most 12 filter launches plus two local infrastructure retries;
every launch and timeout consumes the same budget. Captured sources execute
under Rscript --vanilla, CUDA_VISIBLE_DEVICES=-1 and BLAS/OMP threads=1.
Artifacts: docs/plans/artifacts/iapf-r-plausible-reconstruction-20260921-01.
Driver: docs/benchmarks/run_iapf_r_plausible_choices.py. Every worker has a
unique directory, complete log, source hashes, seeds, command, environment,
wall time and status. Save RDS guides, fit/tail diagnostics and original-prefix
errors. No overwriting prior evidence. Partial cells are reported as incomplete.

Localized harness/serialization failures trigger focused checks and fresh-output
retry without new approval. Stop only for invalid mathematical assumptions,
corrupt evidence, exhaustion of this allowance or an external boundary. One
candidate failing is not a continuation veto; move to the next planned arm.

## Skeptical audit before implementation and execution

PASS with explicit limits. The paper's actual SD/particle/resampling targets
are compared, rather than substituting mean accuracy. All guessed choices are
labeled; changed objective, data, R hardware and smaller sample count preclude
original-code or exact table reproduction claims. Exact Gaussian no-fire and
gradient checks protect the new bounded-fit path. Separate fresh validation
protects against matching calibration noise. The principal alternative
explanation is model/data-dependent luck and underobserved rare weights;
tail diagnostics and uncertainty help, but cannot certify 1000-replicate
performance. A runtime cap cannot be reinterpreted as scientific rejection.

The final result must include decisions, inference status, strongest alternative
explanation, budget ledger and next justified action. Paper-scale reproduction,
nonlinear reliability, method superiority, HMC and GPU readiness remain open.

## Completion

Executed ten launches: nine completed, one planned active-bound rejection.
The frozen delayed QR choice passes the practical screen at d5/d10, fails the
d80 particle limit, and has an uncertain observed d20 conditional heuristic
loss. Original-paper performance is not reproduced. New-setting d40 needs more
than the remaining allowance. Saved-history inspection explains fifteen d80
doubling events; shortening its window is an untested departure from printed
Algorithm 4, not a correction established by this run. The result, inference
tables, terminal review and current budget are in
[result.md](artifacts/iapf-r-plausible-reconstruction-20260921-01/result.md) and
[terminal verification](artifacts/iapf-r-plausible-reconstruction-20260921-01/terminal-verification.json).
Captured pre-run plan and numerical sources remain preserved unchanged.
