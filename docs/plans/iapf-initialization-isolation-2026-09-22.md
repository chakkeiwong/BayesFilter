# Isolate initialization from density-objective scaling

Status: COMPLETE. Both known-target device checks and the fresh consumer screen
are recorded in `artifacts/iapf-initialization-isolation-20260922-01/result.md`.
43 regression tests pass. Five of 24 fresh adaptive candidates fail their caps;
these are preserved candidate failures, not continuation vetoes. Optional
controls remain explicit and defaults are unchanged. Terminal verification
and source/output provenance are in that directory's verification and manifest.
Parent: `younis-kdm-score-master-program-2026-09-14.md`.
Preceding result: `artifacts/iapf-density-scale-20260922-01/result.md`.

## Research intent and evidence contract

Question: does target-aware initialization repair the demonstrated inaccurate
zero-step/local-fit failures while retaining the same profiled Equation-15
density objective? Separate initialization from fixed objective units in a 2x2
factorial: cloud moments / diagonal log-quadratic QR, crossed with native units /
fixed initial peak units. The baseline is the actual existing TensorFlow fitter.
The independent frozen R reference and the exact Gaussian are comparators.
QR is a reconstruction hypothesis from the independent R reference, not an
identified author choice. Keep both new controls optional; existing defaults
and the relative-shape objective retain their identities.

Implement controls in the shared fitter and wire the real recursive/adaptive
consumer to that fitter. An isolated helper that the consumer cannot call is
insufficient. Preserve and test default output parity against the preceding
30 saved fixtures. Expose initialization validity and objective units in the
consumer diagnostics and configuration. No per-model implementation forks.

Stage A: reuse saved known-target inputs as an explanatory experiment, not a
fresh claim set. Require QR arms to recover all exactly representable targets
to max center/covariance error <=1e-9, shape <=1e-12; compare frozen R outputs.
Require baseline/previous fixed-scale arms to match saved outputs <=1e-9.
Test healthy non-harm <=1e-9, target-amplitude invariance and analytical gradients
against finite differences. Repeat same inputs on GPU FP64/XLA. Ill-conditioned,
rank-deficient or nonconcave QR inputs must fail closed with visible diagnostics.

Stage B: exercise the actual adaptive consumer on fresh seeds 81/82, d/o=2/2,
5/5 and 10/10, T=4, N=128, cap=1024, k=2, tau=.5, max iterations=12, the same
six-parameter local linear model used by the preceding conformance phase.
Compare all four arms with identical named random streams where count/iteration
paths permit. Exact Kalman likelihood, BPF/constant guide, and one-step optimal
Gaussian proposal are the constructed heuristic set. Conditional situations
are dimension and fitted versus heldout observations. Small-sample log-value
errors and fit quality are descriptive only; do not rank or promote an arm.
Evaluate guide reconstruction and label-stable finite-program derivatives
separately from marginal model-score accuracy. Stage B can reject a candidate
or identify a further fitting failure; it cannot certify a production default.

Stage B executes on the selected GPU; same-input CPU/GPU arithmetic is checked
in Stage A. Fit and heldout observations have disjoint data seeds. Reuse the
fitted guides on heldout observations as a deliberate transfer diagnostic;
this is an arbitrary positive guide, not a guide trained for those observations.
Evaluate each heuristic at that arm's realized final count, including the
constant-guide version of the same fitted consumer as well as the plain BPF.
For QR+initial-peak, seed81 in each dimension, check both fitted and heldout
finite-program scores against centered differences. Choose the first pair
h/h2 in 1e-5,1e-6,1e-7,1e-8 with all ancestor/mixture labels unchanged;
accept derivative error <=2e-5, without choosing steps by derivative error.
If no stable pair exists, retain the boundary crossing as inconclusive evidence.

Primary mechanism criterion is Stage A recovery, not aggregate likelihood.
Hard promotion vetoes: failed validity, nonfinite accepted results, wrong
analytical derivative, inability to call the shared implementation, default
regression, or losing to the exact oracle/cheap heuristics. Research continuation
vetoes: invalid source/draw identity, wrong mathematical objective/gradient,
corrupted evidence or exhausted budget. A rejected optional initialization,
fit convergence failure or adaptive cap is a repair trigger, not a reason to
abandon the investigation. Preserve the failed case before any local repair.

## Defaults, guards and skeptical review

QR solves log b = c + beta'z + gamma'z^2 in cloud-standardized coordinates;
mean_z=-beta/(2 gamma), variance_z=-1/(2 gamma). Require enough rows and
finite QR factors. A diagonal-R margin relative to its norm and dimension
screens numerical rank; it is an explicit conservative guard, not an exact
singular-value claim. Require gamma < -sqrt(machine epsilon), matching the
concavity margin of the frozen R initialized fit. Reject rather than substitute
weighted moments in this isolated strict-QR candidate. Record failures and
clipping into the preexisting parameter box. No ridge is added.

For initial-peak scaling, subtract the fixed initial log peak from the log
amplitude before exponentiation. The factor is constant in fitted parameters,
so the objective and gradient share it and minimizers are unchanged. It is not
normalization by the current density or a relative-shape loss. Scale selection
can change finite optimization/stopping, which is exactly the hypothesis.

The old box/tolerance/floor/caps are frozen baselines with known failure modes
from the preceding diagnosis; none is promoted as universally appropriate.
QR concavity/rank thresholds are numerical guards with synthetic rejection and
known-good no-fire checks. No stabilization parameter is tuned on likelihood.
Healthy fits and recorded baseline trajectories provide non-harm checks.
The reference initialization is a warm-start hypothesis; diagonal exact-target
success cannot establish accuracy for non-diagonal/non-Gaussian guide targets.
The new optional controls must be included in tuning scope/configuration.

Skeptical review PASS: the 2x2 design removes the R/TF solver confounding for
initialization; saved data are explicitly diagnosis only. Stage B uses fresh
observations but six cells cannot support stochastic ranking. The broadening
from exact diagonal targets to real backward targets is necessary to expose
model-family and clipping gaps. Native objective arithmetic and consumer
reachability have executable checks; source-default parity is preserved.

## Execution and budget

Evidence root: `artifacts/iapf-initialization-isolation-20260922-01/`.
Use tftwogpu Python, CPU hidden GPU and one thread, GPU UUID
GPU-68251639-fe82-8f81-3ccc-2953c32e805b with escalation and verified growth.
FP64/XLA only; existing TF32 veto remains. No R reference file changes.
Suballocation: 2 CPU hours and 1 GPU hour, at most 10 research launches and
three localized repairs, within the recorded remaining campaign balance.
Each launch has a fresh directory, exact command, source snapshots, seeds,
environment and elapsed time. Record tests, results, uncertainty, decision,
failure classification and remaining budget. Stop individual launches at
15 minutes CPU / 10 minutes GPU; revise only if unchanged scope fits the total.

Next action: isolate positive-floor strength with frozen guides and random
inputs, measuring the actual Gaussian proposal mixture probabilities. Neither
the initialized fitter nor a floor hypothesis is promoted by this screen.
