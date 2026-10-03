# R iAPF targeted fitting repairs

Status: complete. The owner-authorized execution used 494.012354 numerical
worker seconds in 29 launches. The [results and terminal review](artifacts/iapf-r-targeted-fitting-repair-20260922-01/result.md)
record the completed tests, one localized optimizer repair, and the remaining
accuracy and replication gaps. No numerical worker remains active. This is an
independent CPU R reference investigation, not an original-author implementation
or a production/default change. The pre-run contract and amendments below are
preserved as the executed plan.

## Question and evidence contract

Can removing the QR prerequisite make warm starts usable, and can a stable,
positive-precision fit survive the actual bootstrap learning clouds? Separate
implementation correctness, guide approximation and terminal likelihood
behavior. The source/code diagnosis is
`artifacts/iapf-r-root-cause-audit-20260922-01/result.md`; its paper anchors and
limitations remain binding. The paper's full Gaussian future guide is an
analytic control; its implemented learning family remains diagonal Gaussian
plus a positive floor. Weighted-log and ridge fits are explicit extensions.

Primary local criteria: the previous-guide route reaches the optimizer without
QR; exact-diagonal response controls and analytic gradients agree to scaled
1e-6; unconstrained healthy fits are preserved by the constraint-only route;
accepted fits have finite positive covariance and satisfy scaled KKT residual
<=1e-6. Report actual Gaussian shape error on independent predictive and
smoothing points, as well as exact Gaussian KL, against the controls below.
Smaller absolute Equation15 loss alone is never success.

Numerical rejection, underflow, rank loss, poor heldout shape or heuristic
underperformance are promotion vetoes. They trigger the next planned repair
or explain a rejected candidate; they do not stop this entire investigation.
A broken algebra/gradient/call-chain check, corrupt input/output, exceeded
compute/deadline or missing provenance is a continuation veto until repaired.
An optimizer iteration/time limit is recorded as incomplete, not success.
ESS, condition numbers, active constraints and timing are explanatory.

No statistical ranking, paper replication, author-choice recovery, default
readiness, GPU/TF parity, LEDH/KDM, gradient or HMC claim follows from this
small diagnostic campaign. Full-filter records are mechanics/diagnostic
evidence; report terminal errors per data set without ranking from few seeds.

## Execution and repair sequence

1. Preserve the old F2 route and add explicitly named independent-start arms.
   When a valid previous Gaussian exists, its mean and diagonal variance define
   the anchor, coordinate scales and fixed density/loss scales; no QR call is
   allowed. Retain the original printed objective, strict/loose tolerances and
   underflow/domain checks. With no previous guide, use the old QR start and
   disclose that remaining dependency. Replay all 25 saved pre-optimizer QR
   rejections under both tolerances. Record reachability separately from valid
   completion, shape change and objective change.
2. On all saved d80 weighted-log failure clouds, compare QR and SVD for weight
   exponents 0, .5 and 1, using the same points and targets. Include an exactly
   diagonal response and deterministic response perturbations of scale
   1e-10 times max(1,response RMS). Report rank, coefficient sensitivity and
   curvature; ESS is not a rank test. Test bounded quadratic coefficients at
   exponent1, then the predeclared ridge repair below. Use actual failure
   clouds for training; never substitute exact-guide clouds.
3. Evaluate the fitted Gaussian shapes on 1,000 fresh predictive and 1,000
   exact smoothing draws for each saved t99 cloud; use independent fixed seeds
   92230000+replicate. Derive the t99 Gaussian component from the already fitted
   terminal guide, measuring the floor correction separately. Report shape
   RMS after removing the irrelevant additive log constant and KL from the
   exact normalized Gaussian component. Predictive/smoothing samples explain
   generalization; they do not select hyperparameters.
4. Compare exact full Gaussian, moment-diagonal, precision-diagonal,
   observation-only and constant guides at d5,d20,d80, T100, N1000 on fresh
   data seeds 92210000+d and 92220000+d. Use two filter seeds per data set,
   92240000+d+1009*replicate. The two diagonal projections minimize respectively
   KL(full||diagonal) and KL(diagonal||full), preserving the exact guide mean.
   This separates family approximation from learning error. Kalman is the
   exact terminal oracle. Preserve prefix diagnostics without conflating them
   with the paper's terminal likelihood claim.
5. If implementation/identity checks pass, run the shared complete learning
   controller with the fixed ridge-bounded exponent1 arm and the existing QR
   comparator on those same six new data sets, one preassigned seed each.
   This arm is nominated in advance, not selected by the preceding results.
   Use N0=1000, T100, k5, tau.5, adaptive resampling .5, tail8 floor, sample SD,
   after-k doubling and k+1 stopping history. Resource limits are 12 learning
   iterations and 4,000 particles, explicitly below the earlier full ladder.
   Log every failed fit and the fresh final estimate. A shape veto remains a
   promotion veto even if a complete filter terminates.
6. Predeclared after all twelve fresh-data runs completed, before examining
   repaired outcomes on the old seeds: bridge the local repairs to the eight
   original tail8/wlog1 d80 failures (data92100180 and92100280, replicates1--4).
   First replay the unchanged wlog1 consumer and require exact agreement of
   its saved failure cloud/target and failure time. Then run the frozen
   ridge-bounded arm from the same initial seed through the shared controller.
   Use the original seed=data+1009*replicate+100003 and population SD for both
   paired arms, retaining the declared 12-iteration/4,000-particle resource
   caps. This is a diagnosis-conditioned repair check, not untouched validation
   or a ranking study. Allocate at most120 seconds per paired case, within
   the unchanged total and30-launch cap. A repaired failure is recorded and
   investigated locally; it does not retroactively change the frozen candidate.

Localized solver repair, declared after paired case data92100180/replicate4
failed at iteration4/time72: L-BFGS-B exhausted its iteration budget with
scaled KKT1.508612e-6, above the unchanged1e-6 criterion. The target regression
has full rank and condition1397.255; lambda is0 under the predeclared rule.
Add an SVD active-set least-squares fallback only after the original constrained
solver fails its KKT gate. It solves the same convex objective and bounds;
retain the original failure, original optimizer diagnostics, and fresh retry
outputs. Check the fallback against exhaustive active-set enumeration on a
small fixture, then replay this exact full consumer with identical seed.
No tolerance, ridge rule, constraint, data or promotion criterion is changed.

## Defaults, mathematical choices and skeptical review

| Choice | Provenance / justification | Risk and earliest diagnostic | Status |
|---|---|---|---|
| Previous Gaussian anchor/scales | Existing valid prior fit; removes measured QR prerequisite | Bad prior guide / density underflow; replay with unchanged guards | Hypothesis |
| SVD rank threshold eps*max(n,p)*smax | Standard backward-error scale, recorded explicitly | Near-null directions amplify omitted terms; exact-diagonal and perturbation controls | Diagnostic solver |
| q_j <= -sqrt(eps)*p_ref/2, p_ref=median(abs(2*q_unweighted)) | In standardized coordinates, an absolute precision error eps*p_ref then has relative reciprocal error at most sqrt(eps); this declares the reference scale, not a guarantee of overall conditioning | Boundary solutions may have useless shape; report reference scale, active bounds and heldout errors | Derived numerical constraint hypothesis |
| Constraint-only fit | Convex weighted quadratic regression with analytic gradient | Solver/KKT error; return an already-feasible SVD solution unchanged | Repair candidate |
| Ridge toward the unweighted SVD coefficients | Preserve the well-conditioned all-particle fit as an anchor | Bias toward a weak guide; evaluate actual shape and exact controls | Repair candidate |
| Ridge strength | lambda=max(0,(smax^2-C*smin^2)/(C-1)), C=eps^(-1/2), for normalized weighted design | Targets Hessian roundoff amplification eps*C=sqrt(eps), not likelihood improvement; report realized strength | Derived numerical calibration, not metric tuning |
| Weight exponents 0,.5,1 | Existing unweighted baseline, half concentration and failed exponent1 | Concentration hides interactions; report each conditional cloud | Diagnostic hypotheses |
| Tail8, k5/tau.5, original controller | Frozen comparator configuration, not recovered author choices | Floor and early stopping effects; preserve values and caps in output | Baseline |
| CPU R / single BLAS thread | Authorized independent R reference | Cannot establish TF/GPU production behavior | Reference exception |

No ridge is used in the constraint-only arm because it isolates the curvature
constraint; the ridge arm separately tests poor conditioning. Constraint and
ridge changes are never silently applied to Equation15 or the existing QR
comparator. The ridge penalty is on standardized regression coefficients,
including the intercept; centering target logs fixes that convention. Its
anchor is the unweighted fit, so an exactly representable diagonal response
is also an exact minimizer of the penalty. Every computed fit diagnostic is
retained. A finite parameter vector is not proof of optimizer convergence.

Constructed heuristic controls: constant guide (ordinary bootstrap filter),
observation-only guide (uses the current likelihood), unweighted QR fit
(simple regression), analytic moment-diagonal and precision-diagonal guides
(remove learning error while retaining the diagonal family), and full Gaussian
guide/Kalman (exact oracle). Evaluate conditional on dimension, data set,
predictive versus smoothing region and the original failure type. Underperformance
against any applicable heuristic is a promotion veto and a reported finding;
the controls are not tuning targets.

Skeptical audit passed for this diagnostic scope: the warm-start test now
actually reaches its claimed operation; no exact-guide training-cloud
substitution is permitted; objective decrease is separated from shape;
solver controls separate roundoff from omitted terms; analytic diagonal
projections have declared probability directions; full-filter seeds are fixed
before outcomes; a failed candidate continues to the planned repair. Main
remaining uncertainty is generalization from selected failures and small fresh
samples. The cheapest falsifications precede the complete learning runs.

## Execution, provenance and budget

Run `/usr/bin/Rscript --vanilla` with CUDA_VISIBLE_DEVICES=-1,
OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Keep executed source snapshots,
Git HEAD/dirty-source hashes, exact commands, input hashes, R version, seeds,
wall time, exit status and result paths in per-attempt manifests.
Output root: `docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01`.
Use a fresh subdirectory for each launch and never overwrite earlier results.

Cap: 4,800 aggregate worker seconds and 30 launches, within the remaining
5,007.613923 seconds of the prior allocation. The unchanged deadline is
2026-09-21T20:04:26Z. No implicit renewal. Initial allocations: 300 seconds for
warm-start replays, 600 for solver/shape controls, 600 for analytic controls,
2,400 for complete learning, 900 for focused checks and local repair. These
may shift within the total cap. Timeouts: at most 120 seconds per d5/d20
complete-learning job and 360 per d80 job. Local harness/solver repairs are
authorized within the same scientific question and budget. Report censored
jobs and remaining work honestly if a true resource veto fires.

Exact commands are generated by the bounded local supervisor with stage,
output directory and fixed data/replicate arguments, and preserved before
launch. The concise active checkpoint is updated at phase boundaries. Finish
with decision/inference tables, strongest alternative explanation, and the
next justified action. No mandatory external review chain is added.
