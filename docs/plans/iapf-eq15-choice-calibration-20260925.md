# iAPF fit and floor discrimination, 25 September 2026

Status: COMPLETE — controls and terminal audit PASS; zero fit nominees.
Untouched confirmation and adaptive pilot NOT_ADMITTED under the stated criteria.
[Result](artifacts/iapf-eq15-choice-calibration-20260925-01/result.md) and
[decision](artifacts/iapf-eq15-choice-calibration-20260925-01/decision.json).
Authority: owner's request to plan, review, and execute. This is master phase I2a.2.

Qualification review, 26 September: **REVISE BEFORE REUSE**. The original
screen was executed correctly but conflated local shape dominance, numerical
validity and floor protection. Its zero-nominee result is conditional on that
screen. See the [criteria audit](artifacts/iapf-eq15-choice-calibration-20260925-01/qualification-review-20260926.md).
Frozen pre-run snapshots and original results remain unchanged.

## Question and source boundary
Which choices yield a useful finite local solution of printed Eq15, and what
floor in Eq16 has a defensible interpretation? Accepted manuscript Section5.1,
local paper text lines911--945, fixes density least squares, diagonal covariance
and a positive constant; optimizer and floor formula are unspecified. This
study can nominate a qualified reconstruction, not recover the author's code.
Independent CPU R diagnostics are an explicit reference exception; TF/TFP
GPU/XLA and full-filter evidence remain required for runtime/scientific status.

## Skeptical review and corrections before execution
First draft: REVISE. It repeated known failures, mislabeled saved failures as
fresh validation, understated its factorial count, proposed an arbitrary 1%
threshold, and omitted a concrete mixture floor. It also confused warm starts
across times with reuse across outer iterations at the SAME time index.
The completed 22September author-choice campaign already tested joint/profiled
fits, QR/moments, warm starts, strict/loose stops, 200/600 iterations and floors
without an all-scope survivor. Its data and the ten saved failures are diagnostic,
not untouched. Revised plan: PASS subject to objective/derivative controls.
This is a skeptical self-review; no independent review is claimed.

## Intent and evidence contract
Question: conditioning/start failure, missing local constraints, or structural
amplitude escape? Floor question is separate because c is added AFTER fitting.
Baseline target: unmodified Eq15, diagonal covariance, joint positive amplitude
or its algebraically identical profile lambda=(p'y)/(y'y).
Cheap adversaries: QR initializer; target-weighted moments; diagonal Laplace
approximation where target derivatives are known. Exact Gaussian truth is oracle.
Evaluate each separately in exact, correlated, heavy-tail, bimodal and saved-input
regimes. These comparisons are falsification checks, never alternate objectives.
Nomination requires finite converged INTERIOR fits, no numerical boundary,
underflow or escape, exact-fixture relative shape <=1e-12 and mean/logvariance
error <=1e-5, and held-out shape no worse than the best valid cheap baseline
within 1e-8 absolute numerical tolerance in EVERY generated fixture class.
This is nomination only: downstream computation remains necessary. Baseline
losses veto nomination without proving a statistical ranking.
Active local constraints veto an unconstrained Eq15 claim; they remain evidence
about the declared constrained problem. Bad gradients, corrupt input, invalid
normalizers, budget exhaustion or missing required output stop dependent work.
Ordinary candidate failures trigger subsequent planned arms and do NOT stop
independent floors. Harness errors allow bounded repair in a fresh directory.
Loss, gradients, iterations, scale/determinant movements, runtime, effective
target count, and floor weights are explanatory. Tiny raw loss cannot pass.
No author identity, paper table replication, stochastic superiority, full-filter
correctness, or GPU/TF default-readiness conclusion is permitted.

## Stages and fixed choices
1. Controls: finite-difference joint/profiled derivatives, profiled value identity,
target-rescaling invariance, exact-start no-harm, proposal normalization versus
independent 1D integration. Stop dependent stages on a failed control.
2. Calibration fixtures: exact diagonal Gaussian d=1,5,20,80; correlated Gaussian
d=5,20,80; broad-tail and bimodal d=1; ten distinct saved Eq15 failures.
Fresh generated calibration clouds have separate held-out points with callable
targets. Saved inputs lack an independent held-out target; preserve that missingness.
3. For every fixture, joint/profiled x nlminb/damped-GaussNewton x QR/moments
gives 8 arms. Add six profiled-nlminb-QR ablations: maxit600, loose stop,
unscaled coordinates, local boxes radius .25/1/4 in fixed standardized mean
and logvariance coordinates. At most266 calibration fits; default maxit200.
Damping starts1e-3 and changes the step only. Fixed per-input residual and
coordinate scaling do not change Eq15. Bounds are probes, not defaults.
Exact-start roundoff guard returns parameters unchanged.
Warm-start behavior already has failed full-filter evidence; test new real
same-time/previous-outer-iteration wiring only if a policy reaches the pilot.
Do not call oracle-parameter starts or adjacent-time starts deployable warm starts.
4. Floor study proceeds regardless of fit failures using fixed Gaussian twists.
N1000,d1/5/20/80,Q=I,parent offsets0/1/3 per coordinate; healthy,
underdispersed and miscentered regimes. Compare tail powers2/4/8, peak/N
powers2/4/8, mixture-delta=1/N and1/N^2, zero control (324cells).
Record actual mixture mass, exact normalization, healthy distortion bounds;
in1D use deterministic quadrature of importance second moments against the
exact conditional. Construct base-transition, no-floor fitted, observation-only
Gaussian comparators; exact conditional is oracle. Compare by regime; do not
tune on these heuristic challenges.
5. Freeze at most2 nominees in selection.json BEFORE drawing untouched
confirmation fixtures. Fixed simplicity order: profiled before joint, QR before
moments, no active constraint; this is not a statistical ranking.
Distinct confirmation seeds; d1/5/20/80 exact and correlated cases.
If no nominee, confirmation NOT_ADMITTED; finish all independent diagnostics.
6. Only after confirmation and a justified floor, run one fresh d5,T100 adaptive
pilot, N0=1000,k5,tau=kappa=.5,max20iterations/max16000particles, exact Kalman,
BPF and fully-adapted APF. Independent final filter draw and previous-iteration
same-time warm-start ablation. One pilot is mechanics only; no convergence or
replication claim. If prerequisites fail, report NOT_ADMITTED, not incomplete execution.

## Floor mathematics and non-harm
Z(mu)=N(mu;m,Q+V); original mixture pi(mu)=c/(Z(mu)+c).
For fixed Q=I, Zmax=(2*pi)^(-d/2)|I+V|^(-1/2).
Constant c=delta/(1-delta)*Zmax gives pi(mu)>=delta, equality at mu=m.
This is a disclosed formula for an unspecified paper choice, not author code.
Use one constant per twist, never parent-dependent c.
For correct conditional q0 and q=(1-pi)q0+pi*f,
integral(q0^2/q)<=1/(1-pi). Record this conservative distortion bound.
Center non-harm screen: pi<=1/N (at most one expected base draw in N);
declared Monte Carlo-scale tolerance, not source assertion. Distant-parent
and misspecified cases expose protection cost; cannot tune primary performance.
Zero is control-only. No floor can be a full-filter recommendation from local
integrals alone. Strictly positive mathematical c need not be numerically effective.

## Assumptions/default audit
- Density residual/diagonal covariance: paper baseline; correlations and modes
  violate exact representability; include both and an exact oracle.
- QR/moments: local hypotheses; QR nonconcavity/rank failure and moments'
  implicit weighting by the cloud distribution are risks; emit rank/ESS failures.
- Profiling: exact algebra; does not remove escape; verify gradients and scales.
- Fixed scaling: conditioning hypothesis; freeze once per input, verify identity.
- Damping1e-3,200/600,stop thresholds: historical numerical hypotheses, not
  author defaults; strict/loose and trajectories expose false convergence.
- Box radii .25/1/4: sensitivity probes; active bound, exact no-harm and radius
  dependence must be reported; never silently accept a boundary optimum.
- Tail/peak powers: historical unverified choices; test dimension-dependent pi.
- Mixture1/N,1/N^2: new scope-specific protection hypotheses; derivation above,
  center non-harm and distant-parent tests; no universal default.
- Double precision/CPU/R: independent reference only; hide GPU before execution.
- Finite tolerances: implementation screens; do not certify downstream correctness.

## Budget, seeds, artifacts
One CPU process/one BLAS thread. Main run<=900wall seconds, repairs and any
conditional confirmation<=300 additional seconds; total1200 worker seconds,
<=400fits and<=400floor cells. Fits remaining historical CPU balance1773.85s.
GPU budget unchanged. Failure consumes budget. New attempt directory per launch.
Root docs/plans/artifacts/iapf-eq15-choice-calibration-20260925-01/.
Seed families: calibration9251000+10*d+case; heldout9252000+10*d+case;
confirmation9253000+10*d+case; pilot9254001. Save all generated fixtures.
Manifest: both commits/branches, exact commands, snapshots/hashes, R version,
CPU-only CUDA_VISIBLE_DEVICES=-1, seeds, wall times, plan/result/output paths.
Write CSV rows incrementally, preserve errors, and account for every scheduled cell.

Premortem: exact QR starts hide optimizer failures; low loss hides amplitude
escape; high dimensions concentrate target weights; bounds hide escape; tiny
positive floors may do nothing while larger floors erase twisting. Poor starts,
source-input replay, class-specific validation and mixture algebra address these.
At closure: decision/inference tables, strongest alternative explanation,
limits/next action, updated master/checkpoint. No GPU campaign is launched here.

## Terminal audit amendment, after attempt01
Attempt01 completed266 fits and324 floor cells in54.9414 CPU wall seconds;
no nominee. Before final interpretation, replay26 profiled QR fits (LM/nlminb,
three correlated andten saved inputs) from its frozen function definitions.
Preserve final parameters, assert original scalar/shape/status parity, and
measure gradients and Hessians relative to final density scale. This is
post-hoc explanatory diagnosis of absolute-gradient stopping, not a changed
selection rule or an untouched validation. Verify floor-mixture algebra and
no-floor quadrature against independent Gaussian second-moment formulas;
record conditional heuristic losses as machine-readable vetoes. Cap240seconds
and26 additional fits, no new floor grid; total292fits remains within400.
Original calibration artifact and pre-run plan snapshot remain preserved.

## Completion record
Both attempts exited zero: 266 calibration fits plus 26 audit replays, and 324 floor cases.
Worker CPU wall charge 63.781892 seconds of 1200; GPU zero. Thirteen source snapshots verified.
No solver/floor policy qualifies for a full-filter default. The remaining
repair is scale-aware termination and a justified finite local Eq15 branch;
see the result for the decision, inference limits and conditional heuristic vetoes.
