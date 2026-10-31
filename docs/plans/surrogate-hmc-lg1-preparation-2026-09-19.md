# LG1 reference and numerical-control preparation

Continue the authorized all-model master at LG1 attempt 4. The question remains
whether imperfect full-DCTR forces give reliable surrogate-HMC parameter
inference. This preparation supplies its independent model-posterior reference
and examines numerical-control hypotheses; it makes no retained-inference claim.
Initially reserve 1 GPU hour and 1 CPU-worker hour within the remaining master
allocation, including unsuccessful preparation. Jobs have at most 900 seconds;
fresh versioned output directories preserve every attempt. No LG1 inference
attempt is consumed until a retained comparison is launched.

First prepare independent calibration, validation and claim datasets using the
existing fixture partition mapping at attempt 4. The observed data for inference
are their FP32 values promoted to FP64, exactly as consumed by the DCTR wrapper.
Freeze all reference centers, scales and CDF thresholds before retained results.
Claim data never select DCTR numerical controls.

Evidence contract: FP64 scalar Kalman likelihood is compared with an independent
joint-Gaussian observation density. Stationarity gives
`Cov(y_i,y_j)=phi^|i-j|/(1-phi^2)+0.5*1[i=j]`; the first state also has the
stationary law because transition precedes observation. The exact normalized
posterior is proportional to this likelihood on `[-.95,.95]` under the uniform
prior. A composite trapezoidal reference on successively doubled interval grids
(1024, 2048, 4096, 8192 intervals) checks mean, variance, fixed CDF values and the
model Kalman conditional `Pr(y_11>0|phi,y)`. Maximum successive standardized
function difference must be <=1e-5. Twice that difference is an empirical
refinement allowance, explicitly not a rigorous quadrature bound.

Finite likelihood and positive normalization/variance are validity vetoes.
Independent log-density agreement <=1e-10 at nine fixed physical points,
analytic score vs centered finite difference <=1e-5 relative to max(1,|score|),
and prior/chart equality at fixed chart points are reference qualification
requirements. Reference failure triggers refinement or mathematical repair,
not rejection of surrogate HMC. Runtime and density shapes are explanatory.
No stochastic method is ranked by this deterministic preparation.

CPU-only TensorFlow/XLA is an explicit independent-reference exception; hide
GPU devices before import. Stable signatures and batch-native recursions are
required. No pfor, NumPy runtime or learned component is used. This tiny data
generation (three ten-observation datasets) stays on one CPU process; worker
startup would exceed generation cost and adds no independent training evidence.

Default audit: grid size and empirical allowance are numerical hypotheses,
checked by the stated refinement. Uniform proper prior, stationary initial law,
chart and fixed model constants come from master section 10.1. Boundary truncation
is part of the declared prior, not a quadrature shortcut. Calibration/validation/
claim seeds are separate. DCTR settings remain unselected warm starts and no
control is promoted by these reference checks.

Pre-mortem and skeptical audit: a likelihood using a fixed initial variance or
different observations could pass internal checks but target a different law.
The independent covariance identity and exact data hash detect those mismatches.
The CDF discontinuities are integrated by interpolation of cumulative interval
masses, not by an unrefined indicator sum. This plan does not authorize tuning
on claim posterior error. Numerical-control selection requires its own explicit
calibration rule before execution. Proceed with the independent reference stage.

## Full-DCTR numerical-control calibration

Evaluate the same full canonical N1024/T10 FP32/TF32/GPU/XLA route at physical
`phi=(-.6,0,.6,.9)` (nominal) and `(-.949,-.94,.94,.949)` (boundary stress), on
calibration and validation data only. Freeze the particle/noise/design bank
within each partition. Keep substeps10, Contract-E epsilon2, Sinkhorn8/balance8,
two correction steps of strength.2, two pairwise steps of strength.02 and cap
power8 as explicit finite-program hypotheses; score and likelihood error against
Kalman remain explanatory. These controls do not change during retained HMC.

Class C evaluation uses a geometric protection-sensitivity curve, with multiplier
`r` in `(1,.1,.01,.001)`. Set ridge=`1e-5*r`, LM damping=`1e-3*r`, scale floor=
`1e-6*r`, trust radius=`.1/r`, pairwise RMS cap=`2/r` and coordinate cap=`16/r`.
Every arm retains all canonical capabilities. The weaker adjacent protection
is a diagnostic comparator, not a reduced algorithm or an admitted default.
The grid measures whether the initial warm start perturbs healthy trajectories;
it is not a search for posterior error or acceptance improvements.

For adjacent pairs require healthy nominal values/forces and all finite trace
quantities, then require maximum absolute change divided by `1+abs(comparator)`
<=1e-3 for likelihood, analytical force, filtering mean/covariance and post-reset
mean/covariance. This conservative local numerical tolerance is a hypothesis:
passing establishes only measured non-harm at the declared probes. It cannot
certify posterior accuracy. Boundary probes must either remain finite/bounded
or emit a nonzero original failure code; masked failures cannot pass. Record
all existing cap diagnostics. A new failure on a previously healthy probe is a
non-harm veto. No tuning to heuristic or retained-claim results is permitted.

Select the first (strongest protection) adjacent pair passing both calibration
and validation in this predeclared order. Do not rank passing settings by loss,
acceptance or runtime. If no adjacent pair passes, emit `NO_STABLE_CONTROL_PAIR`
and preserve the sensitivity curve as a repair trigger; do not issue an offline
selection artifact. A passing selection is only an
`imperfect_proposal_field_experiment` use classification with exact scope,
source, route/reset identity and separate partition hashes. It grants no
canonical exact-score, default-readiness or model-posterior authority. The
untouched claim bank must consume an exactly matching selection and is subject
to fresh sampler tuning and downstream inference checks. Safety outside the
checked nominal/boundary probes remains unproved, so retain runtime status and
health vetoes. This bounded calibration is not a model-general default change.
