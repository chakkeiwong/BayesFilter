# M10 nonlinear model: scope closure and likelihood reference

The goal remains testing surrogate HMC with imperfect full canonical DCTR
forces. This stage establishes the next independent model scope and checks a
likelihood reference needed for its endpoint and posterior comparisons.
It does not run retained HMC or qualify a parameter-posterior reference.

## Model identity and inference task

The actual score-study law is
`x[t] = a*x[t-1] + c*sin(x[t-1]) + sigma_Q*epsilon[t]` and
`y[t] = H*x[t] + b*x[t]^2 + sigma_R*eta[t]`, with independent standard
normal innovations. Initial x[0] is normal with mean m0 and variance P0.
The equations and transition-before-observation convention are in
`bayesfilter/score_study/nonlinear_tf.py:1-46`. Parameters derive from
`bayesfilter/score_study/gaussian_tf.py:112-131`; in dimension one,
H=1.15*theta[3], m0=0.7*theta[4], P0=1.15*exp(2*theta[5]). These factors must
not disappear when defining the surrogate-HMC consumer.
The source converts some Python constants through `tf.cast`, retaining FP32
literal rounding even in its FP64 route. Both independent integrations use
the actual source parameter interpretation and matching curve conversions;
this stage preserves that finite model rather than silently changing it.

This law is distinct from M09 Model B, whose second state obeys
`k[t]=alpha*k[t-1]+beta*tanh(m[t])` and observation is `m[t]+k[t]`
(`bayesfilter/testing/nonlinear_models_tf.py:82-125`). It also differs from
Model C's rational growth transition, deterministic phase coordinate and
observation `x[t]^2/20` (`nonlinear_models_tf.py:248-302`). No deduplication
is justified. Model B's existing proper prior stays with Model B; it must not
be transplanted into M10.

Declare M10-T2, using the existing concrete short fixture:
theta0=(0.62,-0.8,-0.6,0.9,0.25,-0.3), c=0.2, b=0.12, horizon2.
The coefficient/horizon sources are
`scripts/run_younis_iapf_fresh_calibration.py:34-37` and
`tests/highdim/test_younis_score_master_nonlinear_consumer.py:16-17`.
This chooses an existing model instance, not evidence that its numerical
settings transfer to DCTR. N1024/K1024 full DCTR remains the future consumer
scope; controls require separate tuning before retained inference.

Use the master program's proper noise-scale task: infer lambda_Q and lambda_R
with independent N(0,log(2)^2) priors, identity chart. Add lambda_Q/R to
theta0[1:3]; this scales the standard deviations, hence covariance by
exp(2*lambda). Keep a,H,m0,P0,c,b fixed. The posterior score is likelihood
score[1:3] minus lambda/log(2)^2. No scale Jacobian is added because the prior
and integration coordinates are log scales themselves. This prior is the
master's declared hypothesis, not an inherited prior in the score benchmark.

Generate fresh calibration, validation and untouched claim data using source
`make_data_kernel`, seeds (20260920,14100/14200/14300), nominal lambda=(0,0).
Also freeze a stress dataset for each role with lambda=(-log(2),-log(2)), using
second seed components15100/15200/15300. Store once-rounded FP32 observations
and their hashes. Evaluate reference probes only on calibration/validation;
claim data are generated and saved but not used for selecting controls.

## Evidence contract, assumptions and stop conditions

Primary stage criterion: finite FP64 state-grid values and analytical scores,
mesh and domain changes <=1e-6 in log likelihood and <=1e-5 in score/(1+abs
score), initial/predicted omitted-mass and posterior-edge indicators <=1e-8.
Probe the 3x3 log-scale grid {-log(2),0,log(2)}^2 on nominal and stress
calibration/validation data. The grid ladder is (points,radius): (129,9),
(257,9), (513,9), (769,13.5). The final two mesh/domain comparisons must pass;
otherwise preserve the failure and stop this candidate without treating it
as evidence against surrogate HMC.

Independent checks: c=b=0 must agree with a separately coded scalar Kalman
recursion; centered finite differences must match both analytical log-scale
scores. On calibration data, compare the nonlinear likelihood at three fixed
probes (0,0),(-log(2),-log(2)),(log(2),log(2)) with direct integration over
the initial Gaussian and both innovations at Hermite orders64 and128.
Require their own log-value refinement <=1e-6 and grid agreement <=1e-6.
These empirical checks do not yield a rigorous quadrature bound.

The heuristic reference adversaries are the independent linear identity,
coarse state integration and direct innovation integration. They expose
wrong ordering, noise scale, truncation and shared finite-grid bias. This is
reference qualification, not a comparison of learned samplers; no heuristic
dominance claim for HMC is possible until its actual arms are evaluated.

Grid sizes/radii are numerical hypotheses motivated by the existing source
test, not promoted defaults. Gaussian-Hermite orders are an independent
refinement hypothesis. The finite log-scale probe region is not the full
normal-prior support. A successful local likelihood check does not qualify
full-prior posterior integration, tail error, all parameter values, a longer
horizon or a consumer. Those remain separate obligations. Weakly identified
noise scales at T2 are expected; a proper prior makes the target explicit but
does not make a short model representative of longer sequences.

Skeptical audit: correct law and initialization factors are explicit; no
moment projection stands in for a likelihood; calibration/validation/claim
roles are separate; independent nonlinear integration and derivative checks
can falsify the grid reference. The earliest failure reports the actual
reference issue. Missing tails outside the probe scope bar posterior-reference
promotion even when this stage passes. This plan is adequate for local
likelihood preparation, not retained model inference.

## Execution

CPU-only FP64 TensorFlow/XLA independent-reference exception, GPU intentionally
hidden; no NumPy, pfor or autodiff score. Static signatures bound each grid and
Hermite order. CPU reference execution does not change the GPU default for the
future DCTR consumer or HMC. Budget:300 CPU-worker/300 wall seconds across one
run and one infrastructure retry, plus40 CPU seconds for focused tests, within
M10's initial7200 GPU/1800 CPU-second family allocation. Charge all attempts.
Watchdogs and a1.5-times measured forecast before the final integration checks
enforce the cap. Do not shorten the ladder or skip checks to claim a pass.

Run `scripts/prepare_surrogate_hmc_nonlinear_reference.py --output <fresh>`
using tftwogpu, CUDA_VISIBLE_DEVICES=-1 and TF threads2/1. Initial root:
`docs/plans/artifacts/surrogate-hmc-all-models-20260920-m10-reference-01`.
Save data, full logs, per-grid values/scores, manifest, source snapshots,
accounting and result note with decision/inference tables. Next justified work
after a pass is full-prior posterior-reference qualification and the actual
full-DCTR consumer, not an inference claim.
