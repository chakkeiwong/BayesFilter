# M13 canonical likelihood-consumer integration — 2026-09-23

This is the existing surrogate-HMC closure master's canonical-consumer task,
following the qualified nine-point M13 likelihood reference. Its immediate
question is whether the actual canonical analytical DCTR route can consume
the range/bearing model with its wrapped likelihood and total analytical
parameter score. HMC inference still requires a parameter prior, full-support
reference, scoped control selection and public artifact-authority tuner.

## Evidence contract

Use the state4/observation2 constant-velocity model and transition-before-
observation law from the actual M13 fixture. Parameterize Q and R by
exp(2 lambda_Q) and exp(2 lambda_R). This integration stage computes a
likelihood only; it does not silently choose or truncate a parameter prior.
Seed701 is qualification data; subsequent control tuning and posterior claim
data must have separately frozen calibration/validation/claim partitions.

The independent comparator is the qualified FP64 trajectory reference at
nine fixed points, plus exact local Gaussian and wrapped-density identities.
Primary engineering checks are executable callback wiring into
canonical_batch_fused_value_score, complete analytical parameter dependence,
determinism, finite known-good evaluations, total-derivative parity, and
same-law observation values. Likelihood errors explain approximation quality;
a raw absolute log difference is not a posterior rejection criterion.

Numerical-invalid statuses, missing callback dependence, wrong law, incomplete
canonical reset/correction or score mismatch veto consumer qualification.
They trigger localized repair or later scoped-control tuning; they do not
reject surrogate HMC or authorize a different force. Absence of an admissible
endpoint/reference blocks posterior inference only. No speed, posterior,
all-model success or default-readiness claim follows from this integration.

## Call-chain finding and repair

The canonical PerPointScoreModel adapter exposes observation value and tangent
callbacks aligned with particle parameter rows. The shared wrapped kernel
currently only accepts FP64 rows and one common covariance. Extend that same
implementation to batched covariances and FP32, preserving the existing FP64
API. Both paths must call one numerical core; no lane-specific reduced formula.
Tests must compare batched versus common-covariance paths and analytical
tangents against independent finite differences, including radius-angle
correlation. The canonical likelihood callback must call this shared kernel.

Use the explicit polar Jacobian and its directional derivative, including
sqrt(x^2+y^2+1e-12), covariance-scale derivatives and both likelihood residual
and covariance terms. Guard invalid polar geometry and covariance without
changing accepted values. The current wrapped-kernel maximum conditional
sigma is a numerical qualification limit, not a model-support boundary.

## Assumptions and controls

The fixture's linear transition and full-rank Q are checked directly.
FP32/TF32 is the canonical execution hypothesis; FP64 is the independent
parity arm. TensorFlow/XLA, stable signatures and memory growth are required.
Frozen Gaussian particle/noise designs and Contract E residual designs must
be recorded with the complete canonical controls. Existing AR2 controls may
only seed a mechanical check: changed state/observation dimensions and horizon
create a new tuning scope. Such checks cannot issue a tuning artifact or
qualify the controls. Fresh scope-specific offline tuning remains required.

The earliest diagnostic is a small CPU-only callback/parity test, followed by
a bounded GPU canonical-consumer smoke. Small fixture particle counts obey
K=N and establish mechanics only. Full-horizon cost is measured before
allocating a control ladder. Local probes and reference point errors remain
explanatory until the later posterior evidence contract is satisfied.

## Budget, artifacts and skeptical audit

Allocate at most2400 CPU-worker/3600 GPU-wall seconds within the remaining
campaign balance CPU18554.461450/GPU193498.569329. At most3 numerical attempts.
Fresh roots use surrogate-hmc-closure-20260922-m13-consumer-NN.
Archive failed attempts, commands, source hashes, seeds, device policy,
status tensors and charges. Stop dependent work for invalid target/math,
corrupt evidence or exhausted allocation. Continue independent master work.

Pre-mortem: a correct standalone wrapped kernel can be unreachable from the
claim-bearing consumer; a score can omit covariance dependence; a scalar
covariance callback can accidentally read only the first parameter row; and
cross-model controls can be mislabeled tuned. The executable wiring/parity
checks and explicit non-admission of diagnostic controls address those risks.
A successful smoke still does not establish usable HMC performance.

Pre-run audit: the next CPU-only regression checks the shared kernel against
independent alias sums and finite differences, not posterior promotion. GPU
and scope-specific tuning are explicitly deferred to their later stages.
The finite alias scale limit is a numerical veto, not a support restriction.
The bounded kernel regression is an engineering check charged separately.

First numerical attempt: FP64 N16/T2 at (0,0) and (.2,-.2), complete
finite-program score versus central differences h=1e-5, maximum error divided
by 1+absolute FD <=3e-5, plus exact repeatability. Only after passing, evaluate
FP32 N64/T20 at the saved nine reference points. All status diagnostics are
archived; errors against the likelihood reference are explanatory. A failed
warm start triggers M13-scoped repair, not a change of scientific target.

The polar derivative is derived directly: for p=(px,py), s=p.p and
r=sqrt(s+1e-12), dr=p.dp/r and d(angle)=(px*dpy-py*dpx)/s.
The range Jacobian derivative is dp/r-p*(p.dp)/r^3; the angular
Jacobian derivative is rotate(dp)/s-rotate(p)*2*(p.dp)/s^2.
The likelihood residual derivative is minus the observation derivative;
dR=2*R*d(lambda_R), and dQ=2*Q*d(lambda_Q). The canonical routine
propagates these analytical tangents through the same finite value program.
