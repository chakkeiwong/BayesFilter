# SIR concentration and materialized-moment repair

## Active question and trace

Continue on `ledh-refactor-with-policy-fix`, preserving the completed offset
repair and unrelated changes. Does the remaining failure arise upstream of
restoration, and can the existing shared moment-descent protection prevent it?
This is a numerical repair investigation, not guide tuning or efficacy testing.

Saved validation-02 evidence from the preceding repair establishes that the
time-4 posterior log-weight gap is 68.9755788. Its prior/proposal contribution
is 69.7839642; the observation contribution is -0.8083854. Thus the immediate
cause of concentration is the proposal-to-transition mismatch, before OT.
ESS is already 1.113 at time 2. Necessary equal-weight kurtosis upper-bound
margins are negative at times 2--5. Exact matching of those shape targets is
impossible with 64 uniform particles. This does not by itself invalidate a
bounded approximation or prove that the correction caused the concentration.

The endpoint currently filters boolean names ending `_valid` or `_finite`;
it reports but ignores `higher_moment_materialized_shape_available`. Therefore
an explicitly unavailable moment calculation can be accepted as valid. This
is an independently repairable call-chain defect. A physical FP64 cloud
cannot retain offsets below its representable spacing; adding a ridge or
inventing spread is not an equivalent repair.

## Execution and evidence contract

1. Preserve the trace analysis and pre-change sources. Add a relative
   covariance-loss diagnostic without the old unit-scale denominator. Make
   unavailable materialized shape fail closed at the moment-training endpoint.
   Regression: valid cases unchanged; unavailable shape is rejected.
2. Evaluate the existing shared `moment_safety=True` descent safeguard as one
   explicitly non-default Class-C candidate. Keep every other control, model,
   data and particle stream unchanged, including OT (1,5), balance32 and K=N.
   No guide or score accuracy selection. Compare the previously failed and
   healthy endpoints, all trace checks, covariance losses, and three analytical
   score directions with two central-difference step sizes.
3. If the safeguard does not repair the failure, preserve that rejection and
   trace the remaining mismatch; do not claim that finite scalars establish
   moment fidelity, replace the canonical algorithm, or retune on the failed
   holdout. Record a specific next mechanism instead of another blind sweep.
4. Update result, master program and concise checkpoint with actual budget,
   findings, limitations and next action. Update the mathematical account for
   any substantive new conclusion.

Baseline: saved offset-repaired endpoint, same inputs and frozen guide-free
coefficients. Primary numerical criterion: correctly reject unavailable
moments; for the safeguard candidate require finite full traces, available
shape, covariance relative max-norm residual <=1e-6 and analytical/FD
discrepancy <=1e-3 normalized by max(1,abs(score)). These are numerical
engineering thresholds, not guide or score-estimation accuracy criteria.
Healthy non-harm for observability/validity edits means unchanged floating
outputs. The existing safeguard may change the algorithm when its descent
check fires; report those changes, never call that bitwise-equivalent. Its
non-harm question is whether the declared shape loss increases relative to
its same-cap/same-restoration zero-fit comparator.

Invalid inputs, nonfinite target operations, derivative mismatch, loss
increase, or wrong restored covariance veto candidate acceptance. Exhausted
budget, corrupted evidence or a required material target change veto
continuation. Candidate rejection alone does not reject the research idea.
ESS, weight decomposition, feasibility, runtime and moment loss explain the
failure; none proves score accuracy. No default, HMC or efficacy promotion.
Heuristic quality comparisons are not run: this diagnostic does not rank a
learned method; the unguided baseline and zero-fit safeguard comparator answer
the numerical question. Future efficacy work still needs its full ladder.

## Defaults, review and resources

SIR d18, T5/N64, FP64/GPU/XLA, TF32 off: inherited failed fixture, diagnostic
only, not tuned scientific defaults. OT(1,5): owner-frozen, no investigation
or tuning here. Moment safety off: inherited comparability setting, now
explicitly challenged because target shape is infeasible. Safety on: existing
bounded descent implementation, a hypothesis with its own derivative and
non-harm checks, not a new default. FD steps1e-4/1e-5 and threshold1e-3:
preceding repair checks, agreement at both steps required. Covariance1e-6:
diagnostic numerical threshold to distinguish rounding from complete loss,
not a scientific admission tolerance.

Skeptical review: PASS for this bounded investigation. Wrong-baseline risk is
controlled by exact saved seeds and coefficients, not a different guided arm.
Proxy loss is not a promotion metric. A fixed pre-existing safeguard is tested
for numerical non-harm, not selected on heldout quality. Do not infer an
upstream coding error from concentration alone. The main pre-mortem is that
the safeguard still fails because it cannot make an impossible shape or
sub-ULP cloud representable; rejection must remain explicit. A fresh whole
filter representation or proposal redesign is not silently authorized as an
equivalent numerical change.

Artifact root: `docs/plans/artifacts/ledh-moment-collapse-repair-20261020/`,
unique launch directories. Use the remaining preceding stage budget:
6948.324006070 GPU-process seconds, at most4 launches; inherited campaign
remaining27010.281125770 seconds. No training or full comparison resumed.
Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. GPU commands
escalated, `CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true`, verified
growth before initialization. CPU regressions explicitly hide GPUs.

## Reviewed mechanism diagnostic after safeguard rejection

Replay-01 rejects the safeguard as a repair for collapsed physical spread.
The failed time4 prior/proposal logit gap is69.784; observation contributes
-0.808. The next question is whether the shared flow is poorly matched to
its conditional proposal, not whether more Sinkhorn steps help. Read-only
trace analysis has verified that flow coefficients use predicted UKF P while
the zero-guide proposal and transition density use conditional Q. Covariance
carry is the shared row-normalized transport of posterior UKF covariances.
This difference is not by itself a coding error or permission to remove P.

For the two saved fixtures, replay each shared flow on its unchanged incoming
anchors, noise, observations and covariance. Compare the actual P-flow with
a Q-flow diagnostic and the exactly computable conditional Gaussian posterior
for linear SIR observations. The latter conditions on one parent; it is not
an exact nonlinear filtering posterior or score oracle. Record reconstructed
proposal-density and child parity, covariance carry parity, conditional
Gaussian KL, one-step weight ESS, and weight-gap decomposition. Exact
conditional proposal weights are the parent predictive observation density.
No trainable guide, complete filter, default, or canonical lifecycle changes.

Evidence contract: source/trace parity is the engineering pass criterion;
1e-8 absolute child error and1e-8 relative covariance-carry error are diagnostic
replay tolerances. Nonfinite inputs, mismatched proposal densities (>1e-8),
negative KL below -1e-8, or failed parity invalidate interpretation. KL and
ESS explain one fixed incoming cloud only; they cannot promote the Q-flow,
rank trained methods, or establish full-filter improvement. Q-flow is an
explicit noncanonical conditional reference, never a replacement endpoint.
All results go to a new `mechanism-01` directory under this plan's root.

Default audit: Q and R are checked from the actual model, not inherited
guesses;24 flow steps and exact paired noise are frozen replay choices. The
conditional Gaussian equations provide an independent comparator without
requiring score labels. CPU/non-XLA is a small saved-trace reference exception,
explicitly hiding GPU, bounded to180 seconds and one attempt plus one local
repair. It does not spend the GPU-process budget.

Skeptical review: PASS. The main risk is calling covariance mismatch a code
bug, or mistaking better conditional weights for a validated canonical repair.
These inferences are prohibited. A persistent mismatch calls for a separately
reviewed proposal change; a shared replay mismatch instead triggers code repair.

### Follow-through: distinguish a feasible repair from the current guide family

Mechanism-02 passes replay and Gaussian-identity checks. At the failed time4,
mean conditional KL is529.98 for the actual flow, versus8.13e-9 for Q-flow;
paired one-step ESS is1 versus29.04. This is a mechanism diagnostic, not a
full-filter comparison. Mechanism-01 failed on TensorFlow solve batch ranks;
the local rank repair changed no scientific input and mechanism-02 succeeded.

One final CPU diagnostic computes the preimage of the exact conditional
Gaussian through the actual shared affine flow F(z)=b+Bz. For desired output
mean m and covariance V, use preimage mean B^{-1}(m-b) and covariance
B^{-1}VB^{-T}. Feed preimage samples through the same shared flow and check
their equality to the desired samples, log-density change of variables, and
constant conditional importance ratio. Report the eigenvalues of additional
information B^T V^{-1} B-Q^{-1}. Negative eigenvalues would show that a
positive-information pre-flow guide cannot represent this particular exact
repair; they would not disprove simulated-moment training in a richer family.

Review: PASS for a fixed-cloud reference only,180 CPU seconds, one launch plus
one infrastructure repair, source replay tolerances unchanged. No new runtime
endpoint, baseline replacement or scientific default. This checks a concrete
repair algebraically and numerically without silently removing the UKF flow.
An integrated proposal changes the sampling law and needs its own analytical
tangent, full-consumer, tuning and fresh-heldout evidence; reference success
alone does not authorize an efficacy claim.

### Terminal call-chain correction and bounded regression

The terminal audit found an overbroad draft inference. The campaign's kernel
arm explicitly selects mixture_product_moments, not positive_information.
Its actual guided_gaussian_draw branch retains between-component covariance,
which can expand covariance. Thus the negative required information eigenvalue
excludes the trained quadratic adapter, not the kernel-mixture consumer.
Whether that restricted fitted mixture can realize the preimage is not checked.
Correct the manuscript and result accordingly; do not relabel this as a KDM
expressiveness failure.

Reviewed regression: prior N(0,1), equally weighted guide kernels centered at
-3 and +3 with variance1. The product has component means +/-1.5 and variance
0.5, hence moment-matched variance2.75. Exercise the actual guided draw branch
and check output scale, finite density, zero tangents. This is a CPU-only
mechanical identity test, not tuning or efficacy evidence; bound one targeted
suite to180 seconds. The existing full-consumer product-moment test also
checks the shared route. This closes the specific narrative/call-chain error.

## Completion

All four bounded steps and both mechanism follow-through checks are complete.
Result: artifacts/ledh-moment-collapse-repair-20261020/result.md. 102 core
regressions plus10 mixture-consumer tests pass; six healthy GPU analytical/FD
checks pass. The failed fixture still has unavailable shape and is rejected.
No guide/default is promoted. The exact preimage is a fixed-cloud reference,
not an integrated runtime repair. Remaining stage budget6800.396892893 GPU
seconds/3 launches; inherited campaign remainder26862.354012593 seconds.
The next integrated proposal needs its own reviewed numerical contract.
