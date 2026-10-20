# Completed q20 NeuTra training evaluation

**Architecture status, owner directive 2026-09-25:** the revised IAF fits use
the current canonical architecture `bayesfilter_neutra_iaf_author_v1`.
The preserved baseline and legacy-continuation control are **HISTORICAL —
UNFAITHFUL TO THE AUTHOR'S CODE**; their comparisons remain historical diagnostic
evidence, not canonical-map results. Conditional NAF is a separately identified
research alternative. The [policy notice](bayesfilter-neutra-canonical-architecture-policy-2026-09-25.md)
supersedes earlier architecture labels without changing the measurements below.

**The repaired IAF training improved the declared loss and score criteria in
all three seeded fits. NAF did not reproduce that improvement: two fits learned
sharp residual tails despite lower loss. Posterior readiness remains unproved.**

All planned training and verification completed: three IAF fits with 4,096
updates each, three NAF fits with 8,192 each, and 4,096 additional updates of
the preserved legacy map. Every final map passed its required 1,000-point
finite/status checks and exported-map inverse stress check. The baseline and
all seven final maps were evaluated on the common final bank. Bounded HMC
tuning ended without a verified kernel, so posterior sampling did not start.

The target is the four-parameter q20/T30 **UKF approximate posterior**, not the
exact state-space posterior. The [plan](bayesfilter-neutra-precision-training-plan-2026-09-24.md)
specifies parameter provenance, budgets, seeds, criteria and limits. Detailed
[engineering results](bayesfilter-neutra-precision-results-2026-09-24.md)
preserve the precision checks and repairs. Evidence below is under
`artifacts/neutra-precision-training-2026-09-24/`.

## What ran

The shared implementation trained FP32 transport weights and Adam state against
the existing FP64 target, with TF32 enabled for eligible FP32 matrix operations.
Finalization explicitly converted the represented weights to FP64 and diagnosed
that exact exported map. All updates used the batch-native TensorFlow/XLA target
and verified GPU memory growth. Independent jobs used available GPUs 0/1/2;
concurrent worker wall times were summed.

Fits used batch 32, path gradients, learning rate 0.01 and Adam(0.9,0.999,1e-8).
The calibrated emergency clipping thresholds were 31.2098 for IAF, 19.4600 for
NAF and 412.3217 for the control. IAF used three width-16 ELU stages with
source-based masks and free-bias conditional scales. NAF used three conditional
DSF stages, width-16 conditioners and 16 mixture components. The control kept
its four-stage bounded tanh map and started fresh Adam slots from the saved
2,048-update weights. It is a conditional control, not a replicated old recipe.

Actual q20 calibration found FP32/TF32 gradient discrepancy divided by minibatch
gradient variability of 0.0000355--0.0000413 for IAF, 0.000291--0.000304 for
NAF16 and 0.000680--0.001130 for the control. These checks support this precision
choice here; they do not establish equivalence of complete FP32/FP64 training
trajectories or universal TF32 adequacy.

## Untouched final-bank comparison

The original baseline and all final maps used the same 2,048 normal-base points.
The primary quantities were mean negative transformed log density (reverse-KL
loss up to constants) and mean squared vector residual
`||grad_z log pi_z(z) + z||^2`. A fixed map passes only if the adjusted paired
bootstrap upper bounds for both differences from baseline are negative. A
family passes only if all three fits pass. The 4,096-resample percentile
intervals account for up to ten map-baseline comparisons and are conditional
on these maps; three seeds do not establish population-level superiority.

| Map | Loss | Residual RMS | Positive observation weights / 2,048 | Both criteria |
|---|---:|---:|---:|---|
| Preserved baseline |43.5012|4.5135|1,434|Comparator|
| IAF seed 0 |40.5934|0.5448|2,048|Pass|
| IAF seed 1 |40.6383|0.7912|2,048|Pass|
| IAF seed 2 |40.7185|0.9984|2,047|Pass|
| Legacy continuation |40.6564|1.0703|0|Pass for this map|
| NAF seed 0 |40.0323|21.8480|841|Fail: squared residual|
| NAF seed 1 |40.6017|0.8562|2,048|Pass for this map|
| NAF seed 2 |39.9690|32.5173|1,052|Fail: squared residual|

| Map | Adjusted loss-difference interval | Adjusted squared-residual-difference interval |
|---|---:|---:|
| IAF 0 |[-3.035, -2.778]|[-21.210, -18.934]|
| IAF 1 |[-2.988, -2.737]|[-20.866, -18.572]|
| IAF 2 |[-2.912, -2.650]|[-20.641, -18.095]|
| Legacy continuation |[-2.973, -2.711]|[-20.644, -17.717]|
| NAF 0 |[-3.615, -3.302]|[40.336, 1363.123]|
| NAF 1 |[-3.026, -2.771]|[-20.774, -18.487]|
| NAF 2 |[-3.665, -3.396]|[161.892, 2907.744]|

IAF supports the stated replicated local improvement. NAF 0/2 lowered loss but
increased squared residual under the declared test. Their wide intervals show
substantial tail uncertainty; they fail the improvement criterion without
precisely estimating the tails or ranking architectures. The
[paired result](artifacts/neutra-precision-training-2026-09-24/paired-final-analysis/result.json)
preserves the complete statistics and links individual paired contributions.

Sign counts expose a major remaining limitation. Good IAF local geometry
coexists with almost complete concentration in the positive observation-weight
region; the control concentrates in the negative region. These are transport
draws, not posterior probabilities. The two NAF fits that reach both signs have
sharp geometry. The control's own improvement makes additional optimization a
plausible contributor, so gains cannot be attributed solely to architecture or
precision. No family-versus-family ranking was tested.

## Standard 1,000-point verification

These are fresh independent banks on the exact exported FP64 maps. Every row
was finite and valid. Quantiles are descriptive, separate from the paired test.

| Map | Median | Mean | RMS | p95 | p99 | Maximum | Fraction > 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IAF 0 |0.335|0.421|0.517|0.976|1.714|2.513|4.8%|
| IAF 1 |0.436|0.576|0.748|1.540|2.463|4.551|13.6%|
| IAF 2 |0.673|0.809|0.998|1.688|2.735|11.273|22.9%|
| Legacy continuation |0.322|0.455|0.670|1.269|2.814|5.896|7.1%|
| NAF 0 |0.707|4.103|36.454|7.217|70.314|1049.973|35.3%|
| NAF 1 |0.499|0.715|1.014|2.082|3.906|8.336|18.9%|
| NAF 2 |0.569|2.875|19.240|5.437|51.153|434.545|25.0%|

Every inverse stress check on 32 normal draws scaled by four passed the 1e-7
reconstruction/logdet screens. This establishes usable inverses in the tested
scope, not good whitening or coverage.

Clipping no longer affected virtually every update. Counts across accepted
updates, including both portions of each NAF continuation, were 38/4096,
59/4096 and 41/4096 for IAF; 178/8192, 186/8192 and 819/8192 for NAF; and
1/4096 for the control. NAF 2's 10.0% remains a calibration diagnostic.
Initial NAF calibration also showed Adam epsilon changing the first path-gradient
update direction by about 29--31% relative to the zero-epsilon reference. This
is an unresolved optimizer-sensitivity hypothesis, not a demonstrated cause
or evidence that a different epsilon repairs the tails.

## Repairs and tail diagnosis

NAF seeds 0/1 initially failed at updates 2,328/462 because a validity guard
required `exp(log_w)>0`, although the stable mixture consumes log weights.
Finite weights around -118.800 and -107.520 underflowed unnecessarily in FP32.
The repair checks finite log weights and retains finite positive slopes and
finite offsets; it adds no floor, clipping, discarded component or target change.
Both exact failed GPU updates pass after repair. Independent equation and
derivative regressions include a tiny-weight component that dominates a tail.

Seed 2's earlier finalization error arose from editing numerical source during
its live AutoGraph lookup. Its checkpoint was intact. All three resumed exact
Adam/noise state in fresh processes and finished 8,192 updates. Original
failures remain preserved and charged; no later invalid updates occurred.

The final tails are not explained by a demonstrated score bug. A bounded GPU
check inspected four worst and four central validation points per NAF 0/2 map.
Physical and full transformed scores matched central finite differences at
three step sizes. At the smallest step, maximum error scaled by `1+abs(score)`
was 1.14e-7 for the transformed score and 7.66e-8 for the physical score. The
[pointwise check](artifacts/neutra-precision-training-2026-09-24/naf-tail-score-check-01/result.json)
preserves decomposition and all step-size results.

At NAF 0's worst inspected point, the coordinate-2 target pullback was 577.163,
the log-Jacobian score was -56.453 and the residual was 520.935. Its latent
norm is about 0.97, so the issue is not confined to distant Gaussian inputs.
NAF 2 also has large contributions from both terms. These derivatives match the
stated transformed density at checked points; global correctness is unproved.

Reverse-KL averages density values, while score residuals measure derivatives.
A narrow steep region can affect average loss little and HMC geometry greatly.
The loss decrease therefore does not imply adequate whitening. Distinguishing
optimizer, capacity and target geometry as causes needs a targeted follow-up;
these failed fits do not reject NeuTra or NAF generally.

## Bounded HMC result

The predeclared nomination rule chose IAF 0 from the replicated-improvement
family. Its exact frozen map used identity latent mass and four chains started
on both sides of the observation-weight sign boundary. Full-chain qualification
passed. Local curvature proposed epsilon 0.0168302, then the public fixed-map
tuner ran; R-hat did not determine tuning membership.

| L | Epsilon | Stage | Acceptance | Numerical veto | Decision |
|---:|---:|---|---:|---|---|
|3|0.0168302|Pilot|0.9901|None|Increase epsilon|
|5|0.0168302|Pilot|0.9928|None|Increase epsilon|
|9|0.0168302|Pilot|0.9894|None|Increase epsilon|
|3|0.0252454|Measurement|0.9811|None|Increase epsilon|

The evaluations cost 469, 622, 1047 and 422 seconds. The search ended as
`partial_budget` without a verified member. Tuning occupied 2,616.959 seconds:
the synchronous final chunk overran the 2,500-second sublimit by 116.959
seconds before the next budget check. Total supervised downstream cost was
2,765.557 seconds, within the 8,000-second envelope. No threshold was relaxed
and no posterior draws were retained.

This is incomplete tuning evidence, not a mixing failure. The declared training
study is complete; a full HMC attempt needs a newly priced continuation of the
preserved search and posterior work. Independent q20 reference agreement is
also missing. The [downstream result](artifacts/neutra-precision-training-2026-09-24/downstream-queue-01/hmc-assessment/result.json)
records the terminal outcome and links pending repairs.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| IAF improved locally |Both criteria pass for all 3 seeds|Finite/inverse checks pass|Coverage and attribution|Price HMC continuation with sign checks|Correct posterior or architecture superiority|
| NAF family fails this criterion |Seeds 0/2 fail squared residual|No invalidity after repair|Sharp transitions, optimizer sensitivity, capacity|Targeted source-grounded diagnosis|NAF direction rejected|
| Preserve continuation control |Both criteria pass for this map|Finite/inverse checks pass|One fit; negative-sign concentration|Keep optimization as an explanation|Replicated old recipe|
| Defer posterior promotion |No verified kernel or posterior run|Incomplete tuning evidence|Mixing and precision|Reprice pending tuning/posterior work|Convergence from acceptance|

| Inference status | Evidence |
|---|---|
| Hard veto screen |Final maps finite and inverse-valid; original failures repaired and retained|
| Statistically supported ranking |Declared fixed-map comparisons against saved baseline; IAF passes all three, NAF does not; no family ranking|
| Descriptive-only differences |Quantiles, maxima, sign allocations, clipping, timings, acceptance|
| Default-readiness |Configurable precision/execution tested; broad posterior readiness absent|
| Next evidence |Coverage-sensitive downstream assessment and diagnosis of sharp NAF fits|

The strongest alternative explanation for IAF improvement is better optimization
or initialization. The control supports that possibility. The weakest evidence
concerns mass outside the learned sign region and rare NAF tails. Independent
posterior agreement could overturn the coverage concern; replicated training
that removes the tails without losing coverage would overturn the negative
NAF-fit result. These results invalidate neither the target/data nor NeuTra.

## Provenance and accounting

Execution used `/tmp/BayesFilter-neutra-precision-20260924` and
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Initial manifests record their
original commits; repaired runs used pushed merge `ef3e32dd0`. Each manifest
preserves commands, source/native-op hashes, seeds, device, TF32/XLA/dtype,
memory-growth evidence and times. Statistical analysis deliberately hid GPUs.

Starting balance was 143,330.116 worker seconds. Through completed downstream
work, 120,292.958 seconds were charged and 23,037.159 seconds (6.40 hours)
remained before final integration checks. Charges include failed attempts,
engineering/calibration, concurrent workers, the 122.575-second score check and
6.540-second report regression. Waiting for occupied GPUs was not worker time.
No new compute grant was used.

The 46-test repaired equation suite, 15 controller/HMC checks and 36 post-merge
checks passed; scopes overlap and must not be summed as unique tests. The
updated paired-report regression passed. Closing accounting, evidence index
and Git integration are in `terminal-summary-01/` under the evidence root.
