# Zhao-Cui Austria SIR Lane-B T1 Execution Note

Date: 2026-07-30

Status: `EXECUTED_PASS_NEW_FIXED_VARIANT_T1_VALUE_BASELINE`

Baseline identity:
`zhao_cui_austria_sir_fixed_variant_training_base_v1`

This note supersedes only the inactive Lane-B authorization boundary in
`bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-plan-2026-07-30.md`.
It does not change the terminal Lane-A verdict
`BLOCK_EXACT_P88_RECOVERY_EXHAUSTED` and does not rename this baseline P88.

## Scope And Research Intent

The approved campaign is deliberately bounded to B0/B1 and the T1 admission
gate. It asks whether a newly named TensorFlow training-base squared-TT/TTSIRT
step can be made deterministic, measure-correct, scale-identified, reloadable,
and memory bounded for the sealed Austria SIR `y1` target. T2 construction is
permitted only after the T1 gate passes. Score, T20, HMC, and comparisons to
GenUT/SGQF/UKF are outside this campaign.

| Field | Binding decision |
|---|---|
| Candidate | `zhao_cui_austria_sir_fixed_variant_training_base_v1` |
| Fixed parameter | `theta=(0,0,0)` |
| Filtering state | coherent continuous pre-clipping latent state `z_t` |
| T1 joint order | `[z_1,z_0]`, 36 axes |
| Retained T1 prefix | axes `0:18`, the `z_1` filtering density |
| Event order | draw `z0`, transition to `z1`, observe sealed `y1` |
| Value | `log Z_1-c_1`; later sequential value is `sum_t(log Z_t-c_t)` |
| Trainer | existing TensorFlow training-base objective, batch-native XLA update |
| Excluded dependencies | APF, source replica, retained grid, ALS/TT-cross, UKF |
| Output root | `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/attempt-NN/` |

## Source And Extension Ledger

| Operation | Classification | Anchor |
|---|---|---|
| Sequential target `previous * transition * likelihood` | `source_faithful` | Zhao-Cui paper Algorithm 2(a), Eq. (15), `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:693-719`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:72-80,132-135` |
| Squared-TT density and marginal contraction | `source_faithful` | paper Algorithm 2(b-c), same paper text `:703-722`; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:1-87` |
| Physical-to-reference density conversion | `source_faithful` | author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/SIRT.m:50-70`, `ApproxBases.m:263-273`, `Polynomials/Legendre.m:29-35` |
| Conditional KR/TTSIRT transport | `source_faithful` | paper Algorithm 3; author `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100` |
| Frozen arrays, ranks, frames, schedules, seeds, identities | `fixed_hmc_adaptation` | deterministic adaptation of author randomness; no HMC authorization |
| Pre-clipping latent filtering law | `extension_or_invention` | `bayesfilter/highdim/sir_latent_preclip_tf.py`; it is a coherent replacement for the singular clipped-state density mismatch |
| Training-base optimizer, deterministic sample frame, L1 tuning, and scale anchor | `extension_or_invention` | local BayesFilter fixed-variant line; not author TT-cross/ALS |

No whole-route source-faithfulness claim is made.

## Exact Target And Measure

At T1, for physical joint variable `x=(z1,z0)` and fixed `theta=0`, the
unshifted Lebesgue density is

\[
 p_X(x)=p_0(z_0)f(z_1\mid z_0)g(y_1\mid z_1).
\]

Let the deterministic affine frame be `x=mu+Lr`, and apply the algebraic map
coordinatewise,

\[
 u_j=\frac{r_j/a_j}{\sqrt{1+(r_j/a_j)^2}},\qquad
 r_j=a_j\frac{u_j}{\sqrt{1-u_j^2}}.
\]

The local polynomial basis is orthonormal under the uniform probability
measure `nu(du)=2^{-36}du` on `[-1,1]^36`, not under Lebesgue `du`.
Consequently the target density represented by the squared TT relative to
`nu` is

\[
 \rho_\nu(u)
 =e^{c}p_X(\mu+Lr(u))|\det L|
   \prod_{j=1}^{36}\left|\frac{dr_j}{du_j}\right|2^{36}.
\]

This `2^36` factor is mandatory. The author implementation adds it through
`mlogw=-sum(log(0.5))` in `SIRT.potential_to_density`. Omitting it shifts
every T1 log normalizer by `-36 log 2`. The implementation must expose the
affine, algebraic-Jacobian, and inverse-reference-density terms separately and
test their exact sum.

The exact shifted normalizer is

\[
 Z_1=e^c p(y_1),\qquad
 p(y_1)=E_{z_0\sim p_0,z_1\sim f}[g(y_1\mid z_1)].
\]

The density model is `rho_hat=h^2+tau`, relative to `nu`. Weighted
cross-entropy identifies normalized density shape but not the physical
normalizer. Absolute scale is therefore supplied by an independent Monte
Carlo anchor and applied after shape fitting. If `Z_h=int h^2 dnu`, the core
scale is

\[
 s=\sqrt{\frac{Z_{target}-\tau}{Z_h}},
\]

which is valid only when `Z_target>tau` and both masses are finite. The
defensive term is not silently rescaled.

## Evidence Contract

| Role | Rule |
|---|---|
| Primary T1 pass | fresh-process reload identity; exact TT contraction equals serialized calibrated normalizer; untouched MC log-normalizer interval contains the calibrated value under the predeclared rule |
| Promotion veto | incoherent measure; wrong observation hash/event order; identity/tamper failure; nonfinite or nonpositive mass; scale target `<=tau`; validation or untouched scale mismatch; compiled-kernel mismatch; forbidden dependency; memory cap breach |
| Repair trigger | training instability, shape failure, validation scale failure, or insufficient MC precision triggers another predeclared T1 arm within budget |
| Explanatory only | cross-entropy, centered log-shape RMS, raw residuals, runtime, inverse-CDF residuals, and descriptive memory statistics |
| Nonclaims | no exact nonlinear likelihood theorem; no score; no T2/T20; no HMC; no comparison ranking; no source-faithful assembled-route or production claim |

For an iid likelihood cloud with likelihoods `ell_i`, compute the mean and
standard error in scaled linear space. The reported log standard error is
`SE(mean ell)/mean ell` by the delta method. Validation and untouched gates use

\[
 |\log Z_{TT}-(c+\widehat{\log p(y_1)})|
 \le 3\,SE_{log}+10^{-6}.
\]

The calibration cloud sets `c=-log p_hat_cal(y1)`, so its shifted normalizer
target is exactly one and cannot fall below `tau`. It may also set the core
scale. The disjoint validation cloud vetoes the selected arm. The untouched
cloud is read once for the final T1 gate.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| Frame quantiles and expansion | historical factor `4.0`; hypothesis only | tail clipping or wasteful algebraic scale | calibration/validation mapped-coordinate quantiles |
| Algebraic map | author Austria configuration; source-grounded operation | endpoints or Jacobian instability | exact round-trip/Jacobian test |
| Degree/rank | P88 degree-3/rank-4 is one warm start | underfit or excess memory | small rank/degree screen |
| Learning rate | P88 `3e-4` is one warm start | divergence or slow fit | first 16 updates and finite-gradient gate |
| L1 | mandatory scope-specific tuning; zero is comparator only | inherited zero-L1 overfit | validation shape screen |
| Batch size | hypothesis, must remain greater than one | OOM or noisy update | compiled one-update memory smoke |
| `tau=1e-8` | source/local historical hypothesis | dominates tiny shifted target | choose shift `c` so target mass is O(1), test `Z_target>tau` |
| Stopping rule | hypothesis | premature selection on noise | fixed pilot budget, then validation checks only |
| Scale calibration | algebraic post-fit core rescaling | confuses shape with value | exact post-scale contraction and disjoint MC gate |

## Bounded Execution Ladder

1. CPU-hidden engineering tests only: target/hash/event order, latent-law
   coherence, transformed-measure constant, dependency exclusions,
   serialization/tamper rejection, scale recovery, and compiled-kernel parity.
2. Trusted GPU preflight: `nvidia-smi`, TensorFlow device probe, verified memory
   growth before any device initialization, and one XLA update smoke.
3. T1 pilot screen only after engineering admission. At most 12 ten-minute
   arms; ranks, degrees, LR, L1, batch size, and stopping rule are recorded per
   arm. Validation chooses or rejects; audit is not read.
4. At most four 30-minute continuation arms and two finalists if justified.
5. One untouched T1 claim, maximum 120 minutes, in a fresh versioned output
   directory. No retry reads the same untouched cloud after a scientific veto.

Campaign hard cap is the previously approved 24 GPU-hours for T1/T2, but this
B0/B1 execution may spend at most 10 GPU-hours before a new T2 execution note.
GPU live allocation is at most `min(12 GiB, 50% of trusted-probe available
memory)`. Only one trainer may run. Training uses microbatches and never forms
a tensor-product retained grid or retains all-time histories.

### Frozen Pilot Screen

The first pilot uses the following six arms. Each arm is a fresh process and
fresh versioned output directory, capped at ten minutes. All use 8,192 training
points, 16,384 validation points, 32,768 scale-calibration points, batch size
512, 96 compiled Adam updates, expansion factor 4, covariance jitter `1e-5`,
quantile fraction `0.01`, `tau=1e-8`, and XLA. `basis_dim` is
`num_elems*order+1`.

| Arm | Rank | Order/elements | Basis dim | LR | L1 |
|---|---:|---:|---:|---:|---:|
| `p01_r2_b3_lr3e4_l1_0` | 2 | 1/2 | 3 | `3e-4` | `0` |
| `p02_r2_b3_lr3e4_l1_1e8` | 2 | 1/2 | 3 | `3e-4` | `1e-8` |
| `p03_r4_b3_lr3e4_l1_1e9` | 4 | 1/2 | 3 | `3e-4` | `1e-9` |
| `p04_r4_b3_lr3e4_l1_1e8` | 4 | 1/2 | 3 | `3e-4` | `1e-8` |
| `p05_r4_b5_lr3e4_l1_1e9` | 4 | 2/2 | 5 | `3e-4` | `1e-9` |
| `p06_r4_b5_lr1e4_l1_1e9` | 4 | 2/2 | 5 | `1e-4` | `1e-9` |

Seed roles are fixed: training/frame `73101`, validation `73201`, scale
calibration `73301`, frozen references `73501`, trainer initialization `73001`,
and untouched claim `73401`. The untouched seed is forbidden in pilot and
selection processes.

An arm is viable only if it has finite training and validation terms, the
calibration and validation normalizer estimates agree under the declared
combined three-standard-error rule, exact post-calibration TT contraction
passes, peak TensorFlow allocation remains below 6 GiB, and validation
normalized-log-density RMS is at most `0.95` times the constant-density
baseline RMS on the same validation cloud. Among viable arms, the smallest
validation RMS is selected descriptively; no statistical superiority is
claimed. If no arm is viable, that is a tuning repair trigger, not T1 admission
or rejection of Lane B.

The selected arm alone may be evaluated on 65,536 untouched samples from seed
`73401`. A standard-library selection process must first bind all six pilot
result hashes and choose the minimum validation RMS among viable arms. Claim
execution accepts that selection ledger, not an arbitrary artifact path. The
claim passes when the artifact value differs from the untouched log-evidence
estimate by no more than
`3*sqrt(SE_calibration^2+SE_untouched^2)+1e-6`, fresh reload identity and
direct contraction pass, and the memory gate remains satisfied.

## Skeptical Audit Verdict

`PASS_B0_B1_SKEPTICAL_AUDIT`

The plan uses the newly approved baseline rather than unrecoverable P88, does
not treat fit residuals as a value criterion, binds the normalized reference
measure including `2^36`, separates calibration/validation/untouched clouds,
has explicit continuation vetoes and resource bounds, and produces artifacts
that directly answer the T1 question. T2/T20, score, comparison, and HMC remain
blocked until the T1 artifact passes.

## Pre-Launch Engineering Evidence

The CPU-hidden gate passed on 2026-07-30: 11 focused tests covered sealed data
hashes/event order, coherent latent T1 law, the exact `36 log 2` reference
constant, forbidden dependency closure, disjoint Monte Carlo normalizers,
measure-correct batches, exact core-scale recovery, XLA update parity,
fresh reload/tamper rejection, and fail-closed selection. Syntax and diff checks
also passed. This is engineering evidence only, not GPU or T1 admission.

An exact-count CPU diagnostic for the first pilot scope found `c=31.1290512`,
training log-target range `[10.81597,37.12770]`, square-root target range
`[2.2318e2,1.1539e8]`, and rescaled integration-weight range
`[2.8977e-12,1]`. All values were finite and strictly positive. The wide range
is an explanatory conditioning risk and a possible tuning-repair trigger; it is
not a promotion criterion or continuation veto by itself.

Trusted GPU visibility and memory-growth preflight passed. Pilot launch was
deferred while an unrelated repository NeuTra job owned GPU 0 at high
utilization; no competing process was interrupted and no Lane-B GPU budget was
consumed during that wait.

Pilot attempt `p01` then ran for 32.9 seconds and was rejected. XLA, memory
growth, independent normalizer agreement, artifact serialization, and the
74.2 MiB TensorFlow peak passed. The shape gate failed because the generic
random initializer multiplied `0.05`-scale cores across 36 axes: the fitted
density remained exactly the defensive floor `rho=tau=1e-8`, the pre-scale
normalizer was `1e-8`, gradients were about `2.09e-8`, and the post-fit scale
was `3.10e46`. This is an implementation/initialization failure, not evidence
against Lane B. It triggers the localized balanced-initializer repair before
the remaining frozen arms; running them unchanged would not answer the
research question.

Definitive pilot `p02-p04` subsequently ran and were viable, with validation
RMS about `22.8060`, `22.4783`, and `22.4783`. Pilot `p05` failed before
training because the order-2 three-node Lagrange mass helper dispatched its
tiny FP64 Vandermonde solve to GPU and reported a singular matrix; the same
matrix is finite and nonsingular on the CPU reference path. This is a backend
placement failure. The localized repair freezes setup-static mass and integral
tensors on CPU and presents them as constants to the GPU/XLA optimizer. It does
not change basis evaluation, the objective, target, arm table, or selection
criterion. Because source closure changes, all six definitive arms must be
rerun under one common repaired code identity; earlier arms cannot enter the
selection ledger.

Before untouched execution, fresh reload exposed a v1 serialization defect:
JSON erased the tuple type of cloud `joint_axis_order`, while branch identity
correctly distinguishes tuples from lists. No tensor, fit, value, target, or
selected identity was changed, and the untouched seed was not read. A separate
schema-v1 compatibility decoder now restores only the declared `("z1","z0")`
tuple, verifies every tensor and bound source hash, and must reproduce the
original artifact identity exactly. The bound numerical source module remains
unchanged at its pilot hash; the decoder receives its own claim-manifest hash.
