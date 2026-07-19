# Phase A4 Plan: Predictive Calibration And Design Freeze

Date: 2026-07-13 (Asia/Shanghai)

Status: `AUTHORIZED_TIER2_SEQUENTIAL_EXECUTION`

Owner authorization recorded 2026-07-14:

- a standardized predictive-mean change of `0.20` baseline predictive standard
  deviations is materially different;
- predictive-variance ratios `1.25` and `0.80` are materially different,
  equivalently `|log(v_candidate/v_reference)| >= 0.22314355131420976`;
- skew and cross-horizon dependence remain explanatory until prospective
  structural perturbations define approved path-law/MMD material thresholds;
- the total trusted-GPU budget is at most `8 GPU-hours`, consumed
  sequentially rather than by a Cartesian grid; and
- existing HMC artifacts must be inspected first. If none qualify, one
  separately recorded calibration-only four-chain ordinary-HMC acquisition is
  authorized.

## Visible Canary Repair Amendment: 2026-07-14

The first trusted GPU/XLA canary produced a valid finite archive on GPU but
failed the original requirement that every chain move within only eight
retained transitions:

- chain movement: `[false, true, true, true]`;
- per-chain acceptance: `[0.0, 0.375, 0.625, 1.0]`;
- aggregate acceptance: `0.5`;
- finite samples, target values, and log-accept ratios: passed;
- GPU output placement and XLA compilation: passed;
- native divergence: `not_exposed_by_kernel`, not zero; and
- charged trusted-GPU wall time: `424.92757005395833` seconds.

This result did not invalidate the target, transform, archive, GPU placement,
or HMC runtime. It showed that an eight-draw mechanics canary cannot reliably
serve as an all-chain movement/tuning gate. The canary contract is therefore
repaired visibly and prospectively for a distinct retry: at least one chain
must move, all tensors and telemetry must be finite, placement must be GPU, and
archive readback must pass. A subset of unmoved chains is a tuning repair
trigger, not a canary hard veto. All-chain movement and per-chain acceptance in
`[0.20,0.95]` remain mandatory in the 64-draw tuning screen and every retained
acquisition rung. No posterior or calibration criterion is weakened.

The failed artifact remains at
`phase-a4/hmc-acquisition/gpu-canary.json`. The retry uses archive label
`canary_repair_01`, seed `[20260714,1412]`, and output
`gpu-canary-repair-01.json`. Both runs count against the `8 GPU-hour` cap. This
amendment is a post-canary engineering repair, not a reinterpretation of a
retained HMC admission failure.

## Phase Objective

Calibrate a prospective, dependence-aware decision design for comparing the
joint 1-to-10-step predictive laws of ordinary-HMC and exact-corrected
NeuTra-HMC SSL-LSTM draws. A4 will use calibration-only data to freeze practical
equivalence margins, feature scales and roles, aggregate weighting diagnostics,
MMD kernel settings, uncertainty settings, minimum sample sizes, and fresh
confirmation seeds before either confirmatory arm is opened.

A4 is calibration, not confirmation. It cannot decide SSL-LSTM predictive
equivalence, rank samplers, change the posterior target, run NeuTra training, or
promote a default.

## Entry Conditions Inherited From A3

| Entry condition | Required state |
| --- | --- |
| A3 result | `PASSED_FOR_A4_DESIGN_ONLY` |
| Focused A3 tests | `65/65` passed on the CPU-hidden reference route |
| CPU oracle artifact | `A3_CPU_REFERENCE_PASSED` and independently `A3_CPU_REFERENCE_VERIFIED` |
| Trusted GPU/XLA artifact | `A3_GPU_XLA_PARITY_PASSED` and independently `A3_GPU_XLA_PARITY_VERIFIED` |
| Engineering vetoes | None open across formulas, direct replay, hierarchy, decisions, artifact integrity, XLA placement, and parity |
| Power qualification | Identical-law fixture and variance/skew/dependence alternatives were underpowered; A3 fixture constants are not frozen A4 settings |
| Scientific boundary | No SSL-LSTM equivalence, posterior, HMC, NeuTra, model-adequacy, superiority, or default claim exists |

Before A4 execution, recheck that the A1 posterior target semantics, A2
forecast API, and A3 predictive-statistics interfaces required by the harness
have not changed. Source identity is a reproducibility check, not a ceremonial
authorization chain. A semantic source change requires a focused impact review
and affected tests, not an automatic revival of the historical closure
machinery.

The existing-artifact inspection found no qualifying archive. Historical Phase
2V used a CPU-hidden eager screen with 128 retained transitions and one chain
state, while replicated historical diagnostics were separate short runs rather
than one chain-shaped archive. The scalar SSL-LSTM JSON artifacts preserve
summaries or traces, not reusable `[draw, chain, parameter]` samples. No
qualifying scalar SSL-LSTM `.npz`, `.npy`, TensorFlow tensor, or equivalent
four-chain retained archive was found. The historical target semantics are
compatible with A0/A1, but artifact geometry and sampler-validity evidence are
insufficient. The owner-authorized fallback therefore applies: A4 may acquire
calibration-only ordinary-HMC draws under the bounded contract below. A4 must
not borrow A5 confirmatory draws.

## Research Intent Ledger

| Field | A4 contract |
| --- | --- |
| Main question | Can a frozen forecast-law comparison distinguish calibration null variation from scientifically material 1-to-10-step predictive differences at an affordable sample budget while controlling family-wise error? |
| Candidate/mechanism | Horizon-specific standardized mean and log-variance equivalence intervals plus an independent-bank joint-path MMD upper bound; candidate aggregate moment weights are calibration diagnostics |
| Exact baseline | Analytic LGSSM null/alternatives, identical SSL-LSTM draws with independent forecast banks, and split-half ordinary-HMC calibration draws |
| Expected failure mode | Wide intervals, unstable high moments/MGF features, singular long-run covariance, poor MMD bandwidth, invalid chain dependence assumptions, weak alternative detection, or resource growth beyond budget |
| Primary freeze criterion | A predeclared design using mean margins no larger than `0.20` predictive SD and variance margins no larger than `0.22314355131420976` absolute log-ratio meets simultaneous interval-coverage, true-equivalence decision-power, null false-material-difference, false-equivalence, and material-difference-power requirements across fresh calibration seeds and every required mean/variance null or alternative family without a veto |
| Promotion veto | Invalid target/data/artifact; interval coverage below its floor; true-equivalence `PASS` probability below its floor; null `MATERIAL_DIFFERENCE` probability above its cap; false-equivalence `PASS` probability above its cap for any material family; material-difference probability below its floor; invalid chain diagnostics; nonfinite/ill-conditioned weights; post-hoc margin or seed reuse; or failure to keep calibration and confirmation separate |
| Continuation veto | No admissible design meets the criterion within the resource ladder; the target/forecast estimand changes; required sampler calibration draws are invalid; or the actual compute budget exceeds the declared resource stop |
| Repair trigger | A candidate fails for identifiable sample-size, block, bandwidth, ridge, alpha, or numerical-stability reasons while the harness and estimand remain valid |
| Explanatory diagnostics | Average power, interval width, individual alternative effect curves, runtime, high moments, quantiles, covariance entries, aggregate-weight scores, characteristic/MGF features, and quadratic MMD values |
| Must not be concluded | Predictive equivalence, sampler superiority, posterior correctness, HMC/NeuTra readiness, model adequacy, or default/product readiness |

## Evidence Contract

| Field | Prospective requirement |
| --- | --- |
| Scientific question | Freeze a statistically identifiable and computationally feasible predictive-equivalence design, not prove equivalence |
| Comparator ladder | Naive equal weights; diagonal inverse-variance weights; shrinkage/GMM long-run precision weights; bounded MGF/characteristic-function/kernel-inspired diagnostics; all compared against the unaggregated horizon-specific gate |
| Primary pass/fail criterion | On held-out calibration replications: simultaneous interval coverage at least `0.90` with a one-sided 95% exact-binomial lower bound at least `0.85`; overall `PASS` probability at least `0.80` with lower bound at least `0.70` for every true-equivalence null family; null `MATERIAL_DIFFERENCE` probability at most `0.05` with upper bound at most `0.10`; false-equivalence `PASS` probability at most `0.05` with upper bound at most `0.10` for every required material alternative family; and `MATERIAL_DIFFERENCE` probability at least `0.80` with lower bound at least `0.70` for every required material alternative family |
| Promotion veto diagnostics | Target/source mismatch, invalid sampler calibration draws, nonfinite values, malformed hierarchy, fewer than four usable chains, failed stationarity/mixing admission for inferential MMD, non-positive scales/margins/tolerance, covariance condition cap failure after the ridge ladder, zero/nonfinite median distance, bootstrap failure, any `INVALID_HARD_VETO` replicate, seed overlap, or resource stop |
| Explanatory only | Point estimates beyond the hard criteria; average power; runtime; q95/q99/max; high moments; covariance surfaces; score/loss values; aggregate-weight rankings without paired uncertainty; A3 fixture performance |
| Nonconclusions on pass | A passing freeze only authorizes a separately planned ordinary-HMC baseline/confirmation sequence; it does not establish equivalence, convergence, superiority, correctness, or readiness |
| Preservation artifact | One calibration configuration, one run manifest per rung/seed, one aggregate calibration result with paired uncertainty, one immutable freeze manifest, and one A4 result note under `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/` and `docs/plans` |

The numerical thresholds above are calibration operating-characteristic floors,
not practical-equivalence margins. Compute every bound separately by null or
alternative family; a favorable aggregate cannot mask one failed family.
The owner-approved maximum practical-equivalence margins are `0.20` predictive
SD for means and `0.22314355131420976` absolute log variance ratio. Smaller
margins may be frozen if the design has adequate power; wider margins are
forbidden. Skew/dependence diagnostics cannot become promotion criteria during
this execution.
A design cannot rescue low true-equivalence power by widening a margin past a
material anchor, and an always-inconclusive design cannot pass merely because
it rarely emits a false decision.

## Locked Estimand And Decision Hierarchy

### Co-Primary Horizon Features

For every horizon `h=1..10`, retain:

- standardized predictive mean difference
  `Delta_mu,h / s_h`, where `s_h` is a positive calibration-only baseline
  predictive standard deviation with a recorded lower floor; and
- predictive log-variance difference
  `log(v_candidate,h) - log(v_reference,h)`.

The 20 simultaneous intervals are co-primary. A confirmatory feature branch
passes only when every interval lies strictly inside its frozen
horizon-specific practical-equivalence margin. A wide interval containing zero
is `INCONCLUSIVE_UNDERPOWERED`. Any interval wholly beyond a margin is
`MATERIAL_DIFFERENCE`.

### Higher Moments And Dependence

Third and fourth central moments, named quantiles, and the cross-horizon
covariance matrix remain explanatory in the first A4 freeze. Their calibration
curves diagnose blindness to skew, tails, and temporal dependence. They may be
promoted to a future co-primary family only through a visible plan amendment
that defines stable transformations, scientific margins, joint alpha, and
multi-seed coverage/power before confirmation.

The joint-path MMD remains the omnibus inferential branch for distributional
differences not represented by the mean/log-variance family. Overall
equivalence requires both the co-primary interval branch and the independent-
bank MMD upper-bound branch to pass.

### Weighting Boundary

No aggregate weight can override a failed horizon-specific margin or a failed
MMD gate. Aggregate weights nominate a stable summary and help allocate
diagnostic attention; they are not a substitute for simultaneous inference.

## Practical-Equivalence Margins

Freeze margins from the approved scientific effect anchors before fitting
operating characteristics:

1. Mean anchors: use standardized horizon shifts `{0.05, 0.10, 0.20, 0.30}`;
   `0.20` and larger are labeled materially different. The equivalence margin
   cannot exceed `0.20`.
2. Variance anchors: use multiplicative variance ratios
   `{1.05, 1.10, 1.25, 1.50}` plus reciprocal contractions. Ratios `1.25` and
   `0.80` are labeled materially different. The equivalence margin cannot
   exceed `abs(log(1.25)) = 0.22314355131420976`.
3. Structural anchors: perturb each forecast-relevant SSL-LSTM parameter and
   terminal-state coordinate in both directions at fixed, recorded scales; map
   those perturbations to mean/log-variance/path-law changes.
4. The owner labels above satisfy the required prospective mean and variance
   materiality decision. Calibration estimates whether the proposed design can
   detect those labels; it does not redefine materiality from noise.
5. Set each practical-equivalence margin no larger than the smallest approved
   material effect for its feature. If a null-noise interval and the approved
   material effect cannot be separated, increase sample size or stop; do not
   widen the margin past the material anchor.

Record raw-unit, standardized, and log-ratio interpretations for every frozen
margin. Require a positive numerical floor derived before the calibration run;
report whenever the floor is active.

## Candidate Moment And Path Weights

Evaluate these candidates on identical calibration splits and paired seeds:

| Candidate | Definition | Role and safeguards |
| --- | --- | --- |
| Equal horizon | `w_h = 1/10` | Naive transparent baseline; always retained |
| Diagonal inverse variance | `w_j proportional to 1 / max(V_j, floor)` | Uses calibration-only chain-aware long-run variances; normalize nonnegative weights; cap the largest/smallest ratio |
| Shrinkage/GMM | `W_lambda = (S + lambda I)^(-1)` | Use the full long-run covariance; freeze the smallest ridge on a candidate ladder that is positive definite and below the condition cap; report effective rank and condition number |
| MGF-inspired | Finite symmetric grid of centered standardized multivariate log-MGF contrasts `log E[exp(t^T z)]` at predeclared bounded direction vectors `t` | Diagnostic only unless every empirical exponential moment is finite and stable across seeds; weight the finite feature vector with the same chain-aware shrinkage/GMM rule; never estimate an unrestricted empirical MGF optimum |
| Characteristic/kernel-inspired | Symmetric random/Fourier features or a fixed Gaussian-RBF mixture on standardized paths | Preferred robust analogue when the MGF does not exist or is tail-unstable; freeze frequencies/bandwidths and mixture weights from calibration only |

An MGF identifies a law when it exists in a neighborhood of the zero vector,
but it does not itself specify an optimal weight over evaluation points. A4
therefore separates the feature map from the weight: a bounded symmetric set of
direction vectors `t` supplies finite log-MGF features of the standardized
ten-step path `z`, while a regularized long-run covariance supplies a
GMM-style weight. An empirical MGF can be dominated by a few paths. Treat
nonfinite values, excessive maximum-observation influence, or unstable
cross-seed coefficients of variation as an MGF-candidate veto; do not clip or
silently winsorize. The characteristic function is bounded and always exists.
Gaussian-kernel MMD is equivalently an integrated characteristic-function
discrepancy under its spectral measure, so it remains the primary robust
joint-law route.

Selection among admissible aggregate candidates uses a prospective minimax
rule: retain candidates that pass all hard null/alternative screens, then
select the simplest candidate whose paired lower confidence bound for
worst-family power is within `0.02` of the highest admissible lower bound. If
paired uncertainty does not distinguish candidates, select equal weights for
interpretability. Do not call the selected candidate superior; it is the
frozen representative under this calibration contract.

## MMD Calibration

1. Standardize complete ten-step paths using calibration-only center/scales.
2. Compute the pooled calibration-only median of pairwise squared path distance
   without mixing confirmation data.
3. Evaluate fixed bandwidth-factor mixtures around the median, including at
   least `(0.25, 0.5, 1.0, 2.0, 4.0)` as a sensitivity ladder. Freeze a
   parsimonious subset and normalized mixture weights before confirmation.
4. A zero or nonfinite median, non-positive band, duplicate band, non-normalized
   mixture, or unstable cross-seed result is a veto.
5. Use independently generated arm-specific forecast banks for inferential MMD.
   Common-random-number MMD remains a separately labeled diagnostic.
6. Calibrate the MMD practical tolerance from null variation and approved
   material path-law perturbations under the same scientific-anchor rule as
   feature margins.
7. Preserve signed U-form values and biased V-form diagnostics, but use the
   admitted cross-chain linear-MMD interval for dependent-chain inference.

## Calibration-Only Four-Chain HMC Acquisition

### Locked Target And Coordinates

The acquisition targets the accepted A1 posterior in
`bayesfilter/nonlinear/ssl_lstm_posterior_tf.py`, scope
`ssl_lstm_completion:a1:masked_svd_ukf_four_parameter`, target semantic
SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`,
and adapter contract SHA-256
`004f86b5668939febb629c563ca02625998c878d1e74d88c463f93b029a5d556`.
The four free parameters remain indices `12..15` in A1 order.

Use the A0 sampler geometry only as a coordinate transform and initialization
aid:

```text
theta = center + z @ factor.T
center = [0.5704394246369003, -0.1242247342531544,
          0.6609123192759063,  0.1354211218811133]
factor = diag([0.35, 0.35, 0.35, 0.35]) @ factor_z
```

The exact `factor_z` is read from the A0 target-lock artifact and must satisfy
`factor_z @ factor_z.T = covariance_z`. The transformed score is
`score_z = score_theta @ factor`; the constant affine Jacobian may be omitted
from HMC values because it is parameter-independent. This geometry must not
redefine or approximate the A1 posterior.

Use these prospectively fixed latent starts, in chain order:

```text
[[ 0.0,  0.0,  0.0,  0.0],
 [ 0.5, -0.5,  0.5, -0.5],
 [-0.5,  0.5, -0.5,  0.5],
 [ 0.5,  0.5, -0.5, -0.5]]
```

They are deterministic dispersed starts in the local latent chart, not draws
from the posterior and not evidence about stationarity. Vectorized HMC must use
stateless proposal randomness with the declared two-integer seed; the four
chain axes receive distinct momentum components from that draw.

A benchmark-local adapter may declare graph-native full-chain GPU/XLA
authority only for this bounded A4 acquisition after scalar/batch target
parity, transform parity, finite-difference score, XLA, and tiny four-chain HMC
canary checks pass. It must delegate every posterior value and score to A1 and
must not modify A1 capability metadata, production semantics, public APIs, or
defaults.

### Acquisition Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Question | Can ordinary HMC produce a finite, moving, adequately mixed four-chain A1 archive suitable only for A4 forecast-calibration split halves? |
| Exact baseline | Locked A1 target in free coordinates; historical balanced fixed kernel `step_size=0.3925`, `num_leapfrog_steps=4` is the first tuning candidate, with historical constant-trajectory alternatives used only if it fails |
| Primary admission criterion | Four chains; finite `[draw,4,4]` latent archive; deterministic map to A1 free coordinates; in both latent and A1 free coordinates, maximum rank-normalized split R-hat `<=1.05`, every bulk and tail ESS `>=100`, and every mean MCSE/posterior-SD ratio `<=0.10`; every chain moved; aggregate and per-chain acceptance in `[0.20,0.95]` |
| Hard/continuation vetoes | Target/signature/geometry mismatch; malformed archive; nonfinite sample, score, target value, or log-accept ratio; every chain unmoved in the tiny mechanics canary; any chain unmoved in tuning or acquisition; native divergence count positive if a native field exists; GPU budget exhausted; or required diagnostic unavailable for reasons other than the documented TFP divergence limitation |
| Promotion vetoes | R-hat, ESS, MCSE, or acceptance outside its prospective bound. These trigger the next nested retained-draw rung while budget remains; they reject the current archive but do not reject HMC or moment validation as research directions |
| Explanatory only | Runtime, target range, maximum finite log-accept ratio, posterior means/SDs, initialization-memory statistics, and historical CPU screens |
| Native divergence qualification | TensorFlow Probability `0.25.0` plain `HamiltonianMonteCarlo` does not expose a native divergence boolean in this route. Record `native_divergence_status=not_exposed_by_kernel`; never claim zero divergences. Acceptance/log-accept telemetry is not a divergence substitute |
| Nonconclusions | No posterior correctness, convergence proof, HMC readiness beyond this calibration input, sampler superiority, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |
| Preservation artifact | Private TensorFlow retained-sample shards plus public run/diagnostic JSON and an A4 acquisition result under `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/` |

### Sequential Kernel And Draw Ladder

1. Run CPU-hidden scalar/batch/transform/finite-difference checks. These are
   engineering checks, not sampler evidence.
2. Run a trusted GPU/XLA canary with four chains, at most 8 burn-in and 8
   retained transitions, using the first kernel candidate. The canary tests
   compile, GPU placement, finiteness, archive/readback, and at least one moving
   chain. Any subset of unmoved chains triggers tuning attention; it does not
   pass tuning or acquisition admission.
3. Run one 64-retained/32-burn-in tuning screen at
   `(step_size, leapfrog)=(0.3925,4)`. If all finite/movement checks pass and
   every chain acceptance is in `[0.20,0.95]`, select it and stop tuning.
   Otherwise try, according to the observed direction, `(0.19625,8)` for low
   acceptance or `(0.785,2)` for high acceptance. Stop after the first valid
   candidate; do not rank passing candidates by descriptive acceptance.
4. Acquire 250 burn-in transitions and 250 retained draws per chain. If only a
   promotion veto fires, continue from the exact final state with nested
   additional segments of 250, 500, then 1000 draws and no repeated burn-in,
   evaluating cumulative retained rungs `{250,500,1000,2000}`. Stop at the
   first admitted rung.
5. A later rung must contain every earlier retained sample byte-for-byte; do
   not silently replace invalid draws or restart until a favorable result
   appears. A hard veto stops acquisition and requires a blocker/repair record.

Each extension may re-bootstrap the fixed HMC kernel from the prior segment's
exact final position because momentum is auxiliary and refreshed by the HMC
kernel. It must not repeat warmup, change the selected kernel, alter earlier
samples, or treat an extension as a fresh favorable restart. Each segment uses
a distinct predeclared domain-separated seed derived from root
`[20260714, 1404]` and its role/rung index; materialized tensor bytes, not seed
regeneration, are replay authority.

The entire A4 trusted-GPU execution, including canary, tuning, acquisition, and
forecast calibration, has an `8 GPU-hour` cap. Record wall time after every
rung. Before starting a rung, require its conservative projected cost to fit
within the remaining budget; otherwise stop. Do not reserve the full budget for
an unnecessary HMC extension: once the smallest HMC rung passes, stop HMC and
preserve the remainder for forecast calibration.

### Exact Acquisition Commands

All paths are repository-relative and use the `tfgpu` environment. The audit
and CPU checks are:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  audit-existing \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/existing-artifact-audit.json

CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-pycache \
  TMPDIR=/tmp/bayesfilter-a4-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  cpu-check \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/cpu-transform-check.json
```

The trusted GPU/XLA canary and first tuning screen are:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-pycache \
  TMPDIR=/tmp/bayesfilter-a4-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  gpu-canary \
  --archive-dir docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/private \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary.json

# Visible repair retry after the preserved partial-movement canary.
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-pycache \
  TMPDIR=/tmp/bayesfilter-a4-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  gpu-canary --archive-label canary_repair_01 --seed-tail 1412 \
  --archive-dir docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/private \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary-repair-01.json

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-pycache \
  TMPDIR=/tmp/bayesfilter-a4-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  tune --candidate-index 0 \
  --archive-dir docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/private \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary.json \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary-repair-01.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/tune-0.json
```

If `tune-0.json` is `SELECTED`, run the first acquisition segment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-pycache \
  TMPDIR=/tmp/bayesfilter-a4-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py \
  segment --segment-index 0 \
  --selected-tuning docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/tune-0.json \
  --archive-dir docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/private \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary.json \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/gpu-canary-repair-01.json \
  --prior-gpu-artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/tune-0.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/segment-0.json
```

For extension segment `k`, pass every prior GPU artifact exactly once, pass
every earlier `segment-j.json` as `--previous-segment-output`, keep the same
selected tuning artifact, and write `segment-k.json`. The harness rejects
missing budget artifacts, duplicate budget artifacts, nonconsecutive segment
lineage, and extension after an admitted rung. If tuning candidate 0 is not
selected without a hard veto, run only the direction-indicated fallback from
the prospective ladder and bind both tuning artifacts in later budget
accounting.

## Resampling, Alpha, And Sample-Size Ladders

### Block And Bootstrap Design

- Estimate per-feature integrated autocorrelation behavior on calibration-only
  ordinary-HMC chains and predeclare a divisibility-compatible block ladder.
- Include block lengths around the largest stable dependence scale, for example
  `{4, 8, 16, 32}`, subject to at least 20 blocks per chain. The actual ladder
  must be derived from retained draw counts before execution.
- Compare circular and stationary block bootstrap only if both preserve chain,
  draw, forecast-replication, and horizon hierarchy. Never resample the four
  chain identifiers as an empirical population.
- Use at least `999` bootstrap replicates for nomination. Require `3999` for the
  frozen critical value unless a Monte Carlo error calculation prospectively
  justifies a smaller count. Freeze the bootstrap seed.
- Quantify bootstrap Monte Carlo error by repeating the critical-value
  calculation on independent bootstrap seeds; unstable critical values trigger
  more replicates or a stop.

### Alpha Allocation

Freeze a total family-wise alpha of `0.05`. Compare prospective allocations
between the 20-feature branch and MMD branch, including `(0.04, 0.01)`,
`(0.035, 0.015)`, and `(0.03, 0.02)`. Each pair must sum to at most `0.05`.
Select by the same minimax power rule on calibration data. No allocation may be
changed after confirmation begins.

### Sample Size And Replication

Use at least four chains for any inferential candidate. Evaluate a nested
resource ladder over retained draws and forecast replications so every larger
rung contains the smaller rung's materialized inputs. The sequential candidate
rungs are:

- cumulative retained draws per chain: `{250, 500, 1000, 2000}`;
- forecast replications per posterior draw: `{8, 16, 32, 64}`, increased only
  after the current replication count fails a prospective precision/power gate;
  and
- independent calibration seed families: minimum `20` for nomination, then
  enough fresh validation seeds for the exact-binomial operating-characteristic
  bounds. Schedule at least `60` validation replications and compute the final
  count prospectively from the desired lower/upper confidence bounds before
  launch. An invalid replicate is a repair trigger; do not silently replace or
  discard it to manufacture a valid count.

Start with the smallest rung and change only the dimension implicated by the
failed diagnostic. Never execute the full Cartesian product. Promote the first
rung that meets all hard
coverage, true-equivalence power, null false-material-difference,
false-equivalence, material-difference power, ESS/MCSE, and numerical criteria
with stable paired uncertainty. Stop
increasing a dimension once two consecutive larger rungs change every primary
operating characteristic by less than `0.02` and the smaller qualifying rung
already passes. Do not execute the full Cartesian grid when sequential
elimination answers the question.

## Calibration Data And Blinding

| Split | Permitted use | Forbidden use |
| --- | --- | --- |
| A3 unit/oracle fixtures | Verify formulas, shapes, hierarchy, and regression behavior | Freeze final A4 numerical choices from A3 point outcomes |
| A4 nomination seeds | Eliminate invalid candidates; nominate margins, weights, bandwidths, blocks, alpha, and resource rung | Report confirmatory pass/fail |
| A4 validation seeds | One fresh evaluation of the nominated design's coverage and power | Retune after seeing validation outcomes without a new nomination/validation split |
| NeuTra training validation | Choose a checkpoint under a later training plan | Change predictive-equivalence design |
| Confirmation seeds | One-time ordinary-HMC versus NeuTra-HMC comparison | Any tuning or repair of margins, weights, bandwidths, blocks, counts, or alpha |
| Audit seeds | Independent replication after a confirmation pass | Rescue a failed confirmation |

Seed families must be domain-separated by model, null/alternative family,
nomination/validation role, chain, posterior draw, and forecast arm. Persisted
materialized tensors are replay authority; root seeds are provenance.

## Required Calibration Families

### Null Families

1. Independent exact-LGSSM samples from the same forecast law.
2. Identical SSL-LSTM posterior draws with independent forecast banks.
3. Split-half calibration-only ordinary-HMC chains with independent forecast
   banks, after sampler validity screens pass.
4. Common-random-number pairs for variance-reduction diagnostics only, never as
   the sole inferential null.

### Material Alternatives

1. Horizon-local and persistent mean shifts at approved standardized scales.
2. Horizon-local and persistent log-variance shifts at approved ratios.
3. Centered skew and tail-shape changes with finite recorded moments.
4. Cross-horizon dependence changes preserving marginal means and variances.
5. Approved SSL-LSTM parameter and terminal-state perturbations, in both signs,
   that generate forecast-relevant effects.

Required power applies to every approved mean, log-variance, and joint-path
alternative family. Skew/dependence power is required through the MMD branch
only after a minimum material magnitude is scientifically labeled. Smaller
near-boundary alternatives are retained to map power curves, not as hard pass
requirements.

## Planned Artifacts

| Artifact | Required content |
| --- | --- |
| `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/calibration-config.json` | Exact candidate grid, scientific perturbation labels, alpha, seeds, commands, environment, resource limits, and source/interface identities |
| Per-rung manifests | Git commit/dirty state, command, conda environment, TensorFlow/TFP, GPU/CPU/JIT/TF32 state, seeds, sample counts, wall time, input/output paths, and status |
| `calibration-nomination.json` | Per-seed null/alternative decisions; family-specific interval coverage, true-equivalence power, null false-material-difference rate, false-equivalence rate, and material-difference power; paired candidate contrasts; vetoes; and nominated design |
| `calibration-validation.json` | Fresh-seed family-specific operating characteristics and exact-binomial bounds for the nominated design only |
| `confirmatory-design-freeze.json` | Canonical immutable values for all margins, scales, feature roles, weights, bands, tolerance, ridge, block/bootstrap design, alpha, sample minima, and confirmation seeds |
| A4 result note | Decision table, inference-status table, run manifest summary, repairs, nonclaims, and post-run red team |
| `hmc-acquisition/existing-artifact-audit.json` | Candidate paths, target/shape/validity checks, rejection reasons, and the conclusion that no existing archive qualifies |
| `hmc-acquisition/*-run.json` | Exact kernel role, command, target/geometry identities, environment/device/JIT/TF32 state, seed, chain/draw counts, wall time, budget balance, public-safe telemetry, and private archive binding |
| `hmc-acquisition/cumulative-diagnostics-*.json` | Private archive inputs, latent-to-free transform identity, movement, rank-normalized split R-hat, bulk/tail ESS, MCSE, acceptance qualification, divergence limitation, decision, and nonclaims |

Hash only immutable calibration inputs and the final freeze where identity is
needed for confirmation. Ordinary intermediate documents do not require a hash
chain.

## Execution Phases

1. Build a small TensorFlow/TFP calibration harness around the accepted A2/A3
   interfaces and add focused shape, hierarchy, seed-separation, and fail-closed
   tests.
2. Complete the authorized four-chain HMC acquisition contract above, stopping
   at the smallest admitted nested rung.
3. Run the cheapest exact-LGSSM null/material pilot to eliminate broken
   candidates and validate operating-characteristic accounting.
4. Run nomination seeds sequentially across sample-size, block, bandwidth,
   alpha, and weight candidates, pruning only by prospective veto/minimax rules.
5. Admit ordinary-HMC split-half calibration only after its own sampler-validity
   artifact passes; invalid chains cannot calibrate predictive uncertainty.
6. Freeze one nominated design, then evaluate it once on fresh validation seeds.
7. If validation passes, write the immutable confirmatory freeze and A4 result.
   If it fails, record the failed nomination and start a visibly new
   nomination/validation cycle; never tune directly on the failed validation
   rows.

These phases are authorized only within the stated A4 scope and `8 GPU-hour`
cap. They do not authorize A5 confirmation or NeuTra work.

## Required Checks And Review

- In-memory compile and focused unit tests for every new calibration harness.
- TensorFlow/TFP implementation for algorithmic paths; no NumPy, PyTorch, or JAX
  implementation path.
- XLA JIT on by default; CPU-only runs labeled smoke/reference; serious
  calibration runs use trusted GPU/XLA and record provenance.
- Strict finite/shape/hierarchy checks and canonical artifact-schema replay.
- Seed-domain and nomination/validation separation checks.
- Paired uncertainty for candidate comparisons; no ranking from descriptive
  means or extreme quantiles alone.
- One focused review before serious execution because A4 freezes numerical and
  statistical logic. Repair material findings and rerun affected checks; no
  repeated convergence ceremony is required absent a concrete unresolved risk.

Before forecast-calibration nomination, refresh this live plan with the exact
runnable calibration commands, derived final replication count, and exact
block/ridge/condition ladders. The HMC acquisition commands and thresholds are
already prospective above.

## Skeptical Pre-Execution Audit

| Challenge | Current design disposition |
| --- | --- |
| Wrong baseline | Addressed: exact LGSSM, identical-draw/different-bank SSL-LSTM, and split-half valid ordinary-HMC nulls form a ladder; no weak sampler is the sole comparator |
| Proxy promoted | Addressed: validation loss, A3 residuals, quadratic MMD, high moments, runtime, and point power cannot freeze or confirm the design by themselves |
| Missing stop conditions | Addressed prospectively: validity, coverage, true-equivalence power, null false-material-difference, false-equivalence, material-difference power, conditioning, seed, resource, and separation vetoes are explicit |
| Unfair comparison | Addressed: all candidates use paired calibration seeds, identical input subsets, the same null/alternative families, and nested resource rungs |
| Hidden assumptions | Resolved for HMC acquisition and mean/variance materiality. Actual chain dependence is measured rather than assumed. Exact ridge/condition ladders and final replication count remain to be frozen before forecast-calibration nomination, not before the bounded HMC acquisition |
| Stale context | Addressed prospectively: recheck only source/interfaces that affect the estimand or harness immediately before execution |
| Environment mismatch | Addressed prospectively: serious runs target trusted GPU/XLA; CPU is limited to analytic reference and focused smoke roles |
| Commands answer the question | The HMC route directly produces the missing chain-shaped calibration input and its admission diagnostics. Forecast-calibration commands must still be added before that later execution |
| Misleading pass | Addressed: separate nomination/validation seeds, simultaneous null and alternative requirements, and horizon-specific vetoes prevent a favorable aggregate from hiding a material miss |

Audit decision: `PASS_FOR_SEQUENTIAL_HMC_ACQUISITION_IMPLEMENTATION_AND_EXECUTION`.
The baseline is the locked A1 target, proxy metrics cannot establish posterior
correctness, native-divergence limits are explicit, nested stopping prevents a
Cartesian sweep, and the resulting artifact directly answers whether HMC draws
are admissible for A4 calibration. This does not yet authorize forecast
calibration until its exact numerical ladder is refreshed.

## Pre-Mortem

How A4 could pass while misleading us:

- scientific margins could be back-solved from null noise rather than material
  effects;
- split-half HMC could share slow nonstationarity and look equivalent;
- a GMM weight could concentrate on easy features and ignore a material horizon;
- an empirical MGF could be driven by rare paths;
- MMD bandwidth selection could overfit nomination alternatives;
- bootstrap seeds could leak between nomination and validation; or
- aggregate operating characteristics could hide a failed null or alternative
  family; or
- an always-inconclusive design could look safe while having no useful
  equivalence power.

Cheap discriminators are raw per-family tables, effective weight/rank reports,
maximum-influence diagnostics for exponential features, paired bandwidth power
curves, seed-domain audits, and family-specific exact-binomial bounds.

How A4 could fail without invalidating moment-based validation:

- the current sample rung may be too small;
- the block length or alpha allocation may be inefficient;
- one bandwidth mixture may be blind to a dependence alternative;
- ridge choice may be unstable in a high-dimensional feature family; or
- the empirical MGF may be unusable even though characteristic/kernel features
  remain valid.

These trigger the next smallest prospective repair. Only invalid evidence,
changed estimand, irreparable calibration/confirmation leakage, or exhaustion
of the declared resource ladder is a continuation veto.

## Stop Conditions

Stop the A4 run and write a blocker result if:

- a target, forecast timing, feature, or decision semantic change is required;
- calibration ordinary-HMC inputs fail their sampler-validity gate;
- any required null/alternative family, hierarchy, artifact, or seed domain is
  invalid;
- no candidate meets the prospective interval-coverage, true-equivalence
  power, null false-material-difference, false-equivalence, and
  material-difference-power criteria within the declared resource ladder;
- the covariance remains non-positive-definite or above the frozen condition
  cap across the ridge ladder;
- MGF-inspired features are nonfinite or tail-unstable and the plan attempts to
  retain them rather than veto them;
- cumulative trusted-GPU wall time reaches `8 GPU-hours`, or the next rung's
  conservative projected cost exceeds the remaining balance;
- nomination/validation/confirmation separation is breached; or
- execution would require NeuTra training, confirmatory HMC/NeuTra-HMC, a model
  file or dependency change, public/default promotion, or a scientific claim.

Do not stop merely because an individual candidate weight, block, bandwidth,
or sample rung fails. Eliminate it and continue to the next predeclared repair
unless a continuation veto fires.

## Exact Next-Phase Handoff Conditions

A5 ordinary-HMC baseline planning becomes eligible only when:

1. all A4 harness tests and artifact replays pass;
2. the scientific material-effect ledger is signed off before validation;
3. one nominated design passes once on fresh validation seeds under all hard
   family-specific interval-coverage, true-equivalence-power, null
   false-material-difference, false-equivalence, material-difference-power,
   numerical, and resource criteria;
4. no statistical ranking is claimed unless paired uncertainty supports it;
5. the final freeze records every horizon/feature role, margin, scale, weight,
   MMD bandwidth/mixture/tolerance, ridge/condition cap, bootstrap/block/alpha
   setting, sample minimum, and confirmation seed;
6. the result includes decision and inference-status tables, exact manifests,
   uncertainty qualifications, repairs, and post-run red team;
7. one focused review finds no material correctness, feasibility, leakage, or
   boundary issue; and
8. the freeze explicitly preserves the A1 target, A2 forecast semantics, A3
   fail-closed logic, and confirmation/audit blinding.

These conditions authorize planning and running the ordinary-HMC baseline under
its own evidence contract. They do not authorize NeuTra training, NeuTra-HMC,
predictive confirmation, equivalence, superiority, model adequacy, or a
default/product claim.

## Forbidden Claims And Actions

- Do not run ordinary HMC except the separately recorded calibration-only
  four-chain acquisition and nested repair rungs authorized above.
- Do not run NeuTra training, NeuTra-HMC, or confirmation.
- Do not reuse A3 fixture margins, weights, bandwidths, block length, bootstrap
  count, alpha split, sample counts, or seeds as frozen A4 choices.
- Do not infer practical margins solely from observed null noise.
- Do not treat non-rejection of equality as equivalence.
- Do not let aggregate, GMM, MGF, characteristic, or kernel weights override a
  failed horizon-specific co-primary interval or MMD gate.
- Do not promote higher moments, quantiles, covariance entries, quadratic MMD,
  runtime, validation loss, or single-seed power without a prospective plan
  amendment and uncertainty evidence.
- Do not call a viable or selected stochastic candidate best, superior, or
  improved without supported paired uncertainty.
- Do not change dependencies, model files, public APIs/defaults, or make
  scientific/product/release claims under A4 authority.
