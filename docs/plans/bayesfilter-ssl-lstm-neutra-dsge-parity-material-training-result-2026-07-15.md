# SSL-LSTM NeuTra DSGE-Parity Material Training Result

Date: 2026-07-15

Status: `SEED_INSTABILITY_REPAIR_REQUIRED`

## Outcome

The authorized two-seed, 5,000-step source-parity NeuTra program completed on
trusted GPU/XLA in `31,881.50` seconds (`8.8560` GPU-hours), within the
`36,000`-second cap. Both runs were finite, exactly resumable, exactly
reloadable, and free of hard vetoes.

Seed A passed every prospective nomination gate and produced a viable frozen
candidate. Seed B completed cleanly and strongly lowered its heldout reverse-KL
loss, but it was not nominated because its moderate-shell inverse radius was
`4.39594` against the `4.30` cap and its dense-scale saturation was `0.11198`
against the `0.05` cap. Under the prospective contract, this mixed outcome is
`SEED_INSTABILITY_REPAIR_REQUIRED`. Seed A cannot enter the main HMC lane
alone, and an earlier seed-B checkpoint cannot be selected retrospectively.

This is a candidate-stability failure, not an implementation, target, runtime,
or artifact failure. It does not reject NeuTra as a research direction.

## Candidate Evidence

| Gate | Seed A | Seed B |
| --- | --- | --- |
| Decision | `VIABLE_FROZEN_CANDIDATE` | `CANDIDATE_NOT_NOMINATED` |
| Hard vetoes | None | None |
| Promotion vetoes | None | `moderate_shell_missing_support`; `dense_scale_saturation_above_cap` |
| Heldout mean final-minus-initial | `-22.38761` | `-38.73790` |
| One-sided 95% heldout upper bound | `-17.14904` | `-27.29561` |
| Original-neighborhood max radius | `1.90088` | `2.14413` |
| Moderate-shell max radius | `2.85652` | `4.39594` |
| Saturation fraction | `0.00000` | `0.11198` |
| Roundtrip maximum | `5.33e-15` | `4.00e-14` |
| Exact resume/reload | Passed / passed | Passed / passed |
| Runtime | `15,947.83 s` | `15,932.95 s` |

Heldout loss and all continuous A/B differences are descriptive. Seed B's
larger loss reduction is not evidence that it is better; it failed two
prospective promotion screens. No A/B ranking is statistically supported.

## Trajectory Diagnosis

A CPU-hidden, read-only reconstruction of immutable 100-step checkpoints found:

- seed B first exceeded the overall saturation cap at checkpoint 900, while
  still in the original `0.01` learning-rate segment;
- final per-stage saturation fractions were `[0.25, 0.08594, 0.0]` for seed B
  and `[0.0, 0.0, 0.0]` for seed A;
- seed B's shell radius was `4.24359` at step 3,400 and exceeded `4.30` at every
  100-step checkpoint from 3,500 through 5,000;
- the final worst shell point expanded from radius `1.94271` to `4.39594`
  through the inverse of stage 1, localizing the dominant final compression;
  and
- clipping occurred on about `3.3%` of seed B's first 999 steps and `0%`
  thereafter, so persistent clipping is not the observed failure.

Saturation preceded the sustained late shell loss. These facts nominate the
aggressive first learning-rate segment as the smallest repair target, but they
do not prove it is the unique cause. The structured diagnosis is
`docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/trajectory-diagnosis.json`.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not open Phase 5 | Failed: both independent seeds were required; only A passed | B has two promotion vetoes; no hard veto | Whether the hot initial schedule caused instability or merely exposed reverse-KL geometry | Execute a separately authorized, paired schedule-stabilization discriminator followed by two fresh seeds only if it passes | HMC readiness, posterior correctness, or NeuTra failure |
| Preserve seed A as a viable candidate artifact | A passed all declared candidate screens | No A veto | One seed cannot establish stable training | Retain it as evidence/control; do not promote it alone | Superiority or main-lane admission |
| Reject seed B final candidate | B failed shell and saturation screens | Promotion veto, not evidence-invalidity veto | The shell miss is narrow and tail probes remain finite, but criteria were prospective | Diagnose and repair prospectively; do not relax thresholds or select an earlier checkpoint | Implementation failure or research-direction rejection |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Passed for both seeds: no hard veto is supported |
| Viable candidates | Seed A remains viable; seed B is not nominated |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, radii, saturation trajectories, gradients, clipping, runtime, and all A/B continuous differences |
| Default readiness | Not established |
| Next evidence needed | A prospective stabilization discriminator and two fresh full confirmation seeds passing unchanged gates |

## Run Manifest

| Field | Value |
| --- | --- |
| Program git commit | `20835ecf90bff78ca93c5d401f231e4aa94e63ce` (dirty worktree preserved) |
| Seed A recorded commit | `32695caa35c6e660f2fe6ed515bcb2b90123dc7f` |
| DSGE source commit | `d94566c9f70b3143e599a56eba7cb461ff2bda88` |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_dsge_parity_material_training_2026_07_15.py --program-output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; `float64` |
| Device/runtime | `CUDA_VISIBLE_DEVICES=1`; NVIDIA RTX 4080 SUPER; trusted GPU; XLA JIT on; TF32 enabled; soft placement disabled |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Seeds | A init/train/validation `[20260715,4101]` / `[20260715,5101]` / `[20260715,5201]`; B `[20260715,4102]` / `[20260715,5102]` / `[20260715,5202]` |
| Start / completion | `2026-07-15T02:30:41.377863+00:00` / `2026-07-15T11:22:02.418989+00:00` |
| Charged time | A `15,947.83 s`; B `15,932.95 s`; total `31,881.50 s` |
| Budget | Cap `36,000 s`; unused `4,118.50 s`; no overrun |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-plan-2026-07-15.md` |
| Program result | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/program-result.json` |

Artifact SHA-256 values:

- program result: `3b2b1a27c4b9af4d4f10026a111d03ed464c42b3d4f3a4f0a8a93217686c230d`;
- seed A result: `6b0b5ff525e9081870b707784715b6e31e1c8f47d8fec59ff7b96a1bc7bc8186`;
- seed B result: `3cfd5f1d936c99d1f42e4d7f5b4900da9d49403bcac902f55e912ac7b04ab40c`;
- seed A frozen payload: `c9ead9be468b57ca0dcbc20f9936f14ba09e2a4138548b356df105450e3e75b1`;
- seed B frozen payload: `ca0091f898cae98951ca6e966a49b264f0a61c8b06ec551ed9debaea4fd4aa94`.

## Budget Closure

The unused `4,118.50` seconds (`1.1440` hours) remains unused. It does not
authorize a third seed, post-hoc checkpoint selection, repair training, HMC,
or forecasting. The detached program completed and no material runner process
remains active.

## Post-Run Red Team

The strongest alternative explanation is not learning-rate instability but a
reverse-KL mode/support tradeoff intrinsic to this target: lowering the first
rate may only delay the same compression. Conversely, the narrow final shell
miss could be probe sensitivity rather than consequential HMC geometry. The
prospective thresholds still bind; neither explanation licenses relaxation or
promotion. A short lower-rate diagnostic can distinguish immediate optimizer
instability, but only fresh full runs can test late stability, and only later
exact transformed HMC and predictive calibration can assess downstream value.

What would overturn the candidate-stability interpretation is evidence of
source drift, incorrect target/score/Jacobian math, corrupt state, failed
resume/reload, CPU fallback, or invalid artifacts. All corresponding hard gates
passed. The weakest evidence is support completeness: the probe bank is finite
and cannot certify unknown modes or tails.

The next live plan is
`docs/plans/bayesfilter-ssl-lstm-neutra-seed-instability-stabilization-repair-plan-2026-07-15.md`.
Its bounded focused review converged to `VERDICT: AGREE` after one repair; GPU
execution remains pending a separate resource authorization.
