# Weighted forward-KL paper d100 result (2026-08-13)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md`

## Decision

| Target and arm | Primary criterion | Veto status | Decision | Main uncertainty | Not concluded |
|---|---|---|---|---|---|
| Ill-conditioned Gaussian, forward KL | Canonical sequential HMC passed, but the corrected 11-diagnostic exact Gaussian screen did not | One 99% interval, projection-2 mean, excluded zero | Candidate rejected under the frozen plan | One of 11 separate intervals can fail by Monte Carlo chance; no replication was run | No method failure, objective ranking, or default decision |
| Ill-conditioned Gaussian, reverse KL | Smaller-step HMC repair passed, but the same corrected Gaussian screen did not | One 99% interval, projection-2 mean, excluded zero | Candidate rejected under the frozen plan | Same one-seed and multiple-diagnostic uncertainty; the initial tuning failure was repaired | No method failure, objective ranking, or default decision |
| Paper funnel, reverse KL | Canonical sequential HMC passed; exact funnel structural and quantile-law screens did not | `E[y^2]`, lower tail, lower-tail residual moment, and 0.01/0.99 quantile-law intervals failed | Candidate rejected | The frozen transport may compress both tails despite good R-hat/ESS | No claim that reverse KL generally fails funnels |
| Paper funnel, forward KL | Canonical sequential HMC and all nine structural plus five quantile-law intervals passed | No declared hard veto | Viable candidate on this target and seed bank | One training seed and one HMC initialization bank | No superiority, original-paper replication, universal funnel result, or default promotion |

The funnel forward arm is the first d100 result in this campaign that passes
both the current sampler controller and an independent normalized posterior
authority. The correct classification is `viable positive control`, not
`superior objective`: there was no paired replication or uncertainty analysis
for an objective ranking.

## Evidence contract outcome

- Exact targets: source-bound dimension-100 Gamma-spectrum Gaussian and the
  paper funnel `y~N(0,1)`, `x_i|y~N(0,exp(2y))` variance.
- Training: batch-native TensorFlow, batch 4096, float64, TF32 disabled, XLA,
  GPU1, verified memory growth, no scalar or row-mapped fallback.
- HMC: fixed-length TFP HMC only, `L=(3,5,10,15,20,25,32)`, four chains,
  `L>=2`, no NUTS, canonical sequential warm-up/retained controller.
- Posterior gates: maximum modern R-hat `<=1.01`, minimum bulk/tail ESS
  `>=400`, plus target-specific separate 99% exact-law intervals.
- Native divergence is not exposed by this kernel and is not reported as zero.

## Training results

| Target | Objective | Selected LR | Selected update | Heldout selection loss | Untouched audit | Clipped updates | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian | Reverse KL | `1e-2`, paper schedule | 5000 | `-5.4438` reverse objective | recorded in artifact | artifact-recorded | `71.5 s` |
| Gaussian | Forward KL | `1e-2`, paper schedule | 5000 | `197.4720` exact NLL | recorded in artifact | artifact-recorded | `4210.8 s` |
| Funnel | Reverse KL | `1e-2`, paper schedule | 5000 | `50.09316` reverse objective | artifact-recorded | `35/5000` | `69.2 s` |
| Funnel | Forward KL | `1e-2`, paper schedule | 5000 | `142.57124` exact NLL | artifact-recorded | `286/5000` | `4165.4 s` |

Reverse and forward losses are different objectives and must not be compared
numerically. The funnel LR canaries nominated `1e-2` only within objective:
reverse `50.3850` versus `50.5808`, and forward `142.9958` versus `144.0680`.
These differences are selection diagnostics, not statistical rankings.

## HMC and analytic results

| Target/arm | Selected `L` | Epsilon | Warm-up / retained per chain | Max R-hat | Min bulk / tail ESS | Exact-law result | Overall |
|---|---:|---:|---:|---:|---:|---|---|
| Gaussian forward | 25 | `0.4228800` | `2000 / 1000` | `1.00455` | `2352.8 / 1586.3` | projection-2 mean `-0.05009`, 99% interval `[-0.08769,-0.01249]` excludes 0 | Rejected |
| Gaussian reverse repair | 32 | `0.3557668` | `2000 / 1000` | `1.00828` | `593.4 / 1175.6` | projection-2 mean `-0.07216`, 99% interval `[-0.13545,-0.00888]` excludes 0 | Rejected |
| Funnel reverse | 3 | `0.4438449` | `2000 / 1000` | `1.00519` | `841.7 / 683.2` | structural and quantile-law screens failed | Rejected |
| Funnel forward | 10 | `0.4342689` | `2000 / 1000` | `1.00497` | `1562.9 / 863.6` | all 9 structural and all 5 quantile-law intervals passed | Passed |

For funnel reverse, retained `E[y^2]=0.83925`; its 99% interval
`[0.74362,0.93488]` excludes 1. The empirical tail probabilities were
`0.0105/0.0135` instead of `0.02275/0.02275`. The 0.01 and 0.99 exact-quantile
CDF screens also failed, and the empirical quantiles
`(-2.0104,-1.2009,0.0104,1.1957,2.0553)` were visibly compressed relative to
the exact Normal quantiles.

For funnel forward, retained `E[y^2]=0.96044` with 99% interval
`[0.86603,1.05484]`; tail probabilities were `0.02275/0.02325`; standardized
residual second moment was `0.99893`; and all exact quantile probabilities
`(0.01,0.10,0.50,0.90,0.99)` lay inside their separate chain-aware intervals.

## Diagnostic correction

The original Gaussian HMC result marked analytic success merely because it
recorded exact summaries. A hash-bound CPU adjudicator verified the complete
source archive and applied the predeclared 11 intervals without rerunning HMC.
The original forward sampler verdict remains valid; its original combined
analytic verdict does not.

Before funnel training or HMC, the funnel diagnostic implementation was also
repaired. The initial code computed a raw cross-moment instead of the declared
covariance and used the wrong tail-ratio MCSE scaling. Independent-draw
quantile calibration was replaced by the equivalent chain-aware screen
`F_y(q_p)=p`, avoiding an iid assumption for serial HMC draws. Exact-law and
shifted-law tests passed before the claim run.

## Engineering, sampler, and scientific ledgers

| Ledger | Status | Evidence |
|---|---|---|
| Engineering correctness | Passed for all claim-bearing runs | Exact target/score/sampler tests, batch-native XLA tests, verified state and replay hashes, GPU memory-growth manifests, finite archives |
| Sampler validity | Passed for Gaussian forward, repaired Gaussian reverse, funnel reverse, and funnel forward | No hard vetoes; 2,000 discarded warm-up plus 1,000 retained per chain; R-hat/ESS gates passed |
| Scientific target agreement | Passed only for funnel forward | Gaussian arms each failed one projection interval; funnel reverse failed tail-law diagnostics; funnel forward passed 14 separate exact-law intervals |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Gaussian analytic veto for both arms; funnel analytic veto for reverse; no veto for funnel forward |
| Statistically supported ranking | None; one seed bank and no paired replication/interval for objective differences |
| Descriptive-only differences | Training loss, clipping, runtime, acceptance, selected `L`/epsilon, tail point estimates, and cross-objective differences |
| Default-readiness | Not assessed and not promoted |
| Next evidence needed | Replicate the funnel forward pass across fresh training/HMC seeds and run a matched replicated reverse comparator if an objective ranking is desired |

## Budget and manifest

Recorded execution across target generation, canaries, training, adjudication,
Gaussian HMC/repair, and funnel HMC was `11,931.964 s = 3.314435 h`,
within the four-hour d100 cap. The environment was
`/home/ubuntu/anaconda3/envs/tfgpu`, TensorFlow `2.20.0`; serious runs used
GPU1 with XLA, float64, TF32 disabled, and memory growth. CPU replay and
analytic adjudication explicitly used `CUDA_VISIBLE_DEVICES=-1`.

Primary artifacts:

- Gaussian source/replay: `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/source-r1/` and `gaussian-replay-r1/`.
- Gaussian training/HMC: `gaussian-reverse-r1/`, `gaussian-forward-r1/`,
  `gaussian-reverse-hmc-r2-repair/`, `gaussian-forward-hmc-r1/`, and
  `gaussian-forward-analytic-adjudication-r2/`.
- Funnel replay/training: `funnel-replay-r1/`, `funnel-reverse-r1/`, and
  `funnel-forward-r1/`.
- Funnel HMC: `funnel-reverse-hmc-r1/` and `funnel-forward-hmc-r1/`.

Each serious root contains its exact command, git commit, environment/device
provenance, target/replay/state identities, seeds, wall time, outputs, plan
path, and artifact hash ledger.

## Post-run red team

The strongest alternative explanation for the Gaussian failures is one or two
Monte Carlo false rejections among 11 separate intervals, not bad Gaussian
geometry: both arms passed sampler gates and only projection-2 mean failed.
The result would be overturned by a predeclared replication showing stable
Gaussian agreement, but the frozen single-run criterion still rejects these
candidates.

For funnel, the strongest alternative explanation is a favorable forward
training seed and initialization bank. A fresh replicated forward arm that
misses tails would overturn any robustness claim. The weakest part of the
current positive evidence is one seed bank; the strongest part is that the
forward candidate passed both modern sampler diagnostics and an exact
normalized funnel authority, including both tails.
