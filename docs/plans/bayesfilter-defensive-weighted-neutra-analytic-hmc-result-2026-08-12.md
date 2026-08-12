# Defensive weighted NeuTra analytic HMC result (2026-08-12)

## Outcome

One provenance-valid TensorFlow/TFP GPU/XLA fixed-length HMC run behind frozen
weighted-NeuTra replication 1 is **statistically compatible with the analytic
two-mode target under the declared marginal diagnostics**.

This is narrower than a distribution-equality or stationary-sampling claim. It
does not establish cross-transport robustness, general multimodal NeuTra validity,
SSL-LSTM validity, sampler superiority, or default readiness.

The original run-v5 result labeled the candidate rejected because it silently
conjoined every marginal mean and covariance interval. That was an uncalibrated
joint multiple-test veto and was wrong relative to the plan. The immutable run-v5
archive was re-adjudicated without rerunning HMC. The original result and corrected
adjudication are both preserved.

## Research question and computed quantity

The claimed target was the analytic physical law

```text
p(theta) = 0.8 N(mu_1, Sigma_1) + 0.2 N(mu_2, Sigma_2).
```

HMC actually computed the exact transformed target

```text
log p_z(z) = log p_theta(T_phi(z)) + log |det J_T_phi(z)|
```

with an explicit frozen-IAF score pullback. Unit/reference tests compared that
score with TensorFlow autodiff. The candidate runtime did not use a GradientTape
fallback and did not target the learned transport density.

Verdict on target identity: **correct under the checked implementation and frozen
checkpoint binding**. What remains unproved is exact stationary sampling from that
target and behavior for other transports or targets.

## Tuning result

Six fixed trajectory lengths were tuned with four explicit mode-aware latent
initial states. `L=1` was forbidden. Fresh 4,000-draw latent-coordinate verification
used modern rank-normalized split/folded R-hat `<=1.01`.

| L | epsilon | mean acceptance | max verification R-hat | Status |
|---:|---:|---:|---:|---|
| 3 | 0.930110 | 0.7754 | 1.02012 | rejected by R-hat |
| 5 | 0.693847 | 0.7401 | 1.00568 | passed screen |
| 10 | 0.654273 | 0.6015 | 1.03341 | rejected by R-hat |
| 15 | 0.192305 | 0.7445 | 1.01667 | rejected by R-hat |
| 20 | 0.140911 | 0.7376 | 1.00985 | passed screen; selected |
| 25 | 0.111733 | 0.7340 | 1.01584 | rejected by R-hat |

The predeclared selector chose `L=20` because its acceptance was closer to target
`0.70` than `L=5`. This is not statistical evidence that `L=20` is superior.

## Sequential result

The canonical `bayesfilter_neutra_sequential_hmc_v1` controller used the frozen
`L=20`, `epsilon=0.14091138276334744` kernel.

| Diagnostic | Result | Role |
|---|---:|---|
| Warm-up transitions per chain | 2,000 | excluded from posterior |
| Retained transitions per chain | 3,000 | posterior diagnostics only |
| Total retained draws | 12,000 | four chains |
| Maximum retained R-hat over latent and physical | 1.00551 | promotion criterion passed |
| Minimum retained bulk ESS | 6,948.04 | promotion criterion passed |
| Minimum retained tail ESS | 982.80 | promotion criterion passed |
| Hard sampler vetoes | none | passed |
| Target status failures | none | passed |
| Native divergence status | not exposed by kernel | unavailable is not zero |
| Warm-up excluded | yes | required |
| Sequential wall time | 117.42 s | descriptive |

Every chain moved and visited both hard-assignment modes during retained sampling.
Per-chain retained minority occupancies were `0.1830`, `0.1770`, `0.1853`, and
`0.1913`. Retained cross-mode transition counts were `601`, `566`, `620`, and
`645`; longest same-mode runs were `166`, `167`, `89`, and `69` transitions.

## Analytic comparison

The retained soft-responsibility minority mass was

```text
estimate = 0.1841667
batch-means MCSE = 0.0071830
99% interval = [0.1656646, 0.2026688]
truth = 0.2
```

The primary mass screen passed. Both modes were observed overall and in every
chain. All retained tensors were finite.

Marginal moment diagnostics were not combined into a joint test:

- `3/4` analytic means lay in their marginal 99% MCSE intervals.
- `15/16` analytic covariance entries lay in their marginal 99% MCSE intervals.
- The failed mean was coordinate 2: truth `0.45`, interval
  `[0.4519985, 0.4990176]`, estimate `0.4755080`.
- The failed covariance entry was `(2,2)`: truth `0.6745`, interval
  `[0.5251880, 0.6659269]`, estimate `0.5955575`.

These two misses remain evidence of residual uncertainty or mismatch. They do not
form a calibrated joint rejection. Conditional component summaries were close to
their analytic laws but are descriptive only; the minority component had only
about 2,210 effective raw assignments, so its conditional tails remain less
precise.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| One frozen-transport candidate is statistically compatible with the analytic target under declared diagnostics | Sequential R-hat/ESS passed; responsibility-mass 99% interval contains truth; coverage passed | No finite, movement, status, or available-divergence veto | One mean and one covariance marginal interval miss; no independent HMC seed or transport replication | Replicate the same frozen kernel with independent HMC seeds, then test fresh frozen transport replications before broader promotion | No equality proof, no stationarity proof, no sampler ranking, no general/default/SSL-LSTM claim |

## Inference-status table

| Inference status | Result |
|---|---|
| Hard veto screen | Passed for provenance-valid run-v5 |
| Statistically supported ranking | None; `L=5` and `L=20` both passed tuning and were not statistically ranked |
| Descriptive-only differences | Acceptance, runtime, energy-error tails, per-chain occupancy, and marginal moment misses |
| Default readiness | Not assessed and ineligible |
| Next evidence needed | Independent retained HMC seeds for replication 1, then at least fresh transport replications selected without HMC outcome leakage |

## Attempt and repair ledger

| Attempt | Status | Interpretation |
|---|---|---|
| canary-v1 | failed before transitions | custom-gradient callback did not accept captured frozen variables; implementation repair |
| canary-v2 | passed | GPU/XLA finite mechanics evidence only |
| run-v1 | failed before tuning | invalid fallback acceptance ceiling; configuration repair |
| run-v2 | no kernel admitted | physical-coordinate 1,000-draw R-hat gate rejected all candidates |
| run-v3 | no kernel admitted | latent-coordinate 1,000-draw R-hat gate rejected all candidates |
| run-v4 | numerically completed, launch-invalid for scientific use | tuner consumed mode-aware callback starts but artifact falsely reported all-zero starts |
| run-v5 | provenance-valid terminal run | public tuner owned and recorded exact nonzero latent state bank |
| adjudication-v1 | corrected terminal interpretation | immutable archive re-read; no new HMC transitions |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` plus recorded dirty worktree |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_defensive_weighted_neutra_analytic_hmc_2026_08_12.py --mode run --output-root docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5 --device 1 --cap-seconds 3600` |
| Environment | `tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| Hardware | host physical GPU 1 exposed as logical GPU 0; RTX 4080 SUPER; float64; TF32 disabled |
| GPU memory policy | `memory_growth`; verified true before logical-device initialization; full-device preallocation disabled |
| XLA | enabled; runtime emitted `Compiled cluster using XLA!` |
| Target data version | N/A; analytic target `separated_two_mode_unequal_weight_d4_v1` with parameters in manifest |
| Frozen checkpoint | confirmation replication 1, selected update 7,000 |
| Checkpoint SHA-256 | `af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5` |
| Tuning seeds | `(20260812,92001)`, `(20260812,93001)`, `(20260812,94001)` with candidate offsets |
| Sequential root seed | `(20260812,91001)` with phase/chunk derivation |
| Campaign wall time | 646.46 s |
| Plan | `docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-plan-2026-08-12.md` |
| Raw result | `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5/result.json` |
| Corrected adjudication | `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5-adjudication-v2/adjudication.json` |
| Terminal manifest | `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-2026-08-12-run-v5-adjudication-v2/terminal_manifest.json` |

Artifact SHA-256:

- run manifest: `172613cd7a979edc1e324d538890a70c76ec97e54f7cf382c69fd5b68cc3dcfb`
- tuning result: `6dfe2b8145040a18831a08032bfd61854189f2651e76c70842e59d4e4e12eb4f`
- sequential result: `c2c521665a1056ae58a18454e9cdd5c2746054d0033061ce54bd866b79f794cb`
- raw result: `db4ed848e0da72b591796acd7ee8018cfdbf6acf3130caf6e38acb6249289988`
- archive manifest: `d85b21dfb2f55baed07d7edd1f0ef7eb4feee009b90d7bc13b68c82614ec16e3`
- archive checkpoint: `e4fc7b8b1b955d58819d8c483a8a7029efefbddd740a07a658a210e0c778594f`
- corrected adjudication: `9b5fe4358516711a280dd53680e5e8a9693bdc4c76db75b681825a2c62c392fe`
- terminal manifest: `65d37057f4d77084e8fec34fc6507aca938d1643f82f63c4360881bdfff18ef6`

All 120 archive sample/trace/receipt hashes were reverified during corrected
adjudication.

## Post-run red team

The strongest alternative explanation is finite retained mode-weight fluctuation:
the estimate `0.1842` is low, and the only mean/covariance misses are in coordinate
2, whose component means differ by `1.5`. The mass interval still contains truth,
so current evidence cannot distinguish ordinary Monte Carlo variation from a small
residual occupancy bias.

Evidence that would overturn compatibility is an independent-seed replication
whose predeclared responsibility-mass interval excludes `0.2`, recurrent failure
of the same marginal entries beyond expected multiplicity, a receipt/target/hash
failure, or a hard sampler veto. Evidence that would strengthen the claim is
independent HMC replication on this checkpoint followed by fresh frozen-transport
replications.

The weakest evidence is the single HMC random stream and single frozen transport.
The result is therefore a viable candidate, not a default or general solution.
