# SSL-LSTM q=20 seed-B plug-in predictive comparison result (2026-08-08)

## Outcome

The established plug-in comparison completed successfully. The 4,000 retained
HMC draws were mapped to physical coordinates and reduced to one posterior
mean and one coordinatewise posterior median. Each fixed parameter vector was
then forecast independently against the locked true vector using 1,024
forecast-noise paths. No retained draw was propagated as a separate parameter.

The result is a **descriptive predictive discrepancy**: both plug-in estimates
produce an output distribution shifted upward relative to the true-parameter
control and have lower predictive variance. This does not establish posterior
incorrectness by itself, because the check compares a plug-in functional rather
than the full parameter posterior.

## Frozen comparison

| Arm | Free parameter vector `(latent weight, latent bias, observation weight, observation bias)` |
|---|---|
| True control | `(0.350000, -0.080000, 0.650000, 0.050000)` |
| Posterior mean | `(0.644292, 0.163217, 0.609897, 0.167083)` |
| Posterior median | `(0.649384, 0.210217, 0.604353, 0.156786)` |

All three arms used the same q=20 target, ten-step principal-root forecast,
float64/XLA path, and 1,024 independent Philox forecast-noise replications.
The material seed was `(20260808, 82001)`; the canary used the disjoint seed
`(20260808, 81001)`.

## Predictive differences versus true control

| Plug-in arm | Maximum absolute mean difference over horizons | Maximum absolute variance difference over horizons |
|---|---:|---:|
| Posterior mean | `0.2701973` | `0.0543833` |
| Posterior median | `0.2875754` | `0.0616535` |

For the posterior-mean arm, the ten mean differences were approximately
`0.267-0.270` at every horizon. Variance differences were negative at every
horizon, approximately `-0.046` to `-0.054`.

For the posterior-median arm, the ten mean differences were approximately
`0.284-0.288` at every horizon. Variance differences were negative at every
horizon, approximately `-0.053` to `-0.062`.

The mean and median plug-ins are close to one another relative to their
discrepancy from the true control. Their forecast signatures were identical
because the forecast operator and noise seed were shared; only the fixed
parameter vector differed.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete plug-in comparison | All archive, target, transport, finite-value, XLA, and forecast-shape checks passed | No hard veto | Monte Carlo error of 1,024 forecast paths; plug-in is not the full posterior predictive | Treat the discrepancy as a candidate diagnostic and investigate parameter/forecast sensitivity | Posterior correctness, mode-mass correctness, model adequacy |
| Use posterior mean as primary summary | User-requested plug-in summary; median retained as sensitivity | No summary or mapping veto | One seed-B chain campaign | Keep mean and median results together; do not select post hoc | Mean is the true parameter or a universal estimator |
| Do not promote seed B from this result | Plug-in output differs from true-parameter output | No sampler hard veto was newly tested here; native divergence remains unavailable in the HMC archive | True-parameter control is a synthetic generating vector, and the filter forecast is approximate | If stronger inference is needed, run a reviewed parameter/forecast sensitivity or independent sampler check | Exact posterior rejection or broad NeuTra rejection |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for this plug-in execution; prior seed-B archive limitations remain, including unavailable native divergence telemetry |
| Statistically supported ranking | None; mean versus median is a predeclared sensitivity comparison, not a ranking |
| Descriptive-only differences | Parameter vectors, predictive means, variances, quantiles, runtimes, and maximum horizon differences |
| Default readiness | Not established |
| Next evidence needed | A reviewed sensitivity/replication design if the discrepancy must be attributed to sampling, finite-data estimation, or model/filter approximation |

## Research-question guardian

| Question | Verdict |
|---|---|
| Did this answer the established procedure? | Yes. It compares a single estimated parameter forecast with a single true-parameter forecast. |
| Did it propagate 4,000 parameters? | No. The 4,000 draws were used only for mean/median estimation. |
| Did the target or forecast harness fail? | No. Archive binding, physical mapping, finite status, XLA compilation, and 1,024-path forecasts passed. |
| Does this prove posterior correctness? | No. It is a plug-in predictive functional check, not an independent posterior authority. |
| What failed scientifically? | The seed-B plug-in predictive output does not match the true-parameter output descriptively. The result does not localize whether the cause is posterior estimation, target/filter approximation, or finite-data behavior. |

## Run manifest

| Field | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plugin-predictive-comparison-plan-2026-08-08.md` |
| Runner | `docs/benchmarks/run_ssl_lstm_q20_seed_b_plugin_predictive_comparison_2026_08_08.py` |
| Final artifact | `docs/plans/artifacts/ssl-lstm-q20-seed-b-plugin-predictive-comparison-2026-08-08/r4/material.json` |
| Final artifact SHA-256 | `72ba9c7034e36f26e76d0d6542c3aa0ab6699e4d21fe0f727ca5dea275663f09` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; float64; XLA |
| Device | CPU-only; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Wall time | `31.2803 s`; cap `900 s` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Base adapter signature | `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3` |
| Retained archive | 4 chains x 1,000 retained transitions; warm-up excluded |
| Training checkpoint | Seed-B continuation update 4000 / optimizer step 6250 |

## Repair history

Several localized harness repairs were completed under the same scientific
contract and preserved as superseded artifacts:

- `r1`: initial canary artifact; later material attempt hit the static
  1,024-row XLA resource cap.
- `r2`: one-row/replication-axis optimization; numerical output completed but
  retained a stale replication-count manifest field.
- `r3`: corrected replication field; terminal review found stale repeated-row
  wording and shared canary/material seed code.
- `r4`: final independent-seed material receipt used for this result.

The seed-B compatibility loader was also repaired to migrate only absent null
metadata fields in the historical checkpoint configuration:
`fixed_output_scale=[]`, `fixed_output_factor=[]`, and
`chart_signature=null`. The checkpoint tensor values and transport transform
were not modified.

## Post-run red team

The strongest alternative explanation is that the estimated parameter is a
finite-data plug-in estimate under an approximate SVD-UKF target, so its
forecast can differ from the synthetic generating-parameter forecast even when
the inference procedure is behaving as designed. Another possibility is that
the NeuTra/HMC campaign sampled a locally consistent but incomplete region.

This result would be overturned as a discrepancy if a fresh, independently
seeded run under the same target and plug-in contract produced materially
different summaries, or if a target/forecast identity error were found. The
current artifact has no such identity or finite-value failure. The weakest
part of the evidence is causal attribution: this comparison shows a
predictive mismatch but does not identify its source.
