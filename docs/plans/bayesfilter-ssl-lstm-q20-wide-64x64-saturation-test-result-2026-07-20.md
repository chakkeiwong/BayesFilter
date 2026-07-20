# SSL-LSTM q=20 Wide NeuTra Capacity Saturation Test Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-wide-64x64-saturation-test-plan-2026-07-20.md`  
Decision: `WIDE_CANDIDATE_VETOED_SATURATION_NOT_DELAYED`

## Result

The `(64,64)` candidate did not delay the saturation veto relative to the
matched `(32,32)` baseline. With q=20, batch size 100, seed-a, fixed-smoke
optimizer parameters, and the existing 250-step controller, it crossed the
`0.05` aggregate saturation cap at step 500. The `(32,32)` baseline crossed
the cap at step 750; the `(32,32,32)` candidate also crossed it at step 500.

| Architecture | Best step | Terminal step | Saturation path | Stop reason | Status |
| --- | ---: | ---: | --- | --- | --- |
| `(32,32)` baseline | 500 | 750 | `0 -> 0.002604 -> 0.041667 -> 0.075521` | `scale_saturation_above_cap` | `VETOED` |
| `(32,32,32)` depth candidate | 250 | 500 | `0 -> 0.009115 -> 0.092448` | `scale_saturation_above_cap` | `DIAGNOSTIC_VETOED` |
| `(64,64)` width candidate | 250 | 500 | `0 -> 0.039063 -> 0.076823` | `scale_saturation_above_cap` | `DIAGNOSTIC_VETOED` |

For the `(64,64)` candidate, stage-level saturation was:

| Validation step | Stage 1 | Stage 2 | Stage 3 | Aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| 250 | `0.000000` | `0.109375` | `0.007812` | `0.039063` |
| 500 | `0.000000` | `0.230469` | `0.000000` | `0.076823` |

The candidate completed normally in 791.76 seconds. There was no resource,
host-memory, nonfinite-value, target-signature, serialization, support, or
round-trip veto. It used 15,000 trainable transport parameters, compared with
4,440 for `(32,32)` and 7,608 for `(32,32,32)`. The frozen artifact reports
procedure `bayesfilter_ssl_lstm_wide_capacity_64x64_neutra_v1`; all three IAF
components carry hidden layers `[64,64]`.

## Evidence And Interpretation

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Candidate vetoed solely by `dense_scale_saturation_above_cap`. |
| Statistical ranking | None; this is one seed and one fixed-smoke comparison. |
| Descriptive evidence | `(64,64)` saturated at the same validation step as `(32,32,32)` and earlier than `(32,32)`. Its step-250 aggregate saturation was already `0.039063`, versus `0.002604` for `(32,32)` and `0.009115` for `(32,32,32)`. |
| Runtime evidence | The run took 791.76 seconds to reach step 500, while the narrower candidates reached their terminal checkpoints faster. This is descriptive resource evidence, not a statistically supported speed ranking. |
| Default readiness | Not established. `(64,64)` must not replace `(32,32)`. |
| HMC readiness | Not evaluated; HMC was correctly withheld. |
| Next evidence needed | Diagnose scale parameterization and optimizer/initialization interaction; additional width alone has not rescued the failure. |

This result invalidates neither the harness nor the q=20 target. It rejects only
the specific `(64,64)` capacity candidate under the inherited fixed-smoke
training policy and saturation screen. The repeated stage-2 concentration
suggests that the immediate issue is not simply insufficient hidden width. A
lower learning rate, scale-log repair, or stage-specific parameterization test
should precede another capacity-only enlargement.

## Run Manifest

Artifact root:
`docs/plans/artifacts/ssl-lstm-q20-wide-64x64-saturation-2026-07-20/run-01/`

| Field | Value |
| --- | --- |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 64,64 --authorize-material-run --gpu-cap-seconds 7200 --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json --output-root docs/plans/artifacts/ssl-lstm-q20-wide-64x64-saturation-2026-07-20/run-01` |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` |
| Environment | `tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, Optuna `4.6.0` |
| Device | physical GPU `1`, visible logical `/device:GPU:0`, TF32 enabled, XLA enabled, soft placement disabled |
| Dtype | `float64` |
| Workers | 16 CPU-hidden workers, one core each |
| GPU cap | 7,200 seconds |
| Charged wall time | 791.759 seconds |
| Host RAM cap | 64 GiB; no cap breach |
| Hidden layers | `[64,64]` |
| Trainable parameter count | 15,000 |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

## Checks

- `python -m py_compile` passed for the trainer, artifact loader, runner, and tests.
- `git diff --check` passed for touched implementation, test, and plan files.
- CPU-hidden focused suite passed: `52 passed`.
- Contract smoke passed for q=20 with `--hidden-layers 64,64`.
- GPU/XLA run completed with finite values, valid frozen-artifact reload,
  round-trip residual `3.55e-15`, and no HMC launch.

## Close Record

The wide family and runner option remain available as a diagnostic path, but
the existing `(32,32)` family and artifact contract are unchanged. The next
repair should target the repeated stage-2 scale saturation and optimizer/scale
interaction. Increasing width from 32 to 64 is not supported as a remedy by
this result.
