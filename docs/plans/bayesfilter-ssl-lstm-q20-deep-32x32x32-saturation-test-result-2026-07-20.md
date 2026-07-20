# SSL-LSTM q=20 Deep NeuTra Capacity Saturation Test Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-deep-32x32x32-saturation-test-plan-2026-07-20.md`  
Decision: `DEEP_CANDIDATE_VETOED_SATURATION_NOT_DELAYED`

## Result

The `(32,32,32)` candidate did not delay the scale-saturation veto. With q=20,
batch size 100, seed-a, the fixed-smoke optimizer parameters, and the existing
250-step adaptive controller, it crossed the `0.05` saturation cap at step 500.
The matched `(32,32)` baseline crossed the same cap at step 750.

| Architecture | Best step | Terminal step | Saturation path | Stop reason | Status |
| --- | ---: | ---: | --- | --- | --- |
| `(32,32)` baseline | 500 | 750 | `0 -> 0.002604 -> 0.041667 -> 0.075521` | `scale_saturation_above_cap` | `VETOED` |
| `(32,32,32)` candidate | 250 | 500 | `0 -> 0.009115 -> 0.092448` | `scale_saturation_above_cap` | `DIAGNOSTIC_VETOED` |

For the deep candidate, stage-level saturation was:

| Validation step | Stage 1 | Stage 2 | Stage 3 | Aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| 250 | `0.000000` | `0.027344` | `0.000000` | `0.009115` |
| 500 | `0.066406` | `0.210938` | `0.000000` | `0.092448` |

The candidate completed normally in 792.29 seconds, with no resource, host
memory, nonfinite-value, target-signature, serialization, support, or
round-trip veto. It used 7,608 trainable transport parameters, compared with
4,440 for `(32,32)`. The frozen artifact reports procedure
`bayesfilter_ssl_lstm_deep_capacity_32x32x32_neutra_v1` and all three IAF
components carry hidden layers `[32,32,32]`.

## Evidence And Interpretation

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Candidate vetoed solely by `dense_scale_saturation_above_cap`. |
| Statistical ranking | None; this is one seed and one candidate comparison. |
| Descriptive evidence | Saturation occurred one validation cycle earlier and was higher at step 500 for the deep candidate. Loss fell from `79.759855` to `44.011757` at step 250 and `42.570625` at step 500. |
| Default readiness | Not established. `(32,32,32)` must not replace `(32,32)`. |
| HMC readiness | Not evaluated; HMC was correctly withheld. |
| Next evidence needed | Diagnose scale parameterization, learning-rate/initialization interaction, and stage-specific saturation before another capacity-only test. |

This result invalidates neither the training harness nor the q=20 target. It
shows that the specific additional-depth candidate, under the inherited
fixed-smoke training policy and saturation screen, failed to solve the observed
failure mode. The strongest alternative explanation is that the extra layer
increased optimization sensitivity or scale-log growth rather than providing
useful geometric capacity. A lower learning rate or a prospective scale-repair
arm is needed to distinguish that explanation from a capacity limitation.

## Run Manifest

Artifact root:
`docs/plans/artifacts/ssl-lstm-q20-deep-32x32x32-saturation-2026-07-20/run-01/`

| Field | Value |
| --- | --- |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 32,32,32 --authorize-material-run --gpu-cap-seconds 7200 --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json --output-root docs/plans/artifacts/ssl-lstm-q20-deep-32x32x32-saturation-2026-07-20/run-01` |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` |
| Environment | `tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, Optuna `4.6.0` |
| Device | physical GPU `1`, visible logical `/device:GPU:0`, TF32 enabled, XLA enabled, soft placement disabled |
| Dtype | `float64` |
| Workers | 16 CPU-hidden workers, one core each |
| GPU cap | 7,200 seconds |
| Charged wall time | 792.293 seconds |
| Host RAM cap | 64 GiB; no cap breach |
| Hidden layers | `[32,32,32]` |
| Trainable parameter count | 7,608 |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

## Checks

- `python -m py_compile` passed for the trainer, artifact loader, runner, and new tests.
- `git diff --check` passed for touched implementation, test, and plan files.
- CPU-hidden focused suite passed: `34 passed`.
- Contract smoke passed for q=20 with `--hidden-layers 32,32,32`.
- GPU/XLA run completed with finite values, valid frozen-artifact reload,
  round-trip residual `8.88e-16`, and no HMC launch.

## Close Record

The deep family and runner option remain available for further diagnostics, but
the existing `(32,32)` family and artifact contract are unchanged. The next
repair should target stage-specific scale saturation and optimizer/scale
interactions; adding depth alone is not supported by this result.
