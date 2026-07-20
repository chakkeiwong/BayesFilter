# SSL-LSTM Batch-100 q=1/q=20 Pipeline Execution Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-neutra-batch100-q1-q20-pipeline-execution-plan-2026-07-20.md`  
Decision: `TRAINING_COMPLETED_HMC_WITHHELD`

## Summary

The unified wrapper was run for q=1 and q=20 with `batch_size=100`, fixed-smoke
parameters, two independent streams, and the adaptive 250-step controller.
Both training stages completed. Neither q reached the two-stream `ADMITTED`
gate, so HMC tuning and retained HMC were correctly not launched.

| q | Artifact root | Training seconds | HMC status |
| ---: | --- | ---: | --- |
| 1 | `docs/plans/artifacts/ssl-lstm-batch100-q1-pipeline-2026-07-20/` | 348.64 | withheld after training veto |
| 20 | `docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-final-2026-07-20/` | 1912.19 | withheld after training veto |

## q=1

| Stream | Best step | Terminal step | LR reductions | Validation trajectory | Result |
| --- | ---: | ---: | ---: | --- | --- |
| seed-a | 500 | 750 | 0 | loss `79.5933 -> 43.8063 -> 42.6644 -> 42.3736`; saturation `0 -> 0.00260 -> 0.03906 -> 0.05729` | vetoed |
| seed-b | 1250 | 1750 | 1 | loss `63.3088 -> 43.3765 -> 42.5612 -> 42.1280 -> 41.8611 -> 41.6955 -> 41.5735 -> 41.6099`; saturation reached `0.05078` | vetoed |

Seed-b demonstrates the repair branch: after continued improvement through step
1250, the controller reduced the learning rate at step 1500. The stream then
stopped at step 1750 when saturation exceeded the `0.05` cap. Seed-a stopped at
step 750 before repair.

## q=20

| Stream | Best step | Terminal step | LR reductions | Validation trajectory | Result |
| --- | ---: | ---: | ---: | --- | --- |
| seed-a | 500 | 750 | 0 | loss `79.7599 -> 43.9295 -> 42.5170 -> 42.2715`; saturation reached `0.07552` | vetoed |
| seed-b | 250 | 500 | 0 | loss `62.6486 -> 43.1812 -> 42.2301`; saturation reached `0.07031` | vetoed |

## Validity And Interpretation

- GPU memory growth, finite values, worker GPU hiding, checkpointing, support
  probes, and round-trip checks passed for the completed streams.
- The sole candidate veto was `dense_scale_saturation_above_cap`.
- No HMC tuning or retained HMC samples were produced. There is therefore no
  HMC convergence, posterior, or predictive result to report.
- The adaptive training mechanism is operational: q=1 seed-b reached and used
  the learning-rate repair branch. The fixed-smoke batch-100 candidate failed
  the current saturation screen for both q values.
- This does not establish that batch 100 is intrinsically unsuitable. A
  lower-learning-rate or prospective saturation-repair experiment is the next
  discriminating training action before HMC.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Training candidate vetoed by saturation for all four streams |
| Statistically supported ranking | None; no batch-size ranking or HMC comparison |
| Descriptive-only differences | Loss paths, saturation, step counts, repair timing, runtime |
| Default readiness | Not established |
| Next evidence needed | Target-specific batch-100 repair or revised saturation policy, then two admitted streams and HMC tuning |

## Artifacts

- q=1 wrapper summary: `docs/plans/artifacts/ssl-lstm-batch100-q1-pipeline-2026-07-20/pipeline-summary.json`
- q=1 training summary: `docs/plans/artifacts/ssl-lstm-batch100-q1-pipeline-2026-07-20/training/final-summary.json`
- q=20 wrapper summary: `docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-final-2026-07-20/pipeline-summary.json`
- q=20 training summary: `docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-final-2026-07-20/training/final-summary.json`

No predictive-equivalence or posterior-correctness claim is supported by this
execution.
