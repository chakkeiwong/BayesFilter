# SSL-LSTM Batch-100 q=1/q=20 Unified Pipeline Execution Plan

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `COMPLETED_TRAINING_GATES_HMC_WITHHELD`

## Question And Evidence Contract

Question: does the unified pipeline execute the current NeuTra adaptive
training and, where training is admitted, the existing HMC tuning/retained
stages for q=1 and q=20 at `batch_size=100`?

Exact candidate: fixed parameters from
`docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json`,
current three-stage 32x32 dense IAF, existing seeds, validation cadence, and
controller. The same parameter file is valid for both q values because it only
contains optimizer/training settings; q is supplied separately.

Primary stage gates:

- training: two fresh streams, both result rows `ADMITTED`;
- HMC tuning: summary status `KERNELS_FROZEN`;
- retained HMC: existing cumulative checkpoints and finite-sample admission
  status.

Training loss, saturation, wall time, and draw exposure are explanatory or
child veto diagnostics. They are not posterior or predictive-equivalence
criteria. A training veto stops downstream HMC for that q and is recorded as a
candidate failure, not a rejection of NeuTra or q geometry.

## Caps And Stop Rules

Each q is run in a fresh process/output root:

| Stage | Cap |
| --- | ---: |
| Two-stream batch-100 training | 7,200 s |
| HMC preflight/tuning | 3,600 s |
| Retained HMC | 3,600 s |

The child scripts enforce their own non-preemptive reserves and write resumable
artifacts. The wrapper stops before downstream stages when an upstream handoff
fails. There is no automatic resume or second hyperparameter search.

## Skeptical Pre-Execution Audit

- Wrong baseline: fixed-smoke parameters are explicitly an unpromoted training
  hypothesis, not q-specific tuning evidence.
- Proxy promotion: training loss cannot admit HMC; the wrapper requires the
  child `ADMITTED` status and exact artifacts.
- Missing stop: per-stage caps, child hard vetoes, wrapper handoff vetoes, and
  fresh output roots are explicit.
- Unfair comparison: q, batch size, architecture, seeds, cadence, and params
  are held fixed; only q changes between arms.
- Resource mismatch: q=20 HMC may exceed one hour; this is a bounded attempt,
  and a valid resource stop is not a scientific failure.
- Artifact adequacy: each q has stage logs, child summaries, and wrapper
  `pipeline-summary.json`.

Audit decision: `PASS_FOR_TWO_BOUNDED_Q_BATCH100_RUNS`.

## Commands

```text
TF_FORCE_GPU_ALLOW_GROWTH=true python \
docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py \
  --q 1 --batch-size 100 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-batch100-q1-pipeline-2026-07-20 \
  --training-cap-seconds 7200 --hmc-tuning-cap-seconds 3600 \
  --retained-hmc-cap-seconds 3600 --authorize-material-run

TF_FORCE_GPU_ALLOW_GROWTH=true python \
docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py \
  --q 20 --batch-size 100 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-2026-07-20 \
  --training-cap-seconds 7200 --hmc-tuning-cap-seconds 3600 \
  --retained-hmc-cap-seconds 3600 --authorize-material-run
```

## Close Record

Both q=1 and q=20 completed the two-stream training stage. Every stream was
vetoed by the dense-scale saturation cap, so the wrapper withheld HMC tuning and
retained HMC for both q values. The first two q=20 attempts were interrupted
during their first expensive interval and are retained only as incomplete
execution diagnostics; the final fresh q=20 run is the valid result.

| q | training wall seconds | seed-a | seed-b | HMC |
| ---: | ---: | --- | --- | --- |
| 1 | 348.64 | veto at step 750, saturation 0.05729 | veto at step 1750 after one LR repair, saturation 0.05078 | withheld |
| 20 | 1912.19 | veto at step 750, saturation 0.07552 | veto at step 500, saturation 0.07031 | withheld |

The adaptive controller was exercised for q=1 seed-b: it reduced the learning
rate once at step 1500 after continued improvement, then stopped when the
saturation cap was exceeded. No stream reached `ADMITTED`, so no HMC result is
available for either q. This is a training-candidate veto, not evidence against
the q geometry, NeuTra direction, posterior, or predictive law.
