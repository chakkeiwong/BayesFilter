# q=20 SSL-LSTM CPU NeuTra Training Timing Diagnostic

Date: 2026-07-22  
Tier: bounded diagnostic-only CPU run  
Status: `COMPLETED_STRICT_THREAD_BOUND_DIAGNOSTIC_ONLY`

## Evidence Contract

| Role | Contract |
| --- | --- |
| Question | How long would the existing q=20 `(32,32)` batch-100 NeuTra training procedure take if both target evaluation and transport optimization ran on CPU? |
| Exact comparator | Strict CPU fallback: 8 scalar CPU target workers, `(32,32)` three-stage dense IAF, batch size 100, non-XLA parent optimizer diagnostic exception, validation/support every 250 steps. The relaxed 16-worker XLA timing is preserved separately as `r2`. |
| Primary output | Measured cold/startup time, steady full-update time, validation/support checkpoint time, terminal-audit time, and descriptive extrapolations to 250, 1,250, and 2,000 steps. |
| Hard vetoes | Visible GPU; configured TensorFlow compute-pool budget above 50; CPU affinity above 50 CPUs; nonfinite target, score, loss, gradient, validation, or support result; wrong batch/architecture/q. |
| Explanatory only | Individual timing rows, observed min/max, loss values, CPU utilization, native OS housekeeping-thread count. |
| Artifact | Strict result: `docs/plans/artifacts/ssl-lstm-q20-cpu-training-timing-2026-07-22/r3/result.json`; `r1` and `r2` are preserved as repair provenance. |
| Nonclaims | No transport-quality, HMC-readiness, convergence, GPU/CPU ranking, scientific-validity, or default-policy claim. No trained artifact is promoted or retained. |

## Thread And Resource Contract

- Bind the complete process tree to logical CPUs `0-49` with `taskset`.
- Parent TensorFlow: 4 intra-op and 1 inter-op thread.
- Eight target workers: one intra-op and one inter-op thread each.
- Configured TensorFlow compute-pool total:
  `4 + 1 + 8 * (1 + 1) = 21`.
- Set OpenMP, OpenBLAS, MKL, and NumExpr parent pools to one thread; workers
  independently verify the same one-thread library environment.
- Hide CUDA before TensorFlow import.
- Wall cap: 900 seconds. This is a convenience diagnostic cap, not an estimate.

TensorFlow may create non-compute housekeeping threads in addition to these
configured pools. The strict artifact records their realized `/proc` task
counts and fails if the complete process-tree sum exceeds 50. Parent CPU XLA
is disabled because a calibration probe showed that this TensorFlow build
creates about 106 native threads for one tiny XLA function, making a literal
50-thread cap impossible. This is an explicit diagnostic exception, not a
change to the repository XLA default.

## Timing Design

1. Measure Python/TensorFlow import and setup time.
2. Construct the exact q=20 target and `(32,32)` trainer.
3. Start and warm the exact 16-process target pool.
4. Measure the first full batch-100 update separately because it includes the
   parent TensorFlow graph tracing/initialization.
5. Measure five steady full updates. Five is a convenience diagnostic sample;
   observed continuous differences are descriptive only.
6. Measure one validation plus support-probe cycle, matching the 250-step
   cadence.
7. Measure one terminal 256-point audit-equivalent cycle.
8. Extrapolate the measured components to one and two streams at 250, 1,250,
   and 2,000 steps. Report mean-based estimates plus observed step-min/max
   sensitivity bounds; do not call these confidence intervals.

## Skeptical Audit

- Wrong baseline: avoided by using the exact q=20 target, training family,
  batch size, worker topology, and checkpoint operations.
- Proxy promotion: timing is the only output; loss is a finite-execution check.
- Hidden overhead: startup, cold compilation, periodic checkpoint work, and
  terminal audit are measured separately rather than omitted.
- Unfair GPU comparison: no GPU speed comparison will be made from this run.
- Thread oversubscription: explicit environment, TensorFlow configuration,
  affinity, and realized task-count evidence are required.
- Short-sample uncertainty: extrapolations are labeled descriptive and include
  min/max sensitivity bounds.
- Misleading pass: a fast diagnostic does not establish CPU training quality or
  change the GPU-default policy.
- Misleading failure: a slow or failed CPU run says nothing about NeuTra's
  scientific viability.

Audit decision: `PASS_FOR_BOUNDED_DIAGNOSTIC`.

## Result Summary

The relaxed XLA run is recorded at
`docs/plans/artifacts/ssl-lstm-q20-cpu-training-timing-2026-07-22/r2/result.json`.
It respected 50-CPU affinity and configured compute pools but created 213
native OS threads, so it does not answer the literal thread-cap question.
`r3` is the strict non-XLA diagnostic and completed with exactly 50 observed
native OS threads across the process tree.

| Decision | Primary criterion | Veto status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| Relaxed XLA diagnostic preserved | Complete finite timing run under 50-CPU affinity | Literal thread cap failed: 213 native OS threads | Run strict non-XLA `r3` | Does not answer the requested strict thread-bound question |
| Strict CPU diagnostic accepted | Complete finite non-XLA timing run with observed process-tree threads `<=50` | Passed with exactly 50 threads | Use `r3` for strict CPU fallback scheduling | Does not establish transport quality or replace GPU training |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Strict `r3` passed; relaxed `r2` remains historical because it failed the literal native-thread cap |
| Statistically supported ranking | None; this is one bounded timing diagnostic |
| Descriptive-only differences | Startup time, cold compilation, steady-step variation, checkpoint cost, and extrapolated ranges |
| Default readiness | Not assessed; GPU remains the NeuTra training default |
| Next evidence needed | A GPU run for claim-bearing training and the subsequent sequential HMC campaign |

## Exact Command

```bash
timeout 900 taskset -c 0-49 python \
  docs/benchmarks/benchmark_ssl_lstm_q20_cpu_neutra_training_2026_07_22.py \
  --output \
  docs/plans/artifacts/ssl-lstm-q20-cpu-training-timing-2026-07-22/r3/result.json
```
