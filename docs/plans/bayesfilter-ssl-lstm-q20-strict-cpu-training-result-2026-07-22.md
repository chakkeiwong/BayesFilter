# q=20 SSL-LSTM Strict CPU NeuTra Training Result

Date: 2026-07-23  
Status: `BLOCKED_BATCH_NATIVE_CPU_PERFORMANCE_BEFORE_FIRST_CHECKPOINT`

## Outcome

No policy-valid CPU NeuTra training result was produced. The initially fast
route used scalar target workers and is ineligible under the repository's
batch-native training requirement. A replacement TensorFlow batch-native route
passed value/score parity and resource preflights, but it did not reach the
first 250-step checkpoint within about 22 minutes. It was stopped rather than
spending the full campaign cap without observable progress. Because the runner
writes progress only every 250 steps, the completed update count is unknown in
`[0,249]`.

## Attempt Ledger

| Attempt | Route | Outcome | Scientific status |
| --- | --- | --- | --- |
| `r1` | Eight `scalar_eager_value_score` workers | Interrupted after seed-a step 250 | Harness-invalid; not training evidence |
| `r2` | New batch-native worker pool | Worker initializer set TensorFlow threads after context initialization | Infrastructure failure; zero target evaluations |
| `r3` | Repaired batch-native pool, 180-second smoke | Initial validation/support preflight completed; cap exhausted before training | Valid resource preflight only; maximum 45 native threads |
| `r4` | Repaired batch-native pool, full cap | Initial preflight completed; no 250-step checkpoint after about 22 minutes | Performance stop; completed updates unknown in `[0,249]` |

## Checked Evidence

- Batch-native target parity against the scalar authority at q=1, q=2, and
  q=20: maximum value error below `5e-12`; maximum score error below `4e-11`.
- No `tf.map_fn`, `tf.vectorized_map`, Python row loop, NumPy numerical path, or
  scalar worker fallback in the revised target/pool route.
- CPU-only execution with CUDA hidden before TensorFlow import.
- `r3` strict process-tree maximum: 45 native threads, below the 50-thread cap.
- Host-memory cap remained 64 GiB; no host-memory veto fired.
- No worker processes remained after the terminal stop.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not continue the present CPU campaign | Neither stream reached its first 250-step checkpoint | Continuation veto: no observable checkpoint in the bounded window | Per-update timing is absent, so tracing cost and steady-state throughput are not separated | Add per-update timing/progress, profile one 25-row worker evaluation, and optimize the batch-native target; or wait for GPU/XLA training | NeuTra failed; `(32,32)` failed; posterior invalid; HMC invalid |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Scalar route invalid; batch-native CPU performance continuation veto supported |
| Statistically supported ranking | None |
| Descriptive-only differences | Startup duration, thread count, and absence of first-update completion |
| Default readiness | None; CPU artifacts remain diagnostic-only |
| Next evidence needed | A profiled and optimized batch-native q=20 target update, or GPU/XLA training when a GPU is available |

## Engineering, Numerical, And Scientific Ledgers

- Engineering correctness: the batch-native value/score implementation matches
  the scalar authority at checked points. The multiprocessing initializer bug
  was repaired. Long-run progress/resume remains untested because no update
  reached the first checkpoint.
- Numerical validity: checked parity points were finite and close. No trained
  transport exists from this campaign.
- Scientific interpretation: none. This result diagnoses CPU execution cost;
  it does not test NeuTra quality, posterior convergence, or SSL-LSTM adequacy.

## Post-Run Red Team

The follow-up profile resolves the main alternative explanation. Pool startup
was 2.303 seconds, parent serialization/aggregation was under 0.02 seconds per
call, and repeated size-25 worker calls remained 27.0-27.8 seconds. The
dominant cost is steady-state batched UKF evaluation, not tracing. A further
training run under the same CPU budget is therefore not justified without
target-kernel optimization.

The weakest evidence is the approximate `r4` elapsed time, because the manual
stop occurred before the launcher could write its normal run manifest. The
location of the wait is firm from the interruption traceback:
`TFBatchValueScorePool.evaluate` waiting for a training request. The profile
now identifies its steady-state cost and preserves the per-call measurements.
