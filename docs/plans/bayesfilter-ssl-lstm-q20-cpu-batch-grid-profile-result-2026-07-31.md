# q=20 CPU Worker/Shard Grid Profile Result

Date: 2026-07-31
Status: `COMPLETED_DIAGNOSTIC_ONLY`

## Result

The standalone CPU profiles completed with finite, parity-stable values and
scores, exact worker/shard counts, verified CPU pinning, and RSS below the
64 GiB cap. The route was the current status-bearing batch-native q=20 target
with `principal_sqrt_backend="tensorflow_eigh"`, CPU-only, `float64`, and
non-XLA execution.

| Topology | Total rows | Warm calls (s) | Warm mean (s) | Warm throughput |
| --- | ---: | ---: | ---: | ---: |
| `10 workers x 10 rows` | 100 | `13.7335`, `13.4802`, `13.5105` | `13.5747` | `7.3666 rows/s` |
| `15 workers x 15 rows` | 225 | `42.7352`, `42.6858`, `42.5204` | `42.6471` | `5.2759 rows/s` |
| `32 workers x 2 rows` | 64 | `5.3510`, `5.4328`, `5.4706` | `5.4181` | `11.8122 rows/s` |
| `32 workers x 8 rows` | 256 | `37.7408`, `37.6479`, `37.9250` | `37.7712` | `6.7776 rows/s` |
| Historical `25 workers x 4 rows` | 100 | prior mean `14.3649` | `14.3649` | `6.9614 rows/s` |

The `10x10` topology is slightly faster than the historical `25x4` topology
for a 100-row request. Increasing the per-worker shard from 10 to 15 rows
reduced throughput rather than preserving linear scaling. This indicates that
the rows are independent across workers, but the batch-native target cost
inside each worker grows materially with shard size. For a 256-row request,
`32x8` is faster than `16x16` in this standalone measurement, but it consumes
materially more memory and threads. These measurements support independent
worker execution, but not linear scaling.

## Resource And Validity Receipts

- `10x10` startup: `3.0232 s`; peak combined RSS: `7.695 GiB`; native worker threads: `50`.
- `15x15` startup: `3.0702 s`; peak combined RSS: `13.212 GiB`; native worker threads: `75`.
- `32x2` startup: `3.7934 s`; peak combined RSS: `23.539 GB` (`21.92 GiB`); native worker threads: `160`.
- `32x2` first call: `13.0351 s`; warm worker runtime range: `4.16`-`5.46 s`; all warm value/score deltas from the first call were zero.
- `32x8` startup: `4.4830 s`; peak combined RSS: `23.504 GB` (`21.89 GiB`); native worker threads: `160`.
- For the same 256-row request, `32x8` is descriptively `11.7565 s` (`23.7%`) faster than standalone `16x16` (`37.7712 s` versus `49.5278 s`) and has `31.1%` higher observed warm throughput. It also uses about `1.65x` the peak RSS and `2x` the native worker threads.
- All warm-call value/score deltas from the first call were exactly zero in the recorded tensors.
- Worker runtime skew remained material; the wall time was set by the slowest worker, not the median worker.

Artifacts:

- `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/10x10-standalone-r3/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/15x15-standalone-r3/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/32x2-standalone-r1/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/16x16-standalone-r1/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/32x8-standalone-r1/result.json`

## Invalid Attempt And Repair

The first `10x10` and `15x15` processes were launched concurrently while both
pinning workers from CPU 0 upward. Their measurements are retained under
`10x10-r2` and `15x15-r2` as contention diagnostics and are not used above.

Both initial launches also exposed a harness admission mismatch: the shared
CPU pool admitted only the historical policy string
`batch_native_tensorflow_no_row_mapping_v1`, while the current target reports
the status-bearing policy
`batch_native_tensorflow_status_no_row_mapping_v2`. The pool was repaired to
admit exactly those two explicit policies; no open-ended policy bypass was
introduced. The focused process-parallel pytest invocation then hung during
worker setup and was stopped after a bounded wait; Python compilation and
`git diff --check` passed.

## Interpretation

The CPU results are much faster than the current hybrid GPU audit receipt, but
the comparison is not backend-identical. The GPU `256`-row audit receipt is
`2926.534 s` and performs the target once for loss and again for status. The
CPU profiles perform one value/score target call per row batch. Using the
measured CPU throughputs, a 256-row one-call equivalent is approximately
`34.7 s` at the `10x10` throughput and `48.5 s` at the `15x15` throughput;
the direct `32x8` measurement is `37.8 s` warm mean.
The validation-scale `32x2` measurement is `5.42 s` warm mean for a single
64-row CPU value/score call. The GPU 64-row receipt is `741.12 s` and includes
the validation target call plus a separate status call; therefore the ratio is
descriptive timing evidence, not an end-to-end parity result. At the measured
CPU call rate, the six process/pool first calls plus 28 warm calls in the
declared four-arm/two-stream protocol would total about `230 s` (`3.8 min`)
before any status-preserving wrapper or transfer overhead.
Those are descriptive extrapolations, not measurements or a CPU/GPU ranking.

## Decision Table

| Decision | Primary criterion | Veto status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| Characterize CPU worker/shard scaling | Standalone profiles completed and finite | No validity veto; `32x2`/`32x8` have a higher resource footprint | Use `32x2` as the measured 64-row validation-scale point and `32x8` for the measured 256-row point | No production CPU default |
| Predict 75-worker performance | Not tested | No claim permitted | Run a standalone 75-worker profile only if needed | No linear-scaling assumption |
| Compare with hybrid GPU | Descriptively much faster | Backend and call-count mismatch | Prioritize GPU-native `tensorflow_eigh_strict` localization | No CPU-vs-GPU superiority claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for all standalone CPU diagnostics |
| Statistically supported ranking | None; three warm calls per topology are descriptive only |
| Descriptive-only differences | `32x2` has higher observed throughput than `32x8` at smaller shard size; `32x8` has higher throughput than `16x16`; `10x10` has higher throughput than `15x15` and the historical `25x4` point |
| Default readiness | Not assessed; CPU route is a diagnostic exception |
| Next evidence needed | GPU-native strict-eigh parity and timing, or a separately authorized larger CPU topology profile |
