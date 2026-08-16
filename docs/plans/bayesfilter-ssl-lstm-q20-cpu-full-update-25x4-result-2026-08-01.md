# q=20 CPU Full Optimizer Update `25x4` Result

Date: 2026-08-01
Status: `COMPLETED_DIAGNOSTIC_ONLY_WITH_XLA_TOPOLOGY_COMPARATORS`

## Result

The requested complete 100-row CPU optimizer update using `25 workers x 4
rows` completed four times with finite values, scores, losses, and gradients.
The measured route includes transport forward/log-determinant, pooled target
value/score evaluation, external-value/score gradient calculation, and the
optimizer parameter update.

| Call | Wall time | Target value/score | Optimizer/update |
| --- | ---: | ---: | ---: |
| First | `24.1618 s` | `22.5243 s` | `1.6041 s` |
| Warm 1 | `15.5656 s` | `15.5544 s` | `0.0042 s` |
| Warm 2 | `15.6692 s` | `15.6589 s` | `0.0042 s` |
| Warm 3 | `15.6364 s` | `15.6261 s` | `0.0044 s` |
| Warm mean | `15.6237 s` | `15.6131 s` | `0.0043 s` |

The warm full-update throughput is `6.4005 training rows/s`. The first call
includes graph/worker warm-up; the three warm calls are the relevant steady
state measurement.

## CPU XLA Comparator

The matched CPU-XLA run used the same topology, rows, seed, architecture,
learning rate, dtype, and `tensorflow_eigh` backend, with `jit_compile=True`
for both the 25 worker target graphs and the parent transport/optimizer graph.
TensorFlow emitted successful XLA compilation receipts for the workers.

| Call | Non-XLA | XLA |
| --- | ---: | ---: |
| First | `24.1618 s` | `23.8891 s` |
| Warm 1 | `15.5656 s` | `6.9684 s` |
| Warm 2 | `15.6692 s` | `6.7844 s` |
| Warm 3 | `15.6364 s` | `6.8482 s` |
| Warm mean | `15.6237 s` | `6.8670 s` |

XLA is descriptively `2.275x` faster in the warm steady state, saving
`8.7567 s` per 100-row update. The corresponding 2,000-update arithmetic is
approximately `3.82 h` with XLA versus `8.68 h` without XLA, before validation,
support, audit, and startup costs. The first-call timing is not faster because
it includes compilation in both configurations.

The displayed loss and gradient receipts match between the two runs to the
shown precision. This is numerical consistency evidence for this short
synthetic comparison, not a proof of broad backend equivalence.

## CPU XLA `50x2` Comparator

The second matched XLA run kept the 100-row batch and all target/trainer
settings fixed while changing the worker topology from `25x4` to `50x2`.

| Topology | First call | Warm 1 | Warm 2 | Warm 3 | Warm mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `25x4` | `23.8891 s` | `6.9684 s` | `6.7844 s` | `6.8482 s` | `6.8670 s` |
| `50x2` | `23.6691 s` | `6.2976 s` | `5.6709 s` | `5.8820 s` | `5.9502 s` |

`50x2` is descriptively `0.9168 s` (`13.4%`) faster than `25x4` in this
three-repeat sample. The simple 2,000-update arithmetic is approximately
`3.31 h` for `50x2` versus `3.82 h` for `25x4`, before validation, support,
audit, and startup costs.

## CPU XLA `50x4` Comparator

The `50x4` run evaluates a 200-row training batch with 50 pinned workers and
four rows per worker. It completed all four full optimizer updates with finite
values, scores, losses, and gradients.

| Topology | First call | Warm 1 | Warm 2 | Warm 3 | Warm mean | Warm rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `25x4` (100 rows) | `23.8891 s` | `6.9684 s` | `6.7844 s` | `6.8482 s` | `6.8670 s` | `14.5624` |
| `50x4` (200 rows) | `30.6662 s` | `11.2963 s` | `11.4846 s` | `11.0070 s` | `11.2626 s` | `17.7579` |

The 200-row update is `1.640x` the wall time of the 100-row update, not the
same time. It is nevertheless `22.0%` more efficient per sample. At 2,000
updates, the arithmetic is approximately `3.82 h` for 100 rows/update and
`6.26 h` for 200 rows/update, before validation, support, audit, and startup.
For the same 200,000 training samples, 1,000 updates of 200 rows would cost
approximately `3.13 h`, but that is a different optimizer schedule with half
as many parameter updates and cannot be assumed to have the same training
quality.

## Comparison

The measured hybrid GPU `(64,64)` optimizer updates were `568.1132`,
`569.5429`, and `566.3587 s`, with mean `568.0049 s`. On this exact measured
implementation, the CPU `25x4` full update is descriptively about `36.4x`
faster, saving about `552.4 s` per update. A simple 2,000-update arithmetic
comparison is approximately `8.68 h` for this CPU timing versus `315.56 h`
for the measured hybrid GPU timing, before validation, support, audit, and
startup costs.

This is a valid implementation-cost comparison for the measured routes. It is
not a claim that CPU is universally faster than GPU, because the GPU route is
the host-staged `compiled_custom_op` path and the CPU route is non-XLA
`tensorflow_eigh`.

## Resource And Validity Receipts

- Topology: `25` persistent workers, `4` rows per worker, `100` total rows,
  CPUs `0..24`.
- Startup: `3.1111 s`.
- Peak combined RSS: `18,857,844,736 bytes` (`17.56 GiB`).
- Native worker threads: `200` across the parent/worker process tree.
- Warm worker runtime maximums: `15.5367`, `15.6426`, and `15.6094 s`.
- All target values and scores were finite; all optimizer losses and gradient
  norms were finite; clipping was active on all four updates.
- Warm loss moved from `66.4976` to `66.4010` to `66.3039`; this is descriptive
  training telemetry only.
- XLA peak combined RSS: `18,853,199,872` bytes (`17.56 GiB`), effectively the
  same as non-XLA, but native worker threads increased to `675`.
- XLA worker runtime maximums: `6.9348`, `6.7416`, and `6.8072 s`.
- XLA `50x2` peak combined RSS: `36,784,521,216` bytes (`34.25 GiB`), versus
  about `18.85 GB` for `25x4`.
- XLA `50x2` native worker threads: `1,350`, exactly twice the `675` observed
  for `25x4`; warm worker runtime maximums were `6.2492`, `5.6227`, and
  `5.8305 s`.
- XLA `50x4` peak combined RSS: `36,879,327,232` bytes (`34.33 GiB`) and
  native worker threads: `1,350`.
- XLA `50x4` warm worker runtime maximums: `11.2296`, `11.4059`, and
  `10.9399 s`.

Artifact:

`docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/r1/result.json`

XLA comparator:

`docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-r1/result.json`

XLA `50x2` comparator:

`docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-50x2-r1/result.json`

XLA `50x4` comparator:

`docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-50x4-r1/result.json`

## Decision Table

| Decision | Primary criterion | Veto status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| Compare CPU and hybrid GPU update cost | Four complete CPU updates and prior GPU receipts are finite | No CPU diagnostic veto; backend identity differs | Use CPU `25x4` for bounded local diagnostic training if desired | No universal CPU/GPU ranking |
| Compare CPU XLA and non-XLA | Matched four-call runs completed; XLA compiled successfully | Higher XLA native-thread footprint requires resource review | Prefer CPU XLA for bounded timing/training diagnostics if thread budget permits | No production/default promotion |
| Compare XLA worker topologies | `50x2` completed all four updates and is descriptively faster | RSS and native threads are approximately doubled | Use `50x2` only when the host resource budget accepts `34.25 GiB` RSS and `1,350` native threads | No topology optimum claim |
| Compare 100 versus 200 training rows | `50x4` improves rows/s by about 22% | Batch-size effect on optimization quality is not measured | Run a bounded equal-sample/equal-update training comparison before changing the batch | No claim that 200 rows is better |
| Replace hybrid GPU optimizer for claim-bearing work | CPU timing is much lower | CPU route is a diagnostic exception; no claim-bearing admission | Keep claim-bearing route separate; pursue genuinely GPU-native strict-eigh timing | No default or HMC promotion |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed finite, topology, pinning, RSS, and worker-completeness checks |
| Statistically supported ranking | None; three warm repeats are descriptive timing evidence |
| Descriptive-only difference | CPU non-XLA `25x4` is about `36.4x` faster than hybrid GPU; CPU XLA `25x4` is about `2.28x` faster than non-XLA; XLA `50x2` is about `13.4%` faster than XLA `25x4`; XLA `50x4` has 22% higher sample throughput than XLA `25x4` |
| Default readiness | Not assessed; CPU remains a diagnostic exception |
| Next evidence needed | If CPU XLA is used beyond timing, add status-bearing validation/audit receipts and a reviewed CPU resource protocol; compare 100 versus 200 rows on equal samples and updates before changing batch size; for GPU claims, benchmark the GPU-native strict-eigh route |

## Run Manifest

The complete run manifest, command, environment, seed, git commit, source
hashes, CPU-only status, and artifact path are embedded in the JSON result.
