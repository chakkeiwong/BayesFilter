# Phase 6 Result: Trusted GPU Performance And Focused Repair

Date: 2026-07-14

## Outcome

**PASS_PHASE6_AND_CONTINUE.** The certified exact target and full generic
trainer execute on trusted RTX 4080 SUPER GPU/XLA with valid status, no scalar
fallback, and practical rates. No kernel performance repair is currently
justified.

## Trusted Environment

- NVIDIA driver `591.86`, reported CUDA `13.1`.
- TensorFlow `2.19.1`, CUDA build, one RTX 4080 SUPER logical GPU.
- TensorFlow created GPU:0 with approximately 13.5 GB available.
- Trust basis: `owner_designated_managed_session_visible_gpu_trusted`.

## Sequential Target Ladder

| Batch | Warm/compile seconds | Three steady calls (s) | Mean (s) | Valid/status | Peak memory-info bytes |
| --- | ---: | --- | ---: | --- | ---: |
| 8 | 3.339 | 0.0657, 0.0661, 0.0643 | 0.0654 | pass | 355,072 |
| 32 | 3.458 | 0.0840, 0.0789, 0.0867 | 0.0832 | pass | 2,097,152 |
| 128 | 3.415 | 0.1336, 0.1336, 0.1407 | 0.1360 | pass | 8,505,600 |
| 256 | 3.570 | 0.2152, 0.2128, 0.2165 | 0.2148 | pass | 17,003,776 |

Memory values are raw bytes returned by TensorFlow `get_memory_info`, not MiB.
All rows had finite values/scores, status zero, no active floor, and GPU output
placement. With only three timings per row, scaling differences are descriptive
and do not support a statistically ranked batch choice.

An initial `B=32`/`B=128` pair was accidentally launched concurrently on the
same GPU. Those artifacts are retained under `attempt-01` as validity and
diagnostic evidence but rejected for timing comparison. The admitted timing
rows are sequential `attempt-02`, except the earlier isolated `B=8` row.

## Full Training Smoke

The existing strict source-anchor recipe ran five steps at `B=128`:

- decision: `ENGINEERING_VALID_GRAPH_NATIVE_NUMPY_FREE_TRAINING_JOB`;
- compiled training program: `3.7156 s` for five steps, or `0.7431 s/step`
  including the first compilation;
- total harness wall time: `8.9033 s` including setup, freeze, reload, parity,
  closure audit, and artifact writes;
- all trainable variables, Adam moments, and compiled outputs were on GPU;
- all target status rows valid, no floors, no fallback;
- observed recorded loss moved from `84.2411` at step 1 to `75.4357` at step 5,
  which is explanatory only.

The harness runs all steps in one compiled invocation, so it does not expose a
separate warm steady training rate. `0.7431 s/step` is a conservative
compile-inclusive program average, not a claimed steady-state rate.

## Performance Interpretation

The historical row-mapped diagnostic rate was approximately `10.03 s/step`.
Against the conservative compile-inclusive `0.7431 s/step` program average, the
new route is approximately `13.5x` faster. This comparison is descriptive
because the implementation topology changed, but it directly answers the
engineering bottleneck: the scalar row map has been removed and the practical
`<=1.4 s/step` aspiration is met without target-math changes.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit Phase 6 performance | valid trusted GPU/XLA B=128 target and full training pass | no device/status/resource veto | steady full-training rate not isolated | run fresh recipe smokes and 100-step stability | no recipe quality or HMC readiness |
| Do not tune batch size from ladder | only three descriptive timings per batch | no hard failure at any rung | throughput/optimization tradeoff untested | retain protocol B=128 | no statistical batch ranking |
| Do not run component repair | compile-inclusive full rate meets aspiration | no performance repair trigger | long-run stability unknown | Phase 7 stability | no claim that kernel is globally optimal |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | passed |
| Statistically supported ranking | none |
| Descriptive-only differences | all batch scaling and historical speedup |
| Default-readiness | engineering performance-ready at B=128; scientific/default promotion not established |
| Next evidence needed | fresh recipe smokes, 100-step stability, then target-specific screen |

## Artifact Hashes

- `B=8`: `946820693d835dd0ed7cdbd259dc651a59ce2e39ea5a06d0f4f0918007e311f2`
- sequential `B=32`: `e099ddf187b90e9487ad1ffae0d0399705c461cf086ef852fc43c576428dbe64`
- sequential `B=128`: `5d4c4b7feb47a8bcde8fd64f1f67dd6398d6db9db57fa42317e0df7ecaea91aa`
- sequential `B=256`: `0dbbe52f0462c61c897d92fa3b56ad52b3ba75c32e76e1debc5b60d06a7d2deb`
- five-step result file: `2f62aba0f624f3cb51ffc13388fa06f3a866da06fdae4db2e63f30b65096f063`

## Handoff

Phase 7 starts under
`docs/plans/bayesfilter-neutra-batch-native-training-phase7-stability-and-protocol-subplan-2026-07-14.md`.

