# Shared HMC execution: implementation and validation

The shared executor now supports native batches, one process per chain, and
batches distributed across persistent processes. Logical CPU affinity and
TensorFlow threads are configured per worker before target import. GPU/XLA is
the default, with verified memory growth. Available accepted/proposed target
status, scores, momenta and state health are checked; failed calls retain
evidence and reap every worker. Equal batch shapes share compiled runners.

The benchmark API and module CLI compare target values/scores with singleton
evaluation, check replay within a partition, retain failed cells, and report
startup, first-call, warm-call, host RSS and GPU allocator measurements. The
existing TFP runner remains the numerical authority. Its new explicit dtype
option supports float32; omitting that option preserves historical float64
behavior. Diagnostic conversions use the existing float64 casting helper.

The feature branch is `feat/hmc-chain-execution-benchmark-20261001`, based on
`88297ad29`. The evidence manifest records the implementation checksums because
validation ran before the feature commit. TensorFlow 2.19.1 / TFP 0.25.0 ran in
`/home/ubuntu/miniforge3/envs/tf-gpu`; GPU validation used explicit tool escalation
and GPU3, UUID `GPU-b8045e28-4433-ec7a-77a5-db0636748322`. Display GPUs were avoided.
The CPU suite intentionally hid GPUs. All raw worker records, tensor shards and
failure logs are retained in the evidence archives, including the structured
GPU reference failure. No package changes or posterior/training runs were made.

| Engineering check | Result |
|---|---|
| Feature plus existing runner/status regression suite | 52 passed, 104.14 seconds |
| Expanded feature suite after final safeguards | 12 passed, 91.02 seconds; 55 distinct checks overall |
| CPU replay across 1/2 workers at fixed batch partition | Identical tensor hashes and exact direct-runner samples |
| Dynamic states/seeds/epsilon, uneven partitions, float32 and stable tracing | Passed |
| Worker exit/deadline, rejected invalid proposal, dtype mismatch, cross-chain coupling | Correct rejection and retained evidence |
| Default XLA / explicit non-XLA / lazy imports / display avoidance | Passed |
| GPU Kalman value/score singleton-versus-batch parity | Maximum absolute error 4.44e-16 |
| GPU two-chain batch and two one-chain workers | Replay, health, placement, memory growth and XLA passed |
| Lint | New modules/tests clean; touched legacy files retain their existing lint findings |

The CPU checks found a missing scope in the diagnostic example, a missing seed
in the direct-comparator test and float64-only host diagnostic conversions for
float32 results. All were repaired before the successful suites. The first GPU
run compiled the numerical target but failed because tensor serialization had
inherited strict GPU placement. TensorFlow implements this artifact operation on
CPU. Explicit CPU serialization repaired the failure; GPU revisions 2 and 3
passed. No numerical tolerance or HMC health gate was relaxed.

For the final small Kalman CLI run (2 chains, 6 transitions, L=2, epsilon=.03,
one process repetition and two warm replays), the measurements were:

| Layout | Startup s | First HMC s | Warm end-to-end s | Sum of host peak RSS MiB | Sum of TF GPU allocator peaks bytes |
|---|---:|---:|---:|---:|---:|
| Two chains in one batch, one worker, two cores | 3.778 | 6.361 | 0.027093 | 1199.6 | 39,424 |
| One chain per worker, two workers, one core each | 4.057 | 6.568 | 0.037982 | 2391.8 | 78,336 |

GPU allocator bytes exclude CUDA contexts, driver allocations and unrelated
processes. Host peak sums can double-count shared pages and are not simultaneous
live memory. The tiny fixture's process overhead is material. The prior smoke
ranked these layouts in the opposite order under shared GPU load; this is direct
evidence against treating either short timing as a hardware recommendation.
Use several fresh-process repetitions on the actual application target to choose
a layout. The harness records individual timings and observed spread, and calls
its result only the fastest observed eligible layout.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept shared execution feature | Process/direct parity and resource lifecycle pass | No unresolved engineering veto | Arbitrary application batch semantics need their own tests | Integrate tested branch; applications can adopt the shared API | Universal speedup or application migration |
| Accept benchmark harness | Fixed-work parity/replay and structured failures pass | Invalid cells cannot be nominated | Shared-load timing variability, small fixture | Benchmark the user's own target/hardware | Statistical ranking or best HMC kernel |

| Inference status | Finding |
|---|---|
| Hard veto screen | Passed for the two tested GPU layouts and eligible CPU fixtures |
| Statistically supported ranking | Not established |
| Descriptive differences | Reported per layout and repetition |
| Default-readiness | GPU/XLA execution defaults verified for this interface; no posterior promotion |
| Next evidence needed | Application-native batching review and representative machine benchmark |

Terminal skeptical review checked the complete factory → worker → shared runner
call chain through direct executable parity. It also checked cold-versus-warm
timing, CPU affinity before import, failure cleanup, proposed-state visibility,
dtype preservation, stable seed partitioning and source identity. No independent
model review was commissioned. The strongest alternative explanation for the
timing difference is contention and framework overhead, already supported by the
reversal between two short runs. The weakest performance evidence is the tiny
single-repetition workload, so no speed superiority is claimed. This is Linux /
POSIX execution infrastructure; Windows support, migration of existing
applications, and process-topology wiring into the high-level tuner and NeuTra
sequential controller are separate work.

Evidence: [validation manifest](artifacts/hmc-chain-execution-20261001/validation-manifest.json),
[final GPU report](artifacts/hmc-chain-execution-20261001/gpu-kalman-cli-r3/report.md),
[API guide](../reference/hmc-chain-execution.md).
