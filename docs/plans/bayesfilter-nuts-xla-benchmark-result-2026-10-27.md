# Standalone NUTS XLA benchmark: first engineering pilot

Configuration status: **PILOT HYPOTHESES, NOT CANONICALLY TUNED**. These are
short device/timing probes of stock TensorFlow Probability NUTS. The fixed-HMC
arm has adapted epsilon but an unsearched L; it is an untuned contextual
baseline. No table below establishes a sampler ranking, posterior qualification,
or a production/default change. Do not infer that a filtering model will have
the same CPU/GPU ordering.

The deliverable is `docs/benchmarks/benchmark_nuts_xla.py`, with the evidence
contract in `bayesfilter-nuts-xla-benchmark-2026-10-27.md`. The owner request
authorizes this isolated benchmark exception to the previous NUTS exclusion.
The retired implementations and existing campaigns remain excluded.

## What was measured

Stock TFP NUTS 0.25.0 on TensorFlow 2.20.0; float64, TF32 disabled, identity
mass, maximum tree depth 6, unroll 1, initial epsilon .25 and dual-averaging
target acceptance .8. Each of three repetitions has a fresh seed/start, 128
discarded warm-up transitions per chain and 256 retained transitions per chain.
Adaptation covers the first 102 warm-up transitions; the remainder is frozen.
CPU workers request four intra-op threads and one inter-op thread; GPU workers
use the NVIDIA GeForce RTX 4080 SUPER selected as physical GPU 0. The host
reports AMD EPYC 7742. CPU workers intentionally hide CUDA devices; the CUDA
NO_DEVICE startup message in their logs is not a GPU readiness failure.

| NUTS workload | CPU median retained time | GPU median retained time | Observed GPU/CPU ratio |
| --- | ---: | ---: | ---: |
| Gaussian, dimension 2, 4 chains | 0.007285 s | 0.090529 s | 12.43 |
| Banana, dimension 2, 4 chains | 0.013889 s | 0.190009 s | 13.68 |
| Product banana, dimension 32, 32 chains | 0.202665 s | 0.757104 s | 3.74 |

Times cover the entire retained batch and a device synchronization reduction,
excluding warm-up and bulk file output. These ratios are descriptive ratios of
medians, not confidence bounds. CPU/XLA is the provisional execution choice
for these specific simple FP64 workloads. The larger chain batch narrows the
observed gap, but this study does not locate a crossover.

Compilation matters for short jobs. The first retained call, including trace,
compile and execution, took 1.497/5.023 s (CPU/GPU) for the Gaussian,
1.535/5.166 s for the small banana, and 1.883/6.004 s for the larger banana.
The separate first warm-up calls took 1.748/5.608, 1.830/5.633, and 2.179/6.248 s.
These are not estimates of pure compiler time. They are saved alongside warm
warm-up timings in each manifest and replicate record.

The contextual CPU fixed-HMC arm (L=8) took a median 0.003022 s per small-banana
batch, with mean reported leapfrog count 8 versus approximately 4.55 for NUTS.
This motivates measuring NUTS tree/control overhead, but does not prove that
fixed HMC is statistically superior or optimally tuned. Reported chain-local
leapfrogs do not count all masked work in a batched NUTS tree.

## Compilation and engineering validity

The enclosing warm-up and retained functions use stable static signatures and
`tf.function(jit_compile=True)`. They include target evaluation, autodiff scores,
leapfrog integration, NUTS tree control/selection, stateless RNG, dual averaging,
transition iteration, state storage and health telemetry. Each function traced
once. Optimized HLO is saved for both functions in every completed cell. The
graph/library callback scan found no PyFunc/EagerPyFunc or pfor nodes. Output
device placement was checked with soft placement disabled. GPU memory growth
was enabled and verified before initialization.

Python setup, timers, JSON/tensor writing, start preparation and untimed
post-run R-hat/ESS summaries are outside XLA. This is full compilation of the
repeated sampler and adaptation computations, not a claim that Python I/O or
every diagnostic operation is XLA compiled. Diagnostics are not timed as part
of sampler throughput.

Seven cells completed and one GPU attempt failed after successful sampling
because SerializeTensor was explicitly placed on GPU, where it has no kernel.
This was a host-I/O defect, not a NUTS/XLA failure. Moving only serialization to
`/CPU:0` repaired it; the fresh Gaussian GPU attempt completed. The failed
attempt, logs, source and compute charge remain preserved. No eager or partial
graph sampling fallback was used. All completed cells retained finite
state/log-density/score caches and well-formed acceptance telemetry.

The focused CPU/XLA suite initially passed 9 tests, covering the banana value
and analytic score, seed reproducibility/change, compiled warm-up/retention,
HLO availability, stable tracing, and the continued exclusion of other NUTS
routes. The final post-repair run also passed all 9 tests in 22.20 s; its log
is saved in `audit-01/focused-tests-final.log`. Final artifact verification
checked all seven completed cells, 84 nonempty tensor shards, replicate counts,
one trace per compiled function, and the preserved executed sources.

## Statistical limits and remaining hypotheses

The small Gaussian reached maximum rank/folded split R-hat 1.01935 and the
small banana reached 1.03732. These do not pass the repository's 1.01 posterior
screen. The larger banana reached 1.00735 on CPU and 1.00729 on GPU, but passing
that single short-run diagnostic does not establish convergence or reference
calibration. Exact moment errors, inverse-coordinate moments, bulk/tail ESS,
and the banana nonlinear moment E[x^2 y]=2b are retained as descriptive checks.
There is no predeclared uncertainty-based reference-agreement acceptance test
in this pilot; posterior correctness remains unevaluated.

All tree-cap fractions were zero. Native NUTS divergence flags were zero on
the Gaussian, roughly .3%--3.0% across small-banana repetitions and .02%--.07%
on the larger banana. Under the October 26 policy, these flags are reporting
only: they neither reject an epsilon candidate nor prescribe a smaller step.
Safely rejected states remain in the denominator. Invalid retained state or
cache would instead invalidate execution.

CPU/GPU replicas are independently adapted from matched seed identifiers. The
third small-banana replica chose different epsilon and tree paths after
floating-point differences; thus those timings include different realized
workloads. Device order was not randomized, CPU sandbox and trusted GPU worker
contexts differ, host loads changed, and this is FP64 on a consumer GPU. No
paired uncertainty analysis or isolated-device profiler was run. An environment
or launch contribution cannot be separated from numerical kernel cost here.

The inherited dual-averaging settings, inspected in the installed source, are
exploration shrinkage .05, step-count smoothing 10, decay exponent .75,
shrinkage target 10 times starting epsilon, and arithmetic mean acceptance
across chains (`reduce_logmeanexp` on log acceptance). These are third-party
baseline hypotheses, not reviewed universal defaults. The initial start scale
2 and retention seed salt 1729 are convenience choices. The same seed tensor
was supplied to start generation and warm-up in this pilot; no claim of
independent RNG streams is made. Use explicitly split streams in confirmation.
The source and settings are archived so these choices are reproducible.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Deliver standalone harness | Full warm-up/retention XLA and artifact preservation pass | One repaired I/O failure; no completed retained-cache defect | Only simple analytic targets | Reuse for a controlled speed investigation | Filtering paths compile |
| Nominate CPU/XLA for these shapes | Lower observed retained and first-call time in all three workload pairs | No device compilation veto | Nonrandomized order; FP64 and hardware specificity | Frozen-input, frozen-epsilon paired validation | Universal CPU superiority |
| Keep stock NUTS as the baseline | Source inspected; mechanism unchanged | Small-run posterior screen not satisfied in every cell | Statistical precision and actual batched work | Profile tree control and test mechanical optimizations | A faster NUTS implementation has been delivered |
| Stop this pilot | All planned shapes and contextual HMC completed | Eight-launch cap exhausted | No new workload budget in this plan | Design next bounded experiment | NUTS research direction rejected |

| Inference class | Status |
| --- | --- |
| Hard veto screen | Full XLA/device/finite-cache pass for completed cells; failed serialization attempt excluded from timing summaries |
| Statistically supported ranking | None |
| Descriptive-only differences | CPU/GPU wall time, first-call cost, ESS/sec, moment errors, R-hat and contextual HMC comparison |
| Default-readiness | Not assessed; public defaults unchanged |
| Next evidence needed | Identical frozen workloads, randomized device order, independent timing sessions, uncertainty, longer replicated posterior/reference checks |

Post-run red team: the strongest alternative explanation is that cheap FP64
targets and hardware/control overhead dominate the observed ordering. More
expensive native matrix/filter targets or FP32 may reverse it. The weakest
evidence is generalization from three shapes and three short replicas. A
controlled expensive-target comparison overturning the ordering would change
the device choice without invalidating the current measurements. The failed
attempt invalidated an artifact-writing path, not the target, math or sampler;
the planned repair was completed.

## Next speed experiment

Freeze initial states, mass, epsilon, tree depth and seeds across stock and
candidate implementations. Keep a stock NUTS baseline and a separately tuned
fixed/randomized HMC reference. Measure gradient evaluation, tree bookkeeping,
RNG, masked batch work and synchronization with a profiler before altering the
algorithm. Start with computational reuse/fusion, then evaluate a factorial
dimension-by-chain-count matrix. Test FP32 as a separate numerical arm with
reference checks. Changing `unrolled_leapfrog_steps` multiplies trajectory
length and changes U-turn checking cadence; it requires separate validation
and is not a free compiler optimization. Do not shrink tree depth merely to
make timings look favorable. Add a small exact Kalman target after the analytic
controls, then a funnel only with an explicit validity/reference design.

## Artifacts, budget and checkpoint

Root: `docs/benchmarks/artifacts/nuts-xla-20261027/`. `pilot-index.json` lists
all eight exact commands, statuses, worker times and source hashes. Every
completed cell contains manifest, replicate JSON, summary JSON, warm-up and
retained tensor shards, optimized HLO, console log and executed source text.
The original two benchmark source versions were reconstructed and verified
against their already-recorded SHA-256 values. Inspected numerical dependency
hashes still match; those sources and the installed TFP kernel/adaptation
sources are archived in `audit-01/`. Git HEAD was
`7be915869881f1e700708d677dc0e355004f573e`; unrelated dirty work was preserved.

Eight of eight worker launches were used: seven completed, one failed. Total
recorded worker wall time is 151.625734 s, including 26.730564 s for the failed
GPU attempt. This is within the 1,200 worker-second limit. Launch allowance is
exhausted; no further benchmark launch belongs to this pilot. Focused regression
checks are engineering validation, not additional campaign cells.

Checkpoint: harness built, small Gaussian/banana and larger banana tested on
both devices, contextual CPU HMC tested, host serialization repaired, sources
and evidence preserved. Provisional device is CPU/XLA for these workloads.
Next task is a new bounded frozen-workload profiling/optimization plan, not
rerunning this pilot or reopening the retired production NUTS routes.
