# SIR FP32 batch capacity: measured results

GPU native batches 4, 8 and 16 completed the SIR likelihood and analytical
score over all 50 observations. Batch 32 failed because the compiled program
requested one 21.04 GiB buffer. Batch 64 exceeded the compilation time limit;
batch 128 reached the host-memory guard during compilation. The latter two
attempts do not measure their full VRAM requirements. On CPU, only batch 4
completed a full evaluation within the per-arm allowance.

These are capacity and descriptive timing results for the existing filter,
not measurements of a complete NeuTra optimizer update or evidence of trained
map quality. No training or HMC was launched.

## Workload and reproducibility

The workload is SIR, N=4032, T=50, state dimension 18, observation dimension 9,
and three inferred parameters, with uniform FP32 arithmetic, TF32 disabled,
and XLA enabled. CPU children intentionally hid GPU devices before importing
TensorFlow. The Ryzen 9 9950X provides 16 physical cores and 32 logical CPUs;
the CPU lane used 16 intra-op threads in one process. GPU runs used the RTX
4080 SUPER, UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b. Linux sees about
45.9 GiB RAM. Other CPU/GPU work was present.

Each size ran in a fresh process. GPU allocation was capped at 12 GiB using
the repository's verified logical-device limit, an explicit exception to
memory growth. Host guards stopped an owned worker at 24 GiB RSS or below
6 GiB available system RAM. Each arm had 600 monotonic seconds, including
preparation, tracing, compilation and repetitions. These are shared-machine
capacity bounds, not exclusive access to the machine's full memory.

The frozen selection uses 16 flow steps, five reset steps, epsilon 1,
zero correction steps and the direct shared-Gaussian contraction. The exact
selection, beta, four-point parameter bank, data hash, fixed particle design
seed 261008301, controls, commands and source snapshots are preserved in
[the manifest](artifacts/sir-fp32-batch-capacity-20261020-02/manifest.json)
and each arm's result. These settings are workload baselines, not transferred
scientific admission from the original mixed-precision tuning scope.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow
`2.20.0-dev0+selfbuilt`, HEAD `10884528ca8b1a7463895256d30b9655d69c0955`
plus the saved working-tree source closure. All ten prepared workloads have
identical non-parameter inputs and controls; all common imported source hashes
match. The parameter rows repeat the same four-point bank as batch size grows.
The complete executed command is in the manifest. Analysis command:

```
python3 scripts/analyze_sir_fp32_batch_capacity.py docs/plans/artifacts/sir-fp32-batch-capacity-20261020-02 --output docs/plans/artifacts/sir-fp32-batch-capacity-20261020-02/summary.json
```

## CPU measurements

Host peaks are the maximum sampled RSS/high-water mark, including compiler
memory. A peak observed before a timeout is not a full-execution requirement.

| Batch | Peak host RAM, GiB | Completed first call, seconds | Full warmed calls | Outcome |
|---:|---:|---:|---:|---|
| 4 | 5.72 | 338.4 | 0 | Full first call valid; timeout during first warmed call |
| 8 | 10.17 | — | 0 | Timeout before first call returned |
| 16 | 19.33 | — | 0 | Timeout before first call returned |
| 32 | 23.69 | — | 0 | Timeout before first call returned |

CPU batch 4's TensorFlow allocator peak was 2.73 GiB; its larger process RSS
illustrates why allocator memory alone understates host requirements. The
338.4 seconds include compilation and execution. There is no measured warmed
CPU timing and no demonstrated complete fit for CPU batches 8, 16 or 32.
None of the CPU arms stopped because a memory guard fired.

## GPU measurements

The device column is measured TensorFlow allocator peak, not NVIDIA process
reservation. First-call times include compilation. Warmed times exclude
tracing and the first call. All returned calls in this table were finite,
valid, and completed all 50 observations for every row.

| Batch | Peak host RAM, GiB | Peak GPU allocation, GiB | First call, seconds | Warmed seconds | Full warmed calls | Rows/second | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 2.78 | 2.66 | 87.1 | 49.68 | 3 | 0.0805 | Fits |
| 8 | 3.79 | 5.29 | 157.8 | 96.77 | 3 | 0.0827 | Fits |
| 16 | 5.87 | 10.56 | 381.1 | 171.89 | 1 | 0.0931 | Fits; timeout during second warmed call |
| 32 | 9.95 | — | — | — | 0 | — | Allocation failed: requested 21.04 GiB |
| 64 | 18.20 | — | — | — | 0 | — | Timeout during compilation |
| 128 | 21.79 | — | — | — | 0 | — | Available host RAM fell below 6 GiB during compilation |

Batch 4's three warmed calls ranged from 49.25 to 49.80 seconds; batch 8's
ranged from 96.69 to 98.51. The batch-16 number is a single observation,
not a three-repeat median or a supported speed ranking. Within each completed
GPU arm, repeated values and scores were identical.

Batch 32 failed while allocating 22,595,825,888 bytes. This single requested
buffer exceeds even the physical 16 GiB card, so its failure is not explained
solely by the conservative 12 GiB cap. No measured VRAM value is assigned to
batches 64 or 128. Their incomplete attempts must not be called GPU OOMs.

## What limits larger batches

The traced graph has 23,446 / 43,482 / 83,554 / 163,698 / 323,986 / 644,562
nodes at batches 4 / 8 / 16 / 32 / 64 / 128. The factory expands small model
construction and proposal callbacks separately per row at tracing time;
the expensive density and transport operations retain the native batch axis.
Thus this is not a Python loop executing an entire scalar filter per row, but
it still has expensive graph duplication. Batch 64 took 62.5 seconds merely
to trace; batch 128 took 121.6 seconds. The GPU-32 log explicitly reports a
very slow XLA compilation before its later allocation error.

Pairwise particle work is the principal tensor-memory concern. The fixed
chunk policy gives K=2016. The direct shared-Gaussian contraction constructs
a logical residual of shape [B,2016,2016,18], followed by its precision
contraction. One such FP32 residual is about 0.273 GiB per row: 17.4 GiB
at batch 64 and 34.9 GiB at batch 128. These are logical sizes, not measured
device allocations; fusion and buffer reuse affect realized memory. The
actual batch-32 allocation failure is the stronger capacity evidence.
Sources are `make_batched_score_filter` in
`bayesfilter/highdim/covariance_proposal_score_batch_tf.py` and
`shared_gaussian_mixture` in
`bayesfilter/highdim/covariance_proposal_shared_gaussian_tf.py`.

GPU batch 4 or 8 is a conservative starting workload for the subsequent
adapter development. Batch 16 fits the filter but leaves less headroom.
Logical training batches of 64 or 128 would require a checked batched-shard
or gradient-accumulation implementation, or changes to the large temporary
calculations and graph construction. No such implementation was benchmarked
here. Merely generating Gaussian latent draws on CPU does not remove the
cost of evaluating the live LEDH likelihood and score during training.

## Numerical and timing limitations

CPU/GPU likelihood differences at the four shared points are at most
0.0001220703125. One score vector differs materially: at q=[0,0.05,0],
CPU gives [-79.91799, -121.78138, 59.97232], while GPU batch 4 gives
[-91.48097, -116.02063, 58.24422]. The difference is 8.274% of the CPU
score-vector norm. The other three points differ by at most 0.132% in that
relative norm. GPU batches 8 and 16 differ from GPU batch 4 by at most 0.240%.
These are comparisons of the same finite filter program; no independent
score oracle was evaluated. They do not establish which device is more
accurate or whether the discrepancy changes posterior inference.

The primary question was capacity, so this discrepancy does not invalidate
the memory measurements or require abandoning other independent batch arms.
It does require a focused precision/reference check before treating these
numbers as scientific admission of the uniform-FP32 score route.

All reported call durations use Python CLOCK_MONOTONIC. Attempt 02 recorded
5,128.2 monotonic seconds (85.5 minutes), but its UTC endpoints span 5,576.6
seconds (93.0 minutes), an 8.74% discrepancy. Subsequent ten-second checks
in both sandbox and trusted execution showed the clocks agreeing. The cause
of the earlier difference is not established; no retrospective per-call
correction is justified. Shared GPU activity and the clock discrepancy limit
fine runtime comparisons. The saved monotonic values remain descriptive;
they are not calibrated elapsed-time or statistically supported rankings.

The reusable runner now records realtime as well as monotonic duration for
future calls. This is a post-run observability amendment; the immutable source
snapshots in each arm identify the code that produced these measurements.
Existing numerical results were not changed or backfilled.

## Engineering checks and decisions

The first attempt found intentional FP64 inverse-CDF index selection in the
nominal uniform-FP32 path. Converting that calculation to FP32 changes integer
ancestors near boundaries. Instead, the optional fixed-design binding prepares
the exact same integer indices once on CPU and captures FP32 initial means
before tracing. All continuous arithmetic in the measured recurrence is FP32.
The original dynamic route remains the default; beta and uniforms cannot be
overridden on the bound callable.

Eight focused checks passed, including the N=4032 integer-boundary oracle,
small-SIR exact value/score/particle/sensitivity parity, and optimized XLA HLO
without FP64 types. Four watchdog/reporting tests passed again after the
dual-clock logging amendment. Both scripts compile. These checks
establish engineering behavior, not large-SIR score accuracy or convergence.

| Decision | Primary criterion | Veto/stop status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| GPU 4/8/16 fit this filter workload | Full T=50 calls completed | All returned calls finite and valid | Shared hardware; limited parameter region | Use a small native batch for adapter development | Complete training-step fit or trained-map quality |
| GPU 32 cannot execute this compiled allocation on this card | 21.04 GiB single-buffer request | Allocation failure | No alternative contraction/sharding tested | Reduce materialized temporaries or use checked batched shards | Rejection of NeuTra or LEDH as methods |
| CPU 8/16/32 and GPU 64 lack complete fit measurements | First call did not return within allowance | Time cap | Compilation versus full execution footprint | Repair graph scaling before repeating an expensive ladder | Their observed RSS is the final requirement |
| GPU 128 did not fit the available host-memory budget | Compilation unfinished | Available-RAM guard | Other process memory; no VRAM execution | Reduce duplicated graph and bound host use | A measured GPU OOM or impossibility on an idle larger host |
| Uniform-FP32 numerical admission remains separate | No oracle comparison in this benchmark | CPU/GPU score discrepancy is a repair trigger | Which arithmetic is closer to the intended derivative | Compare the affected point with matched references in adapter work | Score accuracy from finite/full execution alone |

| Inference status | Finding |
|---|---|
| Hard veto screen | GPU 32 allocation failure; GPU 128 host guard; time-limited arms recorded separately |
| Statistically supported ranking | None; no uncertainty-supported runtime ranking was designed or obtained |
| Descriptive-only differences | Timing, throughput, memory observations and cross-device score differences |
| Default-readiness | No numerical/training/inference default promoted; GPU 4/8 are capacity baselines |
| Next evidence needed | A complete adapter/training-step capacity check, affected-score reference check, and dual-clock timing under controlled load |

Post-run skeptical review: faster numbers could be misleading if the filter
stopped early; every reported completed call was checked for all 50 observations.
Source and fixed-workload checks found no mismatches. The weakest timing
evidence is the single warmed batch-16 call and the unexplained clock span.
The strongest alternative explanation for small throughput differences is
shared-machine load. The score discrepancy remains unexplained, and the
graph/pairwise-memory diagnoses do not prove that any proposed repair will
preserve the finite-program score. A future change needs a matched check.

All ten requested attempts are closed. Summary data are in
[summary.json](artifacts/sir-fp32-batch-capacity-20261020-02/summary.json).
The broader SIR NeuTra plan still lacks its matched target callback and
frozen-map HMC integration; those are planned integration work, not an
unfinished benchmark or a newly launched training campaign.
