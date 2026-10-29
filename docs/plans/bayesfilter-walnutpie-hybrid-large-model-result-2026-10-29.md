# Hybrid CPU/GPU Walnutpie: execution result and review

The hybrid design works, but complete GPU/XLA execution is faster on the tested
expensive target. The measured hybrid takes 13.8% longer than complete GPU
execution, with a paired 95% interval of 12.3% to 15.4% longer. Do not promote
the hybrid over full GPU for this target. Retain it as an experimental option;
the result rejects this speed hypothesis, not heterogeneous execution in general.

Plan and pre-execution skeptical audit:
`bayesfilter-walnutpie-hybrid-large-model-2026-10-29.md`.
Artifacts: `docs/benchmarks/artifacts/walnutpie-hybrid-large-20261029-01/`,
including result.json, manifest.json, REPORT.md, review.json, HLO, source
snapshots, cpu8/result.json and overhead/result.json. The manifest records
Git 0a1bcd26b6d752c13849435ed78d664745687a8f plus hashes for dirty/new numerical
sources; a commit identifier alone does not describe this working tree.

## Target and compiled implementation

The synthetic Bayesian nonlinear regression has 16,384 observations, 128 latent
bias coordinates, two tanh stages of width 128, fixed generated weights and
four chains. Each gradient recomputes the parameter-dependent activations, a
large dense hidden-layer product, and its derivative. It is a compute-heavy
model with measured GPU benefit, not a claim that a GPU is mathematically
necessary or that all neural-network weights are sampled. The target is
`logp(q)=-||q||^2/2-||tanh(tanh(X+q/sqrt(N)) W) a-y||^2/2`.
All arms use the same generated data, FP64, TF32 off, fixed identity mass and
frozen epsilon=0.2. This coordinate scaling is part of the synthetic target,
not a trained real-model posterior.

The hybrid's CPU/XLA controller handles tree traversal, selection, merging and
stopping. Its GPU/XLA leaf reuses the original R2P integration, local refinement,
reverse checks, model and autodiff. Model data remain on GPU; the tree stack
remains on CPU. Python only schedules compiled calls, copies tensors, checks
the completion flag and assembles output tensors. Numerical model/integrator/
tree updates all compile; the scheduler and final output assembly are outside
XLA. There is no single fused heterogeneous executable. No NumPy numerical
kernel, pfor, or host sampler callback is introduced.

## Primary paired timings

Seven seed blocks, two repeats per block, rotated backend order. Each timed
block contains two transitions per chain across four chains. Compilation is
excluded from steady timing and recorded separately. Every output dependency
is materialized before stopping the timer.

| Execution | Median seconds per block | Evidence class |
|---|---:|---|
| CPU/XLA, four threads | 18.154 | Paired main experiment |
| GPU/XLA, entire sampler and model | 1.826 | Paired main experiment |
| Hybrid CPU tree / GPU integration | 2.063 | Paired main experiment |
| CPU/XLA, eight threads | 10.633 | Later separate-process sensitivity; descriptive |

The paired geometric full-GPU/CPU4 runtime ratio is 0.09948, 95% interval
[0.09891,0.10006]. The hybrid/full-GPU ratio is 1.13812, interval
[1.12254,1.15392]. These are Student-t intervals on seven paired seed-block
log ratios, conditional on this workload/session. CPU8 was not interleaved
with the GPU trials and has no separate ranking claim.

The cost pilot measured median value/gradient time 248.8 ms CPU4 and 21.0 ms
GPU. Fresh seven-seed evaluator measurements give GPU/CPU ratio 0.1121,
interval [0.0829,0.1517], confirming that the selected target benefits from
GPU computation. The maximum device allocator peak was 285,796,864 bytes
(272.56 MiB); fixed model input tensors occupy 17,040,384 bytes.

## Transfer and dispatch interpretation

Each two-transition hybrid block made 30 batched leaf round trips. The
implemented full-span/metric/RNG payload is 70,592 bytes per round trip, or
2,117,760 bytes per block. It does not include the resident model arrays.

A separate seven-block diagnostic measured 11.70 ms median for thirty raw
buffer round trips (returning a slightly larger 74,720-byte total payload per
round trip), and 133.55 ms for thirty compiled echo round trips. The measured
hybrid/full-GPU median-block difference is about 237 ms. This suggests launch,
Python dispatch and synchronization are important costs, while bytes copied
alone are small for this case. These diagnostics are not an additive causal
decomposition; subtracting them cannot identify the exact tree/dispatch cost.
No native C++ dispatcher or reduced endpoint payload was benchmarked.

## Engineering checks and limits

- Two same-CPU replay tests passed in 18.13 seconds: deterministic/R2P, three
  seeds, nonidentity mass and unequal chain steps. They compare all returned
  fields and RNG, with floating tolerance 2e-11.
- The tiny model passed independent direct-chain-rule gradient and row-isolation
  checks; the large CPU/GPU value/gradient comparison passed.
- All fourteen primary blocks have identical discrete evaluations, macro
  counts, RNG and other discrete outputs across backends. Finite state/cache
  checks passed. Largest floating-output difference was 1.82e-12 against the
  predeclared 2e-9 absolute/relative backend tolerance.
- All nine inspected compiled functions have one trace. GPU integration HLO
  contains the model gradient and cuBLAS operations; no host sampler callback
  was present. Physical GPU2 is an RTX4080 SUPER; memory growth was verified
  before initialization. Its earlier unrelated Xid13 was a recorded risk; this
  experiment completed without a CUDA error. No hardware-reset claim follows.
- The run reused current core/R2P arithmetic. Review confirmed its numerical
  source hashes match the launch snapshot after measurement. No running
  replication files or processes were changed.

Actual commands use the tfgpu interpreter. Main:
`python -m docs.benchmarks.walnutpie_gpu_tf.hybrid_benchmark --output ROOT --gpu 2 --seconds 2700`
under an external 2700-second timeout; follow-ups:
`python -m docs.benchmarks.walnutpie_gpu_tf.hybrid_followup --root ROOT --mode cpu8`
(CPU hidden, 600-second timeout) and `--mode overhead --gpu 2` (180-second
timeout). Main wall time 394.08 seconds, CPU8 90.76 seconds, overhead 1.43
seconds: 486.27 seconds aggregate recorded benchmark wall time, within the
2700-second campaign budget. TF startup before the overhead timer is small
and does not threaten the remaining budget. No benchmark repair/retry needed.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep full GPU for this model | Hybrid/GPU interval entirely above one; speed-promotion criterion fails | No engineering veto | One target, short fixed kernel, hardware sharing | Retain full-device baseline | Universal GPU preference |
| Keep hybrid experimental | Numerical and discrete replay passed | No hidden callback/retracing/device failure | Host dispatcher may be optimized | Only optimize if another target or integration need justifies it | Optimal hybrid design or production readiness |
| Do not infer posterior quality | No retained posterior study performed | Posterior validity not assessed | Two-transition timing blocks | Separate scientific validation if method use changes | Convergence, ESS or posterior accuracy |

## Inference status

| Item | Finding |
|---|---|
| Hard veto screen | All stated engineering checks pass; posterior checks not part of this experiment |
| Statistically supported ranking | Full GPU faster than this hybrid and CPU4 under paired runtime design, conditional on target/session |
| Descriptive-only differences | CPU8 sensitivity, microtimings, memory, compilation and work counts |
| Default readiness | No default change; hybrid remains optional experimental code |
| Next evidence needed | Independent target, workload scale and hardware/session replication; native-dispatch comparison if pursuing optimization |

## Terminal skeptical review and reset

The strongest alternative explanation is the chosen compute-heavy synthetic
target: once model gradients dominate, GPU tree overhead is a small component
and splitting can add more cost than it removes. This is exactly what the
experiment tests; it does not invalidate the harness, target or implementation.
CPU sharing, thread count, launch scheduling and short trajectories limit the
ranking. CPU8 narrows the CPU gap without changing the descriptive ordering.
A target with substantially more tree work per expensive integration block,
larger chain batches, or a native dispatcher could overturn the hybrid result.
The weakest evidence is generalization beyond this one fixed-kernel synthetic
case. More seeds from the same case would not close that gap.

The authorized feasibility test is complete. No later repair phase is needed
to explain an engineering defect: the candidate is valid on the checked cases
but misses the speed-promotion criterion. Do not launch another ladder or
change the ongoing million-draw replication based on these timings.
