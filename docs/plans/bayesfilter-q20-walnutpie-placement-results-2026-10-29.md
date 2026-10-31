# q20 WalnutPie execution placement: measured results

Keep the frozen NeuTra transform and the posterior value/score on GPU. For this
four-chain q20 workload, full GPU/XLA is 11.32 times faster than CPU/XLA in the
paired execution comparison (approximate 95% interval 11.05--11.60). CPU graph
tree control with a GPU/XLA macro integrator is essentially tied with full
GPU/XLA. No hybrid demonstrates a speed advantage. Full GPU/XLA is the simplest
supported choice for the next q20 sampling test; this is a scoped execution
recommendation, not a new public inference default or posterior result.

## What was compared

Every arm uses the same source-qualified deterministic WalnutPie core, packed
tree storage, four independent chains, FP64, identity mass and the unchanged
step-3584 two-chart NeuTra target. The posterior is the same four-parameter
q20/T30 rectangular QR SR-UKF approximate likelihood used by the stalled chain.
The inherited epsilon is 0.6835451778053224; maximum tree depth and refinement
count are both five. Warmup adaptation is disabled for timing so every arm
performs the same task. Local reversible step refinement remains active.

There are four recorded seed blocks, alternating ordinary dispersed starts and
the actual stalled state, with four balanced forward/reverse execution orders
per block. Each invocation performs two transitions in each of four chains.
There are 80 synchronized timed invocations, excluding compilation. Input
transfer, CPU/GPU boundary transfers and reading returned tensors are included.
The separate parity calls compare full outputs before repetitions; timed output
objects are released between calls to avoid the prior output-lifetime artifact.

| Placement | Median seconds per two transitions, four chains | Paired time / full GPU time | Approximate 95% interval |
|---|---:|---:|---:|
| Full CPU/XLA, including NeuTra and posterior | 57.529 | 11.3178 | 11.0458--11.5965 |
| Full GPU/XLA | 5.047 | 1 | Reference |
| CPU TensorFlow graph tree; GPU/XLA value/score per evaluation | 5.175 | 1.0242 | 1.0105--1.0381 |
| CPU TensorFlow graph tree; complete GPU/XLA macro | 5.074 | 1.0057 | 0.9997--1.0118 |
| Python/eager TensorFlow CPU tree; complete GPU/XLA macro | 5.177 | 1.0273 | 0.9961--1.0596 |

The ratios are geometric means of the four blocks' median paired log ratios,
not ratios of the table medians. Intervals use a Student-t approximation with
three degrees of freedom across blocks. They are local execution evidence on
one shared machine and this finite set of starts/seeds. The macro hybrid and
Python/eager hybrid cannot be ranked against full GPU/XLA at this uncertainty
level. The per-evaluation graph hybrid has a small measured overhead; CPU-only
execution is decisively slower under these settings. These exploratory multiple
contrasts are not simultaneous familywise intervals.

Block medians expose workload variation:

| Seed/start block | CPU/XLA | GPU/XLA | CPU graph + GPU target | CPU graph + GPU macro | Python/eager + GPU macro |
|---|---:|---:|---:|---:|---:|
| 0, ordinary | 57.309 | 4.953 | 5.124 | 5.001 | 5.161 |
| 1, stalled | 57.694 | 5.138 | 5.219 | 5.145 | 5.190 |
| 2, ordinary | 42.670 | 3.808 | 3.922 | 3.841 | 3.991 |
| 3, stalled | 85.364 | 7.599 | 7.725 | 7.619 | 7.675 |

Forward/reverse-order median CPU/GPU ratios are 11.237/11.247. Macro-hybrid/GPU
ratios are 1.0036/1.0076. The order check does not show the large order artifact
found in the earlier standalone layout study.

## Why the result differs from the funnel benchmark

A complete four-row q20 target value/score call measured 2.177 s on CPU and
0.2044 s on GPU, with input transfer and synchronized outputs. The GPU is about
10.65 times faster for this expensive target. The earlier funnel was a cheap
analytic target and used the R2P variant. Its CPU advantage is real evidence for
that workload, not for q20. This placement benchmark holds the deterministic
variant fixed and does not compare R2P to deterministic WalnutPie.

The cost can be described as target/integrator work plus tree bookkeeping plus
boundary and scheduling costs. Moving the cheap tree control does not remove
the expensive repeated filter and transport evaluations. The measurements are
consistent with target work dominating; they do not independently identify an
exact fraction of runtime attributable to PCIe, launch scheduling or arithmetic.

A hybrid is feasible: CPU tree/control can call a GPU-compiled complete macro,
which includes forward refinement, coarser reverse checks, leapfrog arithmetic,
NeuTra and posterior evaluation. It returns a complete candidate and diagnostics
to CPU. This reduces boundary crossings compared with returning a value/score
after each integration evaluation, but did not produce a measurable speedup
over keeping everything on GPU here.

The hybrid outer control is a TensorFlow CPU graph, not a fully XLA CPU sampler.
Its random primitive is CPU/XLA; its target or macro is GPU/XLA. The Python
variant invokes the shared TensorFlow control eagerly on CPU. It is not a new
NumPy implementation or an optimized native C++ controller. A resumable CPU/XLA
tree state machine calling GPU segments, native C++ plus GPU, alternative CPU
thread counts, R2P, different chain counts and other hardware were not measured.
Thus the recommendation is the fastest supported choice among the tested
placements, not a universal optimum.

## Correctness and harness findings

All 729 frozen q20 closure files matched their recorded checksums. The five
placements use the same frozen target signature and sampler configuration.
All qualified states and all 80 timed calls were finite/valid; identity mass,
tree work, refinement/reverse flags and RNG counters agreed. The largest
cross-placement draw discrepancy over the four checked blocks was 1.8151e-11,
well below the declared 1e-8 absolute plus 1e-10 relative diagnostic tolerance.
The independent small Gaussian placement regression agreed within 2.23e-16.
Final driver/runtime snapshots still match the recorded hashes.

benchmark-01 was excluded from sampler timing conclusions. Its apparent CPU
sampler dispatched to GPU because internal device annotations alone did not
select the must-compile call's device. The first hybrid also used non-XLA
TensorFlow random lowering, so equal seeds did not produce equal work. Repairs
added explicit call-site placement and a shared compiled Philox primitive for
hybrid random calls. A focused five-arm regression passed before benchmark-02.
A suffix typo in that regression's device assertion was corrected; it was a
test assertion failure, not a numerical mismatch. All failed artifacts remain.

Separate target microbenchmarks from benchmark-01 had correctly checked output
placement, but the table above uses only the completed corrected benchmark-02.
This audit shows why device names in a configuration and nominally equal seeds
are insufficient evidence for a fair heterogeneous execution comparison.

First sampler calls including tracing/compilation were 90.17 s CPU/XLA,
44.19 s GPU/XLA, 31.05 s graph/GPU-target, 35.12 s graph/GPU-macro and 28.67 s
Python/GPU-macro. These are observed staged warm-process costs, not comparable
cold starts: target graphs and some hybrid primitives had already been compiled,
and the Python arm reused the macro graph. They must not support a startup
ranking. Separate target first calls were 20.62 s CPU and 18.79 s GPU.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use full GPU/XLA for next bounded q20 test | Lowest observed median; strong paired advantage over CPU; no hybrid advantage | Final placement, source and matched-work checks pass | One machine, frozen short workloads, warmup costs unmeasured | Reprice and assess actual adaptive warmup and longer retained sampling on GPU | Posterior correctness or globally fastest hardware/software design |
| Retain macro hybrid as a diagnostic option | Correct matched trajectories; no supported difference from full GPU | No final numerical veto | CPU graph overhead and different future workloads | Use if easier host control is valuable, with separate lifecycle qualification | Faster execution or a replacement public default |
| Reject CPU-only placement for this four-thread q20 comparison | Approximately11.32 times slower | Numerically valid | Other CPU thread counts and batching untested | Keep CPU as independent reference | CPU inferiority on cheap targets or all hardware |

| Inference status | Finding |
|---|---|
| Hard veto screen | First timing harness excluded; final placement/parity checks pass |
| Statistically supported ranking | Local paired CPU/GPU advantage; small GPU advantage over per-evaluation graph hybrid |
| Descriptive-only differences | Medians, staged first-call times, small macro/Python hybrid differences |
| Default readiness | Scoped q20 execution recommendation only; no public sampler/default change |
| Next evidence | Adaptive warmup, fresh retained chains, modern R-hat/ESS, mode weights, precision and reference checks |

The stronger counterargument is that a tuned kernel, longer trajectory or other
parallelism changes the relative cost. Longer sampling may also revisit much
more expensive regions: the earlier continuing escape chain required 203.11 s for
16 transitions, versus these reset two-transition timing panels. Do not multiply
the 5.047 s median by a desired draw count and treat it as a campaign completion
guarantee. The short panels cannot establish mixing or support ESS/second.

## Artifacts, resources and continuation

Primary evidence: docs/plans/artifacts/q20-walnutpie-neutra-20261029/benchmark-02.
Read result.json, paired-report.json, target-latency.json, parity-block-*.json,
qualification-*.json, manifest.json and accounting.json. The five placement
adapters are in docs/benchmarks/q20_walnutpie_placement_kernels.py; driver and
post-run report are q20_walnutpie_placement.py and
q20_walnutpie_placement_report.py. All numerical source copies are preserved.

Actual completed command from the repository root:

    timeout --signal=TERM --kill-after=30 s 3630 s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/q20_walnutpie_placement.py --output docs/plans/artifacts/q20-walnutpie-neutra-20261029/benchmark-02 --wall-seconds 3600 --transitions 2

Environment: TensorFlow 2.20/Python 3.13 in tfgpu, GPU 1 RTX 4080 SUPER UUID
GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c, four CPU threads with affinity 0--3,
FP64/TF32-off and verified incremental GPU allocation. Other jobs occupied
GPUs 0/2; the measured GPU 1 was idle at launch. This is shared-machine evidence.
The combined process peak host RSS was 5,361,224 KiB and TensorFlow GPU allocator
peak 36,154,368 bytes. These are not independent per-arm memory requirements.

benchmark-02 finished in 1,996.69 s (33.28 minutes), using 3,444.20 CPU seconds.
Including the stopped escape preflight, superseded timing attempt, regressions
and fixed-metric tests, conservative campaign charges are 0.90384 GPU-device
hours and 1.61352 CPU-core hours, below the 4/16-hour allocation. An early failed
smoke lacked measured resource data, so its full 180-second/four-thread cap was
charged explicitly; the CPU test charge similarly uses four times its 123.37 s
measured wall time. No new budget grant was used. Existing ledger balances are
63.8343 GPU-device hours and 381.8185 CPU-core hours after these charges.

No benchmark worker remains running. The original local escape preflight saved
32 draws per chain, with all four moving from the stalled state, before the
owner-directed pivot to this benchmark. Its requested graceful stop completed
after the in-flight compiled call; no forced kill was applied. That escape is
encouraging mechanism evidence only. The64-draw preflight and warmup preview
were not completed, and the longer posterior pilot has not been launched.
It still needs target-specific pricing and the declared warmup/retained
diagnostics. Neither this benchmark nor the partial escape establishes that
WalnutPie already solves the q20 posterior-estimation problem.
