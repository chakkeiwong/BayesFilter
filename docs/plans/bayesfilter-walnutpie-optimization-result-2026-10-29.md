# Walnutpie execution optimization results

The reviewed optimization plan is executed. Packed storage is a qualified optional
GPU/XLA layout; flattening passed correctness checks but failed the performance
criterion. All three benchmark phases completed within the campaign budget.
Plan: bayesfilter-walnutpie-optimization-2026-10-29.md. This standalone study
leaves public inference defaults and the repository NUTS exclusion unchanged.

## Implementation

Config.execution_layout selects fieldwise (unchanged default), packed or flat.
Packed stores the twelve span fields in one contiguous tensor of shape
[depth+1, chains, 8*dimension+4]. This reduces twelve stack updates to one while
preserving field values and the binary-carry order. Flat adds a single leaf/
carry scheduler; it shares every proposal/refinement, reverse check, selection,
random-number and adaptation primitive with the baseline.

The flattened candidate passed the 528 native oracle comparisons on CPU and
GPU plus full-fit composition, runtime-data, continuation, cache and independent
row checks. It failed the performance criterion: Gaussian and banana baseline/
flat ratios were 0.975 and 0.974, respectively. Fewer loops did not imply faster
execution. Keep it as an explicitly experimental reproducibility arm.

Packed is the candidate supported by the measured execution evidence. It
reduces optimized-HLO fusion-operation counts by 26 on the five measured targets;
the nested loop count remains 18. Flat reduces that count to 16 but introduces
conditional scheduling. HLO counts explain structure and are not performance
criteria. Actual kernels and scheduling depend on XLA.

## Paired GPU timings

TF 2.20, RTX 4080 SUPER GPU 0, FP64, TF32 disabled, verified allocation growth.
Command buffers enabled equally for every arm. Four chains, 100 warmup and 100
retained transitions; compilation excluded. The reported ratios are geometric
means of seed-block median paired log ratios, not ratios of pooled median times.
Seven seeds, four repetitions per seed, exactly balanced AB/BA within each seed.

| Target | Fieldwise median s | Packed median s | Paired baseline / packed ratio | Approx. 95% t interval |
|---|---:|---:|---:|---:|
| Gaussian10 | 0.18283 | 0.17216 | 1.0730 | 1.0704–1.0757 |
| Banana10 | 0.20934 | 0.19744 | 1.0681 | 1.0656–1.0706 |
| Centered funnel10 | 0.36861 | 0.35926 | 1.0536 | 1.0467–1.0606 |
| Regression4096x128 | 0.84282 | 0.81803 | 1.0302 | 1.0295–1.0309 |
| Logistic4096x128 | 0.74596 | 0.69624 | 1.0322 | 1.0296–1.0349 |

All pre-timing seed comparisons check the complete fit, including discrete work
and RNG state, against fieldwise with unchanged scaled 2e-11 floating tolerance.
Both ordering strata favor packed. These intervals support execution improvement
on the declared workloads and this run environment. They do not establish a
universal speed ordering, independence from machine load, posterior convergence,
or an ESS/second ranking. Multiple exploratory targets/candidates and one shared
machine limit generalization. Logistic uses mean Bernoulli log likelihood plus
a unit Gaussian prior; it is a tempered synthetic cost fixture, not a standard
full-data posterior study.

## Timing-harness finding

Attempt 05 retained both paired output structures and used three alternating
repeats, giving one order a majority per seed. The second timed call was about
0.18 s versus about 0.11 s for the first regardless of layout. Preserve this
attempt as a harness diagnostic, excluded from performance promotion.
Attempt 06 compares outputs before timings, releases timed output objects, and
balances each order within each seed. The corrected order-stratified ratios
agree. This supports the repair; it does not prove the precise allocator/
command-buffer mechanism causing the original effect.

The packed candidate also passed its independent 528-check native oracle on
GPU (attempt 08), including the same nine public-API checks. The tests preserve
the prior stress interpretation: default trajectories are compared in full;
nondefault extreme funnel adaptation uses controlled native-input replay at the
unchanged tolerance. Neither this result nor paired benchmark agreement upgrades
that stress test into unrestricted cross-backend pathwise reproducibility.

## Equal target-algebra comparison

For the regression workload the target is

    log p(q) = -q'q/2 - (Xq-y)'(Xq-y)/(2n).

With G=X'X/n, b=X'y/n and c=y'y/n, expansion gives

    log p(q) = -q'q/2 - q'Gq/2 + q'b - c/2,
    gradient = -q - Gq + b.

This is the same declared density, including its additive constant. Both TF and
C++ sufficient-statistic arms use these quantities; construction time is recorded
separately from repeated fits. The nonlinear logistic fixture cannot use this
quadratic reduction. Independent checks include nonzero responses and extreme
logits, so zero-response benchmark data cannot hide a missing linear term.

## Chain batching

The Gaussian packed implementation measured 1,651, 4,886 and 9,408 chain
transitions/second at 1, 4 and 16 chains, respectively. All shared chain outputs
agree at the stated tolerance. Latencies increase from 0.121 to 0.164 to 0.340 s;
this is throughput scaling, not lower latency for a fixed number of chains.
First-call costs (tracing/compilation plus one fit) increase from 6.81 to 17.56 to 114.14 s, and optimized HLO size
from 1.76 to 4.26 to 14.67 MiB. The existing per-chain Philox construction grows
the traced graph; larger batches require compile-cost and memory qualification.
Seven fixed-seed repeats are descriptive only for this batching comparison.


| Target | Chains | Fit seconds | Chain transitions/second | First call seconds |
|---|---:|---:|---:|---:|
| Gaussian10 | 1 | 0.1212 | 1651 | 6.81 |
| Gaussian10 | 4 | 0.1637 | 4886 | 17.56 |
| Gaussian10 | 16 | 0.3401 | 9408 | 114.14 |
| Logistic4096x128 | 1 | 0.4124 | 485 | 8.64 |
| Logistic4096x128 | 4 | 0.9625 | 831 | 20.00 |
| Logistic4096x128 | 16 | 1.8352 | 1744 | 120.55 |

All batch-prefix checks passed. Relative to one chain, 16-chain throughput is
5.70x Gaussian and 3.60x logistic in this fixed-seed diagnostic. No uncertainty
ranking is claimed for batch sizes. Peak GPU allocator usage was 16,891,392 bytes
(16.11 MiB); peak host RSS was 4,069,024 KiB (~3.88 GiB). Device allocator bytes
do not measure host compiler memory or all driver allocations. The phase took
604.41 seconds, including compilation and HLO extraction for six distinct graphs.

## Native compiler and algebra measurements

Both C++ builds use GCC 11.4, -O3 -std=c++20 -pthread and the same unchanged
upstream headers. The second adds -march=native, without fast-math. Seven seed
blocks, three cyclically rotated repetitions per block; each native invocation
reports serial and four-thread complete fits. Process startup, fixture parsing,
output serialization and compilation are excluded from the kernel timing.
Different TF/native RNG algorithms prevent a sampler-efficiency ranking.

The regression generic/architecture-aware speed ratio is 1.406 (95% interval
1.396–1.417). Work counts agree between native builds for every measured seed;
retained paths differ by at most 4.44e-9 over the fit. This is consistent with accumulated
roundoff; it is not asserted bitwise or full-path 2e-11 equality. Independent local
value/score checks pass at 2e-11. Native fit/reference means and posterior
convergence have not been established by these short runs.

The sufficient-statistic arm is much cheaper on C++: medians 0.02879 s generic
and 0.01952 s architecture-aware, versus 0.70561 and 0.50423 s for repeated full
matrix evaluation. Add median preprocessing of 0.01085 / 0.00527 s to a fresh
sufficient-statistic fit. TF/GPU medians are 0.53280 s full matrix and 0.40908 s
sufficient statistics, with 0.00084 s steady preprocessing (0.32898 s first
preprocessing call including compilation). The algebra comparison is descriptive;
its two target representations run in separate batches and can differ by
floating-point accumulation. This result demonstrates why the earlier full-data
Gaussian workload cannot support a general GPU superiority claim.

| Workload | GPU packed s | C++ generic, 4 threads s | C++ -march=native, 4 threads s |
|---|---:|---:|---:|
| Full regression4096x128 | 0.53280 | 0.70561 | 0.50423 |
| Sufficient-statistic regression | 0.40908 | 0.02879 | 0.01952 |
| Logistic4096x128 | 0.50235 | 1.01176 | 0.82944 |

These medians come from the separate cyclic-order TF/native comparison; do not
combine them with the layout table's timings to infer another speedup. The
logistic architecture-aware C++ ratio is 1.220 (1.214–1.226) relative to generic;
native work counts agree across builds, with maximum accumulated draw difference
2.02e-9. GPU time is descriptively lower than either native build for this
nonlinear synthetic workload. Different RNGs and work make this a workload
measurement, not an ESS/second or posterior-efficiency result. The comparison
uses Eigen and one thread per chain, not an exhaustive BLAS/threading study.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain optional packed layout | Exact discrete/shared-seed checks; paired intervals favor it on five targets | Native GPU oracle and cache/API checks pass | One machine, finite targets, timing-consumer pattern | Use packed with model-specific qualification | Universal/default readiness |
| Reject flat as performance recommendation | Slower on Gaussian/banana despite fewer loops | Numerical checks pass | Other workloads unmeasured for flat | Retain experimental arm; do not extend performance claims | Failure of Walnutpie |
| Keep chain batching available | Prefix parity and descriptive throughput scale | All measured retained states finite/valid | Fixed seeds; graph and compile scaling | Choose batch size including compilation and latency | 16 is universally optimal |
| Prefer architecture-aware native build for these fixtures | Paired runtime intervals favor it; equal work | Independent scores pass; full paths differ by accumulated roundoff | CPU architecture and libraries | Revalidate build on deployment CPU | Portable binary/default or bitwise equivalence |
| Apply exact algebra where available | Independent value/gradient checks and lower observed work cost | Nonzero responses/constant term checked | Separate timing batches, conditioning at larger scales | Precompute regression statistics equally | Generic nonlinear reduction or general GPU victory |

| Inference category | Status |
|---|---|
| Hard veto screen | All final implementation checks pass; original unbalanced timing screen is excluded; flat fails performance criterion |
| Statistically supported ranking | Conditional paired execution improvement for packed versus fieldwise on five declared workloads, and architecture-aware versus generic native builds; no sampler-efficiency ranking |
| Descriptive-only differences | Batch throughput, cross-engine medians, preprocessing/full-matrix comparison, HLO counts, compilation and memory |
| Default readiness | Original fieldwise and public inference defaults unchanged; packed is an optional qualified standalone layout |
| Next evidence needed | Application targets, independent timing sessions/hardware, batch-aware RNG graph scaling, and convergence-qualified posterior runs before ESS/second claims |

The failed original timing harness invalidated those timing conclusions only.
The flattened candidate failed the declared performance criterion, not numerical
correctness. Neither result invalidates the target, upstream Walnutpie, or the
GPU research direction. The planned packed repair and broader comparisons were
completed. No scientific continuation veto fired.

## Terminal skeptical review

Source review confirms packed indices correspond to the existing eight vector
and four scalar fields. The flat traversal performs one shared macro per leaf,
then the same ordered binary carries; no tree depth, macro tolerance, reverse
check, adaptation cadence or random stream changes. The historical fieldwise
branch retains its arithmetic and remains the default. Native headers are
unmodified. Final tests cover both packed/flat full fits, stack roundtrips at
one and four dimensions, exact discrete fields and nonzero-response algebra.

The strongest alternative explanation for broad speed claims is workload and
execution context: simplifying the Gaussian target reverses the apparent GPU
advantage, and absolute timings differ between layouts/native timing campaigns.
The original order artifact also shows that retained-output lifetimes matter.
The paired within-worker contrasts and matching outputs are the strongest
execution evidence. Replication with a different consumer pattern or hardware
could overturn their magnitude or direction. No fresh profiler trace was used
because the installed profiler disables command buffers; HLO structural changes
are explanatory only, and reduced physical memory traffic is not directly
established. Mixed precision and a persistent CUDA kernel were not implemented.

One CLI consistency defect was caught during final review: worker --mode benchmark
accepted --execution-layout but constructed the default layout. It now passes
the option through, matching its reported field. The new optimization harness
already passed the option correctly, so all measured results are unaffected.


## Reproducibility and completion

Exact commands, source/header snapshots, Git commit, binary hashes, seeds,
TF/CUDA device settings, verified memory-growth flags, compiler/CPU metadata,
wall times and failure records are preserved per attempt under
docs/benchmarks/artifacts/walnutpie-opt-20261029. Compact summaries are in
docs/benchmarks/walnutpie_gpu_tf/optimization_evidence.json. The numerical core
hash is identical across all final qualification and timing artifacts.

The unbalanced attempt 05 is retained but excluded from timing promotion;
all later timing phases use the documented repair. Final review is a skeptical
author review; no independent agent review was requested. Unrelated worktree
changes are preserved. No new environment installation, public sampler-default
change, persistent CUDA kernel, mixed-precision route or claim-bearing posterior
campaign is part of this execution.

The next optimization suggested by these measurements is a bounded batched RNG
implementation that preserves independent streams while avoiding graph growth
linear in chain count. That is future work with its own RNG/parity contract;
we have not changed Philox consumption or substituted a different random stream
in this campaign. For applications now, use optional packed storage, reuse
compiled stable signatures, keep data resident, simplify target algebra when
mathematically exact, and choose native CPU or GPU from the actual workload.

Final validation: 28 tests passed (two existing TFP deprecation warnings),
528 native-oracle checks for flat on each CPU/GPU and for packed on GPU,
plus 18 independent native target value/score cases across both builds.
Total worker time: 2354.62/3600 seconds across
13/18 launches, including the superseded timing screen and final
CLI regression. No worker remains running.

## Prebuilt packaging review (2026-10-29)

The owner requested a local commit and reuse of the small C++ build. Package
the already-qualified generic Linux x86-64 oracle, with its exact source/header
hashes, dependency requirements, source headers and notices. This is a packaging
change, not a new sampler experiment. Check byte identity, relocation to a
temporary directory, all deterministic oracle records and a closed-form target
evaluation. Stop packaging on a source/hash mismatch, missing dependency or
failed check. No GPU work or new performance claim is needed.

Skeptical review: the generic executable has no RPATH/RUNPATH and needs only
system C/C++ libraries; its ELF declares x86-64-baseline. The measured compiler
requirements include GLIBC_2.34 and GLIBCXX_3.4.29. These restrict compatibility;
the local smoke cannot certify other operating systems or library versions.
The -march=native build is machine-specific and remains a local comparison
artifact. The older TensorFlow custom-op bridge embeds absolute environment
paths and has a different ABI and scope, so it is not the reusable binary
selected here. Saving this CPU reference executable does not eliminate XLA
compilation of a user's TensorFlow model. No public inference route changes.

Packaging completed under `docs/benchmarks/walnutpie_gpu_tf/prebuilt/`: the
executable is 249,008 bytes (243.2 KiB) and the source archive is 1,047,371
bytes (about 1 MiB). SHA-256 matches the qualified generic executable. With
library-path overrides removed, a relocated copy emitted the same 528 records;
a closed-form Gaussian value/gradient check passed. All 361 archived source
members matched their hashes. A clean rebuild from only the archive passed
the same checks and reproduced the oracle output byte-for-byte. Packaging
checks took 7.69 seconds, including a 7.35-second source rebuild. The manifest
and `packaging_validation.json` preserve the commands and evidence. These are
installation/reference checks, not new performance or posterior evidence.

The final optimization-test snapshot hashes still match every edited numerical
and harness source plus the test file (28 tests passed). Only documentation and
packaging were added afterward, so no duplicate GPU campaign was launched.
