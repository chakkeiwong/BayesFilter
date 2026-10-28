# Walnutpie execution optimization, 2026-10-29

The owner requests a plan, skeptical review and execution of the proposed
Walnutpie optimizations. This extends the existing standalone exception only;
public NUTS routes and inference defaults remain excluded. Baseline source is
commit 4898c7a06, pinned upstream Walnutpie 5710b033. Preserve unrelated work.

## Research intent and evidence contract

Question: can storage/control-flow changes reduce complete-fit GPU XLA time
without changing deterministic Walnutpie or its generic TensorFlow target API?
Mechanism: pack the twelve tree fields into one contiguous stack; optionally
flatten the leaf/carry scheduler into one loop, retaining the shared macro,
selection, adaptation and random-number authorities. Expected failure modes:
copy/layout overhead outweighs savings, extra conditional scheduling, inactive
chain work, changed random consumption or changed stopping decisions.

Primary engineering criterion: unchanged discrete decisions/work/RNG counters
and scaled 2e-11 FP64 agreement with existing TF and unmodified-header native
oracles, stable signatures, runtime data, independent chains and complete GPU
XLA execution. Preserve the previous distinction between default full-trajectory
checks and nondefault stress controlled-input replay. No relaxed tolerances.

Primary performance criterion: synchronized complete initialization + warmup +
sampling time versus the current fieldwise implementation with command buffers
already enabled. Interleave paired arms, rotate order, retain every timing.
Use seven independent seed blocks, each with three paired repetitions after
compilation; summarize seed-level median log speed ratios with a 95% Student-t
interval (df=6, critical 2.447, derived from the stated interval). A lower
speed-ratio bound above 1 supports only a within-workload execution improvement.
With fewer blocks or inconsistent intervals, report descriptive observations.
Exploratory multiple candidates/workloads cannot establish universal superiority.

Veto/repair: oracle mismatch, invalid retained state/cache, wrong RNG, callbacks,
unverified memory growth, missing synchronization/provenance or corrupted output
blocks candidate promotion and triggers a localized repair. A slow candidate is
rejected for performance, not evidence against Walnutpie. Convergence/coverage
failures veto posterior-efficiency claims, not engineering experiments.
Continuation veto: total budget exhaustion, corrupted baseline/source, missing
trusted hardware, changed mathematical target or unsafe shared resource use.
Explanatory only: HLO counts, gradient/macro counts, first-call compilation,
allocator memory, masked-chain work, short-run R-hat/ESS and throughput.
No posterior/default promotion, R2P reproduction, single-kernel claim, general
GPU/C++ ranking or ESS/second ranking is authorized by these short tests.

## Assumption and default audit

| Choice | Provenance/status | Justification and failure mode | Early check |
|---|---|---|---|
| FP64, depth 5, default adaptation | Existing source-checked baseline | Isolate execution; not target-optimal tuning | Native oracle |
| Command buffers FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL | Measured prior candidate | Already improves baseline; installed TF flags may differ from upstream | Record actual flags and compiled graph |
| Packed / flat storage | New optimization hypotheses | Reduce field copies/loop scheduling; may increase work | Full-fit parity plus small paired timings |
| Four chains, 100+100 initial ladder | Convenience engineering workload | Bound compilation/run cost; cannot establish convergence | Validity, work counts and same inputs |
| Gaussian10, banana10, funnel10, regression4096x128 | Prior fixtures/baselines | Cheap/curved/expensive workloads; Gaussian regression algebra is avoidable | Target value/gradient checks |
| Nonlinear logistic4096x128 | New workload hypothesis | Nonquadratic data-bearing TF target; not a posterior study | Analytic C++ versus TF gradient check |
| Batch 1/4/16; same first four keys/starts | Convenience bounded ladder | Expose occupancy and straggler costs; graph/RNG cost grows | Row isolation, compile/memory/time |
| Native -O3 vs -O3 -march=native | Reviewed build candidates | Same headers/FP64, no fast-math; compiler changes rounding | Native outputs and target parity |
| Regression sufficient statistics | Exact algebraic alternative | Precompute X'X/n, X'y/n, y'y/n equally on TF/C++; preprocessing costs matter | Value/score parity and report preprocessing |
| Seven blocks, three repeats, one GPU | Convenience bounded replication | Paired uncertainty for these workloads, shared-machine drift remains | Alternate order, record all blocks |
| 3600 worker seconds, 18 launches | Convenience campaign cap | Similar to previous engineering campaign, includes failures/builds | Master accounting/timeout |

## Execution sequence and budget

1. Preserve baseline; implement optional packed and flat execution schedules
   sharing all numerical primitives. Add exact storage/schedule, continuation,
   RNG and configuration checks. Keep fieldwise the default pending evidence.
2. Run focused CPU tests and native oracle, then trusted GPU qualification.
   If the flattened candidate fails, repair within budget or reject it honestly.
3. Run paired GPU ladder on the four fixtures plus nonlinear logistic. Screen
   candidates on small cases first; only viable candidates enter the full ladder.
4. Measure batch 1/4/16 on Gaussian and logistic; compare native generic and
   architecture-aware builds. Add and check sufficient-statistic regression
   equally in TF and native C++; report preprocessing separately. Cache compiled
   functions and retain device operands throughout timed repeats.
5. Review numerical, performance and inference evidence separately. Keep only
   justified optional implementations; document rejected candidates and exact
   commands, limits, result table and next justified action.

Environment: /home/ubuntu/anaconda3/envs/tfgpu/bin/python, TF 2.20, GPU selected
after trusted readiness check, FP64/TF32 off. Before import set
TF_FORCE_GPU_ALLOW_GROWTH=true, then verify all visible devices. CPU references
set CUDA_VISIBLE_DEVICES=-1. No pfor, NumPy runtime or environment installation.
Workers run through the new optimization master with hard remaining-budget
timeouts, exact commands, source/header hashes, Git commit, seeds, environment,
device, allocator, output paths and failure classifications in each manifest.
Unique attempt directories: docs/benchmarks/artifacts/walnutpie-opt-20261029.
Result: docs/plans/bayesfilter-walnutpie-optimization-result-2026-10-29.md.

## Skeptical pre-execution review

Reviewed before implementation/benchmark commands. Material risks resolved:

- The old GPU table already enables command buffers: disabling them for the
  comparator would create a false improvement. Both arms use identical flags.
- A smaller tree depth, less adaptation, different RNG, or skipped reverse
  checks changes the algorithm/work. None is an allowed optimization.
- Packed storage alone may be slower; retain a reference option and measure.
  Flattening changes only traversal scheduling and must preserve carry order.
- TF/native use different RNGs. Native timings are descriptive; same-input
  TF pairs and independent native shared-tape checks answer different questions.
- Native -march=native is not exhaustive BLAS tuning. Report this bounded scope.
- Gaussian sufficient statistics must apply to both arms and preprocessing
  cannot disappear from end-to-end accounting. Nonlinear logistic prevents
  treating an avoidable Gaussian workload as general GPU superiority.
- More chains change the workload. Report per-chain throughput separately from
  equal-chain latency and check marginal results for identical keys.
- Short funnel/nonlinear fits are engineering tests; no convergence or sampler
  efficiency claim follows even if numerical checks pass.
- Profiler fallback disables command buffers; unprofiled paired timing is the
  performance authority. HLO reductions explain, but cannot promote a change.

Audit passes for this bounded engineering question. Review is author skeptical
review; no independent agent review was requested. A regression or failed
candidate triggers repair/rejection within the plan, not early termination of
the other discriminating phases. No default execution-layout change planned.

External documentation pointers (web retrieval returned no usable content): OpenXLA command-buffer tutorial (upstream main,
not a promise about installed TF), https://openxla.org/xla/command_buffer_tutorial;
TF XLA slice operation registration, tensorflow/compiler/tf2xla/ops/xla_ops.cc.

## Pre-mortem

A successful run could mislead through warm-cache ordering, shared GPU load,
different stochastic work, expensive output transfers or benchmark-only target
algebra. Use paired alternating order, synchronize without materializing draws
inside timing, match TF RNG/work and report preprocessing and compilation. A
failure could come from changed inactive-lane state or carry order rather than
the method: local stack roundtrips and native shared-tape checks discriminate.

## Execution notes

The initial trusted probe found transient occupancy on GPU 0 and an ongoing
q20 worker on GPU 1. No other job was stopped. A subsequent trusted TF probe
at 2026-10-28T16:21:41Z (October 29 local) passed on GPU 0 with memory growth;
GPU 1 remained busy. The campaign uses GPU 0 and reports that difference from
the previous GPU 1 measurements. Directly paired new runs are the comparator.

CPU tests: 24 passed. The flattened scheduler passed the complete 528-check
native oracle on CPU, including runtime-data/continuation checks. Layout
selection is an execution-only Config field and defaults to the historical
fieldwise scheduler. The numerical macro, reverse, selection, RNG and adaptation
implementations are shared. GPU qualification precedes comparative timing.


### Timing-harness repair after attempt 05

The initial Gaussian screen exhibits a strong first/second-call effect:
approximately 0.11 versus 0.18 seconds regardless of layout. Its three repeats
per seed give one order a majority, producing alternating seed effects. This
invalidates promotion from that screen, not the numerical candidates. Preserve
attempt 05 as a harness diagnostic. Repair: four repeats per seed, exactly two
AB and two BA; compare full fit outputs before timing each seed, release output
objects before the next timed call, and avoid parity assertions between paired
measurements. The candidate equations, targets and 3600-second/18-launch caps
are unchanged. Four repeats are a convenience chosen to balance both orders,
not a retrospectively optimized significance threshold. The seven-block interval
is unchanged. Recheck order effects and require the same conclusion by order.

Attempt 06 Gaussian resolves the order effect: packed/reference ratios favor
packed in both AB and BA orders (~1.07), whereas flat is ~0.975. The flat
candidate remains a correctness-passing, performance-rejected schedule on this
fixture. Carry the packed candidate to the remaining targets; keep flat optional
for reproducibility, not a recommended default. Broader qualification and batch/
model-algebra comparisons remain required. No unchanged target result is a
posterior-convergence claim.

The balanced banana result agrees with Gaussian: packed ratio 1.068 with
95% seed-block interval 1.066–1.071, flat ratio 0.974. Both ordering strata
agree. Funnel packed ratio is 1.054 (1.047–1.061). These are conditional
execution-timing results on the declared fixtures, not sampler-efficiency
or posterior-validity results. The remaining regression/logistic and batch
phases test generalization rather than reinterpret the failed flat candidate.

The logistic fixture uses a mean Bernoulli log likelihood plus a standard
Gaussian prior (the same scaling convention as the regression workload).
This is an explicitly tempered synthetic workload, not an ordinary full-data
Bernoulli posterior. Its purpose is nonlinear target-computation cost; it is
not evidence of mixing on a difficult applied logistic posterior.

Batch phase completed all 1/4/16-chain checks on both workloads. Larger batches
preserve shared chain outputs and increase descriptive throughput, while 16-chain
first-call cost exceeds 110 seconds. This is a compilation-scaling limitation,
not a sampler failure. The target/native comparison now includes independent
value/score tests on nonzero regression responses, extreme logistic inputs and
both C++ builds. Memory growth remains verified; no CPU callback or pfor route
is introduced. Numerical tests and final review remain before completion.


## Completion review

All implementation, qualification, paired-layout, batch and native/algebra
phases have executed. Packed passes the engineering contract and has conditional
paired timing support on five workloads. Flat passes engineering but fails the
small-target speed criterion; it remains experimental. Native architecture-aware
builds and equal algebraic simplification were measured. No continuation veto
fired. Final review found and repaired the legacy worker layout-option reporting
mismatch; the optimization worker used the correct option throughout. The final
result note separates numerical correctness, execution performance and sampler
inference, with decision and inference-status tables. Public defaults remain
unchanged. Further sweeps are not needed to answer this bounded plan.
