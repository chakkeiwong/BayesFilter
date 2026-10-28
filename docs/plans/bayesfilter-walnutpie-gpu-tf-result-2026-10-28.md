# Walnutpie TensorFlow/GPU implementation and results

The standalone deterministic Walnutpie port now runs initialization, adaptive
warmup, frozen sampling, caller-provided TensorFlow target evaluation and
position autodiff inside GPU XLA. It does not invoke the old CPU sampler bridge.
The implementation is under docs/benchmarks/walnutpie_gpu_tf, with a documented
API, independent native oracle, tests, benchmark worker and bounded master.

The performance expectation is only partly supported. Small targets remain
much faster in native C++. The larger regression fixture is descriptively faster
in GPU XLA. Neither a general speedup nor better statistical efficiency is
established by this campaign.

## Implementation and source fidelity

Authority: Walnutpie 5710b0334ba69cf0caa35fc02a2338b61d756953, local unmodified
MIT-licensed headers. The new TF numerical core owns Walnutpie's displacement
U-turns, deterministic refinement, coarser reverse rejection, Barker internal
selection, Metropolis outer selection, per-base-macro Adam, discounted
score/draw metric adaptation and minimum microstep adaptation. It does not
substitute Stan's tree or warmup. Binary-carry storage replaces recursion.

The public standalone API exposes initialize, warmup, sample, fit and
value_gradient, each with an explicit stable input signature and mandatory XLA.
Model data remain runtime tensor inputs. Sampling continuation freezes tuning;
warmup continuation preserves all adaptation state. Input positions are supplied
by the caller. External wall-clock/multichain stopping, automatic random-start
selection, arbitrary Stan-language compilation and the paper's R2P variant are
outside this API. The initial step search has a documented 64-update safety
bound; exhaustion makes validity false. There is no single-CUDA-kernel claim.

Final checks passed:

- 15 focused pytest cases, including configuration, no runtime NumPy/pfor/host
  callback, zero warmup, immutable sampling tuning, corrected diagnostic axes,
  and recovery from an initial negative-infinite proposal energy.
- 528 native-header oracle checks per CPU/GPU backend: 96 transitions, 96 macros,
  96 initializations and 240 lifecycle steps. Cases cover Gaussian, banana,
  funnel and regression gradients. Macro coverage includes 85 acceptances,
  11 rejections, 51 refinement cases and 9 reverse-rejection cases per backend.
- Default lifecycle trajectories are compared without state resets. Nondefault
  target-macro=1 lifecycle checks use controlled native position/step/metric
  replay after each checked transition, with TF adaptation accumulators retained.
  Their final minimum microstep counts reach 3, 4 and 9 across the three targets.
- Exact branch/work/RNG-consumption comparisons and scaled 2e-11 floating checks;
  complete fit composition, runtime data sensitivity, independent-row behavior,
  retained cache agreement, warmup/sample continuation, invalid initialization,
  duplicate stream rejection and explicit rejection of Python callback models.
- Optimized HLO for the full fit has model gradients in loops, one trace and no
  sampler callback custom call. GPU profiling observes device kernels. GPU
  allocation growth is configured and verified before initialization.

Attempts 08 and 09 failed strict accumulated stress-trajectory comparisons:
4.48e-10 absolute/4.68e-11 relative and 6.82e-9 absolute/2.74e-11 relative,
respectively. Position-only replay did not isolate adapted-input perturbations.
Complete exposed-input replay passed without relaxing tolerance. These failures
remain preserved and are not relabeled as successful accumulated-trajectory
agreement. They limit claims about cross-backend trajectory reproducibility in
extreme nondefault funnel states; they did not reveal a changed tree decision
rule. The regular/default source comparisons passed.

## Timings

TF 2.20.0, FP64, TF32 disabled, RTX 4080 SUPER GPU 1; growth allocator. Native
reference: GCC 11.4, -O3 -std=c++20 -pthread, pinned headers and analytic target
gradients. Four chains, matched target/data/starts/default numerical settings.
Gaussian10, banana10 and centered funnel10 use 500 warmup + 500 retained draws
per chain. Regression uses 100 + 100. Medians of seven synchronized complete-fit
runs, excluding first-call compilation and excluding reporting/I/O:

| Target | CPU XLA (s) | GPU XLA (s) | Native serial (s) | Native four threads (s) |
|---|---:|---:|---:|---:|
| Gaussian10 | 0.1206 | 0.8735 | 0.0240 | 0.00849 |
| Banana10 | 0.0871 | 0.9707 | 0.0191 | 0.00869 |
| Centered funnel10 | 0.2028 | 2.1336 | 0.0322 | 0.01184 |
| Regression 4096 × 128 | 2.3639 | 0.8234 | 1.7273 | 0.6574 |
| Regression 32768 × 128 | not run | 2.8324 | 29.7394 | 18.3214 |

Native columns use the measurements adjacent to the GPU comparison; the CPU
run separately repeats the same native comparator and records its own timings.
Native serial/threaded order alternates. TF and native timing batches are not
interleaved, so this remains a descriptive comparison on a shared machine.

The larger regression has an observed 6.47x GPU/threaded-C++ median ratio and
10.50x GPU/serial-C++ ratio. GPU repetitions span 1.686–2.833 s and native
threaded repetitions 13.782–26.132 s. Logical transition gradient work is
8,220 across GPU chains versus 8,376 native evaluations including initialization.
Those counts are close, but RNG algorithms differ and masked GPU lanes still
perform some extra arithmetic. Counts are explanatory, not a speed normalization
or statistical-efficiency measure.

This regression deliberately evaluates the full design matrix at every score.
It is a workload test, not the fastest possible inference formulation for a
Gaussian linear regression: sufficient-statistic preprocessing would change its
cost substantially. The comparator uses standard Eigen/GCC rather than an
exhaustive CPU BLAS/architecture tuning study. The 6.47x observation applies to
this declared build and workload; a broader “GPU beats native Walnutpie” claim
is unsupported. Seven repeats of fixed seeds do not establish sampler ranking.

CPU first-call times were 8.36–10.97 s; GPU first-call times 18.63–20.79 s,
including tracing/compilation and one execution. Larger regression GPU peak
allocator bytes were 100,932,864 (96.3 MiB). Final qualification peaked at
2,438,028 KiB host RSS on CPU and 2,803,156 KiB on GPU; it compiles many distinct
test graphs. No broad memory-scaling guarantee follows.

## Scheduling and next optimization

On a separate 4-warmup/8-draw Gaussian fixture, command buffers preserve draws
exactly. The default unprofiled median is 48.58 ms; enabling
FUSION,CUBLAS,CUBLASLT,CUDNN,WHILE,CONDITIONAL yields a 2.65 ms median, with a wide
1.50–24.82 ms range. This supports using command buffers as a performance
candidate but not a stable universal 18x claim. The benchmark table already
uses command buffers.

The profiler disables command buffers in TF 2.20, so its trace describes the
fallback scheduler. It records 23,001 GPU events, including 2,074 D2H and 7,347
D2D events for the tiny fixture. These counts are per GPU plane; the full copy
listing also contains host records and must not be double-counted. HLO and the
absence of a sampler callback distinguish host predicate scheduling from CPU
sampler/model execution. GPU-native computation is verified; device-only
control flow with no host scheduling is not established.

The next justified optimization is to reduce nested-loop scheduling and tensor
state movement, preserving the shared-tape oracle, and test more independent
chains or expensive nonlinear targets. A persistent CUDA sampler kernel would
need an explicit way to embed each compiled TF target/gradient; simply wrapping
Walnutpie C++ would lose the present generic TF-model composition contract.
No further optimization sweep was launched after the declared attempt cap.

## Sampler diagnostics and reporting repair

The first benchmark diagnostic helper used the wrong axis order for the
repository's chain-major diagnostic API. All ESS/R-hat fields in attempt 05 are
superseded by attempt 06, computed from preserved draws. Timings and draws were
unaffected. A regression test now enforces the correct transpose. Attempt 07
uses the corrected helper.

| GPU target | TF max R-hat | Native max R-hat | TF min bulk ESS | Native min bulk ESS |
|---|---:|---:|---:|---:|
| Gaussian10 | 1.0069 | 1.0036 | 2445.0 | 2552.0 |
| Banana10 | 1.0049 | 1.0051 | 1167.7 | 1019.3 |
| Centered funnel10 | 1.3972 | 1.1425 | 8.83 | 21.58 |
| Regression 4096 × 128 | 1.0720 | 1.0662 | 220.2 | 244.1 |
| Regression 32768 × 128 | 1.0523 | 1.0567 | 263.4 | 317.4 |

Funnel probability v < -3 is 0.1435 in TF and 0.0525 native versus the analytic
0.1587. This is descriptive and strongly affected by short-chain mixing. Both
funnel arms fail a modern R-hat convergence screen; neither is eligible for an
ESS/second efficiency ranking. The regression runs are also too short for
posterior promotion. Gaussian and banana remain viable engineering fixtures,
not evidence that either implementation is statistically superior.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain standalone TF/GPU implementation | Declared source/API/XLA checks pass | No unresolved default-path source mismatch | Finite tests; stress full-path rounding | Use explicit model-specific validation | Public default readiness |
| Keep native CPU for tiny targets | Observed timing strongly favors native | No engineering veto | Shared-machine timing and RNG work | Avoid GPU overhead for these fixtures | Universal backend ordering |
| Retain larger regression as promising GPU workload | Lower observed fixed-seed wall time | No retained-state failure | CPU build/model algebra alternatives | Replicate on intended nonlinear TF models | Statistical sampler superiority |
| Do not promote funnel posterior output | Convergence unsupported | R-hat/mixing veto | Short chains and deterministic variant | Separate convergence/tuning study | Rejection of the Walnutpie research direction |

| Inference category | Status |
|---|---|
| Hard veto screen | Numerical/API checks pass; strict accumulated nondefault stress parity failed; funnel posterior convergence fails |
| Statistically supported ranking | None; no multi-seed uncertainty analysis or interleaved cross-engine timing design |
| Descriptive-only differences | All timings, ESS/R-hat, tail coverage and observed speed ratios |
| Default readiness | No public inference default change; standalone scope only |
| Next evidence needed | Intended-model replication, paired timing blocks, convergence-qualified ESS/second and broader dtype/backend coverage |

Post-run red-team: the strongest alternative explanation of the large-regression
advantage is memory throughput/implementation choice rather than a generally
faster sampler. Optimized CPU sufficient statistics or a different real model
could overturn it. The weakest evidence is generalization from fixed seeds,
short chains and one shared GPU/CPU machine. The work answers engineering
composition and the declared measurements, not universal sampler superiority.

## Reproducibility and handoff

Plan: bayesfilter-walnutpie-gpu-tf-master-2026-10-28.md. Every worker manifest
records the exact command, environment, source/header hashes, seeds, device,
allocation policy and time. Source snapshots, optimized HLO, retained draws and
profiles remain under docs/benchmarks/artifacts/walnutpie-gpu-tf-20261028.
Compact committed evidence is docs/benchmarks/walnutpie_gpu_tf/evidence.json.

Budget used: 16 worker launches, 1,234.39 cumulative worker seconds of 3,600.
This includes the failed strict stress checks and their repairs. Attempt 11 was
refused by the master before spawning a worker because qualification was still
running; no compute was charged and no previous artifact was overwritten.
The attempt cap is exhausted. Stop here rather than silently starting another
sweep. Further research should define a new bounded plan.

The preexisting funnel clarification was committed as 9c352ce55. The new port,
plan, tests, evidence and result note form a separate scoped commit. Unrelated
dirty worktree files and the public NUTS policy remain untouched.
