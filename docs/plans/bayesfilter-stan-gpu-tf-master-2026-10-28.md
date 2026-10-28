# Stan algorithm for TensorFlow models on GPU/XLA

Owner request: commit the completed native bridge, create and review a master
program, then execute a GPU implementation with TensorFlow model integration.
The preceding work is committed as `90617dce2`. This is a standalone research
route; public inference defaults remain outside this task.

Status: executed and terminally reviewed. Eleven worker attempts consumed
1990.90 of 2400 seconds. The numerical integration passed its engineering
checks; posterior and performance conclusions remain qualified in the
[result](bayesfilter-stan-gpu-tf-result-2026-10-28.md).

## Research question and evidence contract

Can the existing source-audited Stan 2.37 Euclidean multinomial NUTS algorithm
evaluate a caller's TensorFlow model and its gradient inside one enclosing XLA
GPU computation, including initialization, windowed adaptation and sampling?
Can removing serial per-chain RNG control improve its execution cost without
changing its random stream or decisions?

The exact implementation baseline is `docs/benchmarks/stan_nuts.py` at commit
90617dce2. Independent algorithm authorities are the installed unmodified Stan
2.37 headers, native transition/adaptation oracles and native CmdStan targets.
Stock TFP is a performance comparator, not an oracle for Stan tree behavior.
The existing CPU custom-call bridge is a reference, not a GPU implementation.

Primary engineering criteria: native-oracle transition and adaptation checks;
matching baseline decisions and random states for the RNG change; compiled
dynamic model inputs and gradients; one trace per static signature; GPU output
placement; no PyFunc, native-sampler custom call, CPU target callback or host
state staging in the new numerical path. Inspect optimized HLO and a GPU
profile, preserving actual fusion counts and execution evidence. Do not claim
that the entire dynamic NUTS program becomes one CUDA kernel.

Numerical mismatch, invalid retained state/cache, hidden fallback, stale source
or corrupted evidence veto engineering acceptance and trigger repair. Posterior
R-hat/ESS/reference mismatch or divergences veto posterior efficiency ranking,
not continuation of engineering repairs. Compilation time, launch count, stack
cost and tree imbalance are explanatory diagnostics. Exhausted budget, invalid
target or unavailable required GPU execution stop the affected campaign.

Timings are descriptive: three repetitions do not establish statistical
superiority. No production promotion, universal speedup, arbitrary Stan-language
support, posterior differentiation or validity of every TF operator is claimed.

## Implementation phases

1. Preserve the completed bridge in Git; audit the direct TF port, native source
   and existing oracle tests. Reuse the single tree/adaptation authority.
2. Extend the shared factory with explicit model tensor input signatures and an
   injectable kernel implementation. Add a GPU-oriented implementation that
   statically batches the per-chain TF Philox calls instead of a serial map loop.
   Preserve keys, counters, inactive-chain masks and supplied oracle tapes.
3. Add a reusable full-fit compiled entry point and separate compiled lifecycle
   functions for continuation. Define model/state/metric contracts and retain
   warmup separately. Add focused CPU/XLA and GPU/XLA tests, independent native
   checks, dynamic-data/row-isolation checks and negative interface tests.
4. Build an executable master runner with fresh output directories, source
   snapshots, commands, budgets, device/memory provenance and a terminal report.
   Exercise Gaussian, banana, funnel and runtime-input Gaussian regression.
   Compare the original direct port, optimized route, and native Stan; preserve
   exact timing boundaries and diagnostic qualifications.
5. Review the implementation/results; record failures, repairs, measurements,
   limitations and next justified optimization. Commit this task's changes.

## Assumptions and numerical provenance

| Choice | Provenance / status | Risk and early check |
| --- | --- | --- |
| Stan 2.37 Euclidean multinomial tree, unit/diag/dense metrics | Installed source and audited direct baseline | Native tape oracle and existing option/adaptation tests |
| FP64, TF32 off | Existing parity baseline | Consumer GPU FP64 speed may be poor; report rather than change precision |
| Explicit fixed chain/dimension/data shapes | Existing XLA contract | Retracing/unsupported model ops; compile dynamic-data test first |
| Static RNG batching | Optimization hypothesis; same TF Philox calls | Graph growth with chain count; test C=1,4,16 and preserve exact streams |
| Stan tuning .8, .05, .75, 10; windows 75/50/25; depth 10 | Pinned native defaults/baseline | Short tuning not convergence; retain full telemetry |
| 4 chains, 500 warmup, 500 retained; 3 timing repeats | Bounded engineering convenience, inherited bridge workload | Short posterior estimates; descriptive only |
| Seed 20261028 plus disjoint per-chain keys | Reproducible convenience | No RNG identity claim versus Stan MIXMAX |
| Numerical tolerance 2e-12 scaled | Existing native transition oracle tolerance | Discrete branch checks exact away from ties; diagnose rounding separately |
| GPU 0, memory growth, stable XLA signatures | Repository execution policy | Fail closed before initialization if growth unavailable |

## Budget, commands and artifacts

Total worker wall budget: 2400 seconds, at most 12 worker attempts including
repairs (convenience cap for this local implementation campaign). Each long
worker has a watchdog; run cheap correctness checks before benchmarking. Native
builds, unit tests and compile/profiling workers count toward this allowance.
Source inspection/editing is outside compute budget. No installation or hardware
expansion is planned. Every launch uses a fresh numbered directory below
`docs/benchmarks/artifacts/stan-gpu-tf-20261028/`.

Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, installed TF 2.20 /
TFP 0.25, native CmdStan 2.37. CPU reference workers hide GPUs before import.
GPU workers run with trusted tool execution and TF_FORCE_GPU_ALLOW_GROWTH=true,
then verify memory growth before device initialization.

Planned master command:
`python docs/benchmarks/stan_gpu_tf/master.py --output <fresh-output> --device both --gpu 0`.
Master emits manifests, logs, HLO, serialized draws/telemetry and a JSON summary.
Terminal interpretation: `bayesfilter-stan-gpu-tf-result-2026-10-28.md`.

## Skeptical review before implementation

Reviewed 2026-10-28. Material traps resolved before execution:

- A wholesale CUDA custom call would hide the TF model and fail the desired
  compiler visibility; use the existing TF numerical authority instead.
- Renaming stock TFP NUTS would not port Stan's transition. Preserve the direct
  tree and run its independent native oracles.
- The prior port already compiles on GPU, so compilation alone is not progress.
  New evidence must show runtime model data integration and measured RNG/layout
  behavior. Exact RNG parity is required for the optimization comparison.
- Passing inputs computed by TF around a CPU sampler does not integrate a TF
  target. Model evaluation and its gradient must occur within leapfrog HLO.
- A scalar CPU comparator versus batched GPU measures hardware/layout differences
  too. Record both direct CPU/GPU results and native timing scope explicitly.
- Short funnel chains cannot justify a speed ranking; preserve diagnostics even
  when they fail. Model data and starting points must match across comparators.
- Do not confuse successful HLO compilation with absence of host transfers inside
  an opaque call. Reject native bridge calls and inspect runtime profiling.
- Static RNG expansion may trade loop overhead for graph/code growth. Keep an
  explicit baseline and measured compilation cost; revert optimization if parity
  fails and record a regression if timing worsens.

Review outcome: proceed with this bounded implementation and evidence contract.
Self-review is recorded; no independent agent review is claimed.

### Profile-triggered control-flow repair

The first GPU validation passed 182 native transitions and full-fit integration,
but profiling captured 450 D2H events during 16 retained transitions. This does
not move the TF model to CPU; it identifies XLA host loop-predicate overhead.
The installed TF 2.20 XLA headers expose command-buffer WHILE and CONDITIONAL
support. Test that backend option before performance comparison. It is an
explicit experimental setting until the native-oracle checks, numerical replay
and a new profile with no host/device copies pass. This repair preserves the
model, method, hardware, numeric policy and campaign budget.

Reviewed repair: enabling command buffers without checking actual transfers
would not answer the question. Preserve the original profile, run fresh GPU
validation, require profiler evidence, and report compilation cost and any
unsupported backend behavior. The master records the exact XLA_FLAGS.

Further source inspection found a material diagnostic flaw: TF 2.20's vendored
`xla/backends/gpu/runtime/command_buffer_thunk.cc`, lines 224-231, disables CUDA
graphs when the TF profiler is active (upstream CUPTI memory-corruption issue).
The second command-buffer validation passed numerical execution but failed the
incorrect no-transfer profiling assertion. The first exited without a Python
error and is retained as an unexplained worker failure. These do not reject the
algorithm or command-buffer execution. Repair the diagnostic: retain fallback
profiles with explicit scope, inspect non-profiled command-buffer runtime logs,
and leave zero-transfer normal execution unproved. Do not override upstream's
unsafe-profiling safeguard. No sampling or promotion criterion is weakened.

### Whole-fit rounding and RNG comparison

The CPU benchmark's longer banana/funnel fits changed trajectories between RNG
layouts despite exact primitive Philox output and short-fit parity. Whole-fit
bitwise trajectory identity is not a valid universal compiler-rewrite criterion:
changed graph fusion can change arithmetic rounding and later adaptive branches.
Preserve this discrepancy, compare common-state one-transition decisions and
numerics, and interpret different-work whole-fit timings only descriptively.
The initial GPU Gaussian/banana fits were exactly equal between layouts. No
rounding explanation is accepted as a proof without direct controlled checks;
the terminal record must retain any unresolved mismatch.

### Terminal diagnostic audit before execution

The diagnostic reader reuses `benchmark_stan_transplant.assess`: finite retained
states/cache, movement in every chain, modern maximum R-hat <=1.01, minimum
bulk/tail ESS >=400, standardized MCSE <=.05, and reference moment agreement
within max(4 MCSE, .05 reference SD). These are inherited engineering screens,
not calibrated guarantees of posterior accuracy. Any retained divergence also
vetoes an efficiency ranking in this campaign. Failure prompts longer sampling
or target-specific tuning in a future campaign; it does not reject the
implementation when independent transition checks pass. The 500-draw chains
and single seed cannot establish a statistical speed or ESS ranking even if
all screens pass.

The funnel neck diagnostic v<-3 is an explanatory count at one scale standard
deviation below zero, with exact marginal probability about .1587. It cannot
certify exploration by itself. CPU layout checks compare 96 transitions at
shared states and seeds; their 2e-12 scaled tolerance is inherited from the
native oracle. The first changed warmup draw is a rounding-localization
diagnostic, not a proof of the cause of every later adaptive branch difference.
Cross-execution replay tests compare saved draws from identical GPU fixtures.
CPU/GPU starts and regression designs differ by roundoff in their separately
generated normal transforms; record the maxima rather than claiming bitwise
cross-device inputs or trajectories. Within each backend, layout inputs are
identical, and native Stan uses the CPU fixture.

Skeptical audit: the diagnostic operations are CPU-only post-run reporting;
they cannot alter tuning, sampling or benchmark measurements. Native
in-process timings do not retain that run's posterior output, so posterior
screen results apply to the separate CmdStan run only. Preserve this gap.
