# Native Stan and Walnutpie XLA bridge: master program

Owner request: implement both native samplers behind CPU and GPU XLA calls,
review thoroughly, and execute. Scope is a standalone benchmark integration,
not a repository sampler default. Baseline commit: e38348bee2b321d695181ec74e7486a611f67df4.

## Research intent and evidence contract

Question: can an XLA call reuse the native Stan and Walnutpie implementations
without changing their seeded complete-fit results, and what overhead does
CPU or GPU host staging add? The sampler remains compiled C++ on CPU. XLA
compiles the enclosing TensorFlow graph; this is hybrid execution, not GPU
sampling or XLA compilation of the sampler internals.

The primary engineering criterion is identical draws and adapted parameters
between direct native, CPU-XLA and GPU-XLA for the same engine, initial state,
seed, model and configuration. Required checks include fixed counts, excluded
warmup, finite retained states, deterministic replay, distinct-seed behavior,
invalid-input errors, actual custom-call HLO and device placement. Errors,
buffer/layout mismatches, changed target values/gradients, or nondeterministic
replay veto implementation acceptance and trigger repair. Unsupported GPU
execution blocks the GPU claim but does not invalidate CPU results.

Timing is explanatory: measure synchronous complete fits including transfers,
setup and warmup, with compilation reported separately. Native execution is
the baseline, not the slower TensorFlow algorithm port. At least three timed
repeats per route provide descriptive overhead; no sampler superiority or
default promotion follows. Centered-funnel coverage, moments, R-hat and ESS
are sampler diagnostics only; poor mixing is not a bridge failure when native
and wrapped draws agree. No short-run funnel convergence claim is permitted.

## Implementation sequence

1. Pin upstream source and review APIs, model/autodiff lifetime, callbacks,
   threading, stopping rules and XLA ABI.
2. Build an isolated C++ native library using the generated existing Stan
   benchmark model; expose a checked C ABI for full fits and target gradients.
   Use Stan's adaptive diagonal-metric service and Walnutpie's upstream
   AdaptiveWalnuts/WalnutsSampler unchanged. Keep all warmup draws separately
   identifiable, adapted parameters, timings and available telemetry in memory.
3. Add a TensorFlow op, explicit static signatures and CPU/GPU XLA lowering.
   GPU entry stages buffers through host memory on the supplied stream, checks
   CUDA results, and synchronizes before host use and before freeing buffers.
4. Validate direct/native and CPU paths, then trusted GPU execution with memory
   growth. Preserve optimized HLO and source/build provenance.
5. Run bounded overhead comparisons on Gaussian, banana and centered funnel;
   record terminal review, limitations and user guide.

## Defaults and assumptions audit

| Choice | Provenance / status | Risk and early diagnostic |
|---|---|---|
| CmdStan 2.37.0 | Installed pinned comparator | Verify generated model and linked source |
| Walnutpie 5710b0334ba69cf0caa35fc02a2338b61d756953 | Downloaded upstream source, hypothesis | Compile and inspect adaptation/transition source |
| Existing stan_targets.stan | Existing target authority | Compare native gradients to analytic Gaussian/funnel |
| FP64, diagonal metrics | Native source baseline | No TF32 or GPU-native performance inference |
| Stan delta .8, gamma .05, kappa .75, t0 10; windows 75/50/25; depth 10 | Existing Stan benchmark defaults | Record all; small warmup gets native window adjustment |
| Walnutpie builder defaults, initialization step .1 | Upstream defaults, baseline only | Preserve and report actual configuration; no universal tuning claim |
| Fixed warmup and sample counts | Static XLA shapes, convenience | Use upstream adaptive sampler directly; document omission of controller's early stopping |
| Explicit starts/seeds, independent chains | Bridge reproducibility requirement | Match seeds and chain IDs within each engine, not across algorithms |
| Sequential chain loop initially | Bounded reference execution | Equal native/wrapped scheduling; no multicore scaling claim |
| 100 warmup/100 retained smoke; 500/500 overhead | Convenience engineering workloads | No posterior promotion; counts and finite values checked |
| Three repetitions and seeds 101/202 | Convenience reproducibility | Descriptive timings only, not statistical ranking |

Arbitrary TensorFlow log-density callbacks, exact checkpoint continuation,
dense metrics and public TransitionKernel integration are outside this initial
complete-fit bridge. Compiled Stan models are required; the benchmark model
supports several geometries. Each call owns RNG, model and output state.

## Skeptical review before execution

Reviewed 2026-10-28 against local native sources. Material issues and repairs:

- Wrapping CPU code does not make it GPU-native: name and report hybrid execution.
- Comparing different seeded trajectories confounds overhead: call the same
  native ABI for all three routes and require output identity.
- Walnutpie's high-level controller permits variable counts and uses a
  log-density R-hat: call its documented per-chain adaptive/sampling classes
  with explicit fixed counts, never use that controller statistic as convergence.
- Stan high-level services restart RNG/adaptation: expose full fits, not false
  continuation semantics.
- Shared reverse-mode autodiff arenas can race: initially run chains serially
  and serialize calls to the native ABI; no shared sampler state.
- Existing custom-call stub redefines private status layout: use the installed
  matching internal status header, and test an actual runtime error.
- CUDA asynchronous copies can outlive host vectors: synchronize the supplied
  stream before reading/freeing them; check every CUDA return value.
- Native-wrapper identity alone cannot verify target construction: retain
  independent analytic value/gradient checks and a CmdStan comparison.
- Walnutpie does not expose Stan's diagnostics: mark unavailable fields NaN,
  preserve native fields/counts; never manufacture zero divergences.

Audit passes for this bounded integration study after these changes. This is a
self-review; no independent reviewer was requested or used.

## Budget, execution and artifacts

Total budget: 90 minutes elapsed development/build/verification; at most 20
minutes sampling across all attempts and 12 benchmark launches. These are
convenience resource bounds, not scientific stopping criteria. Begin with tiny
CPU checks; no large sampler comparison ladder. Repair local build/ABI/harness
failures within budget. Stop for corrupted outputs, unresolved target mismatch,
resource-policy failure or budget exhaustion; preserve partial evidence.

Environment: /home/ubuntu/anaconda3/envs/tfgpu/bin/python, TF 2.20.0, TFP 0.25.0.
CPU runs set CUDA_VISIBLE_DEVICES=-1 before import. GPU launches use trusted
execution, TF_FORCE_GPU_ALLOW_GROWTH=true before import and verified growth.
No environment-wide installs. Build under .localresources/native_sampler_bridge.
Each run creates a fresh directory under
docs/benchmarks/artifacts/native-stan-walnutpie-20261028/.
Exact executed commands, source hashes, environment, seeds, wall time, HLO,
device/memory policy, results and failures go in run manifests and the result
note. A launcher wall-time limit bounds pathological trajectories.

Pre-mortem: a successful custom call could conceal CPU fallback, malformed
layout or omitted warmup; HLO, placement and full-array parity distinguish them.
A funnel failure could reflect native geometry rather than bridge corruption;
native identity distinguishes this. Fast timings without synchronized outputs
would be meaningless; every measured invocation materializes the outputs.

## Checkpoint

CPU and GPU smoke calls completed with exact native parity on four target
families and propagated runtime errors. Stan also exactly matches CmdStan
after restoring its initialization radius 2 RNG consumption (even for supplied
starts). The first failed comparison is preserved under cpu-smoke-01.

Pre-benchmark source review found missing Walnutpie initialization in the first
bridge: its Python wrapper uses gradient-based initial mass and an initial
step-size search. Both are now included through unchanged upstream methods,
using its separate initialization seed convention. Earlier smoke results prove
transport mechanics only and do not establish complete wrapper initialization.
A separate upstream high-level Gaussian oracle will check the fixed-count
adapter against native Walnutpie. Existing upstream source is not modified.

Timing audit found Python list conversion was included only in wrapped times.
Remove that conversion from timing; synchronize by materializing raw output
buffers, then compare bytes outside the interval. Include setup, sampling and
all GPU transfers; report first-call compilation separately. Benchmark repeats
use alternating order. No performance conclusion is based on the old smokes.

Benchmark seeds are 101/202/303/404 for four chains (the initial table's two
seeds described the tiny smoke). Warmup/sample benchmark counts remain 500/500.
Sampling budget remains 20 minutes and 12 launches; initial smokes total under
10 seconds measured process wall time. No posterior/default promotion.

Terminal integration review adds a tiny GPU producer/custom-call/consumer
composition diagnostic: the original GPU tests used already materialized
inputs. A dynamic preceding operation and a transposing/scaling following
operation check stream ordering and layouts in a larger XLA computation. This
is an engineering check under the existing budget, not another sampler study.
The master runner now includes it before GPU benchmarking.

## Completion checkpoint

All five implementation phases are complete for the stated full-fit scope.
`master-02` completed on CPU and GPU with exact native output identity,
independent native oracles, invalid-input and reentrancy checks, and GPU
producer/custom-call/consumer verification. The terminal review and all
measurements are recorded in
`docs/plans/bayesfilter-native-stan-walnutpie-result-2026-10-28.md`.

Engineering acceptance does not promote the short funnel chains: both show
poor mixing. That is a sampler diagnostic failure under the tested budgets,
not a native-wrapper mismatch and not rejection of local adaptation. The
implementation remains standalone; production defaults are unchanged.
