# Walnutpie TensorFlow/GPU master program

The October 28 owner request authorizes this standalone port and comparison,
including its local NUTS tree, without changing public inference defaults.
Prior literature clarification committed as `9c352ce55`.

## Research intent and evidence contract

Question: can the deterministic Walnutpie implementation at
`5710b0334ba69cf0caa35fc02a2338b61d756953` execute its complete numerical
lifecycle with caller-supplied TensorFlow models inside GPU XLA, and when is
that useful compared with optimized native C++? The mechanism is batch-native
TF control flow, autodiff and linear algebra, with bounded binary tree storage.
CPU staging of the sampler does not satisfy this contract.

Engineering pass: independent unmodified-header C++ oracle agreement on
refinement, reversal, tree selection, initialization and adaptation; full-fit
CPU/GPU XLA compilation; runtime model data, continuation and independent-chain
tests; finite retained state and coherent counters. Floating comparisons use
scaled 2e-11 tolerance (engineering hypothesis extending the Stan FP64 2e-12
fixture tolerance); discrete decisions must agree away from ties.

Performance comparator: pinned C++ headers built with -O3, analytic target
gradients, separately measured from compilation. Match target, FP64, starts,
configuration and counts. Report gradient work because different RNGs generate
different trajectories. Also compare CPU XLA and GPU XLA, with synchronization
and seven timing repeats (convenience budget choice). Timing intervals describe
fixed-seed runtime variation only. Speedup is a hypothesis, not an acceptance
criterion that permits changing the algorithm or weakening the comparator.

Invalid retained caches, oracle mismatches, callbacks, hidden CPU sampler/model
execution or missing device provenance veto engineering promotion and trigger
repair. Budget exhaustion, corrupted evidence or a changed scientific target
stop execution. Funnel convergence failure vetoes posterior-efficiency ranking,
but does not stop engineering repairs. R-hat, ESS and observed wall times from
short chains are descriptive. No posterior/default promotion, universal speedup,
paper R2P reproduction, complete Stan-language support or one-kernel claim.

## Defaults and source audit

Source: local unmodified headers `walnuts.hpp`, `adaptive_walnuts.hpp`,
`adam.hpp`, `online_moments.hpp`, `config.hpp`, `util.hpp`. Preserve MIT license.
Use pinned default endpoint tolerance, Adam settings, score/draw metric and
minimum microstep adaptation; these are source defaults, not universal optima.
Port deterministic refinement, not the R2P paper variant. Runtime/model shapes
are fixed per compiled API; dtype FP64 is the parity baseline.

Four chains and 500 warmup/500 retained transitions on Gaussian, banana and
funnel inherit the preceding bridge fixtures. A data-bearing regression adds
an expensive-target engineering arm (4096 by 128, 100+100 initially); these are
convenience workload choices, not convergence claims. Check smaller shapes
first. GPU 1 was idle in the trusted readiness probe; recheck before launches.
All GPU processes use verified memory growth before initialization. CPU
reference processes explicitly hide GPUs. No pfor and no runtime NumPy.

## Program and budget

1. Implement one numerical authority under `docs/benchmarks/walnutpie_gpu_tf`,
   exposing initialize, warmup, sample, full fit and model value/gradient.
2. Compile an independent oracle using unmodified C++ headers. Exercise macro
   refinement, reverse rejection, internal versus outer tree combinations,
   per-macro Adam, discounted moments and initial step search.
3. Verify CPU then GPU XLA, dynamic model operands, chain isolation, continuation,
   finite retained caches, graph/HLO and bounded tracing.
4. Run a versioned master with source hashes, commands, environment, seeds,
   hardware, timing, work counters, allocator stats and retained diagnostics.
5. If warranted, test XLA command buffers under the same numerical checks.
   Review results, document limits, and commit only task-owned files.

Cap: 3600 cumulative worker seconds and 16 worker launches, including native
builds, tests and repairs. These are convenience engineering caps. Each launch
uses a fresh directory under
`docs/benchmarks/artifacts/walnutpie-gpu-tf-20261028`; preserve failures and
account for time. A localized repair does not need renewed authorization.
The master must enforce the remaining budget and individual timeouts.
Result: `docs/plans/bayesfilter-walnutpie-gpu-tf-result-2026-10-28.md`.

## Skeptical pre-execution review

Self-review passed after resolving these material design risks:

- Reusing Stan's tree or adaptation would be wrong: Walnutpie uses displacement
  U-turns, Barker internal selection, Metropolis outer selection, and Adam
  observations at every first macro attempt, including failed proposals.
- Refinement preserves step times minimum microsteps. Reverse coarser-step
  acceptance rejects the macro immediately. Outer selection precedes U-turn
  stopping; internal U-turn rejection precedes random combination.
- C++ lround is not bankers rounding. Initial mass uses the absolute score.
  Initial step search needs a bounded local safety extension with an explicit
  invalid flag, never silent fallback.
- TF Philox is not the native generator. Independent shared-tape tests are
  needed; equal seeds are not cross-language parity evidence.
- Tiny FP64 models are likely launch-bound on GPU. Include an expensive model
  and report a negative speed result if observed. Compilation is not speed
  evidence; host loop-predicate scheduling is distinct from CPU sampling.
- A short funnel run cannot establish convergence or contradict the R2P paper.
- Upstream wall-clock/cross-chain stopping remains outside fixed-count compiled
  numerical functions; this scope must be visible in API documentation.

No independent agent review was requested; this is a skeptical author review.

## Execution refinement after first measurements

CPU/GPU shared-tape parity passed 336 cases per backend. The first comparison
showed a remaining native-C++ advantage even on regression4096x128. This is a
negative performance observation, not an implementation failure. Within the
unchanged campaign budget, add regression32768x128 (100+100, four chains) to
locate the workload-size effect; this convenience size is an engineering
hypothesis, and both sizes remain in the report. Do not claim a universal
speedup from the larger arm. Inspect an 8-draw Gaussian GPU profile and compare
command buffers on/off with exact same inputs and retained-draw checks.

The first benchmark reporting helper passed sample-major arrays to a
chain-major diagnostic API. Its ESS/R-hat fields are invalid and must be
superseded by fresh post-run diagnostics from preserved draws. Numerical
kernels, retained draws and timed intervals are unaffected. Add a focused
axis-regression test. This repair is within the existing contract and budget.

Review of these additions: no target density or method changes; native inputs
remain identical; extra size is labeled rather than replacing the failed
small workload. Profiling is explanatory only and cannot prove a speedup.

Final source review extends shared-tape tests to the regression target and to
macro-count target 1 (test-only setting) so minimum-step adaptation actually
changes. It found an initial-minimum discrepancy for that nondefault target: the
initial estimator's 2/1 pseudo-observation must also pass through the rounding
rule. Repaired before qualification. Initial step growth may encounter -inf
energy differences and recover in the shrink stage; the final state/error, not
that intermediate difference, determines initialization validity. Added focused
regressions and distinct RNG-key/int32-bound checks. These changes do not alter
the finite/default paths used in the recorded timings; final source qualification
will check them on both backends.

Qualification attempt 08 stopped on the nondefault target-macro=1 funnel
trajectory: one coordinate differed by 4.48e-10 (4.68e-11 relative) after
nonlinear accumulation, with numerically matching displayed states. Preserve
this failed strict full-trajectory comparison. Repair diagnostic: replay each
nondefault lifecycle transition from the upstream selected position/cache after
checking that transition, while preserving the TF adaptation accumulators. Keep
2e-11 local tolerance and exact discrete/count checks. Default target=15 retains
unaltered full-trajectory comparison. This controlled check can establish local
source fidelity but does not erase the failed accumulated-trajectory tolerance.

Attempt 09's position-only replay still exceeded componentwise tolerance in the
nondefault funnel stress case (6.82e-9 absolute, 2.74e-11 relative). That replay
left adapted step/metric perturbations in the next transition and therefore did
not isolate the integrator. The next diagnostic resets all exposed native
transition inputs (position/cache, step, inverse mass, log step) after each
comparison; TF moment/Adam accumulators still evolve and are checked against
upstream step/metric outputs. Default full-trajectory checks stay unchanged.
Neither failed stress check is reported as a passing full-trajectory result.
