# Walnutpie initialization, run control and R2P results

The planned initialization, run-control and R2P-transition work is implemented
and passes the scoped CPU/GPU XLA qualification. The source differences and
limits below remain explicit; this is engineering qualification, not a
posterior-convergence or performance result.

## Implemented behavior

The standalone TensorFlow implementation now provides initialization policies,
compiled run control, and an optional R2P transition. The existing deterministic
variant and fieldwise layout remain defaults. Public inference routes and the
tracked prebuilt C++ package are unchanged.

Initialization accepts supplied, zero or normal positions; supplied, unit or
absolute-score diagonal mass; optional geometric averaging; and supplied steps
or fixed-momentum initial search. The bounded search remains the default; an
explicit uncapped mode follows the source selector. Invalid steps, reciprocal
masses and keys are rejected by validity flags.

Run control implements the pinned C++ warmup relative step/mass statistics and
legacy log-density agreement statistic. Numerical loops and decisions compile
with the target into XLA. Synchronized chain advancement replaces asynchronous
C++ workers. Host scheduling handles deadlines, cancellation and callbacks only
between compiled chunks; these are soft boundaries, including compilation time.
Checkpoint continuation preserves RNG, caches, adaptation and online moments.
Warmup and retained arrays are separate, with counts/masks for padded storage.
Zero warmup is supported. A maximum-count stop is not convergence.

R2P uses forward and reverse refinement probabilities and cumulative corrections
at both endpoints, including exact zero weights with preserved geometry. It
uses the optimized endpoint energy criterion and checks intermediate
finiteness. The pinned capped selector may return its cap without meeting the
threshold; the random finer branch can use cap+1. The macro result exposes that
distinction. Optional warmup combines this transition with Walnutpie Adam and
metric adaptation; it does not reproduce Appendix C quantile tuning. Fixed
supplied tuning with zero warmup is also available.

## Source audit and review

Plan: [gap-closure plan](bayesfilter-walnutpie-gap-closure-2026-10-29.md).
The pre-execution skeptical review covered the baseline, missing R2P weights,
upstream source discrepancy, tuning assumptions, synchronized scheduling,
invalid input behavior, evidence limits and the bounded campaign budget.
This was an author self-review, not an independent-agent review.

Pinned sources:

- Walnutpie C++ 5710b0334ba69cf0caa35fc02a2338b61d756953, config.hpp, util.hpp,
  adapt.hpp and sampler.hpp. All 15 local headers match the saved source package.
- Author bob-carpenter/walnuts at 895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a,
  WALNUTSpy/adaptiveIntegrators.py (R2P macro at line 361 onward), constants.py
  and WALNUTS.py. Its MIT license is preserved in WALNUTS-R2P-LICENSE.
- The Within-Orbit Adaptive Leapfrog No-U-Turn Sampler, JMLR 27 (2026),
  25-1452, sections 3.1–3.4, 4.1/4.3 and relevant Appendices A–C; local PDF
  SHA256 68d0e831198a996ce326919980240ab61861d6efbcfcecee8d9be3a62ab9bf4c.

The author backward two-leaf tree code appears to omit the second cumulative
log-weight update before its line-462 weight. Our implementation includes the
term required by equations (13)–(14). Consequently, the comparator is the
unmodified author macro plus an independently written recursive tree using
explicit orbit points and the paper equations, not the unmodified whole author
chain. A negative control explicitly removes the second correction. The author
-700 sentinel is interpreted as mathematical zero probability (-infinity),
which is another documented difference from literal floating-point code.

The diagonal-metric macro extension is checked by conjugacy to the author's
identity-metric integrator. Whole-tree comparisons use identity metric;
whole-tree parity with an independent nonidentity-metric implementation is not
established by that macro check. The tree keeps the deterministic port's pinned
metric/U-turn convention.

## Qualification

Environment: Python 3.13.13, TensorFlow 2.20.0, tfgpu conda interpreter; FP64,
TF32 disabled, stable explicit signatures and XLA enabled. CPU jobs deliberately
hide GPUs. Trusted GPU jobs use physical GPU 1, RTX 4080 SUPER, with verified
memory growth before device initialization. The master records exact commands,
source snapshots/hashes, seeds, device state, wall time and output paths.

| Check | Final artifact | Result |
|---|---|---|
| Focused CPU tests | attempt-06/worker.log | 48 passed, two TFP deprecation warnings |
| CPU XLA initialization/control/R2P | attempt-07/validation.json | Passed |
| Final GPU XLA initialization/control/R2P | attempt-08/validation.json | Passed |
| Saved 528-record C++ regression, CPU | attempt-09/deterministic/result.json | Passed |
| Saved 528-record C++ regression, GPU | attempt-10/deterministic/result.json | Passed |

Artifacts are under ../benchmarks/artifacts/walnutpie-gaps-20261029/. Each
validation covers 144 identity-metric author macro fixtures, 144 diagonal-metric
conjugacy fixtures and 216 whole-tree comparisons (72 fixtures across three
layouts). Targets are Gaussian, banana and centered funnel. These are local
engineering comparisons, not posterior runs. Macro coverage includes baseline
and finer branches, unequal forward/reverse probabilities, zero reverse support
and cap fallback. Discrete exponents, evaluation counts and tape consumption
must agree; local floating comparisons use rtol/atol 2e-11.

The deterministic regression contains 96 macro, 96 transition, 96 initialization
and 240 lifecycle records across Gaussian, banana, centered funnel and regression
fixtures. Its nondefault target-macro=1 lifecycle arm uses controlled native
state replay between steps, as the saved test design specifies; this establishes
local transition/adaptation parity rather than uninterrupted chaotic-path
identity. The default target-macro=15 arm runs uninterrupted. The regression also
checks runtime data, cache agreement, chain isolation, continuation, frozen
sampling tuning, invalid starts and rejection of Python callback models.

The final GPU suite additionally compiled random-start initialization and full
warmup and sampling controllers, preserving their optimized HLO. Each inspected
function has one trace, with no model or sampler host callbacks. All eight saved
optimized graphs have model gradients inside loops and no custom-call targets.
XLA GPU execution does not mean every operation is fused into one CUDA kernel.
The final gap worker reports 163,584 allocator peak bytes and 2,754,808 KiB
peak host RSS for these small fixtures; this is not a large-model memory
bound and does not include every driver/compiler allocation.

## Repairs and attempt accounting

All intermediate evidence remains preserved. Attempts 01–03 qualified initial
implementation versions. Attempt 04 added boundary tests. Attempt 05 reran CPU
parity. Attempt 06 contains the final runtime and 48-test suite; attempt 07
requalifies final runtime CPU parity; attempt 08 adds GPU compilation checks
for random starts and full controlled warmup. Earlier snapshots must not be
substituted for the final numerical revision.

Static audit repairs: invalid zero/negative epsilon could hang uncapped search;
exhausted checkpoints could stall host scheduling; nondefault target-macro
settings could initialize R2P minimum above one; all-zero subtree selection
formed undefined infinity subtraction; reciprocal mass and pre-randomization
key validity needed checking; zero warmup required explicit empty output.
Focused regressions exercise each repair. These were implementation defects,
not evidence against the method or a reason to abandon the planned validation.

Author-source overflow warnings arise in deliberately extreme reference
trajectories. TensorFlow's function-creation warning is emitted because the
fixture loop constructs separate functions; inspected functions each trace
once. Neither warning is suppressed or treated as convergence evidence.

Final budget accounting: 1079.712 cumulative worker seconds across 10
launches; all completed successfully. The limit was 3600 seconds and 18
launches, leaving 2520.288 seconds unused. No performance ladder or
posterior campaign was run. The final runtime hashes agree with attempts
06–10; original prebuilt binary SHA256 remains
ea5d4965854c950c3a8d91f9b4ff5452a6283422edfa0b2e3d51ec15a326f45b.

Compact evidence, exact commands, source hashes and final reports are saved in
[gap-closure-evidence-20261029.json](../benchmarks/walnutpie_gpu_tf/gap-closure-evidence-20261029.json).
Each original versioned attempt retains its complete source snapshot and logs.

| Attempt | Worker seconds | Purpose |
|---|---:|---|
| attempt-01 | 96.479 | Initial tests |
| attempt-02 | 74.171 | Initial CPU parity |
| attempt-03 | 150.865 | Initial GPU parity |
| attempt-04 | 111.067 | Boundary-test expansion |
| attempt-05 | 77.816 | Intermediate CPU parity |
| attempt-06 | 115.787 | Final 48-test suite |
| attempt-07 | 74.494 | Final runtime CPU parity |
| attempt-08 | 168.610 | Final GPU parity and controller graphs |
| attempt-09 | 62.692 | Deterministic CPU regression |
| attempt-10 | 147.731 | Deterministic GPU regression |

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Initialization/control | Pass on scoped CPU/GPU fixtures | No unresolved engineering veto | Asynchronous stopping differs by design | Use documented APIs with target-specific validation | Identical C++ timing or convergence |
| R2P transition | Pass on scoped CPU/GPU fixtures | No unresolved comparison mismatch | Finite fixtures and shared model assumptions | Optional use with separately assessed tuning | Universal equivalence or posterior validity |
| R2P warmup | Explicit local combination | Appendix C identity not claimed | Tuning adequacy is target-dependent | Separate paper-tuning study if required | Reproduction of paper experiments |

| Inference question | Status |
|---|---|
| Hard veto screen | All declared engineering fixtures pass; no unresolved hard veto. Both deterministic and R2P remain viable engineering implementations within this scope |
| Statistically supported ranking | None; no paired performance or posterior ranking was undertaken |
| Descriptive-only differences | Worker/compilation timing and memory are engineering provenance |
| Default-readiness | Existing deterministic default preserved; no public inference promotion |
| Next evidence needed | Target-specific tuning, replicated posterior checks and matched workload timing for any scientific or speed claim |

## Terminal review

Terminal skeptical self-review passes for the declared engineering scope.
No result invalidated the target or research direction; static defects were
repaired and the final numerical sources requalified on CPU and GPU. The
expected callback compilation failure in deterministic logs is the negative
control passing, not a failed worker. The strongest alternative explanation for apparent
correctness is a shared assumption in the author integrator and independent
tree reference, or an unexercised branch. The author backward-weight discrepancy
is explicit and tested; it is not hidden by declaring bitwise author parity.
A reference mismatch, invalid retained cache, unbounded retracing or host
callback would overturn engineering qualification and trigger repair.
Posterior convergence on centered funnel and performance superiority remain
unevaluated by this task.
