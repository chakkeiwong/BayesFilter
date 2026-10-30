# Execution history through the final capacity checks

## Checkpoint

2026-10-21. Owner authorizes committing the preceding benchmark/design,
implementing the proposed batching repair, reviewing the plan, and executing
the tests. This is Phase 1 of the SIR NeuTra integration, not a new posterior
or training campaign. Baseline commit `f6bb90a00` preserves the benchmark,
design, and one complete executed source snapshot; unrelated staged work is
preserved. Active branch: `ledh-graceful-failure-execution-20260917`.

Native model, proposal, reset and correction algebra is implemented. Seven
model-callback variants and ten complete-filter FP64 comparisons passed.
The original 32 focused regression tests passed. The first graph-size assertion
was too strict about TensorFlow constant allocation; inspection shows identical
numerical bodies for B=1 through 128, with differences confined to Const/Fill.
Its original failure is preserved. Pairwise-memory repairs are now undergoing
the combined regression suite before the matched N4032/T50 GPU measurements.

Combined suite: 81 passed in 215.57 seconds. First full-size GPU run completed
valid T50 rows with 0.52 GiB allocator peak and about 20 seconds warm latency,
but the third row score differs from the executed baseline by 2.4314% in L2.
This fails the predeclared 1% numerical criterion. Larger-batch promotion is
paused while diagnostic ablations separately restore the saved direct density
and direct transport formulas. These checks preserve native model/maps and
all workload settings; at most three bounded ablation/repair attempts, charged
to the existing budget. This is a numerical repair trigger, not a new target.

The native-only ablation still differs by 4.0692% on row 3. Restoring the
SIR source observation-density expression reduces the current maximum to
2.2740%, still above 1%. Before selecting another repair, compare old and new
full-horizon analytical recurrences in FP64 as explicitly diagnostic reference
exceptions. Cast identical FP32-prepared inputs to FP64; do not regenerate data
or random design. One synchronized evaluation per variant, B4, same 600-second
arm limit. These reference timings are excluded from FP32 performance results.
They distinguish a finite-program derivative defect from ill-conditioned FP32
recursion; neither an FP64 pass nor faster timings waive the FP32 criterion.

Both full-size FP64 runs completed. Maximum old/new relative score difference
is 4.35e-11; likelihoods agree to rounding. Against that common reference,
the third row's old FP32 score differs by 5.769%, and native FP32 by 8.057%.
Thus the old FP32 result is not a numerical oracle and the recurrence already
has a material precision sensitivity. The native FP32 candidate remains
unpromoted; the 1% requirement is unchanged. This is an engineering result,
not evidence against the model or an authorization to replace FP32 by FP64.

Execution refinement after skeptical review: complete B8/B16 capacity and
batch-consistency diagnostics, then the resource-bounded requested ladder.
Full-size FP64 equivalence and the independent small reference tests establish
the algebra needed to make a capacity experiment meaningful. The unresolved
FP32 veto still prevents numerical/default/training promotion, but does not
make a memory allocation measurement invalid. These runs answer only capacity
and row-consistency, retain all four parameter cases, and cannot waive the
precision mismatch. No HMC or optimizer update will run. This is the existing
phase 4 with an explicit diagnostic-only interpretation, under the same budget.

The native-shard callback is now implemented separately from training. Its
first parity run passed fixed-design results but exposed a history-only
isolation bug: a failed row received frozen ancestor history at later times
only if another row kept the outer loop alive. History after row failure now
consistently contains NaN. The regression compares every output for an uneven
logical batch, including the invalid final row duplicated for padding. This
changes no accepted likelihood, score or particle state. No optimizer adapter
or training-update equivalence is claimed.

Sharding regression after repair: 3 passed in 39.72 seconds. A call-chain
audit additionally exposed a pre-existing scalar wrapper bug: it indexed every
output, including scalar covariance_tolerance. It now removes the leading
dimension only from nonscalar outputs. The executable public-entry comparison
and four supervisor checks passed (5 tests, 21.68 seconds). These localized
repairs do not modify the running batched likelihood/score arithmetic.

CPU4/8/16/32 and GPU8/16/32/64/128 ladders are executing with 600-second arm
caps. GPU4 was already measured. GPU8 and GPU16 have completed; their allocator
peaks are 1.03 and 2.03 GiB. An optional native-shard diagnostic is wired into
the supervisor for a bounded B128/S8 check if the full batch exceeds capacity.
CPU supervisors now omit NVIDIA probes entirely, and workers explicitly hide
GPU devices. No scalar fallback or training update is introduced.

### Final capacity extension and review

GPU B64 completes a first and a warmed evaluation at about 8 GiB, but reaches
the 600-second arm cap before the requested three warm repeats finish. B128
fails allocation under the declared 12 GiB limit (requested 17,264,076,960
bytes). Execute one B128/S4 native-sharded evaluation with the same 600-second
cap. S4 is a measured capacity choice: a full B4 evaluation uses 0.52 GiB and
about 15 seconds warm, giving an estimated 32-shard duration near 490 seconds
plus compilation. S8 is slower per row in current descriptive timings. Neither
choice changes the target or establishes optimal throughput. Each shard retains
a batch dimension of four; no scalar fallback or training update is allowed.
The sharded result must agree with repeated B4 outputs within the same 1%
criterion, and must preserve status/steps and input hashes. This is a capacity
diagnostic under the existing two-GPU-hour budget; the FP32/reference veto
remains. Skeptical review: timeout is possible and will be reported; finite
outputs alone do not prove numerical or training readiness.

The isolated checkout test first failed collection because its existing compiled
TensorFlow custom op was absent. Copy the locally built op as an environment
prerequisite, record its hash, and rerun; preserve the original failure log.
This repairs the test environment without importing unrelated working edits.

