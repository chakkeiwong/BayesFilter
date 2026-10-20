# Accounting-width compiler-context bisection

The real one-iteration control04635--04637 reproduces the first-score
discrepancy. Saved unoptimized-HLO comparison finds6100 computations per arm;
6023 match after SSA renaming, source-metadata removal and content-addressing
callee references. Of the77 unmatched per arm, inspected three-instruction
sum helpers differ in signed32 versus signed64 accounting. This is a lead,
not a count of77 mathematical changes or an optimized-code equivalence proof.

`batched_local_center_tf.py:27--35,79,90,210` deliberately uses int64 for
resource counters/indices because int32 resources cannot be used by this GPU
XLA route. All other numerical code and the floating target remain unchanged.
Question: does changing only these accounting dtypes in an isolated CPU
diagnostic candidate restore the original first score? This does not propose
using int32 GPU resources or changing the production counter contract.

Load the candidate module source from the same frozen r1 snapshot. Build a
separately named diagnostic module replacing only exact `tf.int64` attribute
references with `tf.int32`. Verify the AST differs only by that substitution;
record original/diagnostic source hashes and save both sources. Do not mutate
the shared TensorFlow module, the frozen files or the actual repository
implementation. Bind this diagnostic class to the established one-iteration
driver, and bind its TFP dispatch to the same real one-iteration wrapper.
The callback observer retains its own existing dtype and instrumentation.

Run one candidate CPU worker under the unchanged300-second parent/280-second
child deadline. Compare its first two callback inputs, values, scores and
validity bytes against04635,04636,04584 and04585. Exact input identity,
single trace, no host callbacks, saved HLO/source integrity and at most128
objective batches are mandatory controls. An exact match to the original
score, with the unmodified candidate still matching04585, supports a causal
accounting-width/context effect for this case. A third score or the unchanged
candidate score is a negative result. Neither outcome is a runtime repair or
a reason to waive full-record/convergence criteria.

Allocate at most4 CPU workers/1200 CPU process-seconds in a separate unit,
inside unchanged56/52-hour caps. Two planned workers are the diagnostic and
saved-evidence/policy readback; at most one localized harness repair fits the
remaining slots. Stop on source/input drift, timeout, unexpected changes or
missing evidence. No GPU, training, HMC, new compiler flags, package changes,
numerical tolerance change or full optimizer trajectory is authorized.

Skeptical review: dtype changes can alter allocation, compilation and integer
control together. The run can identify the effect of this narrow intervention,
not the exact compiler optimization or a safe GPU remedy. Counter ranges in
this one-iteration fixture are small; overflow must not explain the result.
All source/float operations and observed first inputs must remain identical.
The primary agent reviewed this bounded diagnostic; no independent reviewer
was used. A negative result will be retained and will not trigger a full
trajectory rerun.

04638 and readback04639 pass: replacing the11 dtype attributes restores the
original first score exactly; the unmodified candidate reproduces the historical
difference. Use the two remaining worker slots for one narrower diagnostic
and readback. Keep every resource/index dtype at int64, replacing only
`reduce_sum(cast(~valid,int64))` by
`cast(reduce_sum(cast(~valid,int32)),int64)` in the isolated module. Verify that
exact single-expression AST replacement and retain all other controls. The
fixture batch size is1, so the inner sum cannot overflow. This is a CPU-only
localization of a potentially GPU-compatible operation; it does not qualify
arbitrary batch sizes, GPU execution or a runtime repair. At most300 seconds
per remaining worker; the same1200-second unit cap and all original stop
conditions apply. No further retry slot remains after these two workers.
