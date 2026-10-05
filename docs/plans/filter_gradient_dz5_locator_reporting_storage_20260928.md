# Reporting-counter storage and arithmetic localization

Through04648 the positive first-score intervention is confined to three
reporting counters: attempts, optimizer calls and replays. The round-budget,
index/call and invalid-row families each preserve the unmodified candidate
score. Continue the unchanged r1 real-one-iteration CPU/XLA fixture.

Run three independent diagnostic modules that change one reporting resource
declaration at a time from int64 to int32. Compare with both historical and
one-iteration original/candidate authorities. Exact source/AST edits, unchanged
floating expressions, identical input/value/status bytes, source/HLO/callback
hashes, one trace, no host callbacks and recorded counts remain required. These
storage interventions are CPU-only and never eligible as GPU runtime repairs.

Then run one distinct diagnostic that retains every resource declaration at
int64 and replaces only four reporting increment expressions (one attempts,
one optimizer_calls, two replays):

`counter.assign_add(1)` becomes
`counter.assign(tf.cast(tf.cast(counter.read_value(), tf.int32) + 1, tf.int64))`.

Independently verify those exact AST substitutions and absence of any other
edit. Save captured variable dtypes and assert no int32 resource in this arm.
Require recorded counts below2**31; no overflow or changed counting semantics
is allowed for this fixture. This is a probe of logical arithmetic with
GPU-compatible storage, not qualification of arbitrary budgets, thread safety,
GPU execution or public API changes. The original program is unchanged.

Allocate6 workers/1800 CPU process-seconds within the unchanged56/52-hour caps:
three individual-counter probes, one storage-preserving probe, one complete
saved-evidence/policy readback, and one localized harness retry if needed.
Each parent timeout is300 seconds and child280 seconds. One numerical worker
at a time; explicit `--device CPU`; groups
`dz5_locator_reporting_{attempts,optimizer_calls,replays,logical_int32}_cpu`
and `dz5_locator_reporting_readback_cpu`. Record ordinary versioned raw runs
and use existing environment/runner approvals. Stop on source drift, altered
float expressions, unavailable evidence, timeout, type error or unit budget.

Skeptical review before execution: a storage-only change can perturb parameter
and loop tuple shapes, scheduling or optimization. A logical-int32 increment
may optimize back to the same program or yield a third score. None of these
outcomes demonstrates the exact compiler cause. A positive one-counter result
identifies a sufficient intervention only. Interaction remains possible if
all individual arms are negative. Keep the one-iteration comparison exact;
never infer full-trajectory equivalence or waive the121 historical record
differences. No GPU, full optimizer, runtime edit, threshold change, training,
HMC or main merge. Primary-agent review; no independent reviewer used.
