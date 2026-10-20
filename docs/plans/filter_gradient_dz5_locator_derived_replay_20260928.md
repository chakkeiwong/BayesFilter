# Derived replay count diagnostic

The isolated int32 replay-resource declaration04651 restores the original
first score. Int64 storage with int32 increments04652 does not. All3/3/2
attempt/optimizer/replay counts match both one-iteration controls.

Inspect `batched_local_center_tf.py`: search_round increments replay once after
L-BFGS; the native outer loop then increments completed rounds once. The final
replay increments once precisely when `has_incumbent` is true. No replay count
is consulted by the algorithm. On a successfully returned invocation,
`replay_batches = rounds_completed + cast(has_incumbent)` is therefore exact.
An exception returns no result and is not converted into a successful count.

Build an isolated r1 candidate module removing only the replay declaration,
reset tuple element and two increment statements. Replace its result read by
`tf.cast(rounds, tf.int64) + tf.cast(has_incumbent, tf.int64)`. Verify these five
exact statement/expression AST edits against literal edits. Retain all other
int64 resources and every float expression, optimizer setting, observer, input
and numerical gate. No shared TensorFlow or runtime source mutation.

One CPU/XLA real-one-iteration probe uses the established280-second child and
300-second parent limits. Require one trace, no host callbacks, no captured
int32 resource, exact counter reports and first-two-row source/input/value/
status/hash controls. Also compare all saved callbacks and every returned
record field against both one-iteration controls; record differences without
waiving them. First-score equality alone is insufficient even for complete
one-iteration equivalence. Preserve the historical full-trajectory gap.

Allocate4 workers/1200 CPU process-seconds within unchanged56/52-hour global
caps: one probe and one complete saved-evidence/policy readback, with two
localized harness-retry/check slots. Groups are
`dz5_locator_derived_replay_cpu` and `dz5_locator_derived_replay_readback_cpu`,
explicit CPU and300-second timeouts. One numerical worker, unique raw outputs,
ordinary source hashes and existing permission prefixes. Stop on source/input
drift, unexpected edits, counter mismatch, timeout or allocation exhaustion.

Skeptical review: the algebraic count identity follows the current completed
control flow, but public edge cases and concurrency/reset behavior still need
qualification before installation. Removing a resource can perturb compiler
context differently from changing its dtype. A third or candidate score is a
negative remedy result, not permission for more full trajectories. A positive
complete short-record result would justify a later CPU/GPU edge-case and full
trajectory plan; it does not itself admit runtime code. No GPU, threshold
change, training, HMC or main merge. Primary-agent review; no independent agent.
