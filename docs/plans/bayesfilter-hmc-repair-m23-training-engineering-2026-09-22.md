# M23 training prerequisites: stable graphs and real batches

Read-only inspection of NeuTraReverseKLTrainer found two implementation gaps:
its four compiled entry points use reduce_retracing without explicit input
signatures, and its direct/external optimizer entry points accept a one-row
batch. Chunked updates also permit one real row plus padding. Neither behavior
satisfies the repository's training policy. This repair is a prerequisite to
target-specific learned-transport experiments, not a training-quality result.

Use a small private fixed-shape graph cache for each compiled entry point.
Every graph has an explicit TensorSpec for every float64 input, including the
actual batch length and target dimension. Keep four signatures per entry point
in an LRU cache: this is a convenience memory bound for ordinary training,
validation and padded tail shapes, not a numerical hyperparameter. Eviction
recompiles a graph and must not change model/optimizer state. Static shapes
preserve existing target adapters that use a known batch length, avoiding a
silent change to a dynamically shaped target contract. XLA remains governed
by the existing jit_compile flag and defaults on.

Reject direct and detached-score optimizer calls with fewer than two rows
before calling the target or changing any state. For chunked updates, require
at least two real rows in the total update; padding is not data. Each physical
chunk must also contain at least two rows, preserving the existing padded-tail
convention. A one-real-row tail is valid when other chunks supply the rest of
the batch. Forward evaluation and validation may still evaluate a single row.
Do not change optimizer equations, clipping, frozen maps, loss definitions or
scientific defaults.

Engineering acceptance requires unchanged multi-row losses, gradients and
optimizer updates on Gaussian and curved targets; direct/detached/chunk parity;
atomic singleton rejection (including padded singleton updates); cached graph
signatures and retracing bounded by the declared cache; restored-state update
parity; and the existing affine/dense/composed trainer tests. CPU tests are
explicit tiny reference exceptions with GPUs hidden, one intra/inter-op thread
and no transport-quality claim. Reserve 600 CPU seconds in M23's existing
14400-second allowance, including localized failures/retries. Keep the shared
two-worker ceiling. GPU/XLA training or quality claims require separate
target-specific resolved designs and a free permitted device.

Skeptical audit: a dynamic [None,D] signature could break adapters requiring
static batch shapes; a total tensor size can count padding incorrectly; and
rejecting every one-real-row tail would break a correct full-batch update.
The proposed checks distinguish these cases. A cache hit is performance
evidence only; numerical parity and atomic rejection establish correctness.
Other trainer classes and staged/curriculum training remain separately
audited migration work; this bounded change must not be described as clearing
every NeuTra training route. Unrelated q20 training-protocol edits are preserved.

The first run had 89 passes and six failures. One fixture still split a
two-row update into scalar physical chunks; pad each tail to two rows while
retaining its original real-row count and anchor-release assertions. Three
reference tests required an older exact sibling commit; use an isolated
checkout of d94566c9 rather than changing the sibling workspace or weakening
the pin. Expose the reference checkout location through a test-only environment
variable. Another test expected a different rejection message for the same
forbidden learning-rate change. The final failure exposed an existing missing
constructor check: the strict DSGE/capacity presets required the paper schedule
and clipping norm but omitted their per-variable clipping mode. Restore that
check, which rejects mislabeled configurations without changing any valid
preset's optimizer. Re-run the affected tests and preserve the first failure
record and its 46.647287479019724-second charge.

The corrected five-file training check passed all 95 tests in
48.46348594396841 seconds. A subsequent isolated-checkout run included those
checks and the new controller, mutation, exact-batch reference and progress
tests: all 152 passed. That run excludes the shared workspace's unrelated
training-protocol/q20 edits and records hashes of all 19 owned code/test files.
Its receipt is `m23-r1/isolated-continuation-tests-r3.json`. The tested repair
is committed in `f9c86f41a` and published on main. This establishes bounded
graph/batch engineering behavior; target-specific banana/mixture training and
downstream quality remain open. Trusted GPU probe r4 again reports no idle
permitted GPU. No serious GPU training or CPU replacement was launched.
