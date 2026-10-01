# Staged joint center execution

`bayesfilter.inference.locate_joint_center_staged` defaults to two XLA-compiled
numerical stages. The first stage evaluates the initial target, runs L-BFGS to
the checkpoint, replays that endpoint and constructs its numerical diagnostics.
The external `checkpoint_validator` then runs once on the host. If it accepts,
the second stage continues the same optimizer state, with the same global
evaluation budget, and completes endpoint replay and incumbent selection.

Repeated calls with the same callback, configuration, dimension and device
reuse one compiled owner. Starting positions and positive diagonal scales are
tensor inputs and may change without retracing. Python bound methods are
identified by their receiver and underlying function, so repeated method
attribute access also reuses the owner. Separate callbacks, receivers,
configurations or devices require separate owners. Only the most recently used
owner is retained by the public cache; active calls retain their own handles.

The callback must be graph-compatible and its Python configuration must remain
fixed while its program is reused. Put changing numerical state in tensor
inputs or TensorFlow variables. When changing captured Python configuration,
create a new callback or call
`bayesfilter.inference.joint_center_staged_tf.clear_staged_joint_center_cache()`
before reuse. Clearing the cache releases Python ownership; TensorFlow's native
executable allocations may remain until the process exits.

The validator can reject, raise an exception, or invoke a nested locator.
Rejection and validator exceptions produce the existing typed result. Nested
execution cannot replace the outer call's saved optimizer/accounting state.
Both stages and the validator are serialized by the owner's reentrant lock.
The result retains the existing frozen derivative boundary; this API does not
differentiate the optimized position or issue a covariance or HMC certificate.

Graph construction failures that the numerical program represents as an
invalid target or optimizer result retain that result. Actual TensorFlow
compiler and device-execution errors propagate to the caller. Such failures
do not cause eager retries or completed records assembled from partial counters.

`jit_compile=False` is an explicit graph diagnostic. A positive
`max_wall_seconds` additionally selects the legacy host-clock diagnostic and
requires JIT to be disabled. For the normal XLA path, enforce wall-time limits
in the parent process. A host clock cannot be a dependency of the XLA kernel.

The public execution evidence and remaining limitations are recorded in the
[September 27 repair result](../plans/filter_gradient_staged_public_result_20260927.md).
