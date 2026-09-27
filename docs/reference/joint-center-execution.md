# Single joint-center execution

`locate_joint_center` runs initialization, bounded L-BFGS, exact endpoint replay,
incumbent selection and score summaries in the shared `make_joint_center_program`
TensorFlow authority. Its default is XLA. The host boundary validates inputs and
formats completed tensors into `JointCenterResult`; returned tensors remain
disconnected from external differentiation.

Starts and scales are tensor operands. One public program is retained for a
callback identity, dimension, complete configuration and device signature.
Bound methods use receiver and function identity. Changing Python configuration
requires a different signature/owner; mutating Python state captured by a callback
does not retrace an existing program. Tensor/resource state can remain dynamic.
The invocation lock covers execution and result materialization, and each public
owner has its own dependency cache scope. Clearing the cache releases Python
ownership; it does not promise native executable eviction.

An explicit `jit_compile=False` selects the graph reference. An additional
`max_wall_seconds` selects the named host-clock diagnostic. Neither is a fallback
for XLA failures. Construction errors already represented by the native program
retain their typed result; real compiler/device errors propagate. The quadratic
initializer catches those errors and records its existing initial-position
fallback. The posterior initializer's enclosing host controller is separate
migration debt.

The [public integration plan](../plans/filter_gradient_single_locator_public_integration_20260927.md)
defines bounded CPU/GPU qualification and original-reference costs. This
execution wiring confers no posterior, covariance, HMC or canonical LEDH status.
