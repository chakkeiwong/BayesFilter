# Paired quadratic refinement execution

`refine_batched_quadratic_center(..., config=BatchedQuadraticCenterConfig(pilot_method="paired_local"))`
runs the complete numerical refinement in one TensorFlow/XLA program. Inputs
and the planned row budget are checked before target evaluation. The host then
formats completed histories; returned center, score, precision and factor remain
TensorFlow tensors. Entries in the paired diagnostic dictionary use ordinary
Python values and lists. `payload()` preserves the serialized numerical schema.

The analytical callback consumes a fixed `[batch_size, dimension]` batch and
returns float64 values and scores plus boolean eligibility. It follows
TensorFlow tracing semantics: Python closure values and attributes are fixed
when the program is traced. Use `tf.Variable` for numerical target state that
must change between calls. If redefining a target through Python, provide a new
callback object or call
`bayesfilter.inference.quadratic_rounds_tf.clear_paired_quadratic_controller_cache()`
before the next refinement. Changing Python attributes on the same callback is
not a supported way to update an already traced target.

Repeated calls reuse the most recent callback, dimension, static configuration
and execution mode. Only one controller is retained in this Python cache. The
seed, center, scale and optional initial evidence remain runtime tensor inputs;
changing the seed does not require another trace. Cache replacement releases
its Python ownership; native TensorFlow/XLA executable eviction and memory
reclamation are not guaranteed. For explicit prepared execution, the uncached
`make_paired_quadratic_controller` factory returns the tensor program. It requires
the same input and row-budget validation described in its docstring.

`jit_compile_trust=False` selects the complete graph reference for paired mode.
The diagnostic fields `jit_compile_refinement`, `jit_compile_fit` and
`jit_compile_trust` record that setting. No XLA failure automatically falls back
to graph execution. `fit_calls` and `trust_calls` count executed operations;
trace counts describe graph construction, including conditional branches that
were traced but not executed. `full_initializer_xla` remains false because
composition with the initial multistart locator is a separate qualification.

Uniform-cloud fitting has separate execution debt and retains its existing
behavior. The paired route's tests do not establish uniform-cloud, DZ5, HMC,
posterior or whole-repository readiness. Current evidence and unresolved costs
are recorded in [the controller repair plan](../plans/filter_gradient_quadratic_rounds_20260921.md).
