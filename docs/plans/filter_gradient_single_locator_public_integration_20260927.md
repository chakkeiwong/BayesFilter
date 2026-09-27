# Public single locator execution repair

After staged-locator qualification, wire `locate_joint_center` to the existing
`make_joint_center_program` numerical authority. The public API currently
constructs new optimizer closures and performs initial replay, endpoint replay,
incumbent selection and score summaries outside the optimizer's graph. The
native program already has original-record, accounting and rounding tests.
This unit changes execution and ownership only, preserving all settings,
callback order, caps, target values, derivative boundaries and returned fields.

Use `3582b4ac` with its complete frozen numerical imports as the reference.
The existing GPU reference helper records its four int64 accounting
substitutions; it must not be described as the untouched GPU original.
The public result formatter consumes only completed native tensors. Use one
retained program for a callback/configuration/dimension/device signature,
including receiver/function identity for Python bound methods and independent
dependency ownership. Starts and scales remain operands. Execute and format
under the program's invocation lock.

The explicit non-JIT host-clock diagnostic remains available under a named
private diagnostic, never as an execution fallback. Construction errors already
represented by the native program retain those result records; true compiler
or device errors propagate. Validate the public rank/finite/positive input
boundary before target use. No tolerance, optimizer, numerical model, NumPy
exception, pfor engine or scientific acceptance criterion changes.

The known internal callers are
`posterior_local_initializer.initialize_posterior_local_location_scale` and
`quadratic_map_covariance._run_locator`. Test their actual call chains as well
as the exported API. The posterior initializer still has an outer host
controller; repairing its locator alone cannot qualify that whole initializer.
The quadratic consumer already catches locator errors and records a fallback;
preserve and inspect that boundary instead of claiming all consumer exceptions
propagate. Unsupported target callbacks must not be silently called eagerly.

Reserve 20 CPU and 16 GPU workers, 7,200 combined charged seconds, inside the
existing global 56/52-hour caps. Use the same registered runner prefix and
fresh numbered output directories, normally 300 seconds per focused worker.
A documented compilation-capacity issue may use a 900-second retry within the
unchanged unit reservation. Run CPU correctness before GPU checks; observe the
existing selector and growth policy. Keep sources frozen during each matrix.

Acceptance requires public complete-original records and exact callback order
at D1/D3 on healthy, nonlinear, invalid, constant, cap and iteration-limit
cases; existing exported API and two consumer checks; unchanged frozen
derivatives; changed-input trace/HLO stability; callback/method/configuration
isolation and Python collection; compiler-failure/no-fallback evidence; policy
checks; and fresh original/graph/XLA public cost observations at two extents.
Costs include construction, completed records and host boundaries. Retain
graph/XLA rounding differences independently; compare each mode with its
original for same-mode numerical qualification. Do not rank failed arms.

Skeptical review: moving initial and final evaluations into XLA can change
finite arithmetic or exception timing. Full records, exact order and actual
consumer tests must expose that rather than comparing optimizer positions only.
Reusing a callback graph can capture mutable Python configuration; only tensor
operands/resource state may change between reuse calls. Changing static
configuration requires a different owner. A single retained Python program
cannot prove native executable eviction. Passing this unit cannot close the
outer initializer, actual DZ5 target, GenUT precision/reporting or terminal
F01--F20 gates. Preserve failures and localize the first mismatch before adoption.
