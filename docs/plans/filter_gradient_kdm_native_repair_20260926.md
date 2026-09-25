# KDM auxiliary native execution repair

The Younis KDM auxiliary is a diagnostic extension, not a canonical LEDH or
NeuTra training route. Its public callable currently invokes the analytical
score once and then uses a Python trace loop, calls `.numpy()` for model
validation, and creates two TensorFlow kernels for every observation. This
violates the repository execution policy even though the underlying mixture
kernels already have stable signatures and XLA options.

The repair preserves the same trace and KDM equations. Tests load the original
implementation from Git `ab431169d` as an independent diagnostic reference;
no runtime copy is retained. The public callable uses one fixed-signature
TensorFlow program. A private stacked-trace option in the shared analytical
executor avoids unpacking and repacking the time axis. Atom/KDM kernels are
built once per owner, and a `tf.while_loop` accumulates the same values, tangents,
validity flags and per-step diagnostics in the original arithmetic order.
An explicit factory enables repeated calls with changed tensor operands and
fixed model callbacks/options. The public compatibility wrapper builds an owner
per call, preserving mutable Python closure behavior rather than silently caching
those closures. Its per-call compile cost remains measured debt for callers that
do not retain an owner. No global cache is introduced.

Model mismatch is a tensor validity result in the numerical program; the eager
public wrapper raises its existing ValueError at the completed assertion boundary.
Inside an enclosing graph the false validity and NaN auxiliary outputs are the
authoritative rejection. The public tuple of complete step dictionaries and all
existing metadata remain unchanged. Their materialization is host presentation,
not a numerical recurrence. Default `jit_compile=True` follows repository policy;
`False` remains an explicit reference test arm.

Acceptance requires the existing auxiliary fixture to agree with the reference
at the current non-JIT test tolerances, exact replay and changed inputs,
canonical-value no-feedback fields, finite-difference agreement for the KDM
score, one trace, no host callbacks and a stable compiled graph. The complete
original fixture, changed operands, invalid model and ownership checks must pass
CPU graph/XLA before GPU renewal. Matched fresh-process costs cover original,
native graph and native XLA with the same fixture; report cold/public replay and
retained-owner reuse, RSS and allocator peaks separately. Three repeats are
required for descriptive medians; any numerical failure excludes speed ranking.
The route remains
diagnostic-only; no canonical LEDH admission, HMC, posterior, scientific or
performance claim follows.

Budget: at most 20 workers / 7200 charged CPU seconds and 12 workers / 3600 GPU
seconds within the existing 56/52-hour campaign caps. Use the stable campaign
runner, explicit device, unique run directories and one numerical worker at a
time; freeze runtime/scripts/tests while a worker is active. Stop integration on
unexplained numerical failure, source drift, absent diagnostics or cap exhaustion.
No seed, numerical method, threshold or canonical algorithm rebuild is changed.

Skeptical review: moving the trace loop into XLA could change evaluation order,
so the reference comparison is mandatory. Prebuilding kernels avoids repeated
tracing but does not make an unsupported callback compilable; graph inspection
must reject host callbacks. An initial untested draft changed public step fields
and error behavior; it was removed before execution. The revised design preserves
the complete public records and classifies only completed formatting/assertions
as host operations. Compiled ownership freezes callbacks by explicit contract;
the public wrapper cannot assume captured Python cells are immutable. The
score's own analytical
trace remains the sole value/score authority, and KDM never feeds back into it.
