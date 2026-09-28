# Enclose score-study analytical direction assembly

Current-source inspection at d8c23fc41 identifies a concrete remaining
filter/gradient consumer violation. `bayesfilter/score_study/adapters.py` calls
six compiled directional kernels through Python list comprehensions inside
`evaluate_gaussian` (lines125–167), then stacks/reduces their values and scores.
This affects LEDH, SGQF covariance, Gaussian-mixture covariance, integrated KDM
and resampling KDM proposals. Compiling each direction does not enclose the
complete analytical score. The nearby six tensor contractions in
canonical_adapter_tf.gaussian_direction_inputs are numerical Python iteration,
not merely schema assembly. No runtime changes are made by this plan yet.

Question: can these actual consumers call one stable enclosing XLA owner for
all analytical parameter directions, preserving their existing finite scalar,
total initial-law sensitivities, status/diagnostics and explicit nonclaims?
Baseline is the same current directional kernel evaluated in a test-only Python
loop on identical frozen operands. No pfor, old LEDH fixture, saved pre-August21
result, algorithm replacement or full canonical rebuild is permitted.

Use one shared TensorFlow parameter-direction loop with declared nested output
signatures and sequential scheduling. Keep theta/data/base noises/reset design
and direction tensors dynamic; freeze only declared dimensions, controls and
the owning directional kernel. The loop must invoke the existing mathematical
authority for each direction, not copy equations or substitute autodiff.
This is a parameter-direction axis, not sample-row mapping for training.
Do not introduce a silent mutable-closure cache. Reuse explicit owners and the
existing bounded factory cache only when configuration identity is immutable;
record native-memory limitations and prohibit cache-eviction claims from Python
weak references alone. One-shot APIs may not silently recompile each direction.

Inspect the full route before editing: evaluate_gaussian -> make_canonical_kernel,
make_covariance_kernel, make_mixture_covariance_kernel or make_kdm_kernel ->
the current shared score executor. Preserve all existing model/covariance/KDM
providers, source-adaptation labels and admission gates. Replace the six
Gaussian tangent contractions with explicit tensor operations. Keep fixed-schema
packing and host artifact materialization distinct from numerical iteration.
In particular, optional diagnostics and heterogeneous nested outputs require
declared shapes, not dummy execution or a Python numerical fallback.

The outer owner must return all direction values/scores plus the same diagnostic
payloads. Native tensor reductions should issue value-invariance and validity
flags; enforce them at the host assertion boundary because XLA may ignore
tf.debugging assertions. Preserve all existing rejection behavior, including
KDM per-direction validity and optional correction/covariance diagnostics.
Return the same public artifact fields; record the enclosing kernel/trace
ownership accurately rather than retaining the misleading six-host-call claim.
Do not self-attest canonical identity or admit an unsupported route through
diagnostic metadata.

Qualification starts with a fresh generic analytical quadratic direction
fixture (including nested auxiliary outputs), then the smallest fresh healthy
actual Gaussian consumers at d2,o1,N8,T2,float64, exact K=N. Declare controls,
seed and complete input arrays before execution. Exercise changed theta, data
and directions, default JIT, explicit CPU reference graph mode, optional
diagnostic payloads and intended invalidity. Keep the prior healthy numerical
gates; at most1e-9 value/directional agreement and2e-6 five-point derivatives at
two step sizes on the same finite scalar. Ill-conditioned or rejected systems
must report invalidity and cannot establish a speed ranking. Do not tune input
choices or tolerances in response to a failed numerical case.

Follow actual evaluate_gaussian wiring with a small recorded context and verify
that all affected proposal branches reach the shared enclosing owner. Generic
helper parity alone does not qualify the consumer. Prove one fixed trace,
dynamic operands, enclosing HLO and no pfor/host callbacks on CPU and trusted
non-display GPU. GPU memory growth must be configured/verified before device
initialization. Scalar/reference loops remain test-only. Neither successful
compilation nor derivative agreement establishes full canonical LEDH, KDM
scientific validity, HMC readiness or training admission.

After correctness, use fresh-process before/after owner costs for representative
qualified consumers: separate cold compilation, warm synchronized calls, host
RSS, allocator current/peak, owner reuse and Python collection. Use a bounded
cohort and matching input/source/environment records; GPU timing waits for
unshared hardware. Numerical GPU checks may use idle shared contexts with
explicit provenance, but cannot become uncontended timing evidence. The existing
paired streaming study and earlier component evidence are not measurements of
this new enclosing consumer.

Initial allocation:16 sequential workers,3600 CPU/3600 GPU process-seconds,
including two localized harness repairs, drawn from the remaining unchanged
56 CPU/52 GPU hour campaign caps. Use the existing registered runner prefix,
300-second workers initially and unique numbered artifacts. Before any longer
worker, identify its concrete compile scope and reserve from this allocation.
Stop/localize source/input drift, missing diagnostics, lost total derivatives,
numerical/status failures, unsupported callback/configuration identity or
allocation exhaustion. Do not rerun huge old matrices, train, sample HMC,
change packages/system settings, edit live MacroFinance or merge main.

Skeptical review: enclosing a map can alter fusion/rounding and memory without
changing equations; exact operands, finite-difference controls and separate
cost evidence address that risk. Nested diagnostic shapes could tempt a scalar
fallback or a fake trace template; both are vetoes. Compiled assertions can be
ignored, so return status tensors and preserve host checks. Existing canonical
names confer no new admission. This bounded consumer repair answers the missing
call-chain criterion; unrelated optimizer/geometry and master gates stay open.
Primary-agent review passes for implementation preparation; record the exact
fixture/controls before the first numerical worker. No independent review asserted.
