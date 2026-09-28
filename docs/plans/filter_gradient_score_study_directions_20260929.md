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

Execution refresh at 54ba5c538: profiling04750–04755 is complete and preserved
separately. The active consumer repair now has frozen, fresh operands and
controls in `tests/fixtures/filter_repair_score_directions_20260929.json`.
These arrays were generated once with a test-only standard-library generator
seed9292026 and serialized before numerical execution; every arm reads the
same values. They are not a historical LEDH fixture or a runtime RNG migration.
Use d2,o1,N8,T2,float64, K=N, theta [.58,-.75,-.55,.85,.2,-.25], observations
[[.35],[-.18]], SGQF level2, mixture within_fraction .6, KDM bandwidth .23,
and the exact positive dual-cap/Contract E controls in that file. Changed theta
and observations are predeclared there. Direction operands are identity6 and
identity6 rolled one row and scaled by .7. Five-point steps are2e-4 and1e-4;
the absolute derivative gate remains2e-6 and scalar/direction agreement1e-9.
Resampling-KDM finite differences must replay the original fixed samples,
proposal log densities and component indices; regenerating them differentiates
a different scalar and is forbidden as the comparator. KDM negative bandwidth
and a rank-deficient reset design test rejection, without adjusting healthy
controls in response to results.

The actual endpoint wiring check replaces only test data/noise generation
with the frozen arrays; evaluate_gaussian still chooses and calls its real
factory and numerical owner. It checks LEDH with/without control diagnostics,
SGQF, mixture covariance, integrated KDM and resampling KDM. Runtime provenance
is supplied from the campaign worker's already configured backend because the
study initializer otherwise tries to change threads after test initialization.
That initializer itself is not changed or qualified by this injection.

Implementation uses the existing immutable configuration factory caches. Each
can explicitly return an enclosing owner, with all parameter directions as a
dynamic tensor. Nested output specifications come from tracing the existing
declared signature; no dummy numerical evaluation, copied equations or global
callable cache is used. Flags return finite/valid status and the existing
10-machine-epsilon direction-value agreement for host enforcement. Explicit
non-JIT generic reference checks do not replace XLA qualification. Group sizes
are bounded so each worker stays inside300 seconds; reserve a longer scope
explicitly if compiler evidence requires it. Costs follow numerical qualification
in a separately recorded bounded allocation if the initial16 workers are used.

Skeptical refresh: the original plan omitted resampling replay in its derivative
wording and the test-worker/study-initializer interaction. Both are now explicit.
Returning all nested diagnostics avoids a reduced score path; reference loops
are confined to tests. The fresh fixture may still uncover an existing validity
or gradient defect; preserve and localize that failure, never tune the fixture
or loosen its gate. Self-review passes for bounded execution; no independent
review or whole-master completion is asserted.

The optional control-diagnostic path also contained a numerical Python loop
over three covariance families. Its eigenspectrum reductions are now batched
over an explicit leading family axis, preserving the named fields. The static
guard covers the seven touched consumer/helper scopes, with six exact reviewed
exceptions only for reading configuration or packing existing named tensors.
No numerical Python loop is allowlisted. This expands the guard from300 to307
sources and1436 to1442 schema/configuration exceptions.

04756 passes the three generic CPU checks. 04757 passes healthy LEDH output and
derivative assertions, then exposes a wrong rejection-fixture assumption: zero
residual design is not necessarily an invalid Contract E input. The actual route
is canonical_score_tf.py:544 -> canonical_reset_score_tf.py:100 ->
ledh_unified_reset_tf.py:221–258. Design only scales the injected residual;
the transported cloud and positive ridge can still yield valid gap, target and
injected Cholesky factors. The reset's validity is their conjunction, not an
independent design-rank condition. No runtime validity policy is changed.
Preserve04757 as a test-specification failure. The corrected test retains zero
design and requires the enclosing owner to match the scalar authority's values
and status; it additionally injects a nonfinite design to exercise rejection.
Healthy inputs, equations, tolerances and derivative gates remain unchanged.
This localized harness retry stays inside the existing worker/time allocation.

04758–04760 pass LEDH, diagnostic LEDH and SGQF CPU qualification with exact
complete-output parity. 04761 identifies a declared-shape defect before any
mixture numerical execution: TensorArray.stack exposes schedule horizon as
None, although this owner fixes T=2. The mixture factory now explicitly binds
that known leading dimension with tf.ensure_shape for every named schedule
tensor. This changes no array values, equations, tolerances or controls, and
avoids either dummy execution or an unknown-shape scalar fallback. The mixture
case is retried in a fresh worker within the same allocation. Other previously
qualified consumer numerical dependencies are unchanged.

04762 preserves a second construction failure: binding schedule T did not bind
the separate per-particle predicted-covariance trace returned as the fourth
result. Flattened output Identity_18 is the first such trace tensor, after two
scalars and16 schedule tensors; its particle dimension remains None. The
factory now binds each returned prediction to its declared [N,d,d] as well.
This is fixed output-schema metadata, not a new numerical recurrence. The
schedule and prediction payloads remain complete. The initial worker count is
refreshed from16 to18 to preserve both localized schema attempts and terminal
readback; CPU/GPU process-second caps remain3600 each, within the unchanged
global budget. No gate, fixture or scientific scope changes. Source review
checks both returned structures before the next retry.

04763 passes the corrected mixture consumer. Pre-GPU review identifies one
host-boundary compatibility requirement: optional LEDH diagnostic failures must
still materialize their diagnostic payload before a generic finite-value veto.
The endpoint now preserves that original ordering. The actual endpoint check
additionally injects a nonfinite design and verifies the existing diagnostic
exception for diagnostic LEDH, KDM validity exception for the KDM proposals,
and host validity assertion for the others. The four prior CPU cases will be
rerun on this final host source; their prior successful numerical evidence is
preserved. Reserve22 total workers instead of18, with the same3600 CPU/3600 GPU
process-second caps, to cover these four checks and the terminal readback.
This is a bounded error-reporting preservation repair, not a new validity rule
or broader scientific claim.
