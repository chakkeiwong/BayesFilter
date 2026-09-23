# Staged locator compilation and residency diagnostic

Question: what compilation, steady-call and memory costs accompany enclosing
the staged locator numerical work? Compare the untouched3582b4ac public staged
locator with the internal two-stage graph and XLA candidates. This is an
internal diagnostic before public integration, not a replacement for the final
repeated public endpoint comparison. Both native arms keep identical TFP
settings, checkpoint validator and explicit continuation state. Graph execution
is an explicit non-default reference; CPU is an explicit reference device.

Use the qualified D1/D3 quadratic fixtures, checkpoint iteration1, total10,
gradient tolerance1e-8, cap120 and the same three starts/scales from the original
record test. Each arm gets a fresh process. Measure factory construction (which
traces the checkpoint), cold complete call, three synchronized repeated calls,
then changed operands. Complete calls include validator invocation and result
formatting. The original intentionally constructs its stages anew per public
call; label the measured reuse difference and do not call this an identical-
graph compiler ablation. Export IR and execute independent original numerical
checks only after timed/memory observations. Require the unchanged full-record
comparison, exact decisions and validator counts; no simplified scalar output
is a substitute for complete records.

Record source/input/configuration hashes, environment, build/cold/warm/changed
times, process RSS/PSS/map counts, allocator current/peak separately from device
reservation, trace counts and unchanged HLO across changed operands. One fresh
process per arm/extent is descriptive localization only. Existing triggers remain:
cold>2x, warm>1.2x, host RSS increase>256MiB or>2x, GPU allocator peak>2x and
continuing warm growth. Triggers prompt attribution/repair; numerical failure
stops that arm without relaxing criteria. These measurements cannot certify
native eviction, leak freedom, general performance ranking, public readiness,
HMC or main merge.

Reserve at most16 workers/2400 charged seconds (six arms/extents per backend,
four localized retries),120s each or300s for a justified retry, under unchanged
32CPU/52GPU campaign caps. Use the existing runner's registered
`staged_center_cost_{prior,graph,xla}_{1,3}_{cpu,gpu}` groups and
`staged_center_cost_cpu/gpu` matrices. Freeze runtime/scripts/tests for each
device cohort. GPU launches require the existing selector and verified memory
growth; preserve sharing observations and exclude shared-device timings from
cost conclusions. Fresh numbered directories retain complete records and errors.

Skeptical review: moving initial replay under XLA can change strict incumbent
ties; the already qualified full-record cases are used without changing
tolerances. Construction must enter cold total because checkpoint tracing
occurs in the constructor. CPU input synchronization and GPU completed result
materialization precede stopping each timer. Source signatures must match across
the six processes. Three warm calls are insufficient for long-term residency;
map/RSS changes are explanatory leads and E4 remains open. Public error and GPU
qualification remain independent prerequisites for integration.

Recovery through03331: the original D1/D3 arms pass03329/03330; graph D1
fails03331 because its best-source field is optimizer_callback rather than
the original XLA checkpoint_replay. The selected value/position agree for
the initial operands; the changed operands also show sub-1e-32 objective
differences. Preserve this failure and keep all comparison criteria. The
graph arm is not eligible for an equivalent-result cost conclusion. Inspect
the optimizer callback and separately rounded replay before attempting repair.
Run the still-unexecuted XLA D1/D3 arms independently under the same reservation;
they answer candidate correctness without relying on the failed graph arm.
The cost fixture adds0.11 to constructed operands, whereas the earlier record
fixture adds it before constructing each operand; their changed D3 tensors are
not assumed bit-identical. This extra operand case must pass its own original
comparison. No source change or fixture replacement is authorized by a failure.

Both XLA arms pass03332/03333. Complete the previously unexecuted graph D3 arm
without changing sources, then use one of the four localized diagnostic workers
to compare the original and candidate separately at each JIT setting on the
unchanged D1 operands. Preserve the original XLA authority for admission and
the failed cross-setting result. Inspect raw callback incumbent, replay and
standardized coordinates; independently calculate fused versus separately
rounded affine positions with standard-library high-precision arithmetic.
Matching the original graph diagnostic may explain03331 but cannot turn it
into an equivalent-to-XLA timing result. Stop attribution if the same-setting
full-record comparison fails; preserve all records for the next repair.
