# Public ordered-block XLA integration

The public `locate_block_coordinate_center` endpoint now invokes the qualified
ordered-block controller once. Its numerical Gauss-Seidel recurrence, exact
replays, transaction decisions, reversal/cycle checks and summaries execute
inside XLA. Configuration/input checks precede construction; completed records
and progress events are formatted after execution. The public signature and
prospective row cap are preserved. Buffered progress cannot enforce a live
deadline; the actual DZ5 consumer still requires E5 independent supervision.

| Qualification | CPU | GPU |
| --- | --- | --- |
| Full original records, scalar/batch, coupled/reversal/partial heterogeneous blocks, changed inputs and target order |6 pass03180--03185|6 pass03213--03218|
| Validation before target use, unsupported callbacks, frozen derivatives, nonfinite replay and row accounting |7 pass03186|7 pass03219|
| Existing consumers and helper comparisons, including migrated native-boundary injections |43 pass03187--03199|43 pass03220--03232|
| Multistart, factor-one/two with score reuse, paired holdout, scaled/orthogonal search and disabled score gating |6 pass03202--03207|6 pass03233--03238|
| Approved resolution error and controlled no-use checks through the public endpoint |4 pass03208--03211; controlled renewal03245--03246|4 pass03239--03240/03243--03244|
| Exact-source policy |129 pass03247|Same static guard|

Original3582b4ac remains the numerical authority. Factor records use the already
approved factor-schema comparator, including its explicit `jit_compile=True`
metadata check and limited factor tolerances; all other numerical comparisons
retain1e-10 and all discrete fields remain exact. Partial-block CPU changed
inputs that trigger the approved resolution error must preserve accounting and
diagnostics and perform neither replay nor later blocks. Unflagged cases still
require full original records and target order.

The six public fixtures reuse one owner across changed centers/scales with one
trace and unchanged normalized HLO. They record every target point and verify
that all progress callbacks observe the final call count. Graphs contain native
While/Case and no PyFunc. Seven existing test fixtures now inject completed
tensor results at the conditional-program boundary; the real compiled sweep
makes every transaction/policy decision. The nested-history fixture separately
injects completed report fields to retain its recursive removal assertions.

The source guard now covers the entire public block module. Its one new exact
exception sums static configured maximum row counts before execution. No new
numerical-loop or NumPy exception is present. Current partial guard coverage is
227 sources/1333 exact exceptions; this does not close the repository audit.

Preserved harness failures:03179 used runtime-valued Range endpoints in the new
target recorder; fixed-length indices plus runtime offset repaired compilation.
03201 compared the new factor `jit_compile` metadata with a schema that predates
it. Exhaustive inspection found no other field/count/order difference. The
existing schema-specific comparator repaired that harness without changing
runtime, numerical tolerances or fixtures.03178's initial boundary pass and
03200's multistart pass remain supporting evidence before those harness fixes.

03241 preserves a GPU backend failure in the controlled scalar injection. Its
standalone conditional controller passes the error/no-mass assertions, but the
public enclosing sweep fails while adding a CUDA graph kernel node with
CUDA_ERROR_INVALID_VALUE. No numerical comparison or no-use assertion was
waived. Repeat the exact source/environment case once under the existing retry
budget to distinguish a reproducible compiler/runtime failure from a transient
launch failure;03242 reproduced it. The test injector now selects only the six
altered fields with the same runtime arm variable.03243 passes the complete
scalar GPU error/no-use and healthy-reuse assertions. Runtime source, target
arithmetic, XLA defaults and assertions are unchanged. Batched GPU03244 and
both CPU renewals03245--03246 pass; this does not repair the backend's general record-wide
conditional limitation.

Review: public integration passes all66 CPU and66 GPU checks, with129 policy
checks. Matched public ordered-block costs and final campaign gates remain open. Startup/native
residency and arbitrary signature churn are not resolved by Python owner
collection or these finite fixture families. E2 initializer parity, E5 actual
DZ5 migration, F01--F20 dispositions and remote integration still prevent a main
merge. This unit reserves72 workers/14,400 charged seconds within the unchanged
32 CPU/52 GPU-hour campaign caps, with one numerical worker and source freeze
during each matrix.

The unit closes at70 workers/2233.332018 seconds, including all four preserved
failed attempts. The receipt `block-public-qualification-checkpoint-03247.json`
checks every manifest and confirms unchanged numerical package sources since
03180. Later test-harness additions and the isolated injector repair are explicit;
final E6 qualification still requires the terminal frozen source. Cumulative
charges are63,689.397439 CPU/63,154.650998 GPU seconds. No worker is active.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain public XLA sweep | Complete original records, events, accounting, changed inputs and failure checks pass on CPU/GPU | No unresolved numerical failure in this unit; synthetic record-wide conditional backend failure preserved | Matched costs and native lifetime | Run reviewed E3/E6 cost unit | No unbounded lifetime or broad performance claim |
| Continue campaign | E2--E6 work remains explicitly mapped | Initializer parity and terminal gates remain open | Native retention and actual DZ5 consumer | E2 coefficient attribution, E4/E5 repairs, E6 audit/integration | Main merge remains blocked |
