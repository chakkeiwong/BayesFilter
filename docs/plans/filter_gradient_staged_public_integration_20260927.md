# Public staged locator execution repair

The public `locate_joint_center_staged` still constructs fresh closures and
performs numerical eligibility, replay selection and reporting outside the
compiled stages. The internal `StagedJointCenterProgram` already has complete
original-record and CPU/GPU cost evidence in the September 23 staged unit.
Wire the public endpoint to that authority, with starts and scales as explicit
operands and reuse for one callback/configuration/dimension/device signature.
No optimizer, objective, derivative, tolerance, tie rule or iteration cap changes.

The reference is original commit `3582b4ac` and its frozen import closure.
GPU reference accounting uses only the four documented int64 substitutions
from the internal unit. Preserve the complete public records and exact target
call order on original, changed and replayed inputs. Validator acceptance,
rejection and exceptions must retain one host call at the checkpoint boundary;
continuation receives the complete saved optimizer/accounting state.

The default is two XLA stages separated by that external validator. It is not
one enclosing XLA function. Explicit graph diagnostics use the same stages.
The existing `max_wall_seconds` route remains an explicitly non-JIT diagnostic,
because a host clock cannot be a numerical XLA dependency. Label its private
legacy implementation accordingly and prevent ordinary calls from selecting it.

Execution failure semantics are explicit: target validation or graph construction
failures already represented by the native stages retain those records. A real
TensorFlow compiler/device execution error propagates to the caller, before the
validator or the next stage can run. Unlike the legacy controller, it cannot
manufacture a completed optimizer-exception record from partially executed
resource counters, or retry eagerly. This is a deliberate fail-closed error
boundary; numerical-success behavior must remain unchanged. Invalid callback
operations rejected by the common callback validator remain errors.

Use a single retained owner, comparing callback identity rather than callable
equality. Key device placement and complete static configuration; starts/scales
are never cache keys. Hold the existing reentrant invocation lock across both
stages and the host validator. Scope callback-dependent program construction
and lazy tracing to the owner, excluding the validator's unrelated work.
Replacing/clearing the owner must release its Python graphs and callback;
this does not claim native executable eviction or a general memory bound.

Reserve up to 20 CPU and 16 GPU workers, 7,200 combined charged seconds, within
the existing 56 CPU / 52 GPU-hour caps. Use registered runner groups and fresh
numbered outputs, 300-second focused limits with a recorded 900-second retry
only for a demonstrated compile-time capacity issue. CPU checks precede GPU
qualification. Follow the existing GPU selector and growth policy.

Acceptance requires public-path original-record/call-order comparisons at D1
and D3, existing staged consumer assertions through the actual export, unchanged
frozen derivative boundaries, stable trace/HLO across changed operands, exact
replay, cache replacement/collection, nested validator behavior, failed native
execution with no fallback, configuration validation, and policy checks.
Reuse the qualified internal numerical cases; add checks only for this public
wiring, ownership and failure contract. Fresh before/after cost workers must
include public construction/materialization and keep HLO inspection outside the
measured interval. Two extents and graph/XLA memory observations remain
descriptive; original numerical failures veto speed conclusions.

Public cost comparisons add an original graph-mode record for the graph arm,
while saving the original XLA records alongside it. The earlier CPU graph/XLA
tie-selection differences are preserved, not waived or redefined as equal.
The unchanged complete-record gate applies within each execution mode. The
original public/XLA arm remains the before cost authority; graph/XLA memory
observations cannot establish a speed ranking across different selected records.

Skeptical review: global reuse can retain old callbacks or choose a program for
the wrong device; mutable start values must remain operands. Activating an
owner scope during the external validator could retain unrelated graphs, so
limit it to construction and stage calls. A successful trace does not certify
an XLA execution. A Python cache bound does not bound the native compiler cache.
Failure records cannot claim target counts after an incomplete device operation.
These are implementation tests, not HMC, posterior, canonical LEDH or terminal
F01--F20 admission. Preserve any discrepancy, stop the affected adoption, and
localize it without changing the original comparison criteria.

CPU 04265--04270 passes public boundary, all ten original-record cases and all
seven existing consumer assertions. In 04271, both real compiler-error cases
pass and nested numerical records/call order match; the final trace assertion
mistakenly inspects a separately constructed, unused internal owner. Repair the
harness to observe the cached public owner. Preserve the failed artifact and
rerun the affected three checks within the existing reservation; no runtime,
comparison criterion or original record changes.

04272 passes all three affected checks. The CPU unit now has 23 distinct
correctness cases. A static production-call search finds no internal caller
beyond the exported API; the external MacroFinance staged-continuation script
calls it with `adapter.log_prob_and_grad`. Its actual target remains unqualified.
Python creates a fresh bound-method object for each attribute access. Extend
cache identity to recognize the same method function and receiver by identity,
without invoking user-defined equality or equating different receivers. Add a
focused repeated-public-call/collection test after the frozen CPU cost cohort.
The resulting cache branch must preserve records, one trace and frozen inputs;
it does not grant the external target numerical or execution-policy status.
The inspected external script's `staged_config()` explicitly sets
`jit_compile=False` and `max_wall_seconds=WALL_CAP_SECONDS`, so it continues
through the documented host-clock diagnostic. Migrating that external consumer
requires its parent deadline and actual target checks; public wiring cannot
silently relabel it as XLA.
