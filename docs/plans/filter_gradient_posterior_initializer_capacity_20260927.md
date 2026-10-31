# Complete posterior initializer reuse and ownership capacity

This follows the public correctness qualification through 04488 and the matched
cost cohort starting at 04490. Execute it after that cohort passes its numerical
and provenance gates. It addresses the remaining twenty-call and native-memory
question for this exported endpoint, not whole-repository capacity.

The question is whether completed calls with the same target reuse their
compiled owner without continuing allocation growth, and what survives when
successful owners are replaced. The existing compiler-error collection test
cannot answer successful-execution retention. In the first cost repeat, the
frozen D3 reference grows from about 2689 MiB RSS after its first call to
5544 MiB after five; the graph candidate stays near 3208 MiB. These observations
motivate longer measurement but do not identify a leak or a native allocator.
The full repeated cost analysis must supersede these preliminary numbers.

Use the identical identifiable Gaussian, seeds, configuration, factor_max=1,
D1/D3 and complete exported payload of test_filter_repair_posterior_initializer_cost.py.
Compare the isolated Git 031692a0b prior, public graph control and default XLA in fresh processes on
explicit CPU reference and a preflight-qualified compute GPU. No numerical
runtime edit, random-stream change, tolerance change or target tuning is part
of this unit. The graph control retains compiled dependencies and cannot serve
as an all-non-XLA authority.

1. For prior, graph and XLA separately, execute a cold original input, one changed
   start/scale, then twenty alternating warm calls. Synchronize and materialize
   complete outputs. Compare every warm result with its matching first result;
   compare both distinct results with the frozen complete reference and the
   independent Gaussian center/covariance after all memory measurements.
2. For XLA separately, replace the target receiver four times while retaining
   identical numerical functions and settings. Execute each complete call,
   check full-result replay, remove all diagnostic strong references, collect
   Python garbage and test weak references to the superseded callback, owner,
   compiled function and dependency scope. Finally clear the last owner and
   record collection and native memory independently. A surviving native
   allocation is not proof that the Python owner remains live.
3. Include an observer-only control on each backend with the same snapshots
   and retained record count but no intervening numerical call. This bounds
   observer-induced growth; it cannot establish compiler or allocator causes.

Record prepared/cold/changed stages and calls 1, 5, 10, 15 and 20 for reuse;
record every owner replacement before and after collection. Save elapsed time,
RSS/HWM, mapping counts and grouped executable/anonymous resident mappings,
TensorFlow allocator current/peak bytes, host load, actual device UUID and
sampled sharing. Keep driver reservation separate from live tensor allocation.
Measure before executing reference comparisons or requesting HLO. Save measured
records before later assertions so failures remain diagnosable. After timing,
require one retained-owner trace, unchanged HLO across the two operands and no
host callback operations. Record the compiled numerical dependency identities.

Correctness, original full records, independent Gaussian checks, source/input
identity, device provenance and Python owner bounds are hard gates. A mismatch,
source drift or compiler failure stops the affected cohort and triggers a
bounded root-cause diagnostic; it does not earn performance ratios. Existing
20% warm, 2x device-peak and 256 MiB/2x RSS thresholds remain attribution triggers.
Continuing warm-call growth after observer attribution is also a repair trigger.
Report observed traces and memory trajectories directly: this finite workload
cannot prove global leak freedom, a hard memory cap or native executable eviction.

Reserve at most 7,200 combined charged seconds and twenty workers within the
existing 56 CPU / 52 GPU-hour totals: sixteen numerical workers, two observer
controls, and two short analysis/policy workers. The frozen prior's observed D3
warm cost is about 21 seconds per call, so reserve a 900-second deadline for each
prior reuse worker, except the D3 GPU prior, which needs 1200 seconds. D3 GPU
successful-owner replacement uses 900 seconds; other candidate and observer
workers use 300 seconds. Localized
harness or infrastructure failures count against this unit. Repartition attempts
only with a recorded repair and without increasing its time/global caps. Use
the existing campaign runner and unique run directories; never run concurrent
campaign workers. No package or environment changes are authorized here.

Deadline refinement from cost worker 04511: the D3 GPU prior passed complete
records in 271.201 seconds, with a 71.587-second cold call and 47.386--47.716
seconds per warm call. Twenty warm calls plus cold/changed calls therefore
cannot fit the earlier 900-second ceiling. After freezing the completed cost
cohort, add a 1200-second runner option restricted to this exact registered
capacity group. Other groups retain their existing limits. This preserves the
twenty-call question, source/inputs and the 7200-second actual-charge unit cap;
it adds no global hours and does not change the current cost cohort's limit.
Cost worker 04513 then measured a 64.226-second D3 GPU XLA cold call. Four
independent successful owners plus the frozen reference can exceed 300 seconds,
so use the existing 900-second option for that one replacement group. Neither
adjustment increases the unit's actual-charge cap; stop before a launch whose
reservation would exceed the remaining allocation.

Final cost analysis through 04525 finds graph-control GPU peak ratios 2.237/2.295
for D1/D3, triggering the existing attribution rule. Actual peaks are
1,236,480/1,302,272 bytes, versus prior 552,704/567,552 and default XLA
185,600/383,744 bytes. Add graph control to the twenty-call CPU/GPU reuse matrix
within the unchanged 7200-second unit; the four extra workers account for the
twenty-worker count above. Record peak/current for every call, completed tensor
payload size and declared nested compilation settings. Separate transient peaks
from retained current bytes, and check whether either grows during replay.
For the native candidate, capture memory immediately after the synchronized
owner return and after completed host formatting using a diagnostic wrapper
that calls the identical owner exactly once. An uninstrumented first call and
unchanged complete records control the observer. The wrapper must not inspect
intermediate numerical values or change their lifetime beyond the existing
returned object. If this only localizes the trigger to the numerical graph,
report that limit; it does not prove which allocator operation caused it.

Skeptical review: capture before and after collection without keeping the
callbacks alive in diagnostic closures. Weak-reference checks must cover a
successfully compiled target, not an unsupported-operation failure. Compiler IR
queries and frozen-reference execution must follow the memory window. Alternate
changed values under one fixed shape; changing configurations would answer a
different cache question. The prior may retain native executables even after
its Python factories return; measure that rather than assuming a cache cause.
Twenty calls and four receiver replacements are bounded probes, not realistic
D23/DZ5 capacity or factor_max=2 qualification. Those actual-consumer checks,
the remaining reporting/precision findings and all F01--F20 terminal decisions
remain separate. This is a local skeptical review; no independent review is
claimed.

Failure 04536 stops the CPU sequence: the frozen D3 prior terminates with
SIGSEGV after 243.995367 seconds inside its original quadratic-geometry fit.
No capacity JSON or JUnit survives, so neither the exact failing call nor its
resource state is established. Preserve that failure and all prior passed
candidate results. The 900-second timeout did not fire. Do not rerun the full
matrix or call this a numerical mismatch, OOM, or mapping-limit failure yet.

Run one diagnostic-only CPU reproduction under the existing GDB wrapper,
bounded at 900 seconds. A new test-only wrapper invokes the identical prior-D3
test and records append-only JSONL before/after every public call, including
RSS/HWM, process-map count, glibc mallinfo2 fields when available, completed
call number, result digest, and replay/Gaussian validity of completed calls.
Snapshot each completed call before starting the next. Load TensorFlow symbols
for the native stack without fetching symbols or modifying the environment.
This instrumentation diagnoses the crash; its timings cannot replace the
uninstrumented cost cohort. No runtime, frozen reference, fixture, threshold,
or completed candidate harness changes. Preserve its wrapper and runner source
in a fresh versioned snapshot; keep the pre-diagnostic source freeze available.

Repartition the same 7200-second allocation to at most 24 workers, counting
failed 04536 and up to three short/localization additions. Global caps remain
unchanged. Stop after the first native backtrace/resource capture and inspect
the evidence before any retry or extension. Only resume unrelated capacity
arms after the failure's scope is understood; no successful replacement of a
failed prior is required to establish the candidate's numerical validity, but
missing prior capacity cannot support a before/after capacity ratio. The
terminal analyzer must preserve invalid/missing arms rather than silently
dropping them. Skeptical review: the GDB diagnostic can alter timing and map
layout, so compare the actual stack and sequence with 04536 and avoid claiming
an exact native allocation owner from an RSS slope alone.

04537 reproduces the fault on call eleven. Ten completed calls pass replay and
independent Gaussian checks; map counts rise from 13,914 after call one to
63,459 after call ten, about 5,505 per call, against vm.max_map_count=65,530.
The log emits LLVM "Cannot allocate memory", then defunct JIT resources and
SIGSEGV in an ORC IRCompileLayer/JitCompiler thread. This establishes a native
compilation allocation failure; the mapping limit is strongly supported but
the failed system allocation itself was not captured. The external map observer
started after the inferior exited and contributes no peak measurement. Do not
raise system limits, change packages or mutate global caches to make the frozen
reference pass. Preserve both 04536 and 04537; no further prior-D3 CPU retry.

Resume the remaining candidate/owner and GPU cases under unchanged fixtures.
The prior CPU D3 capacity cell remains explicitly failed and cannot contribute
a capacity ratio; each candidate still compares two complete frozen original
calls after its own twenty-call window. Continue reporting successful and failed
cells separately. The analyzer must accept a complete disposition matrix with
explicit failed cells, never fabricate a missing numerical record or ignore its
failed JUnit. The GDB group is explanatory and remains outside the matrix.

Source review across this diagnostic boundary: all numerical runtime, ordinary
capacity harness and its imported numerical/reference/observer dependencies
remain byte-identical. Only the runner's GDB registration, a new diagnostic
test and the post-run analyzer/tests change. The analyzer shall compare all
recorded source hashes except these five explicitly listed reporting/diagnostic
files, emit every difference, and still require exact environment/fixture
identity. This is an explicit provenance refinement for descriptive capacity
dispositions, not permission to mix changed numerical sources or to revise the
already sealed cost cohort. The five paths are run_filter_repair_campaign.py,
analyze_filter_repair_posterior_initializer_capacity.py under scripts, and
test_filter_repair_campaign.py,
test_filter_repair_posterior_initializer_capacity_analysis.py,
test_filter_repair_posterior_capacity_crash_diagnostic.py under tests. A difference
in any other source remains a veto. Requalify corruption/source checks before
resuming; preserve both source snapshots and all failed charges.

Terminal-role review on September 28 found that the newly registered frozen
prior capacity groups inherited mandatory-current-test status. That would
incorrectly require rerunning the retired crashing implementation to pass before
merging a repaired candidate. Classify all four prior capacity groups as
explanatory comparators, independently of whether they pass. Keep candidate
graph/XLA reuse, successful owner collection, observer controls and analyzer
checks mandatory. Add a regression that prevents a baseline from replacing
those gates. The complete capacity analysis still requires every matrix cell,
preserves prior failures, and forbids their use in capacity ratios. This changes
test-role accounting only; it does not qualify the failed prior or weaken any
current numerical/ownership criterion. Renew the focused analyzer/runner checks
after the numerical matrix within the existing 24-worker allowance.

The terminal analyzer also requires identical recorded hashes for the full
690-file frozen reference closure, all 22 per-call memory observations, all 21
instrumented candidate return observations, and every successful-owner collection
stage. Reject invalid allocator bounds or device/memory mismatches before using
those observations. Save this expanded report as analysis schema v3; preserve
the earlier CPU v2 analysis and its executed source snapshot. These checks do
not change the numerical workers or their measured records.
