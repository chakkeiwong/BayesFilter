# Filter execution repair recovery, September 19

Worktree: `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`.
Branch: `repair/filter-gradient-xla-validation-20260918`.
Master: [repair program](filter_gradient_repair_master_20260917.md).
Detailed evidence: [execution record](filter_gradient_repair_execution_20260917.md).

September 20 current checkpoint through 01486: scalar and batched L-BFGS
localization, endpoint checks, ordered scalar exact replay and stable selection
execute in enclosing XLA programs. Both preserve the original frozen public
derivative boundary. Optional batched objective progress is buffered and delivered
after execution, with explicit overflow failure and a four-program cache.
External host-mutating DZ5 target callbacks remain unqualified.

The final combined locator suite passes all 33 CPU/GPU cases (01456/01457).
Current-source GPU consumers pass 20 preparation, 40 sequential and 43 block
cases (01483--01485); 63 policy/controller checks pass (01482). The guard passes
for 189 sources / 1,273 exact exceptions, with no new numerical-loop exception.
Inventory 01486 finds 2,869 working-tree Python files, 2,868 parsed and one
unchanged external legacy parse error. Focused new-file Ruff and whitespace pass.

Current-source two-extent measurements have exact baseline outputs:
unbuffered batched public warm time is 276.042/3.994 and 277.522/4.337 ms;
buffered public time is 279.489/6.988 and 294.315/8.694 ms; scalar public time is
523.036/5.543 and 999.783/9.013 ms before/after. Graphs remain 2,367 unbuffered,
3,010 buffered and 2,191 scalar nodes at both extents. Candidate warm device
allocation is constant. Public cold calls and host peaks increase; retain those
costs. Buffered public device peaks of 1,023,488/2,097,408 bytes trigger the 2x
investigation. Analyses are the locator diagnostic artifacts ending in
01463, 01469 and 01481. These are single-process observations; three-process
terminal comparisons remain required.

Run 01422 shows that residual correction improves least-squares accuracy but
does not repair complete CPU fitter records. No runtime correction is installed.
Next isolate projection/inversion/encoding and measure optional trace capacity
costs under the bounded contracts below, then continue structured fit preparation,
outer sequential refinement, block-coordinate and quadratic numerical control.
The uncovered batched quadratic initializer and external DZ5 callbacks remain
in that queue. All F01--F20 terminal dispositions and earlier memory/time
investigations stay open. No merge, HMC or scientific admission is established.

Through 01486, charges are 25,008.772 GPU / 33,922.904 CPU seconds, leaving
162,191.228 GPU / 81,277.096 CPU seconds under the unchanged 52/32 process-hour
caps. No worker is active at this checkpoint. Remote refs were fetched;
origin/main is an ancestor of dcfaa15d, with no divergence to resolve. Preserve
this tested locator checkpoint on the repair branch before the next diagnostics.

The older checkpoints below are historical context.

September 20 continuation through 01421, following pushed checkpoint `d91a1268`:
ordered scalar multistart L-BFGS, endpoint replay, eligible exact replay and
selection now execute in one native XLA program. Original optimizer settings,
smooth-box transform, scalar authority, first ties and target-call order/counts
are preserved. Unsupported host callbacks fail without an eager retry. The
batched locator and outer refinement remain explicitly open.

All 14 focused CPU/GPU locator cases pass (01401/01402), as do 40 sequential,
43 block-center and 12 factor GPU consumer cases (01403--01405), and 63 policy/
controller cases (01420). Current two-/four-start measurements 01408--01419 have
exactly equal outputs. Warm public times are 482.254/5.516 and 961.428/9.151 ms
before/after; graph/XLA times are 51.017/3.884 and 103.620/7.403 ms. Graphs remain
2,181 nodes at both extents; warm device allocation is constant and late host
growth is at most 12,288 bytes. No new trigger fires. Public host peaks rise by
143--146 MiB and cold calls are slower; these costs remain in the analysis.
`sequential-scalar-locator-diagnostic-01419.json` is single-process checkpoint
evidence, not terminal acceptance or a statistically supported timing ranking.

Through 01421 charged time is 23,288.629 GPU / 33,454.492 CPU seconds, leaving
163,911.371 GPU / 81,745.508 CPU seconds under the unchanged caps. No worker is
active at this checkpoint. The partial guard covers 188 sources / 1,273 exact
exceptions. Inventory 01421 finds 2,863 working-tree Python files, 2,862 parsed
and one unchanged external legacy error (2,859 tracked plus four new files).
Focused new-file Ruff and whitespace pass. Remote refs were fetched; main was
an ancestor of `d91a1268`, with no remote divergence to resolve at that point.

Next localize the existing CPU fitter initializer rounding, then continue
batched/outer sequential, block and quadratic control. The explicit bounded
initializer diagnostic and locator measurement contracts are in the master.
External DZ5 callbacks, CPU original-record parity and previous memory/timing
investigations remain open. No F01--F20 terminal disposition is closed. Source-
frozen repeats, full integration/retest and terminal review still gate merge.

Earlier recovery checkpoints follow for historical context.

September 20 continuation through 01399, on fixed-fitting checkpoint `8f334b96`:
initial exact replay and seeded search selection now execute in enclosing native
XLA programs. Eligibility, first-maximum ties, row order, seeded draws and gather
pullbacks are preserved. Optional movement reporting is repaired; its legacy
zero-cloud fixture now enters at the native TensorFlow boundary. All 20 focused
CPU/GPU selection checks, 40 sequential GPU, 43 block-center GPU, 12 factor GPU
and 63 policy/controller cases pass (01344/01346--01348/01373/01398).

Corrected matched comparisons 01374--01397 pass at two extents. Public warm
replay is 4.676/0.408 and 9.087/0.410 ms before/after; search selection is
11.250/0.702 and 16.815/0.800 ms. Maximum compared error is 4.441e-16. Graph
sizes remain 93 replay / 346 search nodes at both extents, with constant warm
device allocation and late candidate host growth at most 12,288 bytes. No new
memory/time trigger fires. These are single-process checkpoint observations;
final three-process evidence remains required. Earlier 01349--01372 measurements
are superseded because the extracted baseline did extra tensor packing.

Current charged time: 22,702.050 GPU / 33,316.018 CPU seconds, leaving
164,497.950 GPU / 81,883.982 CPU seconds under the unchanged 52/32 process-hour
caps. No worker is active at this checkpoint. The guard covers 187 sources /
1,273 exact exceptions and remains partial. Inventory 01399 discovers 2,859
working-tree Python files (2,858 parsed and one unchanged external legacy error;
its tracked-file count of 2,857 excludes the two new files).

Next compile ordered scalar locator starts, then batched locator and outer
refinement/block/quadratic control, under the reviewed plan below. Include the
uncovered `batched_quadratic_center.py` chunk/round loops and non-XLA fit in that
audit. External DZ5 callbacks, CPU fitter parity (01325), and all existing
memory/timing investigations stay open. All F01--F20 terminal dispositions remain
open; no merge, HMC admission or whole-repository compliance is established.

Earlier recovery checkpoints follow for historical context.

September 20 continuation through 01339, based on selector checkpoint `22094f76`:
complete fixed-center replicate fitting, conditional factor escalation, selection,
and audit now compile. Refined XLA eigenpairs preserve the original GPU records;
a shared L-BFGS loss/gradient graph bounds repeated tracing. The original frozen
geometry derivative boundary is restored inside the program; both fitters now
work under an outer tape without exposing unsupported optimizer derivatives.

All 14 GPU fitter cases (01330), 223 affected GPU consumers (01331), and 63
policy/controller checks (01332) pass. Full CPU run 01325 passes 13 but fails
one original-record case: Jacobian condition and principal-angle discrepancies
slightly exceed their unchanged tolerance. Localization 01326/01327/01329 traces
this to initializer rounding amplified near saturated factor loadings. Direct
Jacobian diagnostics agree at identical state; callback sharing and explicit
initialization stage barriers do not cause or repair it. No runtime barrier or
input-specific initializer has been added. Keep CPU parity open.

Fresh two-extent comparisons 01333--01338 reuse valid original baseline runs
01303/01318 after provenance checks. Maximum numerical error is 8.308e-11.
Public warm seconds are 13.533/0.127 and 27.529/0.268 before/after; graph/XLA
seconds are 1.335/0.105 and 2.669/0.224. Native graph sizes stay at 5,341/7,395,
with stable warmed allocation. Public host peaks rise by about 0.47--0.49 GiB,
still an investigation trigger. The graph/XLA pairs remain below 256 MiB added
host peak. Analysis: `fixed-fitting-diagnostic-01338.json`. These are single-
process observations, not final acceptance. Prior failed attempts are preserved.

The partial guard covers 186 sources / 1,273 exact exceptions; no numerical
loop exemption was added. Syntax inventory 01339 discovers 2,857 Python files,
2,856 parsed and the unchanged external legacy parse error. Focused new/runtime
Ruff and whitespace pass; the driver's four pre-existing style warnings and its
test module's two pre-existing C408 warnings remain unrelated cleanup debt.

Next: preserve this checkpoint and continue the compiled sequential candidate
replay/search-selection dependency described in the master plan, then enclosing
sequential/block/quadratic control and external callbacks. Draft only:
`/tmp/sequential_selection_tf_draft.py` is not part of runtime or evidence.
The CPU full-record sensitivity, fixed-fitting host memory, mass timing,
exact-selector overhead, TT assembly and forecast-pool investigations remain
open. All F01--F20 terminal decisions stay open; main remains gated.

Artifacts remain under the primary checkout's
`docs/plans/artifacts/filter-gradient-repair-20260917/`.

The user authorized execution repairs, preserved numerical algorithms and
tolerances, before/after comparisons, and commit/push. Main must remain gated
until full testing. Canonical LEDH rebuilding is excluded; unsupported claims
remain blocked. Only the two geometry initializers have approval to migrate
their random stream. Other seeded draws remain unchanged.

September 19 owner authorization adds 48 GPU / 24 CPU process-hours to the
original 4 GPU / 8 CPU caps. Active cumulative caps are **52 GPU / 32 CPU
hours**; the earlier 16 GPU / 12 CPU proposal is superseded. Through 01339,
charges are 21,725.543 GPU / 33,230.249 CPU seconds, leaving 165,474.457 GPU /
81,969.751 CPU seconds. No further compute approval is needed within these
caps. Use the driver for authoritative accounting, including interrupted runs
and supplemental charge files. No campaign worker remains running at this
checkpoint. Numerical jobs must remain sequential and source must remain fixed
through active tests/measurement matrices.

Latest recovery queue (supersedes historical pending notes below): TP now passes
22 CPU/GPU cases (01173/01174), with its unchanged raw-residual gate and two
converged Richardson estimates at the original derivative tolerance. The old
1e-5 stencil fails equally in both GPU source arms and remains a diagnostic.
Complete symmetric score fitting passes 17 CPU/GPU checks (01178/01197).
GPU consumers pass 40 sequential, 43 block, 26 joint, 33 quadratic-initializer
and five posterior-initializer cases (01179/01182--01185). The selector then
received bulk host transport; all 19 focused CPU/GPU cases pass (01196/01198).
The partial guard covers 180 sources / 1,189 exact exceptions; 61 policy checks
pass in 01199. New-file Ruff and whitespace pass; the existing sequential
outer-loop B023 warning remains part of the open lifecycle migration.

The new registered measurements are `exact_incumbent` and
`sequential_score_fit`, each with graph/XLA kernels and matched public endpoints.
First small public score-fit pair: 19.008/2.302 ms, error <=8.882e-16, no new
memory trigger. Selector public pair: 0.610/38.968 ms, repaired to 2.437 ms by
bulk scalar transport, exact outputs. Its timing and 4.25/25.5 KiB device-peak
triggers remain open. Full XLA selector time is 0.473 ms; do not compare that
kernel timing with the public baseline. Preserved descriptive analysis:
`inference-preparation-diagnostic-01195.json`. Larger extents, causal overhead
localization and final three-process pairs remain pending.

Continue the current master queue: outer sequential search/fitting and ordered
block/quadratic lifecycles; fixed-center and block-score numerical control;
eager mass-matrix construction reached by initializers; external DZ5 callback
compatibility; TT/forecast-pool/selector memory and timing investigations;
coverage/dispositions and final frozen-source comparisons. F01--F20 remain
open. No merge or whole-repository compliance is established by this checkpoint.

Use the existing approved command prefix:

```text
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py
```

GPU3 is the same-class contention alternative under the existing plan. The
driver checks contention, sets and verifies memory growth, hides GPUs for CPU
diagnostics, enforces timeouts and shares accounting/locking across worktrees.
Do not bypass the driver or launch parallel numerical workers.

Checkpoint `c4351f8c` is committed and pushed. Its shared tensor-program helper
builds full pullback graphs only when requested, including coefficients used
only in custom derivatives. Preparation sampling/push/resampling, coordinate
clipping and target shifting now have stable XLA boundaries. Review restored
route/time/shape/nonfinite vetoes before qualification.

The helper passes 22 CPU and GPU checks; six public pullback checks and all 11
sequential consumer checks pass on GPU. Preparation passes 46 checks on CPU
and GPU. The policy/controller group passes 59 checks. These are focused
results, not all-repository coverage.

The two-/four-date public endpoint measurements 01019/01018 preserve prior
candidate outputs exactly and match the frozen baseline within 8.882e-16.
Compared with the prior candidate, single-process host maxima fall by
245.4/409.8 MiB; GPU allocator peaks are unchanged. The four-date candidate
still exceeds original baseline host memory by 362.6 MiB. Harness review
reconstructed the old hash by reversing only an equivalent dictionary syntax
change. The descriptive analysis is `lazy-pullback-diagnostic-01019.json`;
old artifacts remain stale for terminal comparisons. No three-process result
or isolated causal memory attribution is established.

Full preparation run 01020 was stopped at 854.417 seconds after the first 45
checks, with 27.67 GiB RSS observed at 12:01. The first 36-dimensional P59
assembly test had not finished. Run 01024's 45-second stack localizes fitting
update tracing; the run times out at 120.520 seconds.

The next fitter patch defers existing accepted-update derivative construction,
preserving the fixed-design packed-core/target derivative and rejection rule.
All six focused CPU cases, all 37 existing fitting checks (01031), and all 12
scalar adjacent-TT consumer checks (01032) pass. GPU run 01030 times out after
five progress markers without JUnit; it remains incomplete. Run 01028 also
preserves an external-tape TensorList boundary failure affecting the prior
source. Passing tests enclose value and gradient as the actual scalar consumer
does; they do not establish a new external-tape fitter API.

Run 01033 times out at 301.873 seconds. Its 45-second stack has passed the
initial fit and reached model simulation; RSS snapshots are 6.45 GiB at 01:27
and 10.53 GiB at 04:20. The large consumer still has no passing outcome or
final peak. Capture a later stage/stack before another broad retry.

Run 01034 passes all six final CPU fitter checks after the closure edit;
01035 passes all 59 policy/controller checks. The static guard passes for
171 sources with 1,097 exact schema/reference exceptions; coverage is partial.
Focused Ruff and whitespace checks pass.

The coordinate checkpoint `191b9ab6` includes the
budget/watchdog and shared TTSIRT coordinate changes. Run 01038 localizes later graph
growth to retained-sample transport after fitting. The new masked marginal
and coordinate program preserves the grid-CDF extension and total pullbacks;
01041/01043 pass all nine checks on CPU/GPU. Run 01042 passes all six deferred
fitter checks on GPU, closing that focused qualification gap. The current
static guard passes 172 sources / 1,107 exact exceptions, with partial coverage.

Run 01044 completes its assembly assertions in 292.905 seconds (one passing
JUnit case), but the process exceeds the 300-second deadline. The 270-second
RSS high water is 12.50 GiB; final process high-water reporting was added to
the localization worker for the next attempt. Retry this one assembly test
under the existing 900-second ceiling. Runs 01046--01048 pass 6/11/25 public
pullback, sequential and TTSIRT checks on CPU; their GPU qualification is
pending because both GPU2 and GPU3 had unrelated active work.

Uniform log weights and weighted target means now use stable XLA helpers;
all 60 preparation checks pass on CPU (01045). Ten repaired preparation
wrappers were added to the static guard without new exceptions. The policy
group passes all 59 checks (01049). Finish the remaining preparation/callback
audit before freezing source/harness for terminal repeats. Avoid repeating
the broad preparation group until the host-memory regression is localized.

Run 01050 now completes the one-test assembly and exits in 354.963 seconds;
its final process high-water RSS is 16,332,076 KiB (15.58 GiB). The identical
baseline CPU diagnostic 01058 passes in 74.134 seconds at 690,176 KiB
(0.658 GiB). This descriptive single pair triggers graph/cache investigation;
it does not establish terminal timing ratios or isolated XLA overhead.

Checkpoint `de363b43` commits and pushes the P72 preparation repairs, which normalize weights and assemble fit/guard
arrays in stable XLA helpers. Line interpolation, exact first-duplicate
selection and gather use separate compiled stages to preserve comparisons of
realized binary64 columns; the original frozen-design derivative boundary is
retained. Failed attempts 01051--01055 are preserved. All 30 CPU checks pass
in 01057; GPU checks remain pending. Both approved GPUs were
occupied at recovery; do not interfere with unrelated work.

The next fitter repair shares graphs for matching rank-one cores while
retaining higher-rank coordinate specialization. Attempts 01061/01063 showed
condition-history rounding drift when heterogeneous/higher-rank coordinates
shared a graph; preserve these failed attempts and the unchanged 1e-10 gate.
Final run 01064 passes all 15 graph/XLA value/history/pullback checks. Graphs
have 1,089/1,273 nodes for four/eight rank-one coordinates. Runs 01065/01066
pass 37 existing fitter and 12 isolated-baseline adjacent-filter checks.

Assembly 01067 passes in 191.789 seconds at 9,939,348 KiB (9.48 GiB) final RSS
high water. This is descriptively below 01050's 354.963 seconds / 15.58 GiB,
but still exceeds the original baseline substantially. The comparison and
manifest checksums are in `assembly-memory-diagnostic-01067.json`. No terminal
or GPU memory claim follows. The 135-second stack points to repeated coordinate
program construction inside sequential transport branches; investigate safe
reuse while preserving captured derivatives and distinct graph/XLA contexts.
Static coverage is now 172 sources / 1,109 exact exceptions, still partial.

Checkpoint `db3959ac` commits and pushes the rank-one fitter repair. Recovered
run 01069 passes all nine coordinate parity/gradient/graph checks in 94.954
process-seconds after removing identical reference-grid Case arms. The owned
grid ignores its axis; no numerical rule or capture behavior changes. The
current guard passes 172 sources / 1,108 exact exceptions. Both GPU2 and GPU3
remain occupied. Next run one bounded assembly memory diagnostic with the
same inputs before further cache work; no runtime edits during that worker.

Assembly 01070 passes in 188.360 seconds at 9,743,236 KiB (9.292 GiB) peak
host RSS. The small change leaves the assembly memory investigation open.
New P72 support, line-prediction and recorded-spectrum calculations have
stable XLA boundaries. All 81 CPU cases pass (01072), including prior P72
tests and complete baseline records, threshold/empty/nonfinite/index cases.
The six 01071 failures were fixture assumptions about nonfinite predictions:
both public versions reject them during provenance hashing; preserve that.
The 173-source / 1,113-exception guard and all 59 controller checks pass
(01077). Four new source scopes are guarded without numerical exemptions.

Fixture `source_guard_gates` is registered for both extents and all three
execution modes. CPU runs 01073--01076 pass graph/XLA kernels and public
baseline/candidate calls, with 11 exactly matching numeric fields. The
descriptive artifact is `source-gate-cpu-diagnostic-01076.json`. Public warm
medians are 8.511/3.287 ms before/after, with about 172 MiB extra candidate
host peak; numerical graph/XLA medians are 1.325/0.744 ms. These single-process
CPU observations do not close GPU or terminal repeats. Both approved GPUs
remain busy. No worker remains running after 01077.

The refreshed syntax inventory is `source-inventory-after-01062.json.gz`:
2,832 Python files, with the only parse error in external vendored legacy code.
P73 renewal uses the line-gate result to select data, so those reductions remain
active numerical preparation work despite the diagnostic script name.

The recovered 01084 worker has exited successfully: all 59 policy/controller
checks pass. The shared rank-one Legendre marginal preserves the original
paired-core recurrence and all captured interval derivatives. Its initial
01078 failure (variant tapes crossing Cond) is repaired by the existing
complete-pullback wrapper around the common polynomial recurrence. Runs
01079--01082 pass 17 coordinate, 25 transport, six public external-pullback
and 11 sequential CPU checks. Static coverage is 173 sources / 1,117 exact
exceptions, explicitly partial. No cross-context cache was introduced.

Assembly 01083 passes in 151.950 process-seconds with 7,330,768 KiB (6.991 GiB)
host high-water RSS. Relative to 01070, these single-process observations
are 19.3% lower elapsed time / 24.8% lower memory; the original baseline is
still 74.134 seconds / 0.658 GiB. `assembly-memory-diagnostic-01083.json`
preserves all five arms and manifest checksums. Remaining basis/mass graphs
and repeated normalizer construction need investigation; terminal memory
acceptance remains open. GPU2/GPU3 are occupied by unrelated work.

The next reachable audit debt includes the NumPy/Python scalar forecast
pool, plus unclassified `cpu_xla_cloud`, `quadratic_map_covariance` and
`block_coordinate_center` public consumers. The forecast pool repair is now implemented; the other three helpers
remain open. Preserve per-row random streams and byte hashes in any
forecast repair; external CPU sample generation is not NeuTra training.

All F01--F20 terminal decisions remain open. Other carried blockers include
the predator-prey residual about 1.578e-9 against the fixed 1e-10 gate,
core-affine/higher-rank slowdown, centered qualification, incomplete callback
coverage, and missing terminal comparisons. The latest earlier comparison
had 18 valid pairs and 840 missing; this is not a current completion result.
No merge, repository-wide policy-compliance or default-readiness claim is justified.

The subsequent basis/mass repair shares exact owned Legendre schemas with
tensor interval endpoints. Run 01088 passes 29 focused cases, including
complete fitter histories/pullbacks with pinned basis helpers in the baseline
arm. Initial fixture binding and unsupported raw-preparation XLA-gradient
assumptions in 01085 were corrected; no tolerance changed. The four/eight-axis
basis graph drops from 390/742 nodes to 174/186. Runs 01089--01093 pass
37 fitter, 38 density, 25 transport, six public pullback and 11 sequential
checks. Run 01095 passes all 59 policy/controller cases. Static coverage is
173 sources / 1,123 exact exceptions, still partial.

Assembly 01094 passes in 112.504 seconds at 5,586,104 KiB (5.327 GiB) host
peak. `assembly-memory-diagnostic-01094.json` preserves six arms and manifest
checksums. These single-process CPU observations are 26.0% lower elapsed time
and 23.8% lower peak than 01083; the 0.658 GiB original baseline gap remains
open. No current GPU qualification or terminal performance claim follows.

Proceed to the CPU forecast-pool repair recorded in the master. Keep the
existing per-row seeds, raw-byte hashes, row ordering and validity vetoes.
Its numerical iteration must run in a native TensorFlow/XLA shard program;
external forecast generation is not NeuTra training. Forecast-pool source is now repaired and tested as recorded below; the other
newly noted inference debt remains open.


Recovery through 01125: GPU basis/mass qualification passes all 29 cases
(01100). Forecast shards pass all 15 CPU cases after preserving float32 scalar
factory conversion to float64 (01101); the process pool passes all three
startup/exact-replay/uneven-shard cases (01099). The static guard now covers
174 sources / 1,144 exact exceptions, still partial. All 61 policy/controller
checks pass in 01121, including rejected missing/contaminated child provenance
and incomplete final process-memory evidence.

New fixtures are `cpu_forecast_shard` (graph, XLA and public scopes) and
`cpu_forecast_pool` (public process scope only). Both are explicitly CPU sample
generation, not NeuTra training. Shard runs 01103--01114 match original values
exactly in XLA/public mode and within 6.107e-16 in graph mode. The legacy
`.numpy()` boundary cannot trace; those failures remain baseline evidence.
Those shard artifacts use an earlier fixture-harness version and need final
remeasurement. Do not mix their timing scope with the public scalar baseline.

Pool run 01115 exposed a spawned-child source-precedence bug in the harness.
It is contaminated and excluded. The dedicated worker now restores the selected
source before unpickling; both child source closures are verified after timing.
Final snapshots include each child's actual RSS maximum even if it did not
receive the last request. Runs 01122--01125 pass both two/five-row pairs exactly.
`forecast-pool-memory-diagnostic-01125.json` preserves their summaries/checksums.
Two-row warm medians are 13.929/4.703 ms before/after; five-row medians are
37.603/8.459 ms. Peak sums increase by 186.965/279.852 MiB. The latter exceeds
the unchanged 256 MiB investigation trigger. Both workers cache both two-/
three-row shard programs, each with one trace; this is a plausible explanation,
not an isolated cause or terminal memory acceptance. No worker is running at
this checkpoint. Continue pending GPU qualification and reachable inference
repairs, then investigate that bounded signature/compiler overhead before
terminal repeats. The full campaign and merge remain incomplete.


Checkpoint `bbfaf742` (forecast shards and comparison harness) is committed and
pushed. Subsequent GPU source guard/preparation qualification passes all 81
cases in 01127 after correcting the legacy GPU-gather test assumption; the
repaired rejection itself already worked. CPU score-cloud NumPy removal passes
all 11 cases in 01129 and all 61 policy/controller checks in 01130. The mutable
variable snapshot defect found in 01128 is repaired. The current static guard
covers 176 sources / 1,157 exceptions, explicitly partial. The numerical CPU
B=1 worker remains unchanged and is not a NeuTra training endpoint.


Recovery through 01143: all six pending GPU groups pass in 01131--01136
(15 fixed-fit pullback, 60 preparation, six public pullback, 11 sequential,
25 transport and 17 coordinate checks). The CPU score-cloud checkpoint
`a44250a0` is committed and pushed.

Quadratic-map preparation now uses TensorFlow snapshots and stable XLA
callbacks/score/precision/centeredness kernels, with the joint locator's JIT
default restored. Run 01137 exposes a default XLA eigen precision loss and
two inherited seed-specific failures. Explicit binary64 Jacobi precision
repairs the eigen comparison. Run 01138 proves the pinned wrapper shares the
current-cloud rejection statuses. Run 01139 passes all 33 CPU cases: original
acceptance fixtures use their frozen legacy clouds; separate checks retain
current-stream decisions and iteration parity. No RNG, numerical gate, fit or
refinement budget changed. Iteration across whole geometry fits remains open.

GPU run 01140 passes 32 cases but exposes int32 resource counters pinned to CPU
in `joint_center`. Both normal/staged locators now use int64 accounting resources
and matching caps. The exact GPU reproducer passes in 01141; all 26 joint-center
CPU checks pass in 01142. Both GPU2/GPU3 subsequently fail contention preflight.
Pending: full `quadratic_initializer` and `joint_center` GPU groups. All 61
policy/controller cases pass (01143). The static guard is 177 sources / 1,171
reviewed exceptions; the quadratic iterative numerical loop is explicitly
uncovered, not exempted. Focused Ruff passes with the existing joint-center
import-order warning excluded; whitespace checks pass. No worker is running.

The F18/F19 ledger now explicitly tracks block-center and sequential-locator
cloud/search/trust-region numerical loops. External DZ5 callbacks also mix
`.numpy()` reporting into their target callbacks, so complete consumer
qualification is still open. Next repair those actual preparation dependencies
and continue memory investigations. All F01--F20 terminal decisions and merge
remain open. Do not count the compute extension again.

Recovery through 01146: checkpoint `22485606` is committed and pushed.
The TP breakdown 01144 has identical teacher values but 1.058e-16 feature
differences amplified to 3.77e-10--8.25e-10 raw residual differences in XLA.
Frozen-feature runs 01145 match exactly in graph and XLA. The first-factor
Cholesky trial 01146 does not change these errors and is reverted. The original
1e-10 residual gate and full-recursion blocker remain open. GPU2/GPU3 still
retain unrelated allocations; no numerical worker is running. Continue bounded
block-center preparation migration through the existing driver.

Block-center NumPy is removed; cycle/reversal/score summaries, embedding and
complete scalar/batched target callbacks now have bounded stable XLA programs.
Run 01147 exposes a float32 constant inference error; explicit binary64 operand
inference repairs it. Run 01148 passes all 43 original and pinned-record,
threshold, serialization and enclosing-callback checks. Run 01149 passes all
61 policy/controller cases; the partial guard is 178 sources / 1,183 exact
exceptions. The whole ordered sweep remains uncovered and open alongside its
sequential locator dependency. GPU qualification remains pending. Next migrate
sequential cloud and trust-region helper loops without changing seeded streams,
then qualify their actual consumers. No worker is running.

Recovery through 01161: checkpoint `85219330` is committed and pushed.
Sequential scalar-cloud mapping, orthogonal frames and trust-region solves now
use fixed-signature XLA programs and the existing Philox compatibility helper.
Run 01150 passes 20 CPU checks. Run 01151 catches a legacy test's host recording
inside its target; a TensorFlow resource records every callback while preserving
the same box-bound assertion. All 40 consumer checks pass in 01152.
Run 01153 catches an invalid mixed baseline: its old block callback calls the
new compiled sequential locator. Pinning both original modules repairs the
reference; all 43 block checks pass in 01155. No numerical tolerance changed.

GPUs became free. Runs 01156--01160 pass 20 sequential-preparation, 43 block,
40 sequential, 33 quadratic-initializer and 26 joint-center checks on GPU2,
with memory growth verified. These close the previously pending initializer
and locator GPU qualifications. All five groups now require GPU for the final
driver test gate; current focused runs are not final-source terminal evidence.
Run 01161 passes all 61 policy/controller checks. The static guard is
179 sources / 1,183 exact exceptions. New kernels have no numerical exceptions;
the outer initializer/sweep/search lifecycles remain explicitly open. New-file
Ruff checks pass; existing sequential import/closure/style warnings remain.
No worker is running. Continue the master queue and keep main unmerged.
