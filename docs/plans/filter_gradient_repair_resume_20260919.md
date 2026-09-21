# Filter execution repair recovery, September 19

Worktree: `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`.
Branch: `repair/filter-gradient-xla-validation-20260918`.
Master: [repair program](filter_gradient_repair_master_20260917.md).
Detailed evidence: [execution record](filter_gradient_repair_execution_20260917.md).

Current checkpoint through **01953** follows pushed `51bc753a`. The paired
quadratic fitter now uses a stable XLA program; its finite-input reduction has
no Python numerical loop. Residual-refined eigenpairs correct demonstrated
trust, generalized-consistency and nearly repeated raw-spectrum errors.
All 43 focused/full-record checks pass on CPU/GPU, plus 51 quadratic-center,
38 paired-pilot and 25 batch-history checks on each device. All 68 policy/
controller checks pass. See [contract, results and review](filter_gradient_quadratic_numerics_20260921.md).

The original `3582b4ac` dependency closure remains the numerical authority.
Two mandatory singular dense-design tests still fail only `design_condition`
at D3/5 on CPU/GPU. The uniform-cloud fitter therefore remains explicitly
non-XLA migration debt. The separate D5 sequential lifecycle mismatch remains
66 CPU / 53 GPU fields at unchanged atol=rtol=1e-10. No failing field, rejection
rule, threshold, seed or optimizer setting has been removed or relaxed.

Final source costs 01923--01946 match every initial and changed-input field.
XLA paired fitting adds 224/231 MiB observed CPU RSS and about 18/19 MiB GPU
host RSS at D3/5; GPU allocator peaks fall to 19,712/22,272 bytes. A D5 GPU
warm-time trigger was investigated with three process repeats per arm
(01948--01953 plus the first matrix): medians 3.123 ms original, 2.962 ms graph,
2.961 ms XLA. No repeat-median >20% trigger remains; ranges overlap and there
is no statistical speed claim. Earlier 01886--01909 costs predate the final
eigensystem correction and are superseded for the paired candidate. Full
initializer costs and terminal repeated evidence remain open.

Inventory 01947 covers 2,928 working Python files, 2,927 parsed and one
unchanged vendor-reference parse error. Partial guard: 201 sources / 1,281
exact exceptions; the four new exceptions only validate immutable input
schemas. No worker is active. Charged CPU 42695.82916168675/115200 seconds;
GPU 38888.51101641379/187200 seconds. Caps are unchanged. Remote was fetched,
repair branch has no divergence and remote main is an ancestor. Main remains
unmerged. Focused Ruff and whitespace pass; no repository-wide lint claim.

Next: compiled probe generation and fit-round/design-partition controllers;
resolve singular dense-condition and original D5 equivalence; broader block
controllers; complete original lifecycle costs; public sequential outer/
reporting integration; external watchdogs and actual DZ5 target/transition
blocks; terminal F01--F20 dispositions and repeated evidence. No merge until
all gates pass. Historical numbered instructions below apply only to their
own checkpoints.

Historical checkpoint notes follow. Their worker/next-action statements apply
only to their own numbered checkpoints.

Qualified partial checkpoint through 01839 is committed and pushed as
bd36a89b (following ee2d455f). No worker is active. The public outer lifecycle is unchanged;
main remains unmerged. Full result/review:
[sequential eigenpair checkpoint](filter_gradient_eigen_checkpoint_20260921.md).

The two sequential eigensystem consumers now use the existing refined helper
inside an independent stable-signature XLA graph. Directly tracing its custom
derivative retained a factor guard through evaluate's graph-owned resource
dictionary (01823). Isolation releases all actual consumer graphs and tracked
validity variables on CPU/GPU (01825/01828), while preserving HLO inputs, one
trace and execution after factory eviction. Native executable eviction and
broader leak freedom are not established.

Eight focused numerical/gradient cases pass on each CPU/GPU (01824/01827).
All 153 affected GPU consumers pass (01830--01834), and all 67 policy/controller
checks pass (01839). Earlier three-way terminal and complete original lifecycle
qualification is preserved in the checkpoint note. Intermediate cfbc32d2 has a
demonstrated eigensystem defect; original 3582b4ac is the numerical authority.

D5 changed-input original-source records still fail 66 CPU / 53 GPU fields at
unchanged atol=rtol=1e-10. D3 GPU records pass. All 66 CPU failures predate the
eigenpair repair, as archived attribution against 01754 shows. Crossed-data fits
in 01826 reproduce the original failure: 4/2/66 failed fields for current fit on
original data, original fit on current data, and current/current relative to
original/original. Prepared differences are <=1.1883e-16; all use 73 iterations /
217 evaluations. Original-initial-state injection does not fix this (01829).
Native-row decoding passes value/VJP checks but its ForwardAccumulator composition
fails (01836); an objective-only diagnostic with the unchanged vector Jacobian
still fails full records (01837). No diagnostic injection is installed in runtime.

The fresh 01838 inventory covers 2920 working Python files, 2919 parsed, and the
same vendored legacy parse error. Guard coverage remains partial: 198 sources /
1276 exact exceptions. New numerical source/tests pass focused Ruff; the driver
retains its pre-existing import-order, duplicate-key and dict-style warnings.
Whitespace checks pass. No package, OS, external source or source-pin change.

Charged totals through 01839: CPU 41913.46378235762/115200 seconds;
GPU 37772.45033208291/187200 seconds. Do not count the extension again.

Next: preserve this branch checkpoint; continue original-source D5 optimizer
rounding qualification without retuning/tolerance changes; give obsolete
intermediate lifecycle/refinement comparisons an explicit original-authority
disposition; renew original complete costs after numerical runtime qualification;
finish public reporting/outer wiring and independent external watchdogs; repair
broader block/quadratic controllers and qualify actual DZ5 target/full-transition
blocks. All F01--F20 terminal decisions, repeated comparisons and merge remain open.
The original-source cost harness draft at
/tmp/filter_repair_lifecycle_original_memory_draft.py is not installed/tested.
The native-row decoder is diagnostic-only under tests, not a proposed runtime fix.

Historical checkpoint details follow; active/no-worker statements below apply
only to their own numbered checkpoints.

Current checkpoint through 01720: ordered attempts are integrated and all
focused CPU/GPU mechanics, actual two-factor public histories, original consumers
and 67 policy/controller cases pass. Frozen/current XLA records are exactly equal.
Reporting overhead is repaired. D5 host compilation overhead remains observed
and investigated (191 MiB at capacity4, about 291 MiB at capacity32); warm memory
is stable in the tested scope. Strict graph/XLA numerical gates remain failed,
including the inherited D3 XLA eigensystem residual. See the complete
[checkpoint result](filter_gradient_attempts_checkpoint_20260921.md).

No worker is active. Charged GPU 33,426.027247090 / 187,200 and CPU
40,127.291836184 / 115,200 seconds. Commit/push this partial execution checkpoint;
main stays unmerged. Next install and qualify the scratch outer lifecycle before
binding actual refinement/terminal callbacks and public reporting. Drafts:
`/tmp/sequential_lifecycle_tf_draft.py`, `/tmp/sequential_terminal_tf_draft.py`,
`/tmp/test_filter_repair_sequential_lifecycle_draft.py`; none is installed/tested.
They are not completion evidence. All broader controller, external DZ5 actual
full-transition-block, F01--F20 terminal, repeat and merge gates remain open.

Historical progress records follow; any active-session statements below refer
only to the earlier numbered checkpoint.

Historical checkpoint details follow; their active/no-worker statements refer
only to the numbered checkpoints below.

Factor-status checkpoint through 01679: all 40 CPU and 40 GPU synthetic
precedence/boundary/enclosure cases pass (01675/01676), all 14 original/full
fitter CPU records pass (01677), all 223 affected GPU consumers pass (01678),
and all 67 policy/controller checks pass (01679). Invalid eigenvalue rejection
matches frozen f3f47f76 on both CPU/GPU after making positive comparisons
explicit; 01674's failed NaN reduction attempt remains preserved. No threshold,
optimizer, covariance arithmetic, derivative or random stream changed. The
public record now materializes a native tensor status and loading norms.

No worker is active. Charged GPU 31,795.749108093 / 187,200 and CPU
39,841.218343504 / 115,200 seconds. The partial policy guard covers 193 sources
with 1,276 exact exceptions. This small status dependency has no standalone
speed claim; its cost remains included in the forthcoming enclosing-factor and
terminal comparisons. Commit/push before installing the ordered-attempt draft.
Main and every terminal finding remain unmerged/open; the inherited structured
D5 graph/XLA comparison is still unresolved.

Proposal checkpoint through 01673: all 22 complete focused/history cases pass
on each CPU/GPU; all 20 trust/preparation, 40 sequential, 43 block and 12 factor
GPU consumers pass, plus 67 policy/controller checks. Six fresh cost processes
pass: original/XLA outputs are exactly equal at D=3/5, graph/XLA maximum error
2.221e-16. Warm original/XLA medians are 3.875/1.820 and 4.126/1.685 ms;
XLA host peaks are 11.535/12.031 MiB lower, and GPU peaks rise from 15,104 to
21,504 bytes. No declared cost trigger fires. Graphs remain 264 nodes at both
sizes, with one trace and eight unchanged runtime operands; twenty warm calls
retain constant device allocations and at most 24 KiB host growth. The exact
analysis and `proposal-memory-comparison-01673.json` are preserved beside runs.
These are descriptive single-process dependency costs, not final repeats.

No worker is active. Charged through 01673: GPU 31,544.737419723 / 187,200 and
CPU 39,670.321654545 / 115,200 seconds; remaining GPU 155,655.262580277 and CPU
75,529.678345455. Runtime sources did not change during cost measurement. The
partial guard passes 192 sources / 1,276 exact exceptions, with no new exception.
Focused Ruff and whitespace checks pass. This checkpoint can be committed/pushed;
main remains unmerged. The structured D5 graph/XLA comparison remains open.

Next dependency is native factor-fit rejection precedence and numerical report
fields, enabling factor escalation to consume tensor status. A scratch draft
at `/tmp/filter_repair_factor_decisions_tf_draft.py` is not installed. Preserve
original invalid-domain precedence, finite/SPD/condition/rank/holdout/optimizer
status order, boundary comparisons, and missing-condition semantics, including
original NaN behavior; no numerical safeguard or threshold change is authorized.
Compare synthetic conflicting failures/nearest floats to frozen materialization,
then all original records/consumers before enclosing escalation. A status helper
alone will not complete the outer controller.

Earlier checkpoints follow.

Proposal qualification through 01667 passes: all 22 focused/full-history cases
on CPU and GPU against the completely pinned b3334646 dependency (01665/01666),
20 original trust/preparation, 40 sequential, 43 block and 12 factor GPU cases
(01658/01659/01662/01663), and 67 policy/controller checks (01667). No failed or
skipped cases. The six-arm `proposal_memory` matrix is active from 01668; keep
runtime/tests/driver frozen until it completes. The graph/XLA structured-fitter
blocker and all broader terminal work remain open. Main is unmerged.

Checkpoint review through 01655: all 66 policy/controller tests pass after
registering the cost diagnostics. The source guard passes 191 sources / 1,276
exact exceptions; no numerical-loop exception was added. New runtime/tests and
analysis scripts pass focused Ruff; whitespace checks pass. No numerical source
changed after the 441-case qualification. The inherited graph/XLA comparison
remains explicitly open. Commit/push preserves this execution checkpoint and
its unresolved terminal evidence; main stays unmerged.

Recovery through 01654: no numerical worker is active. The 441 focused
qualification cases remain passing. Default XLA before/after and standalone/
enclosed records pass at D=3 and D=5. The eight-arm comparison is preserved in
`structured-memory-comparison-01650.json`; D=5 graph/XLA fails 23 numeric fields
at the unchanged 1e-10 tolerance and is **not accepted**. 01651 crosses prepared
clouds with both fitter modes; 01652 proves the same eight numeric fitter gaps
on identical data in frozen f06fd505 and current compact source. Same-mode
before/after records pass. No tolerance, optimizer or method is changed.

The D=5 resource trigger is explained by two 33-branch compact-shape dispatches
(CPQR initializer and Jacobian QR), versus two five-branch dispatches at D=3.
Fixed-input public host overhead is 446.820 MiB, GPU peak 97,024/195,840 bytes,
and cold 13.883/30.541 seconds. Over six changing active sizes, 01653/01654
retain all records while old/new first-sequence times are 70.215/29.857 seconds
and host peaks roughly 2.79/1.97 GiB. New sizes cost about 11 seconds each in
original compact fitting and about 0.11 seconds in the repaired fixed-capacity
path after the first call. Second-sequence host growth stays below 0.1 MiB.
`structured-eligibility-cost-disposition-01654.json` and its exact analysis
script preserve inputs, source hashes, HLO branch counts and full comparisons.
This is a bounded, explained compilation tradeoff, subject to final repeats;
it does not establish a statistical ranking or arbitrary-capacity memory bound.

Budget through 01654: GPU 30,927.589910224 / 187,200 and CPU
39,575.574552332 / 115,200 seconds (remaining GPU 156,272.410089776 and CPU
75,624.425447668). The extension is counted once. Commit/push the execution
checkpoint with the graph/XLA blocker explicit; continue independent proposal/
outer-controller repair without calling the checkpoint terminal-qualified.
Main remains unmerged. No F01--F20 terminal disposition is closed.

Earlier checkpoint details follow.

Latest September 21 continuation, based on pushed **f06fd505**: preserve the
uncommitted structured-preparation repair. All 441 current qualification cases
pass: 26 preparation and 23 integration on each CPU/GPU (01635--01638), 223 GPU
consumers (01639), 40 sequential GPU cases (01640), 14 original/full fitter CPU
cases (01641), and 66 policy/controller cases (01642). No failed/skipped JUnit
cases in those passing groups. The partial guard covers 191 sources and 1,276
exact exceptions; no numerical-loop exception was added.

Active driver: `matrix --stage tests --test-batch structured_memory
--test-timeout-seconds 300`, beginning with 01643. Recover this sequential eight-
arm GPU matrix before another worker or source edit. The compact public view
and three-row reuse are qualified. Preserve source files through cost runs.

Earlier full-record discrepancies are repaired: two native stages preserve
separately rounded multiplication/addition. The exact affine pullback avoids
XLA TensorList boundary failure; tracing it in its own resource-free graph
avoids TensorFlow's process-level custom-gradient registry retaining the
consuming optimizer graph. 01628 passes derivative, unchanged HLO, lifetime and
release checks. No tolerances, optimizer settings or random streams changed.
All failed attempts 01616--01627 remain preserved with diagnosis in the master.

The active measurement matrix is the reviewed eight-arm `structured_memory` batch with 300-second per-worker bounds.
Numerical jobs stay sequential, same approved driver prefix, idle GPU preflight
and verified growth. The benchmark-only baseline setup bias is repaired before
measurements. Independent analysis draft is `/tmp/analyze_structured_memory_20260921.py`;
it is not evidence until the eight matching arms pass and are compared. Keep
both script and output beside versioned runs. Broader controllers and terminal
three-process comparisons remain open; main is unmerged.

Historical qualified guard checkpoint follows.

September 21 active continuation (after pushed `085baaaa`): covariance domain
checks propagate invalid optimizer evaluations through XLA and reject public/
native fits without geometry. CPU/GPU focused checks pass; the ownership repair
keeps each int64 guard alive through enclosing graph use and releases it with
the graph. No bounds, optimizer, derivative or old parity tolerance changed.

CPU combined crashes 01588/01595 are localized by 01597 to LLVM executable
mapped allocation ENOMEM. 01598 samples 65,021 mappings against limit 65,530,
with ample RAM and unlimited process/cgroup memory. 01599 confirms clearing
Python factory caches and collecting cycles frees no mappings (33,801 and
64,785 unchanged); it crashes again. TensorFlow 2.19.1's installed
`device_compilation_cache.h` explicitly documents unbounded executable retention
without eviction. Evidence: `factor-executable-allocation-disposition-01599.json`.
Do not mutate OS limits/packages or treat Python cache bounds as compiler bounds.

Reviewed qualification profile: all 38 original CPU cases in three fresh
sequential workers, the unchanged combined GPU suite, CPU/GPU lifetime tests,
and all 223 GPU consumers. Long-lived CPU runs across arbitrary distinct
compiled signatures are not admitted by this profile. The rejected combined
CPU stress job remains preserved as explanatory failure.

Current-source qualification completed: 01600 passes 66 controller/policy
checks; 01601/01604 pass two GPU/CPU lifetime cases each; 01602 passes all 38
combined GPU cases; 01603 passes all 223 GPU consumers; 01605/01606/01607 pass
14/18/six CPU cases in bounded fresh processes.

No campaign worker is active. All eight fresh-process cost arms 01608--01615
pass, including frozen 085baaaa and candidate graph/XLA at D=3/5. Every
original before/after field is exactly equal; all healthy invalid counts are
zero, and graph/XLA records pass unchanged atol=rtol=1e-10. Default XLA adds
7.28/5.96 MiB host peak at D=3/5 with warm times 22.090/21.847 and
92.201/93.585 ms before/after. No cost trigger fires and warm allocation shows
no accumulating trend over twenty calls. This is descriptive single-process
evidence, not terminal repeats. Analysis and identical script are preserved
beside the runs as `factor-guard-cost-comparison-01615.json` and
`analyze_filter_guard_memory_20260921.py` (SHA-256
`989038392ffc0ff99cd142351323e5b5048b8c7c7aa4611b6435cc31b8e3fec5`).

Through 01615 the charges are 29,038.68574661859 GPU / 38,711.01263819063
CPU seconds; remaining budgets are 158,161.3142533814 GPU / 76,488.98736180937
CPU seconds. Total caps remain 52 GPU / 32 CPU process-hours. Finalize review
and commit/push the qualified checkpoint before the next runtime changes.

Draft next dependency (not installed or tested):
`/tmp/filter_repair_structured_preparation_tf.py`. The master records its
intended fresh/reused-row preparation contract. Review and qualify it against
frozen compact records before integration; it is not runtime evidence.

The driver now excludes only exact reviewed explanatory jobs from terminal
mandatory tests, with individual reasons and regression checks. New groups
remain required by default. Current runtime/consumer/compiler checks remain
mandatory, including localization groups that are actual regressions. The
source guard passes 189 sources / 1,276 exact exceptions; its new exception is
static graph ownership, never numerical iteration. This is still partial scope.

Main is unmerged. Remote fetch completed; origin/main remains 3582b4ac and the
repair remote remains 085baaaa. Continue structured preparation/refinement,
block/quadratic lifecycles, external callbacks, remaining resource issues,
frozen-source terminal suites/repeats and all F01–F20 dispositions afterward.

The older checkpoints below remain historical.

Current September 20 continuation through 01577, based on pushed `7d08c68e`:
the active-row COD investigation is repaired by binding the CPQR loop to each
compact input shape while sharing rank decisions, the complete-orthogonal
stage and the original full-rank pullback. CPU slice-dot fusion caused the
previous five padded-fitter failures. Isolated-dot and per-step branches failed;
the compact CPQR loop preserves every tested fitter field and discrete count.

Focused evidence: 01560/01562 pass all 32 CPU/GPU solver/derivative cases;
01561/01563 pass all 18 CPU/GPU padded cases at 4/32 spare rows; 01573/01574
pass all 14 CPU/GPU original/full fitter records. All 223 GPU consumer cases
pass in 01575. All 63 policy/controller checks pass in 01576. No worker remains running.

Resource evidence: `factor-cpqr-capacity-memory-comparison-01572.json` compares
fresh processes. Full XLA 32-spare capacity adds 331.219 MiB host peak and
96 KiB GPU peak (2x), cold 11.886/27.013 seconds and warm 166.146/169.506 ms
compact/padded. Twenty warm calls show only 12--32 KiB host growth. At four
iterations, all graph/XLA complete fields agree; graph/XLA warm times are
87.864/30.899 ms compact and 100.031/30.736 ms padded. Graph device peaks are
8.12/8.25 MiB versus XLA 96/192 KiB. Branch graph/compiler memory is explained
at this extent, but the tradeoff and three-process terminal repeats remain open.

New blocker: full graph arms 01567/01568 both hit the existing strict loading
margin assertion during an optimizer trial; XLA ignores that assertion. No full
graph steady timing exists. Do not remove the check, loosen the margin, clip
the state or change optimizer settings to make this disappear. The guard needs
an explicit TensorFlow/XLA validity representation and a no-fire check on
healthy original records. Padded fitting remains unwired into preparation.

Static guard now has one exact fixed-schema row-shape binder exception, for
189 sources / 1,275 exceptions. No Python numerical loop is exempted. Only the
two preapproved geometry initializers may change RNG streams; canonical LEDH
rebuilding remains excluded. Main is unmerged; all F01--F20 terminal decisions,
outer lifecycles, external callbacks and other recorded investigations remain.
The fetched remote repair branch is aligned and origin/main is an ancestor.

The older checkpoints below are historical.

September 20 current checkpoint through 01545, based on pushed `375e7257`:
weight normalization and weighted-loss arithmetic now remain runtime operations
inside enclosing XLA calls. One optimization barrier on input weights repairs
all full CPU record failures without changing formulas, optimizer settings,
thresholds, derivatives or random streams. Full fitter suites pass all 14 CPU
and all 14 GPU cases (01521/01522), including the original 3582b4ac record gate.
All 223 affected GPU consumers pass (01541).

Small-capacity padded fitting also passes all 12 CPU/GPU cases (01542/01530).
Active Jacobian entries had been identical; zero padding changed QR rounding
near the rank threshold. A fixed-shape QR branch selected inside XLA preserves
the original compact factorization. Six/seven runtime HLO inputs and changed-data
executable reuse remain checked. The one new exact allowlist entry covers only
static QR shape binding; the numerical work stays inside TensorFlow/XLA.

The larger default 32-row reuse capacity remains blocked. Runs 01531--01533
show stable warm allocation and about 63 MiB added host peak, but optimizer
counts/records differ. Same-state gradients agree (01535), localizing the
problem to initializer shape rounding. A diagnostic-only compact COD dispatch
restores all records (01536), but its 32.055-second cold call and approximately
847 MiB added same-process RSS trigger further investigation. That COD clone
is not installed. Padded fitting is not wired into sequential preparation.
Next reduce duplicated COD compilation by sharing its control/body and binding
only row-dependent operations, or establish its actual bounded cost with fresh
processes. Preserve every field and tolerance.

Fresh changed-data memory comparison 01537--01540 confirms the compact repair:
checkpoint/candidate XLA median new-cloud calls are 6.403 seconds/9.911 ms;
post-cold host growth is 335,212,544/57,344 bytes. Cold XLA calls are 8.745/8.546
seconds; final GPU current allocations are identical at 10,496 bytes. All
measured public fields pass unchanged 1e-10 comparisons, including graph/XLA.
Artifacts: `factor-compilation-memory-comparison-01540.json` and
`factor-capacity-comparison-01533.json`. One process per arm gives descriptive
mechanism evidence; terminal source-frozen three-process comparisons remain.

All 63 policy/controller checks pass (01545). The partial guard covers 189
sources / 1,274 exact exceptions. Inventory 01544 finds 2,876 working-tree
Python files, 2,875 parsed and one unchanged external legacy parse error.
Focused changed-runtime/new-test Ruff and whitespace checks pass.

Charges through 01545: 26,812.952 GPU / 35,037.820 CPU process-seconds, leaving
160,387.048 GPU / 80,162.180 CPU seconds under the unchanged 52/32-hour caps.
No numerical worker is active at this checkpoint. Continue with the existing
runner prefix, sequential numerical jobs and fresh versioned artifact paths.

Structured reuse and outer sequential/block/quadratic control, external DZ5
callbacks, prior memory/timing investigations, all F01--F20 terminal dispositions,
final frozen-source tests/repeats, remote integration/retest and terminal review
remain open. Main stays unmerged. Canonical LEDH rebuilding remains excluded.

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

September 20 checkpoint accounting through 01577: 27,876.007 GPU /
35,838.542 CPU process-seconds, leaving 159,323.993 GPU / 79,361.458 CPU
seconds under unchanged 52/32-hour caps. Inventory 01577 finds 2,879 working
Python files, 2,878 parsed and the one unchanged external legacy error. Focused
Ruff and whitespace checks pass. No worker is active; main remains unmerged.
