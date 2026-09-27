# DZ5 initializer adapter execution result

Status: adapter fixture qualification and all fresh CDF target checks pass.
Actual r1 rejection is preserved; accepted-r2 and supervisor lifetime
qualification are pending.
This is part of the filter/gradient execution repair master, not a restart of
MacroFinance's closed multi-asset campaign or paused financial/BGL successor.

The [plan](filter_gradient_dz5_initializer_adapter_20260928.md) preserves the
numerical methods, declared recipe, source fixture, seeds and comparison gates.
The candidate external adapter delegates numerical work to BayesFilter's
`SeededDenseInitializerProgram`, enables the enclosing XLA program and writes
completed arrays through the shared tensor NPZ writer. No BayesFilter numerical
source changed in this unit and the live external checkout was not edited.

Snapshot `dz5-initializer-adapter-20260928-r1` contains 732 source files and three
frozen inputs, with manifest SHA-256
`dac56f03566933aefcb0772958b93e1c67609209bb4bc791aca0aa3dd0516c94`.
Its original admission remains historical and is not reused. The supplemental
`dz5-initializer-supervisor-callsite-20260928-r1` preserves the actual caller of
`supervise`, which was outside the original numerical dependency closure.

| Case | CPU / GPU runs | Result | Exact target rows / callbacks | NPZ arrays |
|---|---|---|---:|---:|
| D1 identifiable Gaussian | 04551 / 04555 | Accepted | 17 / 9 | 20 |
| D3 identifiable Gaussian | 04552 / 04556 | Accepted | 41 / 9 | 20 |
| D3 invalid locator | 04553 / 04557 | Rejected before cloud generation | 1 / 1 | 0 |
| D3 invalid cloud | 04554 / 04558 | Rejected after first cloud | 13 / 5 | 4 |

All eight cases compare full original records, callback positions and counts,
archive members and values, and independent known Gaussian marginal scales.
They verify one enclosing trace, stable input signature, unchanged HLO on
changed operands and absence of host numerical callbacks. The pinned original
reference retains its explicit locator-JIT comparison adjustment. The regular
fixture worker verifies loaded BayesFilter hashes against the frozen snapshot;
actual target workers mount both source trees read-only.

Fresh CDF CPU batch4 check 04559 passes graph/XLA value, analytical score and
status comparisons at unchanged tolerances, invalid-row isolation, changed
inputs and exact replay. CPU independent five-point check 04560 passes both
step sizes (maximum error / allowed error 0.391719 and 0.767832), with exact
replay. The score bank has 185 rows and 23 coordinates. GPU batch1/4/46 checks
04561--04563 pass, as does GPU batch68 in 04564. GPU independent score
check 04565 passes both step sizes (maximum scaled errors
0.344380 and 0.625857), with exact replay. Independent readback
confirms the source, device, read-only mounts and memory-growth records.
Actual consumer runs are pending.

| Decision | Criterion / veto status | Main uncertainty and next action | Not concluded |
|---|---|---|---|
| Accept adapter fixture mechanics | Eight full-record/NPZ/callback/XLA cases pass | Complete fresh CDF evidence and invoke the actual adapter | Actual D23 acceptance or lifetime containment |
| Preserve old admission as stale | No hash refresh or old evidence substitution | Issue a new target-only engineering artifact only after all fresh checks pass | HMC, training, posterior or scientific admission |
| Keep main unmerged | Actual consumer and broader master gaps remain | Finish this unit and remaining reporting/precision and terminal reviews | Whole-program completion |

Local review: the Gaussian fixtures exercise the exported adapter with real
configuration parsing and pinned independent references; they cannot stand in
for the CDF target. New target evidence must pass independently before the real
CreditTargetAdapter constructor is used. The complete consumer test stops
before the historical training continuation. Python collection and process exit
answer distinct resource questions; neither is evidence of posterior quality.
No independent reviewer is claimed.

Fresh target readback and exact supervisor mechanics pass 18 checks in 04567
(04566 passed the earlier 16 checks). The final readback independently
recomputes each five-point derivative from saved values, checks declared batch
extents and rejects corrupted/stale/non-XLA evidence. It issues the new
`dz5-initializer-adapter-target-admission-20260928-r1.json` only for isolated
initializer regression use. No historical admission is changed.

Actual initializer attempt 04568 failed before TensorFlow execution because
bwrap could not create the evidence mount point below the read-only source
tree. An empty mount-point directory was added; all 732 source hashes and the
manifest hash remain unchanged. The failure and repair receipt are preserved.
Retry 04569 completed; its acceptance assertion failed on a valid numerical
rejection. Its compilation host RSS reached about 17.5 GiB after
108 seconds and 17.94 GiB after 252 seconds according to host `ps`. These are
observed lower bounds, not a completed peak or resource qualification. A
sandbox `/proc/<host-pid>` probe had no process record and is not process-memory
evidence. The full initializer's capacity cannot be inferred from D1/D3 tests.

The CPU r1 initializer executed in521.691600s and the supervised worker charged
608.215076s. It compiled once, used252 target rows, and returned
`dense_center_score_above_cap`: scaled score1.0324490979441923 against0.02.
The archived original r1 also rejected after252 rows, with scaled
score1.032345508936184. This confirms the rejection path and accounting;
it does not establish full numerical equivalence. One owner and target
batches1/46/68 each traced once; no host callbacks were found. Weak references
cleared after release, while host memory remained allocated.

The initial plan selected an archived rejected recipe as if it could prove
accepted-consumer behavior. The correction freezes the exact successful r2
initializer recipe and its prior-origin start separately, preserving all original
artifact hashes. Only the historical locator limits differ (80->240 iterations,
601->1501 callback batches); numerical thresholds/seeds are unchanged. Fresh
start value/score replay is mandatory before use, with a separate receipt;
old admission, point and optimizer state are never restamped or reused as
current authority. No learned map or training continuation is involved.

Deadline and evidence qualification04570 passes27 checks. The observed CPU
cold cost motivates narrow scope-bound parent limits for later accepted and
lifetime workers under the unchanged14400s allocation and global caps. The
unit worker ceiling is28 after the explicit baseline/harness correction.
GPU rejection and accepted-r2 qualification remain pending.

Saved unoptimized HLO inspection finds774754 instructions in8754 computations
for the complete initializer (196956093bytes), compared with about44334--44511
instructions in453 computations for individual target batches. This is a
compile-size diagnostic, not an exact allocation explanation or a claim that
those instructions all execute. The numerical-return host HWM is21394235392
bytes (19.92GiB); after HLO inspection/release it is24687267840bytes
(22.99GiB). HLO inspection adds
cost and must not be counted as numerical live-tensor memory. The size artifact
is `dz5-initializer-hlo-size-diagnostic-04569.json`.

A source-metadata diagnostic shows 15 copies of each reported `IgammaGradA`
operation group in complete unoptimized initializer HLO versus one per
standalone target (10380 versus692 instructions for each of the four groups).
This narrows a compile-size hypothesis to repeated target call sites after
lowering; it does not prove an exact native allocation source. The source-file
metadata itself collapses to TensorFlow `ops.py` and does not support a more
specific source-line attribution. Preserve these limitations alongside
`dz5-initializer-hlo-callsite-diagnostic-04569.json` and the source-metadata
record. No numerical runtime repair is inferred from this diagnostic alone.

The source/harness bytes for later accepted-case tests are frozen in
`dz5-initializer-harness-20260928-r2`. Remote fetch confirms origin/main is
already integrated; the repair branch is127 commits ahead and zero behind
at checkpoint876960c1b. Main promotion remains gated by the unfinished work.

Complete saved r1 comparison at the target's original1e-8 absolute/1e-7
relative bounds preserves59 mismatched scalar/report leaves in
`dz5-initializer-r1-original-comparison-04569.json`, despite equal rejection
status, callback count and row count. Examples include selected-center
coordinates and their scores. This is comparison with the preserved historical
result, not a new execution of its original target/locator closure; the latter
is required to attribute optimizer/enclosure/source effects. Matching rejection
is not numerical equivalence and cannot waive those differences.

GPU r1 rejection04571 passes the declared rejection/XLA/import checks in
753.627332 supervised seconds. The initializer itself takes666.353274s;
HLO inspection adds61.152172s. It returns the same rejection after252 rows,
with scaled score1.032344485146019. Numerical-return host HWM is20319166464
bytes (18.92GiB), TensorFlow device peak269440256bytes (256.96MiB), completed
current411648bytes; after Python release current is5376bytes while host memory
remains allocated. Source/fixture/default memory-growth records pass.

Full current CPU/GPU r1 records still have59 leaves outside unchanged
1e-8/1e-7 comparison bounds; the complete differences are preserved in
`dz5-initializer-r1-cpu-gpu-comparison-04571.json`. Valid rejection and stable
trace/counts do not close that numerical gate. Accepted-r2 CPU run04572 is
executing; no accepted-case or lifecycle conclusion is available yet.

Accepted-case04572 passes fresh value/score replay of the original r2 start
at unchanged caller tolerances (value18023.47668091579). Original point hash
`a9b93a0e9ab490158a9cfd7c2bce8f50397c73d0e2fde02c38e6c6d48f1231d8`
is preserved; current target admission is linked separately. The main
initializer remains active, so this is start-input evidence only.

Checkpoint archive through04571 reopens and verifies953 files,32621046bytes,
SHA-256 `ea152d55b422976921e05c3c65dbbc5ea98ee2a261aac9435bb5d53b71c632f5`.
It includes both failed attempts, frozen source/input/harness snapshots,
fresh target admission, HLO and complete mismatch diagnostics. The archive
and receipt are committed under `artifacts/filter-gradient-repair-20260917/`
as `dz5-initializer-adapter-04571-evidence.tar.gz` and
`dz5-initializer-adapter-04571-verification.json`. It excludes the active04572
worker, which needs a later checkpoint. Through04571 this unit charged
1131.902120CPU /1284.263164GPU seconds across21workers; no global budget reset.
