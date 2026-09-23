# Public ordered-block costs and memory lifetime

03405 completes the planned buffer attribution and all original public-record
assertions. Both the executed module and the separately exported optimized HLO
report185891 planned bytes, including171360 bytes in one reusable temporary
arena,10411 bytes across407 live-out allocations (including the output tuple),
3665 constant bytes and455 other/input bytes. The temporary arena holds5215
compiler values at926 distinct offsets; its largest array is936 bytes and its
largest tuple3168 bytes. Their address union is86624 bytes; arena alignment and
reuse cannot be interpreted as simultaneously live numerical array storage.

The allocator observation repeats the cohort:283136 peak bytes after cold/warm,
284160 after changed input, and4352/5376 current bytes. Compiler plans and runtime
allocator measurements are different quantities; their98269-byte difference is
not evidence of a live-tensor leak. Many small output allocations and allocation
rounding can contribute overhead, but this diagnostic does not attribute every
runtime byte. See `block-buffer-analysis-03405.json` and its reproducible script
in the artifact root; the raw compiler reports are retained under03405/xla.

Resource disposition: retain the observed150KiB transient peak increase for the
tested D3 signature. It accompanies the complete enclosing control/history
program, stays below0.28MiB in these observations and does not accumulate across
repeated calls. The2x investigation trigger is now answered for this fixture;
it was an investigation threshold, not a required byte-reduction target. No
threshold or numerical behavior changes. Larger signatures and the actual DZ5
consumer still require capacity evidence. Native host executable residency is
separate and requires bounded process lifetimes. Dumping disqualifies03405 timing.
The cost unit closes at41/42 workers; no further attribution retry is needed.

Prepared follow-up after03398: use one of the two remaining42-worker cost-unit
slots for `block_buffer_attribution_gpu`, a300s worker with the unchanged public
XLA D3 fixture and full original-record assertions. Enable compiler text dumps
only for modules matching `inference_execute`, writing into that worker's unique
`xla` directory. Record exact XLA flags, source, compiler memory-usage reports
and buffer assignments. This is memory attribution, with all timing ineligible
because dumping changes compile work. Compare the largest static temporary,
output and constant allocations with the measured cold/warm/changed allocator
peak/current values. No numerical, callback, shape or solver change is allowed.
Missing compiler reports or different complete records invalidate attribution;
preserve the attempt and use the last slot only for localized harness repair.
Do not overwrite or combine this diagnostic with the18-worker timing cohort.

Skeptical review: a relative2x threshold on a130KiB baseline is not itself a
resource limit or an out-of-memory prediction, but dismissing it without
allocation evidence would evade the declared trigger. Compiler planned buffers
are not TensorFlow allocator live bytes, nor peak process reservation. Report
their distinct roles and any unmatched overhead. This diagnostic cannot certify
larger signatures, native host eviction or full initializer readiness.

GPU continuation03380--03385,03387--03398 completes all18 fresh processes:
three repeats of prior/graph/XLA at D3/D5, on one physical GPU2 UUID with verified
growth and no sampled foreign compute. All complete original/changed records
pass. The analyzer's ten checks renew in03386. Sources and numerical inputs are
identical within this GPU cohort. CPU and GPU are separate frozen cohorts; the
only changed imported block dependency is fixed_center_curvature, whose reachable
`_precision_geometry_kernel` AST is unchanged. The source review is preserved in
`block-gpu-cohort-source-review-03398.json` alongside the analysis.

| GPU dimension | Prior repeated call | Graph repeated call | XLA repeated call | Prior / XLA cold time | Prior / XLA maximum observed RSS |
| --- | --- | --- | --- | --- | --- |
|3|20462.85ms|140.18ms|29.99ms|23.06 /27.83s|3459.48 /2875.96MiB|
|5|21612.74ms|159.41ms|65.67ms|24.29 /28.81s|3478.31 /2886.71MiB|

The GPU D3 allocator trigger fires:130560bytes prior versus284160bytes XLA,
a2.176ratio and153600byte increase. D5 is147968 versus292096bytes, a1.974ratio.
The graph arm has larger peaks754432/799744bytes. Candidate live allocation is
4352bytes after cold/warm and5376after changed input, versus6144/7680prior.
Warm peaks remain fixed; the D3 excess is transient allocation in this bounded
observation, not accumulating retained tensors. This locates the concern but
does not identify individual buffers or waive the declared2x trigger. Attribute
the compiled temporaries/output histories and record an explicit resource
disposition before E6 closes. No cold/warm/host-RSS trigger fires relative to
prior; observed warm host growth is at most0.098MiB. Absolute cold compilation
and new-signature native residency remain separate obligations.

Analysis: `artifacts/filter-gradient-repair-20260917/block-public-costs-gpu-03398.json`.
The GPU cohort costs1150.879186seconds; the whole block cost unit uses40/42workers
including the extra analyzer renewal and earlier ineligible shared worker.
The earlier CPU values and03273 sharing veto remain preserved below.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain complete GPU cost cohort |18 full-record and provenance comparisons pass |D3 allocator2x investigation trigger remains open |Small absolute transient increase lacks buffer attribution |Inspect compiled memory and supported lifecycle disposition |No universal ranking or memory limit |
| Retain bounded CPU/GPU containment |Fresh child startup and cleanup reproduce original records |No parent-growth trigger in03290/03373 |Actual full consumer still unintegrated |Apply declared worker lifetime after E5 qualification |No in-process native eviction |

Review: medians compare three fresh processes, each with three complete calls.
Prior recompilation is part of the measured public cost. Sampling cannot prove
uninterrupted exclusivity; these observations do not bound process or allocator
peaks beyond the recorded intervals. All remaining endpoint and main-merge gates
remain required. Earlier checkpoint evidence follows.

All18 CPU matched-cost workers03255--03272 pass complete original records for
both initial and changed inputs. Numerical authority is3582b4ac; mechanism/cost
baseline is aee3ad043. Three fresh processes per arm and dimension compare the
prior public API, current outer-graph diagnostic and current enclosing XLA API.
All processes use the same frozen source, configurations and inputs. Each times
three synchronized complete public calls, including reporting. The prior API
constructs fresh conditional callbacks and recompiles on every public call;
that repeated compilation belongs to its observed cost.

| Dimension | Prior repeated call | Graph repeated call | XLA repeated call | Prior / XLA cold time | Prior / XLA maximum observed RSS |
| --- | --- | --- | --- | --- | --- |
|3|12433.47ms|51.13ms|16.17ms|13.96 /18.38s|3360.48 /2691.17MiB|
|5|12911.47ms|58.45ms|22.62ms|14.51 /19.16s|3407.03 /2716.96MiB|

No declared relative cost trigger fires for the CPU candidate. Cold ratios are
1.316/1.320; additional maximum observed RSS is-669.31/-690.07MiB. Candidate
RSS grows at most0.30MiB during the three-call repeated interval. The graph arm
retains compiled dependencies, so these results do not isolate compiler-only
effects on an identical graph. Three process repetitions support descriptive
cost comparison, not a broad statistical performance ranking. The absolute
startup cost and many-signature residency remain separate concerns.

GPU03273 passes numerically but records foreign PID3257256 during its measured
interval. Its timing is ineligible. The next preflight declines before launch.
All18 GPU cost workers must restart on an unshared physical UUID in a separate
source-frozen cohort after the dependency checkpoint. CPU evidence remains
identified by its original source hashes; the later GPU cohort must explicitly
check unchanged block call-chain source before cross-device interpretation.

CPU03274 and GPU03275 each complete four fresh callback/shape signatures
(D3,D5,D3,D5) and200 alternating-input public calls. Every complete result matches
the original at unchanged tolerances. Each owner traces once, and every measured
owner, graph, dependency scope and callback is collected after release.

| Observation | CPU | GPU2 |
| --- | --- | --- |
| RSS after first /fourth signature release |2690.13 /4447.27MiB|2875.80 /4527.59MiB|
| Fifty-call RSS growth per signature |0.48--0.61MiB|0.45--0.48MiB|
| Mapping count after first /fourth release |6385 /11623|4198 /4201|
| Anonymous executable maps after first /fourth release |1119 /2865|See full category observations|
| GPU allocator at final release |N/A|5376 current /292608 peak bytes|

The GPU probe is explicitly shared-device attribution, not timing qualification.
Native residency persists despite Python collection. These observations support
reusing a fixed signature and investigating process-lifetime containment; they
do not prove a native leak, native eviction, or safety under unlimited callback
and shape churn. Observer snapshots also consume memory, so their small reuse
growth cannot be attributed wholly to the numerical implementation.

Artifacts under `artifacts/filter-gradient-repair-20260917/`:

- `block-public-costs-cpu-03272.json` and individual03255--03272 manifests.
- `block-public-memory-checkpoint-03275.json`, including hashes and sharing veto.
-03274/03275 `block-public-churn.json` and append-only stage JSONL.
- `cost-preflight-declined-20260923T044906081006Z.json`.

The cost unit has consumed21/42 workers, including analyzer/policy prechecks and
the rejected GPU timing. The churn unit closes at2/4 workers and234.452300/1200
charged seconds. Both remain within cumulative32 CPU/52 GPU process-hour caps.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain CPU cost evidence |18 complete original-record comparisons pass |No numerical or relative-cost veto |Small fixtures and three repeats |Complete separately frozen GPU costs |General performance ranking |
| Reject03273 timing |Numerics pass |Observed foreign compute |Unshared GPU costs absent |Fresh complete GPU cohort |GPU cost qualification |
| Keep E4 memory repair open |Reuse and Python collection pass |Native residency grows across signatures |Native allocator ownership and supported process lifetime |Bound signature reuse and verify process containment |Leak freedom or arbitrary-lifetime safety |

Review: full public costs include construction, tracing and result reporting;
neither component timing nor Python collection closes a complete endpoint gate.
The CPU measurements and both churn probes answer their bounded questions.
Remaining initializers, DZ5 supervision, endpoint dispositions, remote integration
and final source-frozen qualification still prevent main merge.
