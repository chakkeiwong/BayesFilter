# Registered LEDH seeded execution and LM precision result

The registered `canonical_value_and_diagnostics` endpoint now uses the shared
seeded TensorFlow/XLA value owner. Runtime NumPy and Python time/stage loops
have been removed from this wrapper. Existing SeedSequence/PCG64/Philox streams
are preserved; this is not the separately approved geometry RNG migration.
Observations and seed words are dynamic operands. A caller-owned factory has
a stable signature and one trace; the one-shot wrapper refreshes Python
callback closures on every invocation. Model strings and trimmed marginal
histories remain reporting-boundary operations. Failed resets produce NaN
public value, false validity and explicit code/index, with raw diagnostics
retained. Full canonical LEDH reconstruction/admission remains excluded.

The CPU endpoint initially passed, but GPU04672 found a healthy dual-trust
ESS difference1.31069892e-5 against the unchanged1e-6 gate. The failure was
preserved and costs held.04673 showed bitwise-identical normal inputs and
identical supplied-input/seeded XLA results.04675 reproduced the mismatch by
compiling only the shared FP32 reset; its trust correction alone showed the
same error. FP64 eager/graph/XLA agreed within4.9e-15.04676/04677 isolated
TF32 multiplication inside the small shared scaled LM solve. Captured normal
matrix errors reached5.04e-4 and coefficient error8.59e-3 against FP64; turning
off TF32 only inside that diagnostic helper reduced first trust-cloud error
from2.23e-4 to5.68e-7. The observed conditions8--196 do not establish severe
ill-conditioning, and all three resets were valid.

The reviewed repair uses explicit tensor products/reductions for the tiny
Gram, right-hand-side and analytical derivative contractions in
`genut_shape_lm_tf.py`. It preserves the equations, float32 dtype, damping,
strength, floor, reset algorithm, tuning controls and TF32-enabled default.
No global precision switch or tolerance relaxation was installed. Independent
FP64 matrices, multiple nonzero analytical directions and converged five-point
differences qualify the helper on CPU/GPU. The previously failing XLA complete
record is bitwise unchanged; the inaccurate eager TF32 helper is corrected.
The new frozen-filter comparison deliberately shares that repaired dependency.
It proves execution agreement with the corrected arithmetic, not equality to
the erroneous prior eager output. The old failure and component references
remain available for inspection.

| Evidence | Result |
|---|---|
|04669 CPU/04671 GPU seeded stream |7 checks per backend; integer states/uniforms exact including160-bit carry; existing FP64/FP32 normal bounds preserved |
|04670 initial CPU endpoint |8 checks pass, including mutable closure, tensor time, changed inputs, replay, HLO and collection |
|04672 initial GPU endpoint |7 pass,1 healthy numerical mismatch; preserved |
|04673/04675--04677 localization |RNG excluded; reset and tiny LM TF32 mechanism isolated; explanatory diagnostics |
|04674 |Diagnostic argument-binding harness failure; one bounded retry04675 succeeds |
|04678 CPU/04679 GPU LM repair |7 checks per backend; independent accuracy and analytical derivatives pass |
|04680 CPU/04681 GPU endpoint renewal |8 per backend pass; healthy and rejected records, registry wiring, one trace, stable HLO and owner collection |
|04682 broad CPU regression |46 pass; one existing static pfor-policy failure in the untouched batch-fused route remains F14 |
|04683 GPU numerical regression |36 pass; seeded inputs, fixed-input value, resampling and analytical-score owner |
|04684--04686 CPU costs |All three fresh-process arms pass numerical checks |
|04687 final readback/policy |162 pass; current dependency hashes checked;278 sources guarded,1436 existing exact exceptions, no new allowance |

The tests use TensorFlow2.19.1 and the existing tf-gpu environment. GPU evidence
used the eligible GPU2/GPU3 UUIDs recorded per run, memory growth verified before
logical initialization and trusted-session provenance. GPU2 was used before
another campaign occupied it; GPU3 numerical checks were shared and supply no
performance attribution. Exact commands, timing, source hashes, seeds, devices,
raw records and logs are retained in the versioned run manifests.

CPU costs use N=8,d=2,T=3,seed123, one fresh process per arm and15 warm calls,
with the same current shared reset. They are descriptive, not a statistical
speed ranking or a capacity qualification. The graph arm is an explicit
non-default reference. Validation and HLO inspection are outside measurement;
HLO export is omitted in cost workers.

| CPU arm | Cold seconds | Warm median ms | Host RSS growth after warm MiB | RSS growth after extra one-shot calls and release MiB |
|---|---:|---:|---:|---:|
| Frozen prior eager wrapper, repaired shared reset |1.342|1197.063|31.69|31.69|
| Candidate graph reference |2.646|11.575|159.77|296.24|
| Candidate XLA |5.029|1.040|521.86|993.70|

The two additional one-shot XLA invocations take5.200/4.826 seconds. The Python
owner is collected, but approximately994MiB additional host residency remains
before process exit; Python collection does not prove compiler/native memory
release. CPU allocator accounting is unavailable. These results expose a real
compile/lifetime cost and favor explicit fixed-configuration owner reuse in the
measured scenario. They do not identify every retained native allocation or
prove indefinite memory growth. Fresh repeated-configuration lifecycle and
streaming seeded-memory/capacity repairs remain necessary: the current seeded
composition materializes a[T,N,d] process-noise array, unlike the original
per-time buffer. No long-horizon memory claim is made.

GPU matched costs remain pending. The final preflight declined before launching
a worker because both non-display devices had another campaign's process.
Receipt:`cost-preflight-declined-20260928T165541177243Z.json`. Desktop devices
were not used. CPU results cannot replace the required GPU timing evidence.

| Decision | Criterion/veto | Uncertainty and next action |
|---|---|---|
| Accept bounded seeded value/LM execution repair |Independent helper/derivative and CPU/GPU endpoint gates pass without tolerance changes |Stream process noise and qualify reusable/one-shot ownership memory |
| Preserve F14 failure |Static pfor ban fails on pre-existing optional batch route |Review approval scope, actual callers and diagnostic/admission disposition; do not delete the test |
| Hold capacity/performance completion |Cold RSS cost and random-buffer scaling are observed; GPU costs unavailable |Bounded lifecycle/capacity phase plus uncontended GPU cost renewal |
| Keep F07/master/main open |Score consumer migration, locator/geometry/source-evidence and terminal gates remain |Continue the master queue; no main merge |

Hard numerical rejection behavior and the corrected LM mechanism have evidence.
No stochastic ranking, canonical method, posterior, trained-map or HMC status
is established. The strongest alternative concern is that tiny fixtures omit
larger-capacity and callback-use patterns; the explicit next phase addresses
that gap. Current-source dependency changes invalidate automatic transfer of
old full-reset cost reports. Primary-agent review completed; no independent
review asserted.

This unit used19/24 workers,300.822211 CPU seconds and456.624632 GPU seconds,
including all failed attempts. Remaining global budget is25.850588 CPU and
25.166452 GPU hours under unchanged56/52-hour caps. No numerical worker is
active at this checkpoint. Archive integrity is reported separately and is
not numerical admission or whole-master completion.

Evidence archive: `artifacts/filter-gradient-repair-20260917/ledh-seeded-public-04687-evidence.tar.gz`, SHA256
`e7b1308380b0dcf7ce50060b1786b77e2bc6804b090ed633d51353a22420ef45`.
All243 members were reopened and checksum-verified. The archive preserves
run artifacts and final numerical/test sources; earlier harness revisions have
source hashes and logs, not a complete byte snapshot of every intermediate edit.
The pre-repair shared LM authority is preserved at Git aabd2b167.
