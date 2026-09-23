# Attribution of the GPU graph-reference timing trigger

The complete SVD cost cohort records a24.7% T=3 GPU graph warm-time increase.
Both arms execute the graph-reference branch, and their node counts/bytes match.
That observation is insufficient to attribute the difference. Preserve the
original cohort and investigate the actual graphs and execution order.

Use the same 3-state T=3 fixture and f4ea46dde original module isolation as the
cost unit. In one process construct the original and candidate graph functions.
Compare their complete GraphDefs after replacing only generated FunctionDef
numeric suffixes and omitting source-debug metadata. Keep operations, edges,
constants, shapes, devices and all execution attributes. Reject ambiguous name
normalization. Save both normalized definitions and their hashes; a mismatch
requires inspecting the exact difference before making any identity claim.

After one cold call per owner, execute40 balanced ABBA/BAAB blocks, synchronizing
all outputs on every call. Preserve all timings/order and bitwise same-arm
replay; independently check both outputs against closed-form Kalman. Observe GPU
sharing throughout, record growth and UUID, and require unshared cost preflight.
Run at most three fresh GPU processes, with repeat IDs0/1/2 and one matching
UUID, then inspect the descriptive ratios. If graphs match and the20% trigger
does not recur, classify the prior cost signal as not reproduced under matched
interleaving; do not assert its environmental cause or erase the original
measurement. If it persists, investigate runtime placement/scheduling before
disposition. No speed superiority claim is authorized by this diagnostic.

Reserve at most5 workers /900 charged seconds for three observations, one
localized harness retry and a CPU-only graph-identity smoke if necessary, under
unchanged32CPU/52GPU caps. Use the stable campaign runner's registered
svd_graph_attribution_gpu group,300-second bound, explicit repeats and frozen
source. Preserve every numbered artifact and stop unchanged retries on failure,
invalid sharing/provenance or exhausted budget. No package, allocator, thread,
numerical or default-policy change belongs in this unit.

Review: separate-process unbalanced order could confound a small kernel timing;
interleaving observes both owners under nearby conditions. Repeated calls within
one process are dependent and cannot be counted as160 independent replications.
Graph comparison is an executable check, but over-normalizing attributes would
make it meaningless, so only nonnumerical names/debug metadata may be removed.
This addresses the graph trigger only; CPU compilation and full consumer
capacity remain separate obligations.

03700 passes both independent Kalman comparisons and same-arm replay with clean
sharing observations, but graph identity fails. Recursive comparison of the
saved complete definitions finds exactly seven differences: five redundant
While formal argument names embed condition-function IDs43/416; a captured zero
Const is named40/413 and its one input edge uses that same name. Its value,
shape and attributes agree exactly; no numerical/execution difference is found.
The observed warm median ratio1.00118 is explanatory only while identity fails.

Repair the diagnostic name comparison to alpha-rename those exact generated
definitions and their corresponding edges, preserving constant values, edge
ports, operation types, device settings and every execution attribute. Keep the
original mismatch and add adverse checks for changes to constants, operations,
edges/ports, devices, argument types and capture shapes. This is a harness
identity repair, not a numerical tolerance or timing-trigger change. Then run
three fresh GPU repetitions with frozen source and matching UUID. The original
failure plus three repetitions remain within the5-worker/900-second allowance.
Review: global string replacement of numeric IDs could erase a real constant
change. Restrict replacement to name definitions and edge identifiers, and make
a string-constant mutation an explicit negative control.

03701--03703 pass all seven checks per process, including the six adverse graph
mutations. Exact source closure2735 files, GPU2 UUID and verified memory growth
match across all three processes. The normalized complete graphs are equal;
independent Kalman numerics, bitwise same-arm replay and sampled sharing checks
pass. Ratios are1.00236572,0.99729573 and1.00213658. A separate standard-library
artifact analysis rechecks every ratio/order, numerical array, graph equality,
source set and device/growth provenance. Immutable receipt:
`artifacts/filter-gradient-repair-20260917/svd-graph-attribution-03703.json`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Original24.7% graph trigger not reproduced under matched interleaving | Three ratios stay below1.20 with identical numerical graphs | Numerical, replay, provenance and sampled-sharing checks pass | Cause of original separate-process difference remains unknown | Preserve original cohort; continue consumer costs and capacity | No environmental-cause attribution or speed superiority |

The four workers including03700 consume35.489403 seconds. This closes this
bounded explanatory experiment. It does not dispose of CPU compilation/RSS
overhead, latest SVD consumer costs, public endpoint capacity or default readiness.
Post-run review: normalization is limited to inspected generated names and has
negative controls; brief unsampled sharing and process-dependent scheduling
remain possible. The observations cannot establish why the original signal
occurred or make three dependent call sequences a statistical ranking.
