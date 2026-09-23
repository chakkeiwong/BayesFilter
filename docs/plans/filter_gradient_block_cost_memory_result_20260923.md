# Public ordered-block costs and memory lifetime

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
