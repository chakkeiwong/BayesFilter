# Complete sequential public cost checkpoint

GPU renewal03160--03177 completes all18 clean, matched public comparisons on
GPU2 UUID541e1e19-2df4-9064-4db9-9d0d2abc3eba. Three fresh processes per arm and
dimension pass full original and changed-input records at unchanged tolerances.
The analyzer verifies source/environment/input identity, memory growth, physical
UUID and sampled device sharing. The CPU and GPU cohorts are separate source
freezes; the later GPU cohort includes the approved resolution guard.

| Dimension | Prior warm median | XLA warm median | Prior cold median | XLA cold median | Extra maximum observed RSS |
| --- | --- | --- | --- | --- | --- |
|3|39.996ms|23.726ms|8.917s|13.304s|476.750MiB|
|5|115.852ms|100.179ms|10.057s|14.202s|482.219MiB|

The host-RSS trigger fires at both dimensions. Representative XLA stages rise
from941MiB prepared to1073MiB built/traced and1714/1721MiB after the first call;
warm RSS grows by at most0.148/0.156MiB over20 calls. This is consistent with
the earlier sequential compilation/residency attribution, but does not establish
native eviction or safe arbitrary signature churn. E4's bounded signature/reuse
and process-lifetime containment investigation remains required.

The D3 GPU allocator peak trigger also fires:58,880 prior versus132,608 XLA
bytes, a2.252 ratio and72KiB absolute increase. D5 is89,856 versus164,864 bytes
(1.835 ratio). The XLA allocator is already near its peak at the cold call,
remains flat across the warm calls, and has4,352 current bytes at warm completion
versus7,680 prior. Changed-input outputs add1,024 bytes of current XLA allocation.
The enclosing program's temporary allocation and retained outputs must be
separated in the E6 disposition; these small-fixture observations do not justify
a general device-memory bound or silently waive the relative trigger.

Receipt: `artifacts/filter-gradient-repair-20260917/sequential-public-costs-gpu-03177.json`.
The complete renewal uses58 workers and2144.744946 seconds below64/14,400.
All43 GPU block consumers and129 policy checks also pass in03158/03159.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain GPU cost evidence | All18 full-record and provenance checks pass | No numerical or sharing veto | Two small extents/three process repetitions | Continue public ordered-block integration | No general speed ranking |
| Keep memory disposition open | Host growth and D3 device-peak triggers fire | Repair triggers | Native residency and temporary allocation by stage | E4 reuse/churn containment and E6 explicit cost disposition | No leak freedom or arbitrary-lifetime claim |

Review: source-frozen checks answer the public sequential cost question. The
graph arm retains mixed compiled dependencies, so it is not an identical-graph
XLA ablation. Sampled sharing cannot exclude every brief competing process.
Costs do not establish canonical LEDH, HMC or terminal campaign readiness.
The following CPU and residency checkpoints remain the supporting history.

All18 CPU cost workers02970--02987 pass complete original and changed-input
payload comparisons. The numerical authority is3582b4ac;48acf5e96 is the prior
public implementation used for mechanism/cost comparison. Three fresh processes
per arm and dimension measure the prior public endpoint, the current explicit
outer-graph reference and the current XLA public endpoint. Construction/tracing,
including work moved into construction by scoped ownership, belongs in cold
time. Twenty synchronized complete public calls include result serialization.

| Dimension | Prior warm median | XLA warm median | Prior cold median | XLA cold median | Extra maximum observed RSS |
| --- | --- | --- | --- | --- | --- |
|3|19.332ms|7.620ms|3.936s|7.548s|555.383MiB|
|5|33.024ms|22.096ms|4.128s|7.874s|562.703MiB|

Warm ratios are0.394/0.669 and cold ratios1.918/1.907. The predeclared host-RSS
trigger fires at both dimensions. Representative stages rise from about582MiB
prepared to714MiB built/traced and1490/1504MiB after the first XLA call. The
twenty-call warm interval grows by at most0.293MiB. The graph-reference peak is
about894MiB and the prior peak935/942MiB. The XLA graphs have6869/6875 nodes and
about1.11MB serialized size. These observations locate the main increase at
compilation/execution; they do not identify its allocation owner or prove a leak.

The saved analyzer and receipt are
`scripts/analyze_filter_repair_sequential_public_costs.py` and
`artifacts/filter-gradient-repair-20260917/sequential-public-costs-cpu-02987.json`.
Numbered manifests preserve commands, seeds/configs, source/environment hashes,
wall time, device scope and complete records. Analyzer/preflight negative checks
pass13 cases in02968;128 policy checks pass02969. The source cohort is frozen
through02987. GPU availability attempts45414/68660 ended before worker launch;
eligible non-desktop devices had foreign compute and desktop fallback conditions
were not met. No GPU numerical or timing result is inferred from these attempts.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain CPU cost evidence | All18 original-record comparisons pass with matching inputs/source/environment | No numerical veto | Small fixtures and three process repeats | Attribute sequential startup/residency | No general speed ranking |
| Keep memory investigation open | Host RSS increase exceeds256MiB | Repair trigger, not a tolerance waiver | Native compilation versus retained Python graphs and observer overhead | Fresh sequential-specific priming/reuse/release controls | No leak-freedom or native-eviction claim |
| Defer GPU costs during contention | Clean preflight required | No worker launched | GPU cost cohort absent | Resume on an eligible uncontended UUID | CPU cannot close GPU/default-target gates |

Post-run review: graph mode retains declared compiled dependencies, so this is
not an identical-graph compiler ablation. Prior and current cold scopes are
complete public work. Snapshots are observed memory, not exact peak bounds;
the earlier posterior attribution cannot close this distinct sequential
endpoint's trigger. Source changes before GPU resumption require a separately
identified cohort and analysis rather than presenting it as the same freeze.
The full campaign remains open, with main unmerged.

Sequential residency qualification completes in02989--02993 CPU and03000--03004
GPU after the device-registration repair. Eight probes pass9,600 alternating
complete public calls against original records, one trace and collection of
root, graph, scope and callback owners. Two observer-only controls pass with no
numerical calls between memory snapshots. Policy renewal02999 passes129 checks.
The saved receipt/analyzer are sequential-residency-analysis-03004.json and
analyze-sequential-residency-03004.py under the shared artifact root.

| Observation | CPU | GPU2 |
| --- | --- | --- |
| Minimal XLA control RSS increase |64.83--64.91MiB|26.01--26.08MiB|
| Endpoint first execution after priming/tracing |713.76--725.56MiB|630.74--638.25MiB|
|1,200-call observed RSS growth |1.51--1.60MiB|1.37--1.67MiB|
| Observer-only retained snapshots |1.86MiB|1.96MiB|
| Python owners after final release |All collected|All collected|
| Executable-map counts during reuse |Flat|Flat|
| GPU allocator during reuse |N/A|5376 bytes current;131840 bytes peak|

The primedD3 CPU trace-to-cold mapping difference includes559.24MiB heap,
112.15MiB other anonymous memory,35.21MiB TensorFlow library residency and6.10MiB
executable mappings. This attributes the increase to compilation/first execution
at the stage and mapping-category level, without identifying the exact allocator
owner. Memory remains resident after Python collection. Observer-only growth
is comparable to the reuse growth; it is not valid to subtract independently
measured process values and claim zero leakage. The small signature family
does not close the known many-signature native mapping-exhaustion concern.

GPU2 UUID541e1e19-2df4-9064-4db9-9d0d2abc3eba has verified growth and recorded
foreign compute throughout these probes. These are resource attribution and
correctness observations, not uncontended cost measurements. The CPU/GPU source
cohorts differ only in the declared routing correction and its regression test.
Five accidental duplicate CPU references02994--02998 remain preserved and
charged; none is classified as GPU evidence. All17 workers in this unit used
956.52 seconds under the2400-second ceiling. Exact cumulative charges through
03004 are59647.997930025376 CPU/59401.31348332534 GPU seconds.

Reviewed next action: continue the ordered-block dependency, retaining startup
RSS/native-residency as open cost limits. Clean GPU costs, wider signature churn,
remaining endpoint repairs and terminal integration are still required.
