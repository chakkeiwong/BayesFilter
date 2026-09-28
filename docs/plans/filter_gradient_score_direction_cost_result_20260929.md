# Complete direction owner cost screen

Runs04778–04782 pass the four fresh CPU process arms and162 final
cost/qualification readback and policy checks. Complete shared numerical,
status and diagnostic outputs are exact in both pairs. Fixture, numerical
sources, worker environment, affinity and framework provenance match. The
screen consumed80.150181 CPU process-seconds across five workers, zero GPU
seconds, within its6-worker/1800-second allocation.

| Consumer | Prior warm median | Enclosing warm median | Enclosing/prior | Prior cold total | Enclosing cold total | Observed warm RSS increase |
|---|---:|---:|---:|---:|---:|---:|
| LEDH with diagnostics | 7.877906 ms | 5.026228 ms | 0.638016 | 10.598161 s | 11.399576 s | 96.348 MiB |
| Resampling KDM | 10.584185 ms | 5.460592 ms | 0.515920 | 12.797217 s | 13.632661 s | 90.863 MiB |

Cold totals include owner construction and first synchronized execution. The
enclosing factory traces the inner declared schema during construction, so
timing only its first call would understate cold cost. Each process used three
conditioning calls and30 synchronized warm calls. The prior arm stacks only
values/scores during timing, as its original endpoint did; auxiliary tensors
are normalized for comparison after primary measurements. All shared nested
outputs are synchronized. These are instrumented owner costs, excluding data
generation, complete study orchestration, HLO export and comparison-owner
compilation. One fresh process per arm cannot support a statistical ranking.

Factory reuse and one trace pass. Sampled RSS growth between cold and warm
measurements was0.090/0.449 MiB for prior/enclosing diagnostic LEDH and
0.102/0.285 MiB for prior/enclosing resampling KDM. Python collection did not
reduce the recorded resident memory while owners remained retained. These
bounded observations do not establish an unbounded leak or compiler eviction.

There is a measurement inconsistency to retain: the artifact field
`primary_peak_rss_bytes` comes from resource.getrusage(RUSAGE_SELF).ru_maxrss,
and was below the separately sampled `/proc/self/status` VmRSS in all four
processes. It must not be used as a strict peak upper bound. Observed VmRSS
samples are separately preserved and are the basis for the table; exact peak
capacity is unproved. No GPU allocator/capacity claim follows from CPU RSS.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Preserve execution repair and measured tradeoff | Exact outputs; warm owner medians lower | Numerical/execution checks pass | One process per arm | Continue nonlinear consumer repair, then bounded matched cost follow-up | Statistically established speedup |
| Keep cost acceptance open | Cold time and RSS rise | Required follow-up remains | Compiler/function ownership versus process noise | Inspect construction/compile residency and confirm matched pairs | Memory improvement or whole-master completion |
| Preserve memory-metric discrepancy | Independent OS measures disagree about peak | Strict peak-capacity claim blocked | Rusage/VmRSS accounting and sampling | Cross-check VmHWM and smaps_rollup outside primary timings | Live tensor memory inferred from process RSS |

| Inference status | Finding |
|---|---|
| Hard veto screen | Full record parity, status, source and environment checks pass |
| Statistically supported ranking | None; one process per arm |
| Descriptive differences | Warm medians lower36–48%; cold totals higher6–8%; observed RSS higher91–96 MiB |
| Default readiness | Whole-program readiness remains open; this is a CPU-reference screen |
| Next evidence | Matched repetition, explained residency/peak accounting and uncontended GPU cost/capacity |

The next memory diagnostic should compare `/proc/self/status` VmRSS/VmHWM,
RssAnon/RssFile, smaps_rollup and rusage at synchronized construction/cold/warm
boundaries. Reserve at most two CPU workers/600 process-seconds when activating
that diagnostic; no numerical source change or cache mutation is needed.
For any compiler-ownership intervention, first inspect the retained inner
ConcreteFunction and outer direction-loop definitions. A test of invoking the
same underlying Python numerical authority inside the enclosing graph versus
its concrete function call must remain an execution representation change,
preserve all fields and statuses, and pass current CPU/GPU derivative checks
before adoption. Opcode counts or weak-reference collection alone cannot prove
the cause. Freeze a matched cohort and its allocation before running it; do not
change RNG, tolerances, dtype, scientific target or filter equations.

Skeptical terminal review: host-dispatch elimination plausibly explains lower
warm medians, but instrumentation and scheduler noise remain alternative
explanations for magnitude. Extra function/graph/compiler ownership plausibly
explains higher residency; it is not yet a demonstrated cause. The strongest
limitation is the unreplicated cost comparison and inconsistent peak counters.
The source-bound readback now scans only the current unit, avoiding the initial
full-history reporting overhead. This repair does not transfer Gaussian
evidence to nonlinear consumers, accept the separate streaming RNG slowdown,
resolve native residency generally, or authorize merging main.
