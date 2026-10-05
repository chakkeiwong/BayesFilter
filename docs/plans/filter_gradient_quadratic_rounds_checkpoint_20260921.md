# Paired refinement public XLA checkpoint

The paired-local public refinement endpoint now runs its initial evaluation,
replay, ordered probes, model/factor decisions, trust proposals and bounded
round recurrence in a stable TensorFlow/XLA controller. Original seeds, target
analytical scores, thresholds, strict selection, rejection precedence, padding,
callback order and numerical histories are preserved. Center/score/factor/
precision results remain tensors; host iteration only formats completed reports.
The API's default uniform-cloud mode is still separate migration debt.

The complete dependency closure at `3582b4ac` is the numerical authority, with
its trust solve in graph mode because original XLA trust precision was wrong
in demonstrated cases. The original graph reference is an engineering baseline,
not an admitted default. Every numerical field uses unchanged atol=rtol=1e-10;
discrete decisions, callback counts and ordering agree exactly. No algorithm,
optimizer, sampling distribution, seed or tolerance was changed.

Qualification through 02059 includes 102 complete public/raw/original CPU/GPU 3
cases at dimensions 1/3/5, runtime-input HLO/enclosing-XLA checks, six cache
identity/ownership/mutable-target cases, 280 public consumer cases, and 72
policy/controller tests after fixing repeat resumption. No failures or skips
occurred in these final groups. Earlier failed attempts remain preserved in
[the execution contract and investigation](filter_gradient_quadratic_rounds_20260921.md).
The driver previously ignored the requested repeat when resuming test matrices;
it now matches the repeat key, with regression coverage for repeat 0/1/2.

Public measurements 02060--02095 use three fresh processes per original/graph/
XLA arm at each dimension and device, with twenty warm calls per process. They
include input validation, controller construction/cache lookup, complete
refinement, report materialization and payload serialization. Cold includes
tracing and compilation. All initial and changed-input records match, all input
and baseline dependency hashes agree, graph reference contains no nested XLA,
and XLA keeps every explicit input as a runtime operand with one trace.
GPU arms all use idle-preflighted GPU 3 (RTX 4090), verified memory growth and the
recorded managed-session trust basis. CPU reference arms hide GPUs. Exact
commands, source hashes, environment and wall times are in each run.json.

| Device / dimension | Original warm ms | Graph warm ms | XLA warm ms | Original cold s | XLA cold s | Extra XLA host RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU / 3 | 379.58 | 30.19 | 15.79 | 0.419 | 4.807 | 518 |
| CPU / 5 | 428.84 | 47.65 | 28.84 | 0.482 | 5.357 | 528 |
| GPU 3 / 3 | 497.89 | 118.01 | 25.97 | 2.777 | 8.639 | 277 |
| GPU 3 / 5 | 569.49 | 144.23 | 42.87 | 2.793 | 9.257 | 280 |

These are medians across three process medians. XLA warm ranges are
15.35--15.88 / 27.60--28.87 ms CPU and 25.93--26.29 / 42.59--43.65 ms GPU 3.
They are descriptive engineering observations, without an uncertainty analysis
supporting statistical superiority. GPU XLA allocator peaks are 104704/139008
bytes; warm current remains 6400 bytes. Reservation is not live tensor memory.
Analysis: `quadratic-public-repeated-costs-02095.json` under the campaign root,
produced by `/tmp/analyze_quadratic_public_repeats_20260921.py 2060 2095`.

The host increase exceeds the declared 256 MiB investigation threshold. It is
concentrated in tracing/first compilation, with real anonymous/private memory
growth in smaps. Graph size is 3734 nodes at D3/5; capacity 1/4/8 has
3734/3734/3738 nodes, ruling out round-count graph unrolling in that scope.
The isolated controller's 3000 alternating-call measurements (02029--02034)
retain exact records/HLO with final 1000-call host growth 0--16 KiB and constant
GPU live allocation. Higher cold memory is an explicitly retained compilation
tradeoff for this checkpoint, not a repaired or absent cost. No claim is made
that it is unavoidable or acceptable at arbitrary dimensions.

A separate fresh-construction test (02035/02036) found >1 GiB additional host
RSS across eight controllers. Python graphs/functions/target variables are
collectible, but process RSS remains high after collection; native executable
release is unproved. Public calls therefore reuse one most-recent controller
for identical callback identity, dimension, static configuration and execution
mode. Seed stays dynamic. Twenty reused calls show small host drift, and cache
replacement releases old Python ownership. Changing many target/configuration
identities may still grow native memory; no general eviction claim is supported.

The cache follows TensorFlow tracing semantics. Mutable numerical target state
can use tf.Variable; its changed value/score/factor behavior is tested. Redefining
Python closure/attribute state requires a new callback or explicit cache clear.
Details and the complete-graph reference option are documented in
[the API note](../reference/paired-quadratic-refinement.md). Trace counts describe
traced branches; executed fit/trust counters are reported separately.
`full_initializer_xla` remains false pending composition qualification.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain paired public controller on repair branch | All original numerical records, HLO and consumers pass | No demonstrated paired numerical failure remains | Larger dimensions and actual DZ5 target | Preserve checkpoint; continue campaign | Whole initializer or repository compliance |
| Retain bounded single-target reuse | Same program/trace across dynamic inputs; old Python resources release | Fresh-construction growth rules out naive rebuilding | Native executable retention during target turnover | Keep cache contract explicit and target-specific memory gates | General leak freedom |
| Record higher compile RSS/cold cost | Three-repeat public costs and sparse growth evidence | Host threshold fired and was investigated | Larger complete workloads may magnify overhead | Carry this cost into whole-lifecycle/DZ5 comparisons | Zero regression or universal acceptability |
| Keep main unmerged | All master terminal gates must pass | Dense diagnostic, original D5 sequential, broader controllers and external evidence open | Full current-source terminal qualification | Continue authorized repairs | Campaign completion |

| Inference status | Disposition |
| --- | --- |
| Hard veto screen | Paired final numerical/consumer tests pass; earlier failures preserved |
| Statistically supported ranking | None claimed |
| Descriptive differences | Lower observed warm costs, higher cold and host compile costs |
| Default-readiness | Paired branch execution qualified in tested scope; no whole-repo/default scientific claim |
| Next evidence needed | Larger complete initializers, actual DZ5 target/transition blocks and all master terminal gates |

Primary-agent result review finds no dropped numerical field or hidden fallback.
Full records and actual public wiring support the execution claim; warm-only
benchmarks would have hidden both cold overhead and construction retention.
The weakest evidence is low-dimensional synthetic scope and native memory
ownership. Independent terminal review remains pending. No package, OS,
external source/pin, HMC tuner/sampler or canonical LEDH admission changed.
The singular dense-condition definition proposal is separate and awaiting the
owner's response; its failing tests and uniform non-XLA exception remain intact.

Final inventory 02096 covers 2939 working Python files,2938 parsed and the same
one vendored-reference parse error. Partial guard passes204 sources/1289 exact
exceptions; completed reporting is the only new exception role. Focused new/
touched numerical/reporter tests pass Ruff and whitespace checks. The older
campaign test file retains two unrelated C408 warnings. Charged through 02096:
CPU 45398.626980266694/115200 seconds; GPU 42122.65920439076/187200 seconds. No
numerical worker is active. A read-only status lock briefly blocked the first
audit invocation before launch; retry after status exit produced02096. No
package/OS/external source or pin was changed. Branch commit/push follows.
