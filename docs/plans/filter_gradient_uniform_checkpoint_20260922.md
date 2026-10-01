# Uniform refinement execution checkpoint

Worktree: `/tmp/bayesfilter-filter-gradient-xla-validation-20260918`, branch
`repair/filter-gradient-xla-validation-20260918`, based on `415eaa8f`.
Status: CPU public qualification and costs complete; public GPU checks/costs wait
for GPU3 idle preflight. Main remains unmerged.

Uniform-cloud refinement now calls the same native TensorFlow round controller
as paired-local refinement. The Python round/partition computation is removed.
The fixed-capacity program preserves the original Philox float64 draws, ordered
batched target calls, exact incumbent selection, replay, fit, factor validation,
trust updates, stopping decisions and complete histories. Seed, center, scale
and optional initial evidence remain runtime inputs. One cached callback and
configuration avoids rebuilding the program on repeated public calls.

The original `3582b4ac` dependency closure, with graph-mode trust solving, remains
the numerical authority. Its old XLA trust errors are not a comparator to match.
The uniform 33-row fixture exposed a 5.24e-10 SVD condition error. The repaired
XLA SVD uses binary64 convergence tolerance and the standard values-only SVD
pullback; it passes independent values, condition derivatives, full dense and
controller records. No rank threshold, optimizer setting or tolerance changed.
The graph reference uses ordinary TensorFlow SVD.

Public center/value/score/factor/precision results remain tensors. Completed
diagnostic leaves use the host-list representation already used by paired mode;
the serialized numerical schema is preserved. `jit_compile_trust=False` selects
the complete graph reference, and execution metadata records the selected mode.
Composition with multistart localization remains a separate compilation boundary.
See [API semantics](../reference/paired-quadratic-refinement.md).

Native qualification includes 48 original uniform records at D1/D3/D5 on each
device, changed-input/HLO and resource checks, nondefault batches and partial
clouds, and shared paired regressions. GPU runs 02287--02293 pass 111 checks,
following 69 focused GPU cases in 02279--02286. Public CPU runs 02341--02345 pass
204 checks: 18 original/cache/state, 51 quadratic-center consumers, 38 paired,
25 batch records and 72 policy/controller tests. Public GPU qualification remains
pending because its first preflight found an unrelated workload; no worker ran.

The source guard now covers the whole public module. Its four exceptions only
validate a fixed configuration schema or serialize completed results. There are
no numerical-loop exemptions in the new controller. Guard coverage remains
partial across the repository: 204 sources and 1,292 exact exceptions.

## Costs and memory

Native controller measurements include complete record materialization. Each
row below is one fresh process per arm and is descriptive only. Public endpoint
measurements with three fresh processes per arm are complete on CPU; GPU remains
pending. The table reports the separate native measurements.

| Device/dimension | Original warm ms | Graph warm ms | XLA warm ms | XLA extra host RSS versus original |
| --- | ---: | ---: | ---: | ---: |
| CPU D3 | 436.89 | 45.96 | 25.68 | 517.8 MiB |
| CPU D5 | 446.79 | 57.00 | 34.91 | 517.6 MiB |
| GPU3 D3 | 618.14 | 178.85 | 37.30 | 267.4 MiB |
| GPU3 D5 | 627.65 | 204.19 | 54.98 | 266.0 MiB |

Total XLA cold cost, including construction, tracing and first execution, is
4.11/4.21 seconds on CPU and 8.39/8.57 seconds on GPU3. All four ratios exceed
2x the original. The first analysis retained those times but omitted the cold
trigger from its list; the corrected version preserves the old artifact and
reports it explicitly. GPU allocator peaks fall from about 8.5 MB to 126/160 KB,
which does not cancel the separate host-memory increase.

Six D5 capacity checks cover round limits 1, 4 and 8 on both CPU and GPU, with
3,000 alternating-input calls each (02278 and 02300--02304). Complete records,
one trace and unchanged HLO pass. Graph size is 3,695/3,699/3,699 nodes; HLO is
about 3.21--3.23 MB. Final 1,000-call RSS growth is 0--12 KiB. GPU current remains
7,168 bytes; peak allocation increases with the retained output history. Most
host allocation appears at first execution. These results support bounded warm
reuse in the tested scopes; they do not prove arbitrary target turnover, native
executable eviction, exact process peak memory or general leak freedom.

Renewed dense dependency measurements (02305--02340) use three fresh processes
per original/graph/XLA arm at D3/D5 on both devices. Every initial and changed
record agrees at the unchanged criteria. D5 original/XLA warm medians are
0.574/0.623 ms CPU and 2.503/4.241 ms GPU. The GPU penalty remains 69.4%, with
repeat ranges 2.458--2.507 versus 4.223--4.276 ms. The accurate SVD and related
linear algebra explain this standalone cost. The complete controller's lower
dispatch cost does not mean the dependency penalty disappeared. CPU dense
compilation also exceeds the 2x cold threshold. Retaining numerical accuracy
and the native controller is the current candidate tradeoff, subject to public
consumer and complete cost qualification.

All evidence is under the shared campaign artifact root:
`docs/plans/artifacts/filter-gradient-repair-20260917/`. Analyses:
`uniform-native-costs-02299-v2.json`, `uniform-capacity-memory-02304.json`, and
`dense-numerics-repeats-02340.json`; exact analysis scripts are archived there.
Run manifests retain commands, source hashes, seeds, environment and durations.
GPU3 uses idle preflight and verified growth; CPU references hide GPUs.

## Decision and review

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue public candidate qualification | Native records, independent derivatives, CPU public consumers and bounded reuse pass | No numerical veto in the tested scope; standalone GPU and compile/host costs retained | Public GPU consumers and three-repeat public costs | Complete those tests and review actual endpoint tradeoffs | No whole-repository, DZ5, posterior, HMC or merge readiness |

Primary-agent review only; no independent reviewer or subagent was launched.
The main alternative explanation for apparent warm gains is excluding setup or
reporting. Public costs include validation, cache lookup, complete refinement and
payload; cold costs retain the full first call. The weakest evidence is workload
coverage: small target fixtures do not qualify external DZ5 transitions or
arbitrary target changes. Those remain in the master program along with public
sequential lifecycle, block/iterative controllers and F01--F20 terminal checks.


Public CPU costs02346--02363 pass all18fresh workers and complete initial/changed
records. D3 original/graph/XLA warm medians are443.75/47.53/27.18ms;D5
445.28/57.28/36.11ms. XLA repeat ranges are26.18--27.88 and35.38--37.10ms.
Full first public XLA calls take4.259/4.364seconds versus0.491/0.536original;
extra observed host RSS is520.6/524.9MiB. Both cold and host triggers remain,
consistent with native profiling. Analysis`uniform-public-costs-02363.json`.
No statistical ranking is inferred from these three process repeats.

Inventory02364 covers2959working Python files,2958parsed and the same one vendor
error. No terminal finding is closed. Through02364 chargedCPU48907.31598352069
andGPU46001.71196921611seconds, within32/52process-hour caps. Focused lint,
whitespace and policy pass. Remote fetch confirms no branch divergence and
remote main is an ancestor. Preserve this checkpoint on the repair branch;
publicGPU, remaining master execution and final merge gates stay open.
