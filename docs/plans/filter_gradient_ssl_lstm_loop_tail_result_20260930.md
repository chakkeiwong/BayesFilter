# Fixed replay loop-tail result

The replay now executes T-1 unconditional observation/transition steps and one
final observation. This removes the terminal conditional from every recurrent
step while preserving chronological order, all RNG draws and the analytical
score. Both original execution contexts and public manifest/error semantics
remain intact. This is the same existing local SSL-LSTM replay algorithm.

Qualification05060/05061 passes17 tests per CPU/GPU, including T1 and two-state
cases, complete records against original c4950a827, independent derivatives,
changed-input reuse, native control and one enclosing XLA trace. The analytical
directional derivative differs from independent finite differences by4.05e-13.
The preceding conditional owner is preserved at c05f2c2cc with its full failed
cost-trigger evidence. No numerical tolerance or policy exception changed.

Fresh source-frozen costs05062--05133 cover72 workers: CPU/GPU, T2/T8,
before/after, public default/explicit graph/explicit XLA and three repetitions,
reversing order in repetition1. Independent readbacks05134/05135 pass all36
records per backend, source/device/memory-growth integrity, exact replay,
changed-mode RNG context identity and complete numerical comparisons.
Maximum absolute difference is2.67e-15; GPU explicit-XLA comparisons are exact.
GPU3 is UUID GPU-b8045e28-4433-ec7a-77a5-db0636748322 with clean sampled sharing;
float64/TF32-off settings and CPU reference role are unchanged.

| Device/mode | T | Before warm ms | After warm ms | Paired geometric ratio (95% interval) |
|---|---:|---:|---:|---|
| CPU public default |2|36.745|0.995|0.027 (0.025--0.028)|
| CPU public default |8|168.814|1.065|0.006 (0.006--0.007)|
| GPU public default |2|57.916|1.714|0.028 (0.020--0.038)|
| GPU public default |8|258.946|1.799|0.007 (0.006--0.007)|
| CPU explicit graph |2|1.197|1.445|1.209 (1.189--1.230)|
| CPU explicit graph |8|3.368|4.326|1.177 (0.838--1.652)|
| GPU explicit graph |2|4.071|6.385|1.411 (0.807--2.466)|
| GPU explicit graph |8|8.731|11.299|1.287 (1.178--1.405)|
| CPU explicit XLA |2|0.579|0.593|1.030 (0.970--1.093)|
| CPU explicit XLA |8|0.714|0.686|0.952 (0.899--1.008)|
| GPU explicit XLA |2|1.373|1.394|1.081 (0.717--1.629)|
| GPU explicit XLA |8|1.460|1.435|0.974 (0.721--1.316)|

Times are medians of three process medians; ratios/intervals use paired log
ratios with Student-t df2. They are conditional small-cohort evidence, not
proof of a universal speed bound or statistical superiority. The original
public default is eager. Graph-reference slowdowns remain real observations;
the explicit graph route remains a non-default diagnostic exception. The
paired GPU T8 estimate improves from1.235 in the preceding cohort to0.974,
but variability prevents asserting that all overhead is below10%.

Graph size stays independent of T:1451 direct/default,1486 explicit graph,
1306 explicit XLA nodes at both extents. The original enclosing graph grows
1169 to5201. The tail duplicates one observation site, a fixed-size cost,
and removes the repeated transition conditional. GPU explicit-XLA T8 cold
time falls5.987 to2.154 seconds against the original. CPU equivalent cold
time falls3.277 to1.080 seconds. No static graph count proves the timing cause.

CPU public-default host RSS rises194.1--195.8MiB, reflecting the newly compiled
owner; GPU public-default RSS falls22.8--27.5MiB. For matched explicit-XLA T8,
host RSS falls120.2--122.1MiB CPU and114.8--115.7MiB GPU. T2 matched-XLA adds
about12--14MiB. Device allocator memory is separately recorded in every worker;
host RSS and driver reservation are not interpreted as live tensor memory.

Final-source lifetime05136/05137 passes2000 alternating-input public calls
with exact replay and one trace. Late1000-call RSS growth is0.226563MiB on
both devices; GPU allocator current/peak stays5632/33536 bytes. Twenty seed
configurations obey the16-entry LRU cap and retained owners do not retrace.
Their additional sampled host RSS is928.4MiB CPU/826.5MiB GPU, within the
predeclared2GiB tiny-scope capacity bound. Ordinary cache clear does not release
observed native RSS. Repeated owner reuse is bounded in this test; large sets
of configurations require bounded process lifetimes. No native-eviction or
unlimited-capacity claim follows.

| Decision | Criterion/veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|
| Retain loop-tail execution repair | Numerical/derivative/RNG/error/signature gates pass both devices; no policy exception added | Other shapes and callbacks remain outside the fixture | Preserve qualification and final source guard | Whole-repository or canonical/scientific admission |
| Accept the measured cost tradeoff for this repaired owner | Public default is substantially faster; fixed graph replaces forbidden unrolling; warm reuse/capacity checks pass | GPU paired intervals still cross1.10; graph reference is slower | Record these costs and require new capacity evidence for materially larger configuration sets | All calls have less than10% overhead |
| Close this bounded performance investigation | Concrete conditional-removal intervention preserves output and improves the previously regressing T8 estimate; no consistent final XLA slowdown established | Process/compiler scheduling remains a source of variance | Continue current core caller/evidence closure | Exact causal attribution or no memory retention |

This disposition uses the master's existing permission to record a justified
execution/memory tradeoff after investigation. It does not relabel triggered
intervals as passing a10% bound, relax a numerical gate, revive Python
unrolling, or accept unrelated streaming/geometry resource findings. No further
microbenchmark tuning is required to prove an unrequested universal speed
bound. Serious configurations exceeding this measured scope remain subject to
their own bounded capacity plans.

Inference status: numerical/provenance vetoes pass; no general statistically
supported ranking; descriptive default speedups and mode-specific costs are
reported; scientific/default-readiness claims remain outside this repair.
The strongest alternative explanation for part of the speed improvement is
fresh-process scheduling variation. A current-source caller regression or
continuing fixed-owner growth would reopen this unit. Primary-agent review;
no independent-agent review, HMC/training run or algorithm substitution.

Final05138 passes164 readback/policy checks. Focused Ruff and whitespace pass;
guard coverage stays317 sources/1465 unchanged exact exceptions. The79-worker
unit uses278.564120 CPU and409.940881 GPU process-seconds. Archive
ssl-lstm-loop-tail-05138-evidence.tar.gz preserves333 reopened/verified members,
14840430 bytes, SHA256
073062f6860d4810b71f8890ffd4d1f4dde6fe2a3b01c967c34adc56ba6b521c.
The adjacent verification JSON binds all records under
docs/plans/artifacts/filter-gradient-repair-20260917. Main remains unmerged;
current core caller/evidence and other terminal resource work remain open.
