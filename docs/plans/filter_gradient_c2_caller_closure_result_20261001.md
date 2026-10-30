# C2 public helpers and actual caller result

The remaining standalone Gaussian sampler now performs random generation and
the shared transform/density calculation inside a retained fixed-signature
TensorFlow owner with XLA enabled by default. Proposal mean, factor and seed
remain live inputs. The K1 convenience endpoint retains its existing compiled
sampler across calls. Both caches contain at most four static configurations.
The numerical sampling methods and shared analytical-score authority are
unchanged. The Phase0 ordinary-random reference helper now explicitly documents
its independent reference role and incomplete enclosing-XLA scope.

CPU05529/GPU05530 pass18 checks each, including original helper streams,
large/negative seeds, changed inputs, accepted empty/singleton results,
invalid-size/seed ordering, one trace, no captured proposal values and the
affected prepared-APF regressions. Gaussian records agree exactly in these
fixtures. CPU05532/GPU05533 execute all11 actual benchmark candidate factories
and compare full branch/diagnostic/manifest/value/analytical-score records with
direct qualified endpoints. Reference and standalone fallback helpers are
blocked during those calls. The tested families cover K1, fixed K2/K4,
defensive K1/K2/K4, bootstrap, Student, Gaussian, stationary and retained Hermite.

The source inventory `filter_gradient_c2_callers_20261001.json` records29 direct
compiler calls in five benchmark files. They use explicit JIT true except the
mixture benchmark's labeled smoke calls, which propagate its JIT option. This
is inspected wiring; the actual11-family factory test is separate execution
evidence. No full benchmark, proposal training, tuning or HMC campaign was run.
`compile_k1_apf_proposal` is used only by Phase0 checks and their unit test in
the inspected repository. Its device-specific ordinary categorical stream is
preserved as a reference, not represented as complete default XLA preparation.

Fresh GPU workers05534/05535 measure the complete standalone helpers with20
synchronized warm calls. The same source snapshot and GPU2 UUID are used,
with verified growth and uncontended observation. These are descriptive
one-pair costs, not a statistically supported ranking. Gaussian runs first in
each worker; K1 RSS therefore includes that preceding helper's retained memory.

| Helper | Original / current warm ms | Original / current cold s | Original / current RSS MiB | Original / current allocator peak bytes |
|---|---:|---:|---:|---:|
| Gaussian |1.0144 /0.7195|0.2343 /0.6477|1029.76 /1031.34|8960 /8960|
| K1 convenience |980.3127 /1.5447|1.7023 /1.7423|1264.44 /1086.05|14592 /14336|

Gaussian cold compilation adds0.4134s and1.58MiB measured host RSS in this
pair; accept that bounded cost for enclosing XLA. The K1 wrapper previously
constructed a new compiled sampler on every call; retaining the owner removes
that repeated cost. Each current helper passes128 additional calls, one trace,
4096-byte late64-call RSS growth and exactly stable live device allocation:
5632 bytes Gaussian and7424 bytes K1. Worker exit and absence of its GPU
allocation are verified. This does not prove native eviction or arbitrary
shape capacity, and does not replace the45-worker complete preparation cohort.

Two failures are retained.05528 has15 passes/3 failures because the first
Gaussian draft narrowed accepted large seeds to int32. Preserving TensorFlow's
inferred seed width and widening to int64 before the retained owner repairs
that defect;05529/05530 pass without changed tolerances.05531 fails before
candidate execution because the test used optional None gate values where the
benchmark expects selected numeric controls. Using the benchmark's existing
constants repairs the fixture;05532/05533 pass. An unsupported --gpu-index CLI
attempt was rejected before a worker and corrected to --test-gpu-index2.

Terminal05536 passes161 checks and independently compares helper outputs,
source/device identities, growth, costs, lifetime and process exits. It also
compares AST bodies against d91e269c8 to establish that the previously qualified
complete preparation owners in the two edited modules are unchanged. The
change is limited to the public wrappers, added retained owners and reference
documentation. Source enforcement remains335 modules/1552 exact allowances;
no allowance was added. New diagnostic tests pass Ruff. The two pre-existing
runtime files retain exactly the same15 pre-existing Ruff findings, verified
against the preceding commit; unrelated numerical code was not reformatted.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action / limit |
|---|---|---|---|---|
| Close C2 public-helper/caller unit |18 CPU/18 GPU checks,11 actual families per backend and161 terminal checks pass|Seed-width and fixture failures preserved and repaired|External dynamic callers and larger shapes untested|Reuse this scope; finish final F01--F20 reconciliation|
| Retain bounded compiled helpers |Original streams/errors preserved; one trace and stable allocation|Gaussian cold overhead accepted for complete XLA|One descriptive cost pair, no universal speed bound|Reuse owners; process exit contains native residency|
| Preserve complete preparation costs |All unchanged callable AST bodies verified against cost checkpoint|No numerical preparation change hidden by wrapper edits|AST/source evidence does not prove arbitrary callbacks|Keep separate45-worker resource result|

The plan is `filter_gradient_c2_caller_closure_20261001.md`. Exact commands,
TF2.19.1 environment, input/seed/source identities, trusted GPU2 UUID, CPU
reference hiding, wall times, logs and JUnit are in raw runs05526--05536.
The terminal result is `run-05536/c2-public-closure-readback.json`. The verified
archive is `artifacts/filter-gradient-repair-20260917/c2-public-05526-05536-evidence.tar.gz`.

Primary-agent review checked whether a reference label concealed an active
candidate call, whether the seed fix changed ordinary draws, whether cache
reuse captured a stale proposal and whether complete preparation costs became
stale. Blocked fallback calls, independent original records, changed live
operands and AST/source witnesses answer those tested questions. Canonical
LEDH, proposal quality, arbitrary capacity and deferred iAPF/KDM remain outside
this result. Main is still unmerged pending final scoped closure.
