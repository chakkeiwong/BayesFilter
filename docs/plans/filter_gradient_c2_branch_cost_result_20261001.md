# C2 complete preparation memory and performance result

All 45 source-frozen cost workers05480--05524 pass. Terminal readback05525
passes161 checks, including30 original/graph versus XLA full-record comparisons
at the unchanged1e-10 FP64 bounds, exact discrete fields, source/fixture/device
identity, verified memory growth and clean worker exits. This closes the
five-family complete preparation resource unit; public helper/caller review
and final finding reconciliation remain separate.

Each family has three fresh-process original/graph/XLA blocks with rotated
order and20 synchronized warm calls per worker. Inputs are T4/N20/D2/seed-814
from the frozen plan. Stationary and Student geometry construction is inside
every timed public call; supplied Gaussian/Hermite construction is outside
for all arms. Original source is immutable385a348b9. Numerical runtime is
43ec55f86 and measurement harnessv2 is fbd0b0474; documentation-only checkpoint
d555e79f6 does not change the frozen measured source set. All workers use
GPU2 UUID GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba with TF2.19.1, recorded
TF32/trust provenance, verified growth and no observed competing GPU process.

The table reports medians across process medians. Ratios use paired log means
and approximate95% Student-t intervals with two degrees of freedom, rather
than treating the20 calls inside each worker as independent replicates.

| Family | Original / graph / XLA warm ms | XLA/original ratio [95% interval] | XLA/graph ratio [95% interval] |
|---|---:|---:|---:|
| Bootstrap |5309.390 /43.796 /14.426|0.002695 [0.002537,0.002863]|0.3196 [0.2848,0.3586]|
| Stationary Gaussian |5156.766 /57.592 /20.676|0.003918 [0.003568,0.004302]|0.3611 [0.2700,0.4829]|
| Mixed Gaussian/Hermite |5189.817 /232.484 /31.513|0.006040 [0.005857,0.006228]|0.1355 [0.1333,0.1376]|
| Transformed Student |5190.621 /127.467 /25.709|0.004968 [0.004741,0.005207]|0.2002 [0.1816,0.2208]|
| Equal-bank DMIS |5259.338 /618.689 /65.391|0.012305 [0.011845,0.012783]|0.1043 [0.0988,0.1101]|

All intervals pass the inherited1.10 warm-regression criterion against both
controls. The large original/current difference includes removal of Python
time feedback and repeated prefix/evaluator construction. The graph control
disables the enclosing owner's JIT switch but retains existing compiled inner
primitives; it is not a pure no-XLA process. The original is the actual mixed
host/compiled implementation, not an already complete XLA baseline.

| Family | Cold original / graph / XLA s | Warm RSS original / graph / XLA MiB | Allocator peak original / graph / XLA KiB |
|---|---:|---:|---:|
| Bootstrap |6.421 /4.564 /4.243|2843.47 /1167.06 /1215.00|608.00 /558.75 /543.75|
| Stationary Gaussian |6.825 /4.976 /4.636|2826.23 /1223.34 /1239.47|617.25 /559.75 /544.50|
| Mixed Gaussian/Hermite |11.768 /8.915 /9.242|2956.37 /1410.00 /1470.50|633.75 /579.50 /549.50|
| Transformed Student |11.976 /6.142 /6.212|2935.71 /1290.64 /1330.14|629.75 /576.00 /546.50|
| Equal-bank DMIS |27.432 /14.499 /11.837|3262.53 /1554.03 /1609.41|647.25 /601.50 /562.50|

XLA host residency is16--61MiB above the graph control for these fixtures,
while both current arms are much lower than original. Mixed and Student XLA
cold cost is respectively0.327s and0.070s above graph; the other XLA cold
medians are lower. Retain these bounded compilation tradeoffs for complete
native execution. No unexplained ongoing-growth trigger remains in this
cohort. Host RSS and live TensorFlow allocation are distinct from GPU process
reservation, which includes context/compiler/library memory.

Every XLA worker executes128 additional public calls with one model-owned
configuration and one trace. With the same first result, inputs and compiled
owner retained, calls64 and128 have exactly equal live allocation in all15
workers:9984--20992 bytes across families. Late64-call RSS growth is8192--36864
bytes, below16MiB. All reaped workers leave no owned GPU process. These tests
establish stable measured fixed-shape reuse and process-exit containment,
not native eviction after Python collection or arbitrary-T/N capacity.

The first protocol remains preserved.05474/05475 are superseded original/
graph measurements;05476 failed equality with the replaceable last result
held.05477 showed synchronization and collection alone did not change the
variation;05478 isolated the varying output allocation, while identical
retained roots gave9984 bytes at all checkpoints. Thev2 harness reports both
output-held and fixed-root measurements and keeps the exact equality and RSS
thresholds. No numerical implementation or tolerance was changed.05525
explicitly excludes those old measurements and records05476 as failed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action / limit |
|---|---|---|---|---|
| Close this scoped resource unit |45 workers and161 terminal checks pass; both warm-control criteria pass|No unexplained numerical, provenance, memory or cleanup failure; earlier failure retained|Three process blocks and tiny finite fixtures|Reuse unchanged numerical/source scope; finish public callers and helpers|
| Retain enclosing XLA preparation |Same full numerical records, lower repeated-call cost and bounded reuse|16--61MiB host increment over graph explicitly accepted|Large shapes and native eviction unmeasured|Retain bounded owners and worker lifetime; no universal speed/capacity claim|
| Keep main unmerged |Final caller/finding reconciliation is incomplete|Whole-program completion not established|Remaining public sampling and integration obligations|Execute the already reviewed follow-up; no canonical LEDH or deferred-algorithm claim|

| Inference status | Result |
|---|---|
| Hard veto screen |All qualified workers pass; superseded/failed measurements remain visible.|
| Supported timing comparison |Approximate paired intervals show lower XLA warm time against both controls within the five frozen fixtures.|
| Descriptive-only differences |Cold time, RSS, allocator peaks and all generalization beyond these fixtures.|
| Default readiness |Scoped preparation execution/resources qualify; final program closure remains pending.|
| Next evidence |Standalone public-helper and actual candidate-caller checks, then terminal per-finding reconciliation.|

The evidence contract is `filter_gradient_c2_branch_cost_20261001.md`.
Raw manifests, complete records, JUnit and logs are in the main workspace's
`docs/plans/artifacts/filter-gradient-repair-20260917/run-05480` through
`run-05525`. The verified archive is
`artifacts/filter-gradient-repair-20260917/c2-branch-cost-05480-05525-evidence.tar.gz`.
The terminal readback is `run-05525/c2-branch-cost-readback.json`.

Primary-agent review checked measurement boundaries, unchanged source and
device identity, full records, rotated process order, lifetime roots and
preserved exclusions. The strongest limitation is small-fixture coverage:
repeated tiny preparation cannot prove production-scale capacity or proposal
quality. The near-constant original warm cost reflects the public call's
repeated construction; an inner-sampler comparison would answer a different
question. No training, HMC, method/RNG substitution, canonical LEDH admission
or deferred iAPF/KDM work occurred.
