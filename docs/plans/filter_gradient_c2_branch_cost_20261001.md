# C2 complete-public preparation cost and lifetime qualification

This continues the authorized XLA execution repair under the existing global
56 CPU/52 GPU process-hour caps. It follows the seven-family original-record
qualification in filter_gradient_c2_branch_preparation_20261001.md. It changes
no algorithm, draw stream, numerical gate or scientific scope.

The question is whether the native preparation fixes improve the complete
public calls over immutable original385a348b9, and what XLA costs relative to
the same current public call with its enclosing JIT switch off. Numerical
qualification alone cannot establish this. All calls return branch tensors,
proposal records and real fingerprints, with synchronization included. Shared
analytical values/scores are checked after the timing interval.

Freeze five materially different cost families before measurement: bootstrap;
stationary Gaussian including its geometry construction; supplied mixed
Gaussian/Hermite proposals (both dispatch bodies); transformed Student including
batched guide construction; and equal-bank Hermite/Student DMIS including its
Student construction. Pure Gaussian/Hermite calls share those qualified
numerical authorities; their exact standalone costs are not inferred from the
mixed case. Use the larger frozen tiny input T4/N20/D2/seed-814 and the exact
coupling/theta/observations/proposal inputs in the branch qualification tests.
Supplied retained/Gaussian proposal construction is outside timing for every
arm; no proposal training, tuning or snapshot generation is measured.

Each family has three counterbalanced fresh-process blocks: original/graph/XLA,
graph/XLA/original, and XLA/original/graph. Twenty synchronized warm calls follow
one cold call. Each XLA worker then performs128 reuses, recording64/128-call
RSS and live TensorFlow allocation, one retained trace and one model-owned
configuration. Shared primitives/geometry keep their existing compiled owners
in the graph arm; it is an enclosing-graph reference, not a pure no-XLA process.
Original remains the original mixed host/compiled implementation, not a
fictional fully-XLA baseline. Record the realized sampler JIT flag in artifacts.

Retain complete source/fixture/GPU UUID identity, trust and memory-growth
provenance, TF32 setting, synchronized cold/warm time, host RSS, TensorFlow
current/peak allocation, external GPU reservation samples and worker exit
observations. Allocation and reservation are distinct quantities. Compare
full branch/diagnostic/value/analytical-score/manifest records at unchanged
1e-10 FP64 bounds with exact discrete results. Do not stamp historical hashes;
compare actual values and validate current compiler identities. A numerical,
provenance, unexpected competing-GPU-process or memory-growth failure stops
that matrix for diagnosis. No costs from a failed worker qualify.

Report paired log-ratio95% intervals for warm time, and cold/RSS/allocator
changes against both controls. The inherited1.10 warm-regression threshold is
not relaxed; any failure stays failed unless an explicit bounded engineering
tradeoff is justified by the separate lifetime/placement evidence. XLA compile
memory and cold cost must be reported even when warm throughput improves.
Reuse must retain one trace/configuration, have late RSS growth at most16MiB,
and unchanged live allocation between calls64 and128. Verify process exit and
absence of its GPU allocation. These tiny checks do not prove arbitrary-scale
capacity or that TensorFlow releases every native allocation while alive.

Allocation:60 serial workers,1800 CPU/10800 GPU process-seconds,900-second
per-worker limits, inside the remaining global budget.45 cost workers plus a
terminal readback/policy worker are planned; remaining slots are bounded repair
retries. Use the existing trusted runner, GPU3 if available (or recheck and pin
another non-display device for the complete matched cohort), and fail closed on
contention. Exact registered commands use matrix --stage tests
--test-batch c2_preparation_branch_cost_block_K --repeat K --test-gpu-index 3
for K=0,1,2. The terminal CPU group is
c2_preparation_branch_cost_terminal_cpu. Outputs stay in unique run directories
under docs/plans/artifacts/filter-gradient-repair-20260917.

Skeptical review: freezing inputs outside timing must not hide the newly repaired
Student/stationary construction, so those builders are inside every invoke.
Repeated IDs alone cannot prove numerical equality, so readback compares full
records against the independent original. A graph control with compiled inner
primitives must be labeled accurately. Three tiny paired blocks provide scoped
engineering evidence, not a universal performance ranking, proposal-quality
claim, HMC result, or canonical LEDH admission. No adaptive iAPF/KDM, live
MacroFinance edits, training/HMC, package/system changes or main merge occurs
in this unit. A failure triggers a localized retry within the same budget,
with original failures retained and no tolerance or method substitution.

Launch note: GPU3 became occupied before the first numerical worker. The
runner declined before launch and preserved
cost-preflight-declined-20260930T202855264470Z.json. No cost observation was
collected. GPU2 was rechecked free and selected for the complete matched
cohort; commands use --test-gpu-index 2. This is the same approved hardware
class and global budget. Source checkpoint43ec55f86 is frozen.

05474 original/bootstrap and05475 graph/bootstrap pass.05476 XLA/bootstrap
failed the unchanged exact live-allocation gate:16384 bytes at64 calls versus
15360 at128. The matrix stopped. This decrease is not accepted as a pass or a
leak diagnosis. Register one bounded GPU lifetime probe using the same148
post-initial calls and input, recording allocation before/after full device
synchronization and after garbage collection without changing reachable first/
last results or the retained owner. This distinguishes pending executor work
from Python object lifetime. Preserve all results; only an explained harness
repair permits retry. No numerical method or tolerance may change.

05477 repeats the allocation pattern (15872/16384/15360 bytes after20/84/148
calls); neither full synchronization nor garbage collection changes it. This
excludes those two proposed explanations. The retained first result, inputs
and single compiled owner stay fixed; the final returned record is replaced
each call. The next bounded probe additionally releases only that replaceable
result before each snapshot, retaining the same first record/input/owner roots.
It tests output-allocation layout versus growing owner residency; no equality
threshold is changed. Preserve the first probe and distinguish output-held
allocation from residency after disposable outputs are released.

05478 identifies the changing roots: while the last returned record is held,
current allocation is15872/15616/15616 bytes. Releasing only that record gives
exactly9984 bytes at all three snapshots, with the same first record, inputs
and single traced owner retained. No runtime numerical change is needed. The
harness now records both output-held allocation and residency with those fixed
roots, synchronizes all visible device work, and enforces the same exact
64/128 allocation equality and16MiB RSS threshold on the fixed-root snapshots.
It also writes the complete result before lifetime assertions. This repairs
measurement comparability;05476 remains a failed output-held equality check.
The cost schema isv2; retain05474/05475 as superseded protocol evidence and
renew all three arms under one unchanged final source snapshot. No older or
failed worker is silently promoted. Full-source matching is still mandatory
inside each accepted block; readback reports excluded source versions.
