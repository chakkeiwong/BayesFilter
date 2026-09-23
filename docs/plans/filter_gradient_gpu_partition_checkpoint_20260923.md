# GPU qualification and exact partition validation checkpoint

Through03407, the repair branch qualifies the native staged/batched locators,
target-failure numerical boundary, dense cloud evaluator and partition checker
on GPU. Public/actual-consumer integration is still required. Main2921c2ffd is
unmerged. This is an engineering checkpoint, not terminal campaign acceptance.

| Evidence | Result and limit |
| --- | --- |
| Staged locator |26 CPU/24 distinct GPU checks pass; complete original records, exact calls, changed-input HLO, compiler failures, nested reuse and ownership. GPU reference changes exactly four int32 accounting occurrences to int64; CPU original unchanged. Public error/reuse wiring remains. |
| Batched locator |9 CPU/9 GPU checks pass, including enclosing invalid-to-valid recurrence. Public and actual DZ5 wiring remain. |
| Target failure |57 CPU/57 GPU checks pass. Arbitrary Python exceptions remain an explicit host diagnostic adapter. |
| Dense cloud |14 CPU/14 GPU cases pass against exact external2f386f75 statements with identical frozen offsets. Full call order/count, histories, decisions, HLO and owner collection pass. Seeded RNG is not yet qualified. |
| Partition validation |68 scenarios per backend pass against exact original3582b4ac helpers;121 affected fixed-geometry checks pass per backend. The current public subnormal duplicate-row defect is repaired using bitwise signed-zero normalization. |
| Policy/discovery |129 policy checks pass03406;231 guarded sources/1333 exact exceptions, no added waiver.03407 scans3056 Python files/3055 parsed; the single historical vendor syntax error remains. Counts do not prove compliance. |
| Ordered-block costs |18 clean GPU workers, three repeats per arm/extent, pass complete original/changed records. D3/D5 XLA warm medians29.99/65.67ms versus20462.85/21612.74ms for the recompiling prior API. Results describe the fixture and are not a pure compiler ablation. |
| Block GPU memory |D3 peak130560→284160 bytes.03405 identifies171360 bytes of compiler temporary storage. Stable repeated peaks/current allocation support retaining this bounded transient cost; exact runtime byte attribution and larger-consumer capacity are not established. |
| Staged costs |All six GPU prior/graph/XLA D1/D3 cases pass. XLA warm5--6ms versus7.6--7.8s; one process per arm, no relative cost trigger. Public repeated costs remain. Earlier CPU graph failures are preserved. |
| Process containment |Two sequential GPU children reproduce full original records, four signatures/200 calls each, and return to a fresh startup footprint. Parent growth3.781MiB. CPU also passes. This supports process cleanup, not in-process native eviction. |

The result notes retain all failures.03338 is original GPU int32 placement;
03350 loses a static callback row extent;03351/03352 reveal actual changed-input
HLO specialization from dynamic StridedSlice;03399 is a reference-loader
dependency failure;03400 exposes the subnormal bug. Corresponding repaired runs
use unchanged comparisons.03386 renews the ten cost-analyzer tests. The frozen
external source fixture keeps its original import formatting and byte hash; a
blanket Ruff command reports its inherited I001. All nine edited Python sources
pass focused Ruff with that immutable reference excluded. No runtime waiver is
introduced.03405 dump timing is explicitly ineligible.

The receipt `artifacts/filter-gradient-repair-20260917/gpu-partition-checkpoint-03407.json`
binds all manifests3338--3407, exact commands, sources, environments and charges.
Total charged time is66508.902306 CPU/66488.573124 GPU seconds;13.53/33.53 hours
remain under unchanged32/52-hour caps. The partition unit uses8/12 workers and
449.310532/1800 seconds; block cost/attribution closes at41/42 workers.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Commit qualified dependencies/bug fix |Complete original records and affected CPU/GPU suites pass |No renewed numerical/policy veto |Enclosing caller not yet qualified |Compose validated fitter and attempts |Full initializer readiness |
| Retain clean GPU cost evidence |Full records and unshared sampled UUID provenance pass |D3 transient trigger investigated and bounded for fixture |Compiler plan does not reconcile every allocator byte |Qualify actual consumer capacity/lifecycle |General memory or speed ranking |
| Keep terminal campaign open |E1--E6 still lists every remaining gate |Main merge blocked |Public E2/E5/E6 integration, costs and complete endpoint audit |Continue authorized repair |Repository-wide compliance or scientific admission |

Skeptical result review: dependencies do not establish call-chain compliance.
The exact cloud reference is preserved rather than rewritten; padding never
adds target rows. Invalid partitions must gate fitting, not merely report errors
after a fit. Original stability errors must precede selection errors in the
enclosing controller. GPU memory growth is verified in manifests and sharing
vetoes remain intact. Bounded clean cost observations cannot prove arbitrary
signature capacity. The E2 reporting-count proposal remains pending with no
revised comparator or residual correction installed. Independent agent review
was not used; this local review and focused evidence do not replace terminal
result review. Canonical LEDH rebuild and completed CDF sampling remain excluded.
