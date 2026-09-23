# M30: distinguish first-call costs before changing runner reuse

Status: M30 complete and audited. See
`bayesfilter-hmc-m30-runner-reuse-result-2026-09-23.md` and the execution note
for the preimplementation corrections and terminal evidence.
M29 profiles and pairs are complete and must not be rerun merely to reproduce
their successful audit. This phase consumes the existing campaign allowance.

## Question and evidence contract

Can reuse of compiled runners reduce the measured cost of the same complete
tuning experiment while preserving target, four independent chain streams,
epsilon/L proposals, all verified members, numerical health, receipts and
posterior decisions?

The exact baseline is M29's frozen source-r3 and its unprofiled Gaussian and
beta-binomial designs under `m29-r1/designs-r1/`. The saved first-call ledger
`m29-r1/cost-attribution-r2.json` attributes 313.707 of 354.947 measured
Gaussian observation-chunk seconds and 369.978 of 399.670 beta-binomial seconds
to first calls for their cache key. There are 29 and 33 observation cache keys.
These times include construction, tracing, compilation, execution and capture.
They do not by themselves measure avoidable compilation.

The code already caches by (L, chunk count) within
`HMCCandidateExecutionBinding._run`; epsilon and seeds are runtime inputs.
Claiming that every candidate currently compiles a new graph would be wrong.
The serial independent-chain wrapper creates four scalar runners per key.
The underlying scalar reusable runner already has an optional dynamic-L
capability, which is a candidate mechanism to investigate, not a reviewed
default for the ordinary tuner.

| Evidence | Role |
| --- | --- |
| Exact replay of all saved chunk states and complete traces, including acceptance and momenta | Primary engineering criterion for the diagnostic and any proposed cache change |
| Same full-fit candidates, observations, receipts, tensors and posterior decisions, with declared clock/source/path normalization | Required before admitting a runtime optimization |
| Missing evidence, wrong source/design, changed target or streams, uncontrolled retracing, invalid GPU memory policy, abnormal exit | Continuation veto for the affected comparison until repaired |
| First-call and warmed-call times, graph counts, host RSS and allocator peaks | Explanatory; nominate a repair and quantify its practical limits |
| Posterior R-hat, ESS and MCSE | Posterior-only criteria; never tuning membership or selection |
| Exhausted phase/campaign allowance or changed scientific scope | Continuation veto |

One replay per model does not establish speed superiority. No diagnostic
promotes coverage, global exploration, subtle-defect power, learned transport
quality or a numerical default. The unchanged confirmation denominator stays
384 independent fits per target unless a separately justified statistical
design replaces it; replay calls and profiling arms never count toward it.

## Bounded sequence and refresh

1. Freeze a concise diagnostic manifest before launching. Resolve first saved
   observation chunks for L=3 and L=25 at counts 8, 136 and 256 in each model,
   using the recorded observation order, not work-creation ordinals. Those
   twelve cells exercise short residual, initial and full chunks. Record each
   original work ID, evidence and chunk checksum, starts, per-chain seed
   folding, epsilon, L, count, transformed target and source identity. Use the
   existing adapter reconstruction and execution binding; do not replace the
   model by a standard-normal approximation or omit its preparation layers.
2. First run a tiny CPU reference/debug replay and compare repeated calls on
   that same CPU backend, including every trace tensor. Cross-backend exact
   equality to archived GPU values is not required. Keep GPU hidden. Then use trusted GPU/XLA on the same
   hardware class, memory growth before import, one process per model and
   serial four-chain execution. For each cell, build a fresh existing runner
   and run its exact saved chunk once, then twice more with identical inputs
   and explicit synchronization at the same tensor boundary. Label these
   mechanics replays, not retained draws. Save construction, first-call and
   warmed-call costs separately; verify exact equality to the archived chunk
   for each repetition. Stop the affected cell on any mismatch and diagnose it.
3. Refresh the plan from that diagnostic. If first-call costs dominate and
   shared dynamic-L graphs preserve exact transition behavior, prototype
   narrowly scoped reuse within one frozen binding and fixed chunk count.
   Keep the baseline static implementation available for comparison. Bind
   cache lifetime to target, geometry, dtype, shape, trace policy and backend;
   never reuse a graph across unrelated targets or coordinate maps. Preserve
   scalar-chain seed folding and telemetry. Never obtain reuse by switching
   to batched-chain random streams, merging work items, reducing L coverage,
   thinning evidence, or lowering posterior counts.
4. Test any prototype on static/dynamic L replay, changed epsilon, all three
   chunk counts, separate target/geometry bindings, tracing bounds, corruption
   and restart, then actual Gaussian and beta-binomial public pipelines.
   Source identity participates in work seeds: simply changing source and
   normalizing identity fields cannot produce a valid paired full-fit test.
   The saved M29 chunks remain the exact transition comparator. Before full
   fits, resolve a same-source static/dynamic comparison with identical seeds
   and independently recorded implementation selection; retain all source
   validation. Reallocate the existing ceiling explicitly if fresh baseline
   fits are needed. Require the same decisions, archived tensors and seed
   lineage. If exact parity fails, preserve the prototype as
   unpromoted diagnostic work and investigate before further expensive fits.
5. If warmed execution dominates, or dynamic-L cannot preserve the required
   behavior, reject that optimization hypothesis and record the smallest next
   discriminating repair. Checkpoint writing costs 61.9/73.5 profiled seconds,
   but removing that entire cost still cannot fund confirmation. Do not drift
   into a broad serialization refactor or weaken integrity checks to claim
   affordability. Treat cost limits as limits of the current design, not
   rejection of the tuning research direction.
6. End with a terminal audit, all attempt charges, updated master and an
   executable next phase. Reprice using complete unprofiled fits if an
   optimization passes; graph counts or short replay speed alone cannot price
   the full confirmation inventory. No full-fit baseline needs rerunning
   unless hardware/source comparability or unresolved uncertainty requires it.

## Environment, allocation and numerical provenance

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` and fresh outputs under
`docs/plans/artifacts/hmc-repair-master-2026-09-16/m30-r1/`. Freeze owned
source changes separately from unrelated work. Each manifest records the exact
command, Git/source identity, original design and seed, runtime/memory policy,
elapsed time, tensor checksums, plan and result paths.

Allocate at most **1800 CPU and 4800 GPU worker-seconds**, bounded by the
terminal M29 remainder. These are inherited development ceilings, not new
compute authority or scientific adequacy thresholds. Reserve up to 600 GPU
seconds for two baseline replay workers capped at 300 seconds each, up to
600 for two prototype replay workers, up to 3000 for two complete prototype
fits capped at 1500 each, and 600 for local infrastructure repair. Keep at most
two numerical workers and one GPU fit active. The complete-fit cap is a
convenience margin over M29's roughly 10--11-minute fits, not a timing bound.
All failures and retries count once at the outer receipt boundary.

The choice of L=3/25 and counts 8/136/256 comes from M29's observed cache
keys and tests both range endpoints and every actual count. Two warmed repeats
are a convenience check for cache reuse and deterministic execution, not
uncertainty evidence. M29 starts, epsilon, seeds, posterior counts and lugsail
settings remain target-specific development baselines. Profiling stays off for
timing comparisons; optional diagnostics remain outside design identity.

## Skeptical audit

The plan passes after rejecting three misleading shortcuts: attributing all
graph-call time to compilation; assuming the existing tuner lacks a runner
cache; and treating fewer graph objects or a cheap checkpoint rewrite as proof
that the full confirmation inventory is affordable. Saved first-call flags use
execution order, not proposal ordinal. The exact chunk replay checks seeds,
states, target layers and all trace tensors before timings can nominate a
repair. GPU memory growth, XLA and scope-bound reuse remain mandatory.

A profile can pass while missing the real cost: first-call time contains useful
execution and TensorFlow may retain caches beyond one Python object. Fresh
per-model processes and the saved M29 graph inventory bound that risk.
Resource measurements and replay checks can veto a cache proposal; a slow but
valid proposal simply fails its cost hypothesis. No arbitrary numerical
tolerance is introduced to force equality, and no optional prototype is
silently promoted into a global default.

The next-phase refresh must continue to list the other open requirements:
adequate stopping/MCSE coverage; global modes with same-target reference
uncertainty; calibrated subtle full-fit defects; exact MacroFinance source,
data, prior, coordinates and independent reference; and learned-map quality
upstream of supplied-map tuning. M28's successful supplied residual fixture
remains closed for its tested two seeds, not a universal geometry guarantee.
