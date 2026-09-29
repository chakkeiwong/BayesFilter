# SSL-LSTM fixed replay execution repair

The current audit found an in-scope numerical Python time loop in
`ssl_lstm_zhaocui_fixed_adapter._fixed_replay_value_and_score`. The public
`tf_ssl_lstm_zhaocui_fixed_score` runs it eagerly unless its caller supplies an
enclosing function. The Phase6 benchmark calls it directly (lines727--741)
while its artifact builder records XLA (lines516--525). The artifact builder's
own `jit_compile=False` default is debug/report metadata, not this runtime bug.

Baseline: c4950a827. Preserve the existing fixed replay likelihood, analytical
first-order score, parameter unpacking, complete numerical records, weighted
frame diagnostic and both original seeded execution streams. Replace only the
numerical time recursion with TensorFlow control flow and supply a reusable,
bounded, signature-bound public XLA default. Permit explicit non-JIT reference
execution. Parameter unpacking, noise, replay, score and numerical diagnostics
belong inside the enclosing program; metadata/dataclass assembly stays outside.
Use the shared compiled-program and Philox compatibility authorities. The raw
helper keeps the original enclosing-XLA stream; the newly compiled public
default preserves the previous ordinary-TensorFlow stream. No new RNG version
or changed seeded cloud is authorized. Test both contexts, including T=1.

Source classification remains `extension_or_invention` for the existing local
fixed-replay likelihood, not a new source-faithfulness claim. Inspected the
local Zhao--Cui paper, `.localresources/papers/zhao-cui-tensor-train-sequential-
learning-jmlr-2024.txt`, Section2.3 / Algorithm1, lines457--520: it constructs a
nonseparable adjacent-state target, reapproximates it by TT and marginalizes.
Inspected author `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/
full_sol.m:21--43,90--135` and `computeL.m:24--47`: TT/SIRT reapproximation and
sampling differ from this local replay; weighted-frame vocabulary alone does
not make it the author algorithm. Preserve the existing manifest/nonclaims.
There is no TT/LEDH rebuild, new HMC/training run or scientific promotion.

Qualification compares the pinned original implementation with the candidate
on identical data, theta, seeds and settings, CPU reference and trusted GPU,
with TF32 disabled for binary64 comparisons. Check every numerical output and
parameter component at 1e-10 absolute/relative, independent finite differences
of the same finite likelihood, changed-input reuse, original error boundaries,
default XLA/HLO, one trace per fixed signature, native time control and absence
of host callbacks. Preserve any failed record before assertions. Existing
adapter/derivative tests remain applicable; a monkeypatched independent dense
derivative oracle must get a fresh compiled owner, not a cached candidate.

Cost comparison: fresh processes at T=2 and T=8, nine replay samples, scalar
latent/hidden/observation dimensions, identical frozen theta/observations and
seeds. Before/after ordinary public defaults and matched graph/XLA enclosing
calls are distinct arms. Three interleaved repeats per arm/device measure cold
call, steady-state median, sampled host RSS/high-water mark and TF allocator
current/peak. Numerical equality is required for each relevant pair; graph and
XLA seeded streams are not compared as if identical. Preserve the full result
and source/device/memory-growth provenance. Timing requires uncontended GPU
preflight and recorded sharing; classify compile residency separately from
live allocator memory. Existing 10% warm-time and 64MiB host-residency triggers
require disposition, not automatic acceptance or broader optimizer research.

Register groups under `ssl_lstm_replay_` in the existing runner. Use:

```sh
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test --group GROUP --device CPU --test-timeout-seconds 300
```

GPU groups use `--device GPU --test-gpu-index INDEX` after availability checks.
At most120 serialized workers,12000 CPU and12000 GPU process-seconds, including
qualification,72 cost workers, readback and at most four local repairs/retries,
within the existing campaign balance. Stop on changed RNG/algorithm semantics,
unexplained numerical failure, exceeded allocation or unusable measurements.
No gate or allowlist numerical exception may be loosened. Exact metadata-only
source exceptions may cover existing schema/parameter-name traversal and the
debug artifact default; each is reviewed and hash-bound by the existing guard.

Primary-agent review: the affected benchmark actually calls this implementation,
so the recurrence and misleading execution label are relevant to the rewrite.
The proposed public wrapper must not silently change stateless random draws,
freeze theta in a closure, duplicate numerical authority or hide a reference
behind the candidate's cache. Matched explicit modes and actual public defaults
answer different comparisons and are recorded separately. HMC and the source
author's algorithm are not executed or certified. Repair and qualify first;
cost workers run only after numerical gates pass.

Compatibility follow-up before terminal costs: review found that the original
manifest accepts list-valued seeds and integer-coercible sample counts. A raw
dataclass LRU key cannot preserve those inputs. Reproduce with the pinned
original, normalize only the numerical cache key, retain the caller's complete
manifest in public records, and requalify. Preserve04944--04979 as an earlier
owner cohort; renew final costs after this repair so default-call metadata
overhead is measured. The worker-count cap increases to120 to include the36
preserved cost workers and this bounded retry; CPU/GPU second caps, campaign
budget, target, data, algorithm, seeds and gates are unchanged.

Recovery review: run04980 reproduces the original-accepted list-seed input
failing at the candidate's raw-manifest cache lookup. The repaired key contains
only normalized count, immutable seeds and ridge; original metadata stays in
public records. TensorFlow validates seed configuration before cache lookup,
preventing Python's integer/float key equality from admitting invalid seeds.
Run04981 passes CPU qualification, including metadata-only cache reuse and the
invalid-float-seed regression. Requalify on GPU before renewed cost workers;
do not change numerical source or harnesses during those cohorts.

Resource follow-up declared after CPU repetitions04983--05018: the direct
default gains approximately194MiB sampled host RSS against the former eager
default. Explicit-XLA comparisons have much smaller or negative deltas, so
compilation is a candidate explanation, not proof of a leak or its absence.
After the cost matrices finish, use one CPU-reference and one trusted GPU
worker (300-second timeout each) for bounded lifetime/capacity diagnostics.
At T8 alternate two dynamic theta/observation inputs for2000 complete public
calls, recording memory at calls0,10,100,500,1000,1500,2000 and requiring exact
per-input replay, one trace and unchanged final1000-call allocator state.
Late sampled RSS growth over16MiB is a repair trigger. Then evaluate20 distinct
T2 numerical seed configurations, require the16-entry LRU cap, revisit retained
entries without retracing, and record memory before/after ordinary cache clear
and collection. CPU host growth over2GiB or GPU allocator peak over256MiB is a
capacity veto for this tiny declared scope; these bounds are resource limits,
not numerical tolerances or general production-capacity claims. Bound outputs,
not process reservation, and preserve measured post-clear native residency.
Run these workers only after unchanged-source cost completion. They fit the
existing120-worker and CPU/GPU-second caps. The key question is whether a
fixed owner grows during reuse and what the bounded specialization costs;
universal native-executable eviction is not required. No HMC/training runs,
algorithm changes or gate relaxations follow from this diagnostic.
