# Remaining core filter cost renewal

The question is whether the currently qualified core filtering callables have
an applicable matched cost and memory disposition after their XLA rewrite.
Core numerical/API checks through05169 remain applicable: numerical sources
have not changed. Source/cost reconciliation finds only early smoke/single-pair
records for the ten fixtures below, including no general adjoint cost record.
Do not rerun the obsolete whole-repository matrix. No numerical implementation,
algorithm/RNG, tolerance, training/HMC, canonical LEDH rebuild, adaptive iAPF
or KDM is included.

Merge9d8202b77 accidentally removed the68-line TT fixture block while the
runner continued advertising its four names. Restore that exact block from
its first parent. The adjacent obsolete GenUT fixture stays retired. This
is a benchmark harness repair; the TT implementations and public callers remain
active and qualified. Repeated successful measurements also verify the restored
fixture dispatch. No source-faithfulness claim is made for these engineering
fixtures or their diagnostic models.

Compare source3582b4ac against current, and current graph versus current XLA,
using existing size1 fixtures and their complete prepared numerical owner:

| Fixture | Measured owner / scope | Before mode |
|---|---|---|
| contract_e | Batched LGSSM value and all manual score directions, T2,D3,N4,K=N | XLA |
| tt | Bounded TT value, T3,degree2,rank2,64 rows,2 sweeps,seed7781 | Existing eager public route |
| tt_adapted | Same TT value with fixed predictive means/covariances | Existing eager public route |
| tt_gaussian | Gaussian TT value with fixed initial/predictive moments | Existing eager public route |
| tt_adjoint | TT value and manual adjoint with explicit parameter callbacks | Existing eager public route |
| tt_actual | Actual-SV batched analytical value/score/status,T4,degree10,rank2,order25 | XLA |
| tt_scalar | Fixed scalar adjacent TT value/finite-program AD diagnostic,T3,degree6,order7 | Existing eager public route |
| apf | Frozen-proposal APF complete value/manual score time recurrence,T3,N8,D2 | XLA |
| dns | Complete DNS curve recurrence and fixed quadrature preparation | XLA |
| retained_moments | Complete retained moment/normalizer contractions,D2,degree3,rank2 | XLA |

The generic worker records preparation separately from trace, synchronized
first execution, output copy and20 warm calls. Report total setup+cold as well
as the numerical call boundary. Prepared tensor hashes/shapes/dtypes and all
common returned values/scores/statuses must match before speed ranking at the
existing1e-10 absolute/relative bound, with exact discrete fields. The existing
public/factory tests through05169 cover API/error semantics outside this pure
owner boundary. This renewal does not replace those checks or imply full
end-to-end cost for an external caller's unmeasured preprocessing.

Use three counterbalanced fresh-process blocks, each original/current-graph/
current-XLA, for90 workers. Known original TT host routes are measured as the
existing eager reference; do not repeat known impossible graph-compilation
attempts. The original scalar fixture must load its complete frozen closure,
not an old filter over current fitting. On an original failure stop the affected
scope and inspect it; do not substitute a new method or select a different
fixture. Primary comparisons happen as each three-arm block completes.

Add sampled uncontended-device observations and process reservation at stage
boundaries using the existing provenance helper; parent records process exit.
Primary cold/warm samples precede HLO inspection. Each current XLA worker also
gets128 fixed-input reuse calls; compare64/128 RSS and allocator occupancy,
exact replay and one trace. Current graph is an explicit nondefault reference.
GPU3 is preferred only after availability checks; avoid GPU0/1 display devices.
Require verified growth before initialization. All scopes are FP64 under the
same existing TF32 setting, frozen per-arm input hashes and environment.

Allocate at most108 serialized workers,14400 GPU and1200 CPU process-seconds,
300 seconds per worker, including bounded harness repairs and terminal checks.
This remains within the56 CPU/52 GPU hour caps; after05305 about23.57 CPU and
22.93 GPU hours remain. Numbered versioned artifacts share the existing campaign
root. No second numerical worker runs concurrently. Run from the repair checkout:

```sh
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py matrix --stage repeat --selection core-resource --measurement-gpu-index 3
```

Stop an affected unit on numerical/status mismatch, changed source, missing
baseline applicability, timeout, contention or uncontrolled fixed-shape growth.
Preserve failures and use only remaining allocation for localized repairs.
Master investigation triggers remain20% warm regression,2x allocator peak,
256MiB extra host RSS,2x cold cost and continued fixed-shape growth. Reuse tests
add a16MiB late-growth/2GiB peak ceiling; exceeding it requires attribution,
not an automatic claim of scientific failure. Report medians/ranges and paired
log-ratio intervals, with only three independent blocks. A necessary explained
cost may get an explicit scoped engineering tradeoff; no unexplained trigger
closes. No arbitrary-shape capacity, universal performance or canonical/method
quality conclusion follows.

Preflight05306 preserves142 passing checks and18 failures in older test-matrix
fixtures: the new optional selection branch assumed every internal Namespace
had the CLI selection attribute. Use an absent-safe lookup; CLI selection and
all numerical behavior are unchanged. Rerun the bounded preflight before the
cost matrix. This harness-only failure consumes15.012 CPU seconds.

05308/05309 pass original-XLA/current-graph Contract E costs.05310 completes
current-XLA numerics and128-call reuse (late RSS growth12288 bytes), but one
in-run nvidia-smi observation exceeds its existing five-second deadline. No
foreign process is observed. Preserve the failed32.407-second worker and retry
the same input/source/device job. No observation, numerical or resource gate
is relaxed; the two qualified arms remain reusable under the source freeze.

Primary-agent skeptical review: imported modules overapproximate executed
dependencies, so current API evidence is not invalidated merely by unrelated
imports changing. Conversely a lone unchanged public file does not prove its
callbacks are unchanged. These costs load complete frozen/current closures
in fresh workers and preserve imported provenance. Preparation, output copies,
HLO and host status handling have different boundaries and must be reported
honestly. Known old eager endpoints cannot be presented as old fully-XLA
baselines. The measured scalar AD path cannot establish analytical-score
claims. The ten scopes address concrete missing comparisons; scientific
retuning and historical optimizer research remain outside this renewal.
