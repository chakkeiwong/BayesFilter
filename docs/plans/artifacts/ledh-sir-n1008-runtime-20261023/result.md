# Fresh SIR runtime comparison at 1008 particles

23 October 2026. Complete on `ledh-refactor-with-policy-fix`.

The required particle count gives a relevant new measurement, but it does not
remove the conditional-proposal slowdown. On fresh identical N1008 inputs, the
current code takes about 36% longer than pre-merge for the conditional proposal
and 21% longer for plain LEDH. The physical-mixture arm takes about 27% less time.
These are newly measured comparisons; no N64 measurement enters the result.

All previous N64 results in this investigation are disqualified, including
speed, training, moment, score and guide-rejection conclusions. The master,
checkpoint, primary old reports, LaTeX source and rebuilt PDF carry the notice.
The previous learned weights and calibration were not reused. See
[machine-readable withdrawal](disqualification.json) and the
[reviewed plan](../../ledh-sir-n1008-runtime-20261023.md).

## Workload and validation

SIR d18, theta0, T=5, N=1008, float64/TF32-off diagnostic exception, GPU/XLA on
RTX4080SUPER; chunk K=1008. OT remains epsilon1, Sinkhorn5, terminal balance32.
All physical controls remain fixed between sources. Two fresh simulated
datasets, fresh calibration and fresh paired particle/reset designs feed all
three code versions. The current kernel uses determinant group size 12 at every
step (24 flow substeps total). Each endpoint is the actual shared analytical
LEDH consumer, with guide=None. Conditional denotes beta1 current-observation
Gaussian proposals; physical denotes beta0.9 mixture proposals. No trained
KDM/cSMC guide, optimizer update or native parameter batch is timed.

Three workers compiled/warmed separately; subsequent requests alternated source
order and rotated cases. Sixteen complete paired blocks produced 576 timed
calls, plus 36 warm calls. Each call materialized every output before its timer
stopped; serialization was measured separately. First calls took 16.6--23.2s
and were excluded from warm latency. There was no retracing: one trace per arm.
Runtime source snapshots stayed unchanged. No algorithm or protection was edited.

All 612 calls returned valid results and repeated each source's warm output
exactly. All 24 cross-source dataset/arm/direction comparisons passed
1e-8 absolute +1e-8 relative tolerance for value, directional score, normalized
and raw moments, and losses. Common flags match; current/checked-repair common
numeric diagnostics also match, excluding intentional grouping metadata.
Maximum differences across these comparisons: value 9.20e-11, score 3.78e-8,
raw moments 8.28e-9. The largest normalized tolerance ratio is 0.00157, well below 1.
The maximum raw squared-loss difference 2.38e-6 also passes its scale-aware test.
Agreement here establishes engineering parity, not model-score accuracy.

## Fresh timing results

Mean synchronized milliseconds per five-step filter call. The nonzero direction
is e0; it evaluates value and one analytical score direction, not all three
parameter directions. Confidence intervals are pointwise 95% paired bootstrap
intervals using 4096 whole-block resamples. No familywise or all-workload ranking
is claimed.

| Arm | Pre-merge | Current | Current/pre-merge, 95% interval | Observed change |
| --- | ---: | ---: | --- | --- |
| Plain, e0 | 422.54 | 512.69 | 1.2133 [1.1478,1.2981] | 21.3% slower |
| Conditional, e0 | 373.26 | 507.08 | 1.3585 [1.3300,1.3937] | 35.9% slower |
| Physical mixture, e0 | 717.23 | 522.32 | 0.7282 [0.7145,0.7461] | 27.2% faster |
| Plain, zero direction | 424.25 | 515.68 | 1.2155 [1.1845,1.2628] | 21.6% slower |
| Conditional, zero direction | 384.72 | 524.82 | 1.3641 [1.3340,1.3994] | 36.4% slower |
| Physical mixture, zero direction | 713.11 | 522.40 | 0.7326 [0.7179,0.7513] | 26.7% faster |

Zero direction still executes the analytical score recurrence; it is not an
independent value-only path. Both datasets separately give the same direction
of change with intervals excluding 1 for each pre-merge comparison. For e0, the
two plain ratios are 1.2085/1.2184, conditional 1.3697/1.3472, and physical
0.7218/0.7345. Dataset-specific intervals are retained in the analysis JSON.

The checked repair before determinant grouping is a distinct comparator:

| Arm | Checked repair ms | Current ms | Current/checked, 95% interval |
| --- | ---: | ---: | --- |
| Plain, e0 | 528.21 | 512.69 | 0.9706 [0.9407,1.0140] |
| Conditional, e0 | 492.36 | 507.08 | 1.0299 [1.0044,1.0597] |
| Physical mixture, e0 | 544.02 | 522.32 | 0.9601 [0.9213,1.0189] |

The grouping change does not establish uniform acceleration or the declared
<=5% non-harm criterion: the conditional e0 upper interval exceeds 1.05.
The other e0 intervals include 1. The zero-direction physical interval favors
grouping; the zero-direction plain/conditional intervals include 1.

The large N1008 plain/conditional gap to pre-merge remains compatible with the
source finding that current validity checks keep determinant computations live.
This run does not measure exclusive operation costs, so it does not establish
what fraction of the gap comes from determinants, derivative solves, OT or host
scheduling. It supplies no basis for removing numerical checks.

## Environment, budget and repairs

The actual run's own snapshots show the target GPU idle with 11MiB allocated at
both boundaries. The earlier preflight found unrelated jobs, but those earlier
observations do not demonstrate contention during this run. GPU process owners
were not polled during timing. Three resident benchmark workers shared the
device while executing one request at a time. Their TensorFlow allocator peaks
were 149267712/149276416/149276416 bytes (about 142.4MiB); memory growth was verified
for every worker before initialization. No OOM or device failure occurred.

The first CPU preparation failed because tf-gpu's TensorFlow ABI did not match
the existing Sylvester extension. Its log is preserved; no measurements resulted.
Switching to the bound tftwogpu environment repaired preparation in 6.68s without
installing or rebuilding anything. Eight focused harness tests passed. The GPU
run completed in 509.35s (8.49 minutes); summed resident-worker lifetimes were
1327.79s (22.13 minutes), below the 45-minute elapsed/135-minute worker caps.
No task GPU process remains. There is no need for an additional timing retry.

A reporting-only correction removed an unsupported hardcoded assertion that
external contention occurred. The executed driver and original analysis remain
preserved. All numerical comparisons, times and intervals are unchanged.
The PDF was rebuilt with pdflatex (latexmk unavailable); its withdrawal notice
was visually inspected, and the final build has no unresolved-reference warning.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept fresh N1008 engineering measurement | Complete matched run and parity pass | No numerical, source, tracing or resource veto | Two datasets, fixed theta/T5/FP64, block-timing assumptions | Use these fresh measurements for this scope | Particle convergence or scientific accuracy |
| Reject a uniform speed-recovery claim | Plain/conditional are slower than pre-merge | <=5% non-harm fails | Exclusive operation costs unmeasured | Profile the accepted N1008 conditional endpoint if repair continues | Every merged route is slower |
| Recognize physical-mixture speed reduction in this scope | Pointwise intervals below1, both datasets | Parity passes | Different FP32/batch workloads unevaluated | Preserve the code and valid measurements | Guide quality or production-wide superiority |
| Withdraw N64 scientific conclusions | Owner minimum N1008 | Previous evidence disqualified | Guide training at eligible N not evaluated | Fresh N>=1008 training/tuning plan before efficacy claims | The scientific guide idea failed |

| Inference class | Status |
| --- | --- |
| Hard veto screen | All fresh numerical/source/repeatability checks pass; uniform speed non-harm fails |
| Statistically supported ranking | Pointwise timing intervals support slower plain/conditional and faster physical versus pre-merge, conditional on the block resampling assumptions |
| Descriptive-only differences | Cold cost, memory, individual timings, and checked-repair comparisons whose intervals include1 |
| Default readiness | Not evaluated; unchanged scientific controls are diagnostic comparability settings |
| Next evidence needed | Eligible-scope guide training for efficacy; actual native FP32 parameter batches for that workload's performance; operation profiling for exclusive cost attribution |

Post-run skeptical review: the strongest alternative explanation for small
current/checked differences is scheduling variability and correlation between
neighboring blocks. Source parity cannot certify posterior accuracy. The weakest
generalization is transferring this FP64/T5/single-cloud result to the other
agent's FP32/T50/native-batch workload. An eligible repeat with intervals crossing1
would overturn a speed ranking for its matched scope. These limitations do not
revive any disqualified N64 result.

Raw and structured evidence: [campaign manifest](campaign-manifest.json),
[analysis](run-02/attempt-01/analysis.json),
[completion](run-02/attempt-01/result.json), [progress log](run-02/launch-01.log),
[fresh input manifest](run-02/preparation.json),
[current worker manifest](run-02/attempt-01/current/manifest.json),
[checked worker manifest](run-02/attempt-01/checked/manifest.json),
[pre-merge worker manifest](run-02/attempt-01/premerge/manifest.json),
[executed driver](run-02/benchmark_sir_n1008_runtime.py),
[harness tests](harness-tests.log), [PDF build](document-build.log).
