# Matched CPU streaming cost follow-up

The prior single-process capacity ladder measured T128/N64 streaming XLA at
43.913ms against buffered XLA36.918ms. That is descriptive evidence from one
worker per arm and three warm calls, not an accepted regression. Determine
whether the difference persists across five randomized fresh-process pairs
at each of T32 and T128, N64,d2,float64. CPU is an explicit reference-cost
experiment; GPU remains the default and its uncontended measurements remain
pending. Do not change runtime code during this cohort.

Baseline: the buffered seeded owner frozen at c7c0b88c2; candidate: the current
streaming seeded owner, retaining the same callbacks, Philox/PCG64 streams and
shared numerical recurrence. Reuse the exact qualified seed13 LGSSM fixture,
process seed123, resample seed17, flow_substeps3, Sinkhorn2, balance2 and enabled
dual-cap/trust-region controls from the capacity cohort04695–04702. This is fresh measurement of already qualified September
owners; no pre-August21 LEDH evidence is reused and no canonical LEDH claim is
sought. Keep K=N and fixed enclosing XLA signatures.

For each horizon, pair and arm, use a separate process. The old queue suggested
ten workers, but two horizons times five pairs times two arms requires twenty;
this review corrects that counting error without increasing the global budget.
Freeze a standard-library randomized order (seed81130), with adjacent arms for
each pair and randomized horizon order per pair. Register the exact order in
the runner and save it in the result. Same TF environment and two intra-op/one
inter-op threads per process; record affinity and load averages. Do not alter
system affinity, other jobs, clocks or caches to obtain a preferred result.

Measure owner creation plus first synchronized call separately as cold cost.
After three unmeasured conditioning calls, measure thirty synchronized calls.
Sample RSS and available TF allocator statistics before creation, after cold
execution and after warm calls. Record process high-water RSS at those points;
compiler exports and a second comparison owner happen afterward and cannot be
used as the primary memory peak. Verify one trace and enclosing XLA/HLO with no
host callbacks, complete healthy output/status records and exact replay. After
timing, execute the other September owner on identical inputs and require the
existing complete-record comparison to pass; any invalidity/drift excludes the
worker and stops the cohort for diagnosis. Save inputs/configuration/source
hashes and every raw timing; do not trim slow samples or rerun on timing alone.

The unit of replication is the fresh-process pair, not the thirty warm calls.
Report each arm's within-process median, all five paired ratios, the geometric
mean ratio and a95% Student-t interval for the five log ratios (df4,2.776445105).
Also report an exact two-sided sign-flip test on paired log ratios; with five
pairs its minimum nonzero p-value is0.0625. The t interval is conditional on
independent approximately normal log effects, not distribution-free proof.
Report raw RSS differences, cold compilation times and source/device scope.

Primary repair trigger: a median paired slowdown greater than10% at either
horizon, or an inconclusive/noisy interval extending above10%, requires bounded
RNG/loop-fusion profiling before accepting costs. A small effect does not prove
GPU performance or universal capacity. Exact numerical records, unchanged
sources/settings, trace/HLO identity and complete artifacts are vetoes; load
averages explain possible timing variability but do not authorize discarding
inconvenient pairs. A failed owner is a repair trigger, not a reason to rank
its timing. No claim about leak freedom, posterior/HMC readiness or whole-master
completion follows from this experiment.

Allocation: twenty measured workers, one analysis/readback worker and up to two
localized harness retries, at most24 workers/1200 CPU process-seconds; no GPU
allocation in this unit. The24-worker ceiling includes a possible extra static
policy check. Global56 CPU/52 GPU hour caps are unchanged. One worker at a time,
initial300-second timeout, stop if the next reservation would exceed this unit
or global allowance. Use the already approved stable runner:

```
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py matrix --stage tests --test-batch streaming_paired_cpu --test-timeout-seconds 300
```

Artifacts stay in fresh numbered campaign directories. No new runtime/default
implementation, dtype, seed, tolerance, training, HMC, subagent, package/system
or main-merge change. Uncontended GPU costs remain their own evidence gate.

Skeptical review: corrected the worker-count error, specified per-process
replication and timing boundaries, preserved every pair, and separated valid
numerics from speed. Small sample uncertainty cannot establish equivalence or
justify accepting the earlier regression; the next action is declared before
measurements. Existing code and fixtures suffice, so new scope-wide numerical
matrices are unnecessary. Primary-agent review passes; no independent review
asserted.
