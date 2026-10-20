# Complete posterior initializer matched costs

Results through 04525 pass all 36 fresh CPU/GPU workers: isolated Git
031692a0b prior, public graph control and public default XLA at D1/D3, each in
three process repeats. The complete result records, unchanged-input replay,
changed starts/scales, independent Gaussian mean/covariance and compiled-owner
trace/HLO gates pass. All accepted workers share 3441 source hashes. The GPU cohort 04508--04525
uses preflight-qualified compute GPU 2 with verified memory growth and no
other compute process in the sampled observations. Analyzer 04526 passes 14
checks and policy 04527 passes 147. No whole-program or main-merge qualification
follows from these fixture results.

The first attempt, 04489, failed before timing because TensorFlow Probability
could be imported but had no distribution metadata under the queried package
name. Recording the imported module's __version__ repaired the harness without
installing or changing packages. Preserve its 5.878002-second charge; it is not
a numerical or performance result. Accepted sources are frozen in
posterior-public-cost-04490-source, based on pushed commit 976c33552.

The measurement is the complete exported call, including construction, CPU
cloud preparation, validation and payload materialization. Each worker records
a cold call, three warm calls and a changed-input call before reference
execution and compiler inspection. These are factor_max=1 identifiable Gaussian
fixtures, not default factor_max=2 or actual DZ5 capacity. The graph control
preserves original XLA movement/curvature dependencies and is not an all-non-XLA
compiler ablation. The prior already contains earlier campaign repairs and
does not replace the oldest-original terminal comparisons.

| Device / dimension | Arm | Median cold seconds | Median warm seconds | Maximum observed RSS MiB |
|---|---|---:|---:|---:|
| CPU / D1 | Prior | 12.501 | 9.5047 | 3169.8 |
| CPU / D1 | Graph control | 15.052 | 0.0670 | 1839.6 |
| CPU / D1 | Default XLA | 13.962 | 0.0611 | 2204.7 |
| CPU / D3 | Prior | 35.575 | 20.5810 | 5544.3 |
| CPU / D3 | Graph control | 34.229 | 0.1121 | 3209.5 |
| CPU / D3 | Default XLA | 35.947 | 0.0961 | 3803.5 |
| GPU / D1 | Prior | 22.943 | 17.1552 | 3427.1 |
| GPU / D1 | Graph control | 24.143 | 0.1368 | 2252.7 |
| GPU / D1 | Default XLA | 22.737 | 0.0928 | 2496.9 |
| GPU / D3 | Prior | 71.430 | 47.6094 | 5589.7 |
| GPU / D3 | Graph control | 59.607 | 0.2827 | 3533.8 |
| GPU / D3 | Default XLA | 64.226 | 0.1895 | 3930.3 |

Relative to the prior, default XLA cold ratios are 1.117/1.010 and warm ratios
0.00643/0.00467 for D1/D3. The maximum observed RSS differences are
-965.1/-1740.8 MiB. None of the default-XLA CPU/GPU ratios crosses the predeclared regression-attribution triggers.
GPU default-XLA cold ratios are 0.991/0.899, warm ratios 0.00541/0.00398,
and observed RSS differences -930.2/-1659.4 MiB for D1/D3. GPU allocator peak
ratios are 0.336/0.676. The graph control triggers allocator attribution at
2.237/2.295 times prior: 1,236,480/1,302,272 bytes versus 552,704/567,552 bytes.
Default XLA peaks are 185,600/383,744 bytes. Preserve this graph-control finding;
its small absolute size does not waive the existing trigger. The capacity unit
now includes graph reuse and synchronized owner-return/payload boundaries.

These ratios compare complete implementations, not identical graphs with only
a compiler flag changed. No statistically supported performance ranking is
claimed from three process repeats.

The prior's RSS continues growing across the five measured calls; candidate
graph/XLA RSS is nearly flat after the first call in this window. This does not
prove leak freedom or attribute retained native memory. The next
[capacity unit](filter_gradient_posterior_initializer_capacity_20260927.md)
measures twenty alternating calls, four successful owner replacements and an
observer-only control. Python collection and native allocation remain separate.
CPU TensorFlow allocator telemetry is unavailable; it is not reported as zero.

The raw root is
/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917.
CPU analysis posterior-initializer-cost-cpu-04507.json was reopened and
independently recomputed with exact equality; SHA-256 is
1505fc014631f27c738f73a18cbe6430e02143b447513789e62ddba4e0c13470.
Its accepted interval is 04490--04507. Complete CPU/GPU analysis
posterior-initializer-cost-cpu-gpu-04525.json also reproduced exactly, with hash
abb8503be66c6bfc23ec8d7bfb33ae9cd8cfb84aad7ebbf5fa9e14dae9ea0027.
The archive posterior-initializer-cost-04527-evidence.tar.gz preserves 04489--04527,
all raw manifests, measured/completed records, logs, JUnit, source snapshot and
both analyses. Its verification receipt records every reopened member checksum.
The unit used 39 workers including the metadata failure and two short checks;
the reopened archive verifies all 214 members. Charges total 4227.008302
seconds, within the 10800-second allocation.

| Decision | Criterion/veto status | Uncertainty and next action | Not concluded |
|---|---|---|---|
| Retain CPU/GPU fixture results | Full records, independent Gaussian and source/device checks pass | Longer reuse and actual callers | Whole-program repair |
| Investigate graph allocator peak | Graph control crosses 2x; default XLA crosses no cost trigger | Twenty-call and return-boundary attribution | Harmlessness, native eviction or hard memory bound |
| Keep main unmerged | Actual consumers and terminal F01--F20 remain open | Continue master repair sequence | Scientific/default readiness |

| Inference status | Evidence |
|---|---|
| Hard veto screen | CPU/GPU numerical, replay, compiler and provenance gates pass |
| Statistically supported ranking | None; three process repeats are descriptive |
| Descriptive differences | Warm time and observed RSS decrease; cold cost retained |
| Default-readiness | Not established by these bounded fixtures |
| Next evidence | Longer reuse, graph allocator attribution, actual consumers and terminal checks |

Local skeptical review: construction and retained caches explain why this is
an endpoint comparison rather than a pure XLA ablation. The independent
Gaussian and isolated numerical reference reduce shared-implementation risk.
The weakest evidence is long-lived successful-owner capacity and actual target
coverage. A failure in either requires repair; small-fixture ratios cannot
override it. No independent reviewer is claimed.
