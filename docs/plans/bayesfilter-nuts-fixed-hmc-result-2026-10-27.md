# Tuned fixed-length HMC versus NUTS: banana result

Configuration status: **FINITE-GRID BENCHMARK TUNING; NO CANONICAL TUNING
AUTHORITY**. Both methods were tuned on the same 2D banana target, with separate
validation seeds. Fixed-length HMC passes the planned validation screens. The
selected NUTS configuration fails the convergence screen because one chain
does not move. Consequently, wall-time differences and ESS rates below are
descriptive; no posterior-efficiency superiority claim passes the evidence
contract. Do not remove the failed NUTS replica or interpret the sign test and
bootstrap interval as overriding this validity failure.

## Comparison and measured result

Both arms use the existing `benchmark_nuts_xla.make_chain` authority: TensorFlow
2.20.0 / TFP 0.25.0, CPU/XLA, float64, identity mass, four chains and banana bend
b=.5. CPU was nominated by the earlier device pilot; this experiment does not
repeat a device search or establish the best device for filtering models.

HMC searches fixed L in {1,3,5,8,9,13,18,25}, including the naive L=1 and prior
L=8 arms. Both HMC and NUTS search target acceptance {.65,.8,.9}, with three
separately seeded tuning repetitions per cell, each having 512 warm-up and
1,024 retained transitions per chain. Epsilon adapts with stock dual averaging.
The nomination metric is median minimum mean ESS/sec over x,y,x^2,y^2,x^2*y.
These observables cover coordinate means, scales and banana dependence.

The selected fixed-L HMC has L=3, target acceptance .9, and frozen
epsilon=0.6089081806811285. Selected NUTS has target acceptance .8, frozen
epsilon=0.7586082110008886, depth cap6 and unroll1. These are the nominees of
the declared finite grid, not global optima. Epsilon is the median adapted value
from each selected cell. Both selections were saved before validation began.

Validation uses 12 paired seed IDs, common dispersed starts with independent
start/warm-up/sampling substreams, randomized method order within each pair,
1,024 discarded transitions using the frozen kernel, then 4,096 retained
transitions per chain. No held-out adaptation or retuning occurs.

| Held-out measure | Fixed-L HMC | NUTS |
| --- | ---: | ---: |
| Fixed/mean reported leapfrog steps per transition | 3 | 4.8323 |
| Median retained time, 4,096 transitions x 4 chains | 0.0273044 s | 0.1405021 s |
| Median time including discarded frozen warm-up | 0.0349223 s | 0.1952029 s |
| Median minimum five-observable mean ESS/sec | 106,892 | 10,235 |
| Mean acceptance probability | .9131 | .7992 |
| Maximum coordinate/observable rank/folded R-hat | 1.00136 | 1.52671 |
| Zero-movement retained chains | None | One chain in replica1009 |
| Required tail diagnostics available | All | One upper-tail coordinate ESS unavailable |
| Predeclared validation screens | Pass | Fail |

Observed retained execution is 5.1458 times faster for HMC. The ratio of median
primary ESS rates is 10.4439. These are different quantities: the first measures
execution throughput and the second combines execution with estimated
autocorrelation. The latter is not an admissible statistical ranking because
the NUTS comparator failed validation. The scalar minimum also does not say
that every observable improves by the same factor.

HMC's validation means for (x,y,x^2,y^2,x^2*y) are approximately
(0.000145,0.001724,1.00874,1.49937,1.01042), compared with exact
(0,0,1,1.5,1). All lie within 1.23 replicated MCSE of expectation, below the
predeclared four-MCSE screen. All HMC replicas have maximum R-hat below 1.002.
This qualifies the selected HMC candidate under this bounded check, not all
fixed-length HMC targets or tuning policies.

NUTS replica1009 retains a completely stationary chain1 during all 4,096
transitions. Its finite retained state/cache and well-formed acceptance records
do not establish mixing. The other 11 replicas have maximum R-hat below 1.003.
NUTS pooled x^2 is 1.4071 against expectation1, and pooled x^2*y is 1.5569
against expectation1. Its replicated MCSE expands because of the outlying
replica, so the four-MCSE mean screen passes despite these errors. That is
precisely why the independent convergence screen is required.

The paired geometric-mean ESS-rate ratio is 15.5481, with bootstrap95%
interval[7.4313,50.5976] and exact two-sided sign p=.0004883. These numerical
inference summaries are preserved for audit but **do not authorize a ranking**.
The stationary NUTS chain drives a very large paired ratio, and the protocol
requires both methods to pass validation first. The failed replica is included
in every summary; attempt01 and its deterministic replay are not pooled as
independent repetitions.

## Cost, compilation and correctness

All repeated target/gradient, integrator, tree/selection, RNG and transition
work executes within the same full-chain XLA functions used by the earlier
benchmark. Tuning warm-up is also fully XLA compiled. Static signatures,
optimized HLO hashes, output CPU placement and one trace per function were
verified. Python scheduling, diagnostics, timing and file I/O are outside the
sampler timings. The multiple-function TensorFlow warning concerns constructing
distinct configurations; each recorded function individually traces once.
CPU runs deliberately hide GPUs. No GPU workload or environment change was
launched for this comparison.

Selection, including imports, compilation, diagnostics and I/O, took
75.9785 seconds. It is not included in the retained times in the table. HMC
searched 24 configurations and NUTS searched 3, so setup effort is unequal and
HMC's greater search cost must be amortized. This study measures efficiency
after finite-grid tuning; it does not establish lower end-to-end cost for a
single short inference problem. Per-function first-call costs are recorded in
the manifests. A fair cold-start comparison must include search and compilation
under a predeclared equal budget.

All completed sample records have finite state/log-density/score caches and
well-formed acceptance telemetry. Native NUTS energy/divergence flags and tree
cap hits remain reporting-only under the October26 rule. The invalid comparison
comes from a stationary chain and failed R-hat/diagnostic availability, not
energy-flag counts. No rejection is removed from the denominator and no step
direction is inferred from a native divergence flag.

## Repairs, decisions and remaining evidence

Attempt01 completed all 27 tuning configurations, froze both selections, and
completed nine held-out pairs before strict JSON rejected a NaN upper-tail ESS
in replica1009. State tensors for that replica were already saved. A diagnostic
read confirmed the stationary chain and finite cache. The repair represents
unavailable diagnostics as JSON null with exact field paths and records
per-chain movement; it does not replace missing values with zero or omit them.

Attempt02 reuses the byte-identical frozen selection, replays the same twelve
held-out seed IDs, and completes. This is a localized reporting repair, not
fresh tuning or twelve additional independent repetitions. No kernel source
changed between attempts. The first engineering test pass had a test-order
initialization conflict; configuring TensorFlow before the new observable test
repaired it. The prelaunch combined suite passed14 tests; the post-repair
focused serialization/observable/sign-test suite passed6 tests.

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep L=3 HMC as viable for this banana | Held-out primary rate measured; reference and convergence screens pass | No retained-state or validation failure | Finite grid, one target/dtype/device | Use as comparator in a new bounded NUTS investigation | Globally optimal HMC or broad default readiness |
| Reject this frozen NUTS candidate for posterior qualification | Timing/ESS available but insufficient for promotion | One stationary chain, R-hat1.5267, undefined tail ESS | Initialization sensitivity, frozen epsilon transfer, or kernel behavior | Diagnose the saved start/path before choosing a repair | NUTS algorithm or research direction is invalid |
| Withhold posterior-efficiency ranking | Predeclared joint-validity requirement fails | Comparator veto remains | Large ESS ratio depends on failed mixing | Fresh validation after independently designed repair | 10x or15x valid posterior superiority |
| Complete current comparison | All selected settings and12 planned pairs preserved | Reporting failure repaired | Mechanistic explanation not measured | Start a separate bounded repair experiment if continuing | More unplanned tuning is part of this finished validation |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Completed sampler state/cache valid; selected NUTS fails posterior screen |
| Statistically supported ranking | None under the evidence contract, despite computed interval/sign test |
| Descriptive-only differences | 5.15x retained-time ratio and10.44x ratio of median observable ESS rates |
| Default-readiness | Not evaluated; all public defaults unchanged |
| Next evidence needed | Diagnose replica1009, separately tune repair, fresh independent reference/convergence checks, then compare rates |

Post-run red team: a few tuning replicas may nominate a high-throughput setting
that is sensitive to dispersed starts. Freezing a median adapted epsilon is a
specific deployment choice; per-run adaptation or different initialization may
change this result. A reference-valid NUTS configuration that consistently
escapes this start and matches HMC efficiency would overturn the practical
ordering. The weakest evidence is generalization from a mild 2D banana and a
single finite search. Tree bookkeeping overhead is a plausible explanation
for execution cost, not an identified cause without profiling.

Before stopping: the serialization defect invalidated an intermediate report
write and was repaired. The stationary chain rejects the current frozen NUTS
candidate under the stated posterior screens. It does not invalidate the
banana target, checked score, HMC implementation, or NUTS research direction.
No later scientific repair phase was part of this comparison; held-out data
must not be reused to retune and report a replacement as confirmatory.

## Reproducibility and checkpoint

Plan: `bayesfilter-nuts-fixed-hmc-comparison-2026-10-27.md`. Driver:
`docs/benchmarks/compare_nuts_fixed_hmc.py`. Artifact root:
`docs/benchmarks/artifacts/nuts-fixed-hmc-20261027/`. Attempt01 retains all
tuning data, original selection and partial validation; attempt02 contains the
complete replay, `summary.json` and `heldout.json`. Both manifests preserve
commands, source hashes/snapshots, environment, seeds, HLO and durations.
Both launch manifests record Git HEAD
`e2c183d5a9a4e9d8a054025e097fd212aa7f8a37`. The final preservation audit
corrected an earlier draft's commit identifier against those manifests;
executed source snapshots and their hashes are also preserved.

Two worker attempts consumed 91.831236 +19.361544 =111.192780 seconds, within
the900-second/three-attempt budget. The comparison is complete; unused budget
does not justify changing the predeclared scientific design after validation.
The artifact-root `audit-20261027-01/index.json` preserves five copied
console/test logs and verifies 18 source snapshots, 44 optimized-HLO hashes,
one trace per recorded function, identical frozen selections and identical
sampling-kernel source across attempts. It confirms 27 tuning cells, 12
validation pairs and 404/96 tensor shards in attempts 01/02.
`audit-20261027-01/artifact-sha256.json` records checksums for all 641 attempt
files. The archived test logs record outcomes and durations but not the exact
test invocation commands; no post-repair full combined-suite run is claimed.

Checkpoint: selected HMC L3/epsilon.6089 passes this banana screen and has
lower measured cost; selected NUTS epsilon.7586 fails one dispersed-start
replica. Report this distinction directly. Next useful work is a bounded
diagnosis of that saved NUTS trajectory, followed by fresh validation of an
explicit repair, not further tuning on these held-out seeds.
