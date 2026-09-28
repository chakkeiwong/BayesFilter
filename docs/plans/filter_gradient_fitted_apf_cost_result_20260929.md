# Fixed fitted-APF CPU cost screen

Four fresh-process arms04858–04861 and162 combined readback/policy checks04862
pass. They measure the actual fixed-fit adapter, including live seeded inputs,
all fitting, final analytical score, synchronization and ordinary result
formatting. Original archived source is4f0dfeb3d; repaired source is4629f3bd8.
Both arms share the frozen fixture, source closure, environment and CPU
affinity. No numerical source changed during the cohort. See the exact commands
and contract in `filter_gradient_fitted_apf_cost_20260929.md`.

| Case | Original warm median | Enclosing warm median | Enclosing/original cold ratio | Extra sampled RSS after warm calls |
|---|---:|---:|---:|---:|
| Gaussian |8.078214ms|1.297788ms|0.977217|99.105469MiB|
| Nonlinear scalar |8.103275ms|1.423909ms|1.034161|100.503906MiB|

Original/enclosing cold totals are4.248453/4.151662 seconds for Gaussian and
4.011291/4.148322 seconds for nonlinear. All30 warm calls per arm replay
exactly and reuse one trace. Full shared output/history differences are at most
8.881784197e-16 and4.440892099e-16, respectively. Fit digests are checked against
each actual fitted payload, rather than pretending rounded payloads must hash
identically. Discrete statuses and seed identities match.

Both sampled RSS increases exceed the predeclared64MiB attribution trigger.
VmRSS/VmHWM, smaps_rollup and rusage samples are preserved before/after setup,
cold/warm calls and ordinary Python collection. They do not measure exact
simultaneous maxima, live allocator tensors or compiler eviction. No HLO export
or comparison-owner compilation contaminated these samples. Add this result to
the existing enclosing-owner compiler/lifetime attribution work; do not erase
it with the unrelated rusage accounting explanation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Retain valid before/after screen | Source/input/output gates pass | No numerical mismatch | Single fresh process per arm | Replicate and inspect retained compiler residency | No statistical speed ranking |
| Keep cost acceptance open | Extra RSS approximately100MiB | Memory attribution trigger fires | Large-scope and unshared GPU costs | Shared owner-lifetime/capacity work | No memory-leak or eviction conclusion |

| Inference status | Finding |
|---|---|
| Hard veto | No numerical/provenance failure; memory trigger requires follow-up |
| Statistically supported ranking | None |
| Descriptive differences | Lower warm medians, similar cold totals, higher sampled RSS |
| Default readiness | Numerical repair qualified; terminal costs/capacity remain open |
| Next evidence | Matched replication, compiler/lifetime attribution, unshared GPU capacity |

Five workers consumed45.472253664 CPU seconds and no GPU time. Remaining global
budget is25.017350 CPU/24.776033 GPU process-hours. CPU is an explicit reference;
GPU stays the default target. The unchanged GPU numerical qualification does
not answer unshared cost/capacity. Adaptive iAPF remains a separate open repair.

Post-run review: small N16/T2 emphasizes Python overhead. Thirty calls in one
process do not constitute thirty independent process replications. The cost
scope excludes the score-study endpoint's separate physical-data/oracle work;
it includes the complete fixed-fitted adapter actually repaired. These results
support the recorded engineering observations only, not method-quality,
posterior, HMC, canonical LEDH or whole-program completion claims.
