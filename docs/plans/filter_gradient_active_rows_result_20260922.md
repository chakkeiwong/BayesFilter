# E1 result: active geometry counts and conditioning rejection

E1 numerical qualification passes **83 CPU and83 GPU checks** in
02642--02644,02646--02660. These cover complete original3582b4ac records at
D1/D3/D5, changing inputs and active counts, poisoned inactive storage,
rank/holdout boundaries, exact Philox permutations, stable HLO with runtime
counts, fixed-fit regressions and22 existing public geometry cases per device.
GPU qualification uses UUID `GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba` with
verified memory growth. The existing public numerical controller has not yet
been replaced; public tests here establish a no-fire guard regression.

The active fitter uses exact compact QR/projection/reductions and a shared SVD.
Inactive rows do not become observations. Invalid counts cannot trigger target
calls. The new retained-solve guard rejects the inherited condition~1e15 case
as `fit_design_ill_conditioned`, reports rank/condition/roundoff margin, exposes
no usable geometry and makes no refinement/replay calls. Its threshold is
`eps * max(active_design.shape) * retained_condition <= sqrt(eps)`, an
engineering roundoff indicator, not a certified error bound. Tests1% above/below
the analytically derived crossing pass in graph/XLA, fixed/active, CPU/GPU.
Rank truncation and every accepted-result tolerance remain unchanged.

Preserved failures:02638 discovered the near-rank discrepancy;02639 attributed
identical raw results to the pinned ca920bac5 compact XLA implementation.
02641 incorrectly counted an empty rank-zero basis as an XLA runtime operand;
the test now counts nonempty inputs.02645 used Python identity against a NumPy
boolean in the new threshold test; converting the independent threshold to a
Python float repairs that test. Neither test repair changes numerical code.

All12 fresh cost workers02661--02672 pass complete original records, full
payloads and changed-count reuse. The matched CPU analysis is
`artifacts/filter-gradient-repair-20260917/geometry-active-costs-cpu-02672.json`.
The benchmark includes full result/hash/payload construction, one cold and20
warm calls, then changed count13 and20 alternating calls. Both extents start
with the same12 training/2 holdout observations. The numerical authority remains
3582b4ac; the cost baseline is ca920bac5 compact XLA.

| CPU storage capacity | Compact XLA warm ms | Active graph warm ms | Active XLA warm ms | Active/compact cold ratio | Extra observed RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|16|5.767|7.080|6.134|1.998|74.707|
|24|6.212|7.523|6.491|2.568|153.980|

At the changed count, compact XLA builds another program and takes1.900/1.873
seconds; active XLA reuses its program and takes0.0065/0.0064 seconds. These are
single-process descriptions, not a statistically supported ranking. Active
graphs have3972/5052 nodes versus1663 for compact XLA. Active graph-reference
and XLA use their declared distinct SVD implementations.

First-execution host RSS is658.38/679.13MiB for active graph and
1034.92/1114.42MiB for active XLA. Most added residency appears at first
execution, after tracing. The reported extra-RSS comparison above includes the
compact baseline's second shape compilation, so it is smaller than the
first-execution-only difference. Twenty alternating calls after the count
change add0.082/0.023MiB to active XLA RSS. This is neither a native-cache
attribution nor evidence of leak freedom. Capacity24's cold2x trigger remains
open, and larger public capacities still require measurement.

The GPU comparison is **not qualified**:02669--02672 recorded another compute
process on the selected UUID despite0% sampled utilization. The analyzer
explicitly rejected shared-device timing; it did not silently combine these
with the two clean compact runs02667/02668. All six GPU records remain
preserved as numerical/descriptive observations. Renew a complete clean GPU
comparison under E6, with unchanged physical UUID and frozen source/inputs.
The observer subsequently saw the other process on both non-desktop GPUs, so
no performance retry was launched. Continue E2 implementation under the same
budget and device-sharing policy.

| Decision | Primary criterion | Veto / uncertainty | Next action | Unsupported conclusion |
| --- | --- | --- | --- | --- |
| Use E1 as the dependency for E2 |166 numerical/regression checks pass|Full public outer execution is still unwired|Implement exact active pilot extents and enclose geometry|Whole-repository policy completion|
| Reject unreliable retained solves|CPU/GPU margin and no-target-use checks pass|Indicator is heuristic, not a certified forward-error bound|Keep the error/status guard and healthy comparisons|Identifiability or posterior correctness|
| Retain CPU cost observations|Six matched records and payloads pass|Capacity24 cold trigger; single fresh process per arm|Measure public capacities and investigate compile/native residency|Timing superiority or leak freedom|
| Reject current GPU cost comparison|All six numerical checks pass|Four shared-device preflights fail timing eligibility|Renew matching clean runs in E6|Qualified GPU performance|
| Continue campaign with main unmerged|102 policy/controller checks pass in02673|E2--E6 and all F01--F20 terminal dispositions remain open|Finish endpoint wiring, ownership, consumers, costs and integration|Merge readiness|

The policy guard covers215 sources and1306 existing exact exceptions. Both new
runtime modules are fully guarded, and six existing pure geometry kernels now
have Python-iteration coverage. No numerical-loop or NumPy exception was added.
The twelve cost groups and redundant rank-attribution alias are explicitly
explanatory; all active-count/guard/original-record groups remain mandatory.

Post-run review: complete component records establish the active-count repair,
but cannot establish public endpoint compilation. Static branch growth and
native compiler/allocator retention are the strongest alternative explanations
for extra host memory. The full endpoint at real capacities and coordinated
ownership tests can distinguish them. The GPU sharing veto prevents timing
claims from contaminated comparisons. The canonical LEDH rebuild stays outside
this campaign and unsupported claims remain blocked.

Charges through02673 are **53099.95632481277 CPU /50344.81445407226 GPU seconds**,
leaving17.25 CPU/38.02 GPU process-hours. E1 used37 workers and1301.995 seconds,
within its42-worker/7200-second tranche and unchanged global32/52-hour caps.
