# Complete filter and gradient execution repair

Status: executing on `repair/filter-gradient-xla-20260917`; merge is gated.
Owner request: repair all findings in the September 17 audit, compare memory
and performance before/after, review this program, and merge only when tested.

## Question and scope

Can every owned runtime route identified in findings F01–F20 preserve its
declared value/score semantics while removing Python numerical iteration,
nonreference NumPy, implicit pfor, and incomplete/default-off XLA boundaries?
Does the repaired complete calculation change trace/compile cost, host/device
memory, or steady execution time on the same inputs and hardware?

The source baseline is `3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf` (the prior
Kalman/UKF repair). Commit `39ad4f67` preserves the audit; its numerical source
is identical. No old LEDH result is reused. Baseline execution here is an
explicit engineering diagnostic, including any legacy implementation that
fails current policy. It grants no scientific or runtime admission.

The inventory and F01–F20 report define the initial scope. Follow actual
consumers, callbacks, preparation and imports into shared dependencies. Add
newly discovered reachable violations to the ledger rather than leaving them
outside the gate. Reference and archived implementations remain available as
such; relabeling an active runtime or deleting its tests cannot close a finding.
No algorithm substitution, changed score definition, hidden stop-gradient,
new reset rule, relaxed tolerance, or numerical retuning counts as a repair.

Owner-approved September 17 exception: migrate the two geometry initializers'
NumPy PCG64 probe-cloud RNG to the versioned TensorFlow stream
`geometry_tf_philox_cpu_xla_v1`. Seeds intentionally produce different clouds;
record the stream ID, seed, call order, shape and TensorFlow environment.
Numerical before/after comparisons must inject identical frozen clouds into
both versions. This does not authorize changing the sampling distribution,
thresholds, holdout roles or fit algorithm.

Scope confirmed by the owner on September 17: repair execution and block
unsupported canonical LEDH claims. The full canonical LEDH rebuild is outside
this campaign. Existing AD and finite-program manual-JVP scores retain their
explicit diagnostic semantics and cannot advertise training/HMC or canonical
admission. Compilation evidence does not remove that restriction.

## Completion ledger and order

Each finding must have source changes or a supported non-runtime classification,
an executed consumer/fixture check, relevant parity and compilation evidence,
and a recorded terminal decision. A passing subset never means complete.

| Phase | Findings | Work and required checks |
|---|---|---|
| 0 | all | Preserve baseline; inspect current callers; freeze comparison inputs; review plan and runner; establish resumable budget/logging. |
| 1 | F09–F14, F17, F20 | Native SQMC/initialization recurrences; TF quadrature and retained moments; SGQF preparation; TF admission/serialization; explicit analytical model derivatives or approved non-pfor model-local derivatives; tensor state simulators; remove indirect NumPy control. Check ordering, quadrature, moments, exact serialization and derivatives. |
| 2 | F01–F02, F07–F08 | Tensor time/particle/pseudo-time/parameter recurrences and batch-native algebra. Audit analytical score provenance through consumers. Preserve all Contract E moment/weight/transport terms. Existing finite-program AD/manual-JVP diagnostics cannot become canonical analytical scores by renaming. Canonical consumers must enforce the current rebuild contract. |
| 3 | F03–F06 | Pack heterogeneous TT core/checkpoint state as necessary; native forward/reverse time and ALS loops; preserve fitting branches, ranks, schedules and frozen proposals. Inspect paper and author-source anchors before any Zhao–Cui route behavior change; record mechanical execution refactors separately from algorithm changes. |
| 4 | F15–F19 | Batch-native CPU shards and TF serialization; XLA-default stable-signature factories; remove runtime NumPy from reachable geometry/identity/failure paths; isolate reference imports. Check all consumer signatures and role classification. Consult the HMC capability registry; no sampler/tuner algorithm change. |
| 5 | all | Repeat paired measurements, complete static/consumer guard, affected suites and integration tests; review changes and results; fresh remote integration check; merge only after all gates pass. |

Work within a phase may proceed independently when dependencies allow. Fixes
are committed in reviewable groups; artifacts record the exact source hashes
even before their source commit. Keep recovery status in the campaign ledger.

## Comparison contract

Use fresh processes for before/after × graph/XLA, same pinned fixture builder,
inputs, seed, shapes, dtype, outputs, device, TF environment, thread counts and
output lifetime. The baseline kernel comes from the pinned Git source, never
the edited checkout. Preserve actual failures: a baseline that cannot compile
has no invented XLA timing. Compare its existing valid execution separately.

Fixtures cover every repaired numerical family (complete endpoints whenever
available): Kalman/SRUKF, SGQF/model derivatives, SQMC, GenUT/Contract E,
particle/flow, TT forward/adjoint/APF, preparation and CPU score consumers.
Primitive evidence cannot substitute for complete endpoint evidence. Start
small, then use at least two horizon/parameter/particle extents for changed
recurrences and three fresh-process repeats for reported timing ratios. Use
FP64 parity authorities and relevant FP32/TF32 default execution checks.
TF32 tolerance must be justified before its fixture executes. Use fixed random
streams without retuning; obey the exact transport chunk rule.

Record preparation, trace, first execution (including compile), optimized HLO,
GraphDef node count, warmed execution, and output-copy time separately. Warm
timing synchronizes the returned tensors; serialization is outside timing.
At least 20 warm calls test fixed-shape allocation stability. Report /proc
host RSS/HWM and TF allocator current/peak; driver reservation is separate.
Inputs and output lifetimes match across arms. Check nested XLA boundaries so
the graph diagnostic is genuinely non-XLA. No Python callbacks are permitted
in a passing numerical graph. Check bounded tracing and graph growth.

Primary pass criteria: preserved values and analytic derivatives against
baseline plus independent identities/finite differences where meaningful;
finite outputs/status and branch parity; unchanged downstream consumer
semantics; policy guard clean for every admitted path; successful enclosing
GPU/XLA execution. Default FP64 comparison is atol=rtol=1e-10 unless a
documented existing numerical test requires a stricter or condition-aware
criterion. Discrete ordering and status outputs must match exactly.

Performance/memory ratios are explanatory, not mathematical validity. More
than 20% repeat-median warm-time regression, 2x device peak, 256 MiB additional
host memory, continuing fixed-shape growth, or horizon/parameter-dependent
trace unrolling triggers investigation before merge. A justified unavoidable
tradeoff must be recorded; unexplained regressions remain open. With three
process repeats, report descriptive medians/ranges, not statistical superiority.
The program does not certify posterior convergence, HMC readiness, LEDH
canonical scientific admission, or Zhao–Cui source-faithfulness from compilation.

## Environment, budget and stop conditions

Use the existing `/home/ubuntu/miniforge3/envs/tf-gpu` environment. No installs,
environment mutation, network research downloads or paid compute are included.
GPU2 is the initial device; check contention before launching. One GPU process
at a time; memory growth must be set/verified before any numerical import.
Two intra-op threads and one inter-op/OpenBLAS thread. CPU reference/test
processes explicitly hide GPUs. Test orchestration can use Python loops;
numerical implementations cannot use them to evade the policy.

The initial total budget is 4 GPU process-hours and 8 CPU process-hours, with
at most 300 seconds per benchmark worker, 900 seconds per focused test group,
three unchanged-fixture retries, and fresh numbered artifacts for every attempt.
Elapsed failed attempts consume budget. The runner records commands, source,
environment, hardware, outcomes and wall time in ordinary JSON/log files under
`docs/plans/artifacts/filter-gradient-repair-20260917/`. No overwrite of prior
evidence. Crash recovery charges unfinished attempts their reserved timeout.
No posterior chains or learned-transport training campaigns are in this budget.

### Proposed budget amendment, September 18 (not yet authorized)

The comparison audit at run 00834 has only 37 current pairs and 695 missing
pairs. Earlier measurements are preserved but many have stale harness or
shared-source hashes. GPU2 still has unrelated work; completing matched groups
on idle GPU3 requires both arms and every repeat on that same device.
Enumeration of the registered matrix finds 1,521 pending GPU jobs and 24 CPU
jobs. Using historical per-fixture/arm/mode/size median durations where known,
and 30 seconds for 708 jobs without a measured duration, estimates 33,764 GPU
and 720 CPU process-seconds. This is a planning estimate, not a runtime bound.
Passed test-group durations sum to 4,262 seconds; nine groups have no passing
duration, including the unresolved TP residual check.

Propose increasing the cumulative caps to **16 GPU process-hours and 12 CPU
process-hours**, inclusive of all time already charged. Through 00834 the
charge is 8,676.105 GPU and 20,882.313 CPU seconds. The existing caps remain
active until the owner approves this amendment. No extra device class, package
change, posterior sampling, training campaign, algorithm change or relaxed
criterion is authorized by this proposal. One GPU worker at a time, the
300/900-second worker limits, three-repeat rule and contention checks remain.

Complete the remaining source/callback audit before the terminal repeat
matrix. Use only focused qualification during implementation, then freeze
source and harness for final repeats and affected suites. A new repair after
freezing invalidates its affected measurements and consumes the same total
budget. This addresses the avoidable evidence churn found in the recovery
review. A failed gate still blocks integration; budget authorization cannot
substitute for source coverage, numerical parity or final review.

September 18 isolation repair: validate in linked worktree
`/tmp/bayesfilter-filter-gradient-xla-validation-20260918` on branch
`repair/filter-gradient-xla-validation-20260918` when concurrent unrelated
edits occur in the primary checkout. Commit campaign changes separately from
those edits. All worktrees share the original artifact root, lock and cumulative
budget through Git's common directory. Use the same bounded driver in that
worktree; its absolute program path is the only additional approval prefix.
The target, fixtures, seeds, hardware, tolerances and promotion gates are unchanged.

September 18 contention repair: focused correctness tests may explicitly use
`--test-gpu-index 3` on the idle RTX 4090 while GPU2 has unrelated work. This
uses the same GPU hardware class, contention thresholds, process limit, and
cumulative budget. The option is restricted to tests and their matrix stage;
all before/after measurements remain pinned to GPU2. Test logs record the
visible device, TensorFlow version, TF32 state and verified memory-growth
policy. This is an infrastructure repair, not a change to the comparison
contract or authorization for additional compute.

September 18 sustained-contention amendment: new measurement groups may also
select GPU3 explicitly, preserving the RTX 4090 hardware class and original
compute budget. Within each fixture/size/mode, both source arms and all three
fresh-process repeats must use the same physical GPU. The driver must not
resume a GPU2 arm into a GPU3 pair, and the comparator must reject mixed-device
repeat aggregates. Existing GPU2 measurements remain usable only in complete
matched groups. This supersedes the temporary test-only restriction above;
GPU2 remains the default selection. The contention thresholds, memory-growth
policy, scientific fixtures, tolerances and stop conditions are unchanged.

Extend numerical preparation coverage to the complete core-affine loss/gradient,
additive and pair fitted initializers, seeded balanced/residual initialization,
and prefix-score training targets. Use fresh exact constant-density parents
and frozen tensor inputs, the original settings/seeds, two extents and three
repeats with 20 warm calls. No fitted historical parent is reused. Each result
includes every returned numerical field relevant to the endpoint. A baseline
that cannot trace retains its failed attempt and uses its valid eager reference
for parity, without inventing a baseline compilation time.

Generic stochastic TT coverage uses fresh fixed heterogeneous cores, four/eight
axes and four/eight rows. Compare the complete density objective/gradient and
one Adam update for both density fitting and square-root prefit. Every timed
update restores identical input parameters and optimizer slots in both arms;
reset time is included and no optimizer trajectory is treated as a frozen
input. Preserve all returned numerical terms, updated parameters, and optimizer
state. Use the existing loss, clipping, regularization, and seed semantics;
these fixtures establish execution parity, not a trained-model quality claim.

Stop the affected measurement on invalid comparison, corrupted artifacts,
numerical mismatch, uncontrolled allocation or GPU contention. Repair local
harness defects within the same scope/budget. A failing candidate prevents its
merge, not independent repairs or a bounded retry. Budget exhaustion requires
a recovery report and further authorization; never mark remaining work done.

## Permission allow list

Use one reusable tool-approval prefix for the bounded campaign driver:

```
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /home/ubuntu/workspace/BayesFilter/scripts/run_filter_repair_campaign.py
```

Its allowed actions are status, pause, a fixed test group, a registered measurement,
the sequential registered matrix, audit, comparison and gate verification.
The matrix has qualification, three-repeat and current-source test stages;
it resumes current evidence, stops on source changes or candidate failure,
checks GPU2 contention before each launch, and uses the same cumulative budget.
The pause action lets the active worker finish and stops before the next
worker. A subsequent explicit matrix command resumes the campaign.
It has no arbitrary shell/command
argument, package installation, network operation, deletion, Git merge or push.
This process allow list is distinct from the source policy exemptions: those
must name exact reference/reporting/schema functions with reviewed reasons;
there is no blanket module or numerical-loop exemption.

Existing Git approvals cover branch/commit/fetch/merge/push. User authorization
already covers execution and conditional merge. Request the driver approval
once when it is concrete and reviewed; reuse the exact prefix for all retries.
No broad Python, bash, or external-review-tool approval is needed. A platform
prompt can still be required; this plan cannot waive it.

## Skeptical review before execution

Review outcome: proceed with the above gates. The following defects in a naive
program were identified and addressed before launch:

1. Merely wrapping loops in XLA leaves unrolled graph growth: native tensor
   recurrence plus horizon/parameter graph checks are required.
2. Reusing the edited checkout for “before” contaminates the comparison: use
   the pinned source snapshot and record imported source hashes.
3. A faster different score is a wrong comparison: preserve semantics and
   test total derivatives and consumer wiring; AD cannot establish analytical
   LEDH admission.
4. A primitive-only benchmark misses full TT/GenUT compilation memory: require
   complete endpoint coverage before closing those findings.
5. Output copying, allocator reservation and import cost distort measurements:
   separate phases, synchronize identically, and retain both host/device data.
6. A hand-maintained pass flag could permit premature merge: the final gate
   must require all finding closures and passing, current-source test and
   comparison artifacts. Missing/skipped required checks fail the gate.
7. A permissive allow list could hide violations: only bounded orchestration
   is approved; source exceptions remain specific and cannot admit runtime debt.
8. Remote integration can invalidate prior tests: inspect the integrated tree
   and rerun affected gates before advancing main; no force push/reset.

This is the primary agent's plan review. No independent-agent review is claimed.
Terminal review must revisit these failure modes against executed evidence.
