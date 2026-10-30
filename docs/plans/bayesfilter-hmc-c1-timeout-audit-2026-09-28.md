# C1 timeout diagnosis from saved execution records

## Question and bounded audit

Why did ten beta-binomial C1 fits time out? Inspect the original frozen-source
supervisor, all 256 beta-binomial exit/telemetry receipts, and the ten failed
fits' preparation, tuning and posterior checkpoints. Completed beta-binomial
fits from the same frozen design/source provide a descriptive timing and
workload comparator. No fits are rerun, no GPU is initialized, no timeout is
relaxed, and no historical outcome is replaced.

Primary engineering criterion: explain each stop from its saved hard deadline,
extension decision, progress and active numerical stage. A missing/mismatched
receipt vetoes a definite explanation for that fit. Foreign GPU process IDs,
utilization, wall time and numerical evidence counts are explanatory diagnostics;
they cannot by themselves identify a job's owner or prove the counterfactual
uncontended runtime. Existing posterior coverage failures remain a separate
question. No statistical ranking, convergence or coverage repair is inferred.

Skeptical audit before aggregation: distinguish per-fit ceilings from cell or
grant exhaustion; check the actual frozen source rather than assuming the live
code ran; avoid treating retracing warnings or high utilization alone as a
cause; preserve censored outcomes and the 256-fit denominator. All reported
thresholds are inherited from saved receipts, and any timing comparison is
descriptive. The saved artifacts are sufficient for this read-only diagnosis;
paired reruns would be needed to quantify the exact contention slowdown.

Use standard-library JSON inspection only, capped at 60 CPU worker-seconds for
the aggregation. Write the command, elapsed time and extracted evidence to a
fresh `artifacts/hmc-c1-timeout-audit-2026-09-28/r1/` directory, then append the
findings here. This inspection does not authorize a repair campaign or consume
the remaining GPU grant.

## Findings

The immediate cause of all ten stops is the **per-fit wall-clock ceiling**,
under sustained competition from other processes on the selected GPU. Every
receipt says `fit_budget_exhausted`, `allocation_limiter: fit`, and exit `-15`
(supervisor termination). No cell or total-grant deadline caused these ten
stops. Nine consumed the 750-second base allowance plus 120 seconds of grace;
fit 82 received no extension and stopped at 750 seconds.

The evidence strongly supports contention as the main source of the slowdown.
All ten have foreign GPU processes in 87–100% of saved workload observations.
The workers still recorded numerical progress 1–21 seconds before termination.
All completed preparation in 33–70 seconds and already had verified tuning
candidates. Nine were still performing candidate measurement/verification;
fit 86 had finished tuning and started its first 500-transition posterior
warmup chunk, with no committed posterior chunk at termination. The records
therefore do not support failed R-hat, a prolonged retained posterior run, or
an inability to find any acceptable kernel as the reason for these timeouts.

| Fit index (zero based) | Wall seconds | Samples with foreign GPU processes | Verified candidates already saved | Last stage |
| --- | ---: | ---: | ---: | --- |
| 78 | 870.343 | 27/30 | 6 | Candidate search |
| 79 | 870.319 | 30/30 | 3 | Candidate search |
| 80 | 870.380 | 30/30 | 5 | Candidate search |
| 81 | 870.384 | 30/30 | 4 | Candidate search |
| 82 | 750.348 | 25/27 | 4 | Candidate search; cooperative budget stop also saved |
| 83 | 870.315 | 29/30 | 5 | Candidate search |
| 84 | 870.340 | 30/30 | 6 | Candidate search |
| 85 | 870.379 | 30/30 | 4 | Candidate search |
| 86 | 870.364 | 26/30 | 17 | First posterior warmup chunk |
| 198 | 870.292 | 29/30 | 4 | Candidate search |

The nine consecutive fits 78–86 started between 16:05 and 17:59 Shanghai time
on September 26, with the last stopping around 18:13. Fit 198 started at
01:57 on September 27 and stopped around 02:12. These are localized busy
periods, rather than timeouts scattered evenly over all seeds.

Of the 246 completed beta-binomial fits, 226 had no observed foreign GPU
process. Their median wall time was 249.653 seconds and their maximum was
290.598 seconds. The other 20 completed fits had at least one contended sample;
their median was 302.015 seconds and maximum 506.054 seconds. These are
descriptive observations, not a randomized contention experiment. They do not
prove that all ten failed seeds would finish under 750 seconds on an idle GPU.

Adjacent completed fits 77, 87 and 197 recorded 155–172 tuning evidence items
in 77–88 summed evidence-call seconds. Timed-out fits generally recorded fewer
items (43–106, excluding completed-search fit 86) while spending 679–795
summed evidence-call seconds. Seeds and evidence-rung workloads differ, so this
is explanatory evidence rather than a paired speed estimate. Graph reuse was
enabled in every inspected fit. Retracing warnings also occur in adjacent
successful fits and do not by themselves explain this localized slowdown.
No inspected timeout log contains a traceback or an out-of-memory marker, and
none has a worker failure receipt. The saved host load does not indicate CPU
saturation. GPU telemetry preserves process IDs, but not enough provenance to
identify the other jobs' owners or purposes.

## Why the existing load-aware logic did not prevent it

The supervisor does detect machine load. Its GPU rule identifies other
processes on the selected UUID, excluding the worker and its process group.
Near the 750-second deadline, recent numerical progress plus a fresh contended
sample can grant the fixed 120-second extension. A fit still stops when that
870-second ceiling is reached, even if the recorded progress is healthy. The
grant budget and per-fit budget are separate: unused grant time does not
automatically increase a fit's declared cap.

There are two concrete limitations in that policy:

1. It gives only a fixed extension, independent of how much earlier time was
   spent sharing the GPU. It does not measure the worker's usable compute share
   or scale the allowance with cumulative contention. The coordinator then
   starts the next fit without a resource-admission wait, explaining why nine
   consecutive slots were exposed to the same busy interval.
2. It tests the **latest** contention sample rather than the earlier history.
   In fit 82, 25 of 27 observations showed another GPU process, but that
   process had disappeared by the samples at 746.146 and 749.352 seconds.
   The extension eligibility condition was therefore false at the first
   decision opportunity. The final record at 749.497 seconds also says the
   cooperative boundary had passed. The worker saved `ExecutionBudgetExceeded`
   at 749.397 seconds. Calling this solely a late-delivery race would be wrong:
   current-only contention eligibility had already denied the extension.

The decisive source is the frozen
`source-confirmation-r1/bayesfilter/testing/inference_validation/fit_supervision.py`,
especially lines 88–120. Both the current checkout and the prepared SSM
campaign snapshot have the **same supervisor file**, so this is still a
remaining scheduling limitation. The prepared model implementations and CPU
checks remain useful; they do not repair this execution behavior.

## Interpretation and next repair

Before another long shared-GPU campaign, add a bounded resource-admission wait
so new fits do not enter known sustained contention, retain cumulative
contention evidence when deciding allowance, and publish any extension before
the cooperative stop boundary. Pause new admissions after a contention-induced
timeout instead of spending consecutive slots during the same busy period.
Keep the enclosing wall/compute grant hard, and test these rules with controlled
clocks and workload probes before using GPU time. No uncalibrated utilization
percentage should be treated as an exact estimate of lost compute.

These are proposed scheduler repairs, not code changes made by this audit.
Any resumed diagnostic would use a fresh result directory and preserve original
C1 missing outcomes and costs. Finishing ten previously missing fits would not
by itself repair the Gaussian mean/joint coverage failures. Those statistical
failures require a separate analysis; no changed denominator or selective
replacement is justified by this timeout finding.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Timeout classification established | All ten receipts identify per-fit wall cap and supervisor termination | No conflicting source or stop-reason receipt found | None for immediate stop mechanism | Preserve original outcomes | Sampler rejection or cell/grant exhaustion |
| GPU contention is the leading slowdown explanation | Sustained foreign processes, healthy progress, much slower numerical calls | No recorded crash/OOM; preparation passed | Counterfactual idle-GPU runtime and other-job identity unknown | Repair admission and accumulated-contention handling | A measured causal slowdown factor |
| Confirmation remains unsuccessful | Original incomplete inventory and coverage screens retained | Ten missing fits; separate coverage failures | Terminal scientific audit beyond these timeouts is pending | Analyze coverage separately; price any new work | Correct coverage or a promoted default |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Ten administratively timed-out fits remain unavailable in the original denominator; no numerical failure established by this audit |
| Statistically supported ranking | None |
| Descriptive differences | Contention observations, progress, stage counts and wall times |
| Default readiness | No new timeout or inference default validated |
| Next evidence needed | Deterministic scheduler regression; bounded complete-fit check under declared load if a repair proceeds |

The strongest alternative explanation is seed-dependent candidate workload or
compilation variation correlated with the busy interval. It cannot be excluded
causally without a controlled comparison, but all ten fits had active foreign
GPU processes and the saved calls, not startup, consumed most of their budget.
The weakest evidence is historical process ownership and the unobserved
uncontended runtime. A matched replay under controlled load could overturn the
leading slowdown explanation without changing the directly observed cap rule.

The [extracted audit](artifacts/hmc-c1-timeout-audit-2026-09-28/r1/audit.json)
contains all 256 exit summaries, ten detailed failures, four adjacent controls
and ordinary input hashes. The
[execution receipt](artifacts/hmc-c1-timeout-audit-2026-09-28/r1/execution.json)
records 0.736 CPU worker-seconds for aggregation, zero GPU work, the exact
command/environment and inspected C1 source identity. No experiment was rerun.
