# C1 timeout recovery followed by the state-space campaign

## Active funding and continuation amendment, September 28

The owner added 50 hours and asked whether the 48-hour plan is ready. In the
context of the preceding GPU balance this is an additional 180,000 GPU seconds.
The September 25 ledger is settled at 31,448.932 seconds remaining. A new
additive ledger preserves its five charges, adds the grant once, and starts
with 211,448.932 seconds (58.736 hours), including the existing 1,200-second
NeuTra reserve. No additional CPU grant is inferred.

The first recovery finished fit 78 in 237.381 seconds. Fit 79 encountered foreign
GPU processes in 48 of 50 samples, received its full 750-second extension, and
still timed out at 1,500.301 seconds with recent progress. Eight later fits
were deferred. The recovery used 2,337.993 seconds. State-space admission then
waited 480.004 seconds without launching a numerical worker. Both charges are
settled. These are resource failures; they do not invalidate the numerical
target or the prepared state-space tests. Original C1 coverage failures remain.

At the new check all three GPUs were busy. A framework-free queue will scan
the three recorded same-class GPU UUIDs, in original-C1/previous-recovery/third
device order, every 30 seconds. It waits at most four hours across queue
admissions, within the existing four-hour contention reserve and original
state-space wall deadline. The 30-second cadence is inherited telemetry policy;
four hours is the existing convenience reserve, not a forecast of availability.
Unknown telemetry cannot admit a device. Waiting is charged conservatively
once to the GPU ledger, but does not consume the short numerical preflight
budget. The selected GPU becomes fixed before state-space preflight and remains
the device for pricing and main fits. Another job can still arrive later;
bounded per-fit admission and accumulated contention allowances remain active.

The next version preserves successful fit 78 and attempts only the remaining
nine original seeds, once each, with the unchanged C1 source and criteria.
This includes one localized resource retry of fit 79 under the new grant; it
supersedes the earlier per-attempt one-retry rule only for this continuation.
Recovery retains the 4,800+30-second ceiling. After settlement, the existing
state-space controller executes preflight, full-fit pricing and funded main
cases. The original SSM start time (September 28 05:03:52 Shanghai) is retained;
additional funding does not reset its 42/46/48-hour limits. Total SSM work,
including the prior 480-second wait and new queue time, remains capped at 36
GPU hours. The 22-hour main ceiling and all 32 denominator slots remain.

Skeptical audit: larger funding cannot resolve contention by itself. Repeating
the same eight-minute admission-only stage would produce no numerical evidence,
so capacity waiting moves ahead of numerical stage launch and is separately
bounded/charged. Completed fits cannot be selected for rerun based on posterior
results. Frozen package, data, seeds and criteria are unchanged; only the
external queue/recovery launcher needs changes. Tests must cover reuse without
rerunning completed fits, corrupt/mismatched prior evidence, busy/unknown/idle
capacity, one-time queue settlement, exhausted budgets and retained wall
deadlines. Passing these engineering checks permits launch; main-matrix
readiness still requires the actual GPU preflight and uncensored prices.

CPU launcher changes, tests and reporting have a further 900-worker-second
ceiling from the remaining CPU allowance. The new artifacts live under
`artifacts/hmc-ssm-funded-2026-09-28/`; every old artifact remains intact.
The question, baseline, promotion criteria, scientific vetoes and nonclaims
below continue to apply. This amendment passes the pre-execution self-audit.

The owner requested rerunning the ten timed-out fits under a revised policy,
then starting the prepared 48-hour state-space test. This authorizes the ten
specified recovery attempts and the following preflight/pricing/main sequence
within the existing grant. It does not create another GPU allowance.

## Question, evidence and preserved outcomes

Can the exact ten original beta-binomial seeds finish when resource admission
and contention allowances address the recorded execution failure? The baseline
is C1's frozen numerical package, unchanged design/data/seeds and the original
256 beta-binomial outcomes. Fresh reruns use indices 78–86 and 198, once each,
in a new output tree. Restarting complete fits is simpler to audit than moving
partial source-bound checkpoints and is priced by the 226 uncontended completed
fits (observed maximum 290.598 seconds; median 249.653). Those observations do
not guarantee the failed seeds' runtime.

An external framework-free supervisor uses the repaired resource policy while
each numerical child imports the unchanged C1 source. No source-identity check
is bypassed. The broad search, all-member retention, fresh verification,
posterior counts, precision and coverage criteria remain unchanged. R-hat,
ESS and MCSE remain posterior checks. Original reports are immutable.

Primary engineering success is ten clean exits with original-source/seed/data
identity and complete independent assessments. Posterior delivery/coverage
are reported separately, including failed results. An additional recovery
summary may combine the 246 unchanged completed records with all ten recovery
dispositions using the original 256 indices. It must be labeled post hoc
recovery, not a newly prospective confirmation. The original failed study and
Gaussian coverage failures remain visible; no changed stopping-coverage claim
follows merely from finishing ten fits.

| Diagnostic | Role |
| --- | --- |
| Source/data/seed mismatch, corrupt/missing control records, non-budget worker crash | Continuation veto for the affected recovery; preserve and diagnose |
| Candidate rejection or posterior failure | Promotion veto only; continue other declared fits |
| Busy or unobservable GPU before admission | Bounded waiting; if still busy, defer remaining work rather than launch competing fits |
| Recent durable progress and observed contention intervals | Engineering eligibility for additional time; not numerical validity or estimated GPU share |
| Enclosing budget/deadline exhaustion | Hard stop; record missing dispositions without resetting allowances |
| Runtime/utilization and recovered coverage summaries | Descriptive diagnostic evidence; no superiority/default claim |

## Revised execution policy

Keep the old policy available for historical compatibility. Add an explicit
`observed_intervals` extension mode: accumulate only intervals bounded by two
trusted contended observations, cap each interval at the declared telemetry
cadence plus probe allowance, and retain this evidence after the foreign job
leaves. Recent durable numerical progress is still required. Publish the
bounded accumulated allowance as soon as it is earned, rather than waiting
until the last few seconds of the base deadline. Both the per-fit extension
cap and outer cell deadline remain hard. Observed busy time is a scheduling
allowance, not an estimate of actual lost compute.

Before each numerical child launch, an explicit GPU admission policy waits
for a trusted observation without foreign processes. Waiting is charged to
the enclosing stage budget; it does not consume a numerical child's base
allowance. Unknown telemetry is not idle capacity. Poll at the existing
telemetry cadence. A bounded wait that ends without capacity defers that and
later slots; it must not create ten consecutive failed numerical launches.
Ordinary optional-default behavior is preserved unless the campaign requests
this policy. Never kill another job, disable growth or relax numerical gates.

## Budget and numerical-choice audit

The settled grant has 34,266.930 GPU seconds, including the separate
1,200-second NeuTra reserve, leaving 33,066.930 for this sequence. Recovery
has a 4,800-second enclosing cap plus 30 seconds for shutdown/accounting.
This 80-minute allocation is a convenience ceiling: ten times the observed
uncontended maximum is about 2,906 seconds; the remainder provides restart,
admission and contention headroom. It is not a calibrated runtime quantile.

Each recovery child keeps the inherited 750-second base allowance and may earn
at most 750 additional seconds from observed contention with progress. The
equal additional cap is a conservative scheduling hypothesis, not a sampler
default. All such time must fit the same 4,800-second enclosing recovery cap.
Admission can wait at most 600 seconds per requested launch, also inside that
same stage cap; this inherited telemetry cadence times 20 is a bounded
capacity retry policy, not an estimate of job duration. There is at most one
new numerical attempt per failed index. An incomplete allocation remains a
recorded outcome; no automatic new grant or reset occurs.

CPU implementation and focused deterministic/integration checks have a
1,800-worker-second ceiling from the existing CPU balance. Tests use GPUs
hidden, the tfgpu environment and two CPU threads. GPU work uses a recorded idle
UUID of the same hardware class, trusted execution, verified growth, TF/TFP and XLA. Enclosing
GPU stages are charged once, not their nested fits again.

Before launch, the prepared state-space data and references are copied unchanged
into a new version with the revised supervisor and suites. After recovery
settlement, that version starts its GPU stages.
Its existing preflight, complete-fit pricing, 32-slot inventory, 22-hour main
ceiling, four-hour reserve and 42/46/48-hour deadlines remain, further limited
by the actual remaining grant. Preflight/pricing numerical settings do not
change. Admission waiting fits their existing stage envelopes and cannot
produce a cheap censored price. All funded stages execute sequentially; no
competing background campaign starts.

## Skeptical audit and execution order

The audit rejects copying repaired supervisor code into C1's numerical package:
its whole-package identity would change and saved results would no longer have
the stated source. Instead use the frozen child source and separate parent
policy. Fresh fits avoid checkpoint relocation while preserving exact identities.
The other risk is assuming 48 wall hours means 48 GPU hours: the grant remains
33,066.930 usable seconds before this recovery, so full SSM affordability is
conditional and every unfunded slot remains in the 32-slot report.

Test cleared contention before deadline (the fit-82 case), stale/untrusted
telemetry, no progress, early allowance publication, outer bounds, busy-to-idle
admission, unknown admission telemetry, no competing child launch, process
cleanup and immutable seeds/source. Include actual Gaussian/beta-binomial
CPU public-pipeline subprocess tests and the affected SSM planner tests.
Do not rerun entire numerical ladders unchanged by this scheduling repair.

The revised plan passes this self-audit. First implement and verify the
optional supervisor policy, then freeze its source and the refreshed SSM
launcher. A bounded detached sequence runs all ten recoveries, settles the
single enclosing receipt, writes original-versus-recovery results, and begins
SSM GPU preflight followed by complete-fit pricing and affordable main cases.
Expected posterior or coverage failure does not veto those independent SSM
engineering checks; shared source/artifact invalidity does. Record any localized
repair and fresh attempt within the same remaining budget.

A pre-mortem: another GPU job may appear after admission; accumulated allowance
and the outer cap handle that only for bounded delays. Too much waiting may
leave some slots unfunded. Successful retries may still fail posterior checks,
and cannot erase earlier missingness. The terminal result must distinguish
engineering completion, posterior delivery, unchanged original confirmation,
post-hoc recovery, remaining funding and the next justified action.

## Implementation review and launch, September 28

The revised optional policy passed 103 controller/executor/planner tests in
114.45 seconds. A further 29 tests passed in 28.72 seconds, including actual
Gaussian and beta-binomial isolated public pipelines, restart preservation,
original-source recovery validation, exact ten-seed selection, one-time
accounting and an independently bounded recovery process group. The sets
overlap in planner checks; these are engineering results, not sampler claims.
JUnit receipts are under `artifacts/hmc-c1-recovery-ssm-2026-09-28/`.

Final skeptical review caught and repaired the older C1 snapshot's different
JSON layout. Its complete Python source map now verifies against the original
fit identity `1ad92f2b86f625e0362ce1fe47b9c5df1f7a455cb9f4a2228c9c00235f07b549`.
Only four framework-free scheduling modules differ from the prior prepared
SSM package. C1 numerical children use the original package, not those modules.
An independent systemd control-group limit bounds recovery at 4,800 seconds;
shutdown/accounting fits the 30-second reserve. Each SSM stage has its own
control-group limit and grant reservation. A failed/invalid result never
authorizes a repeated fit or another grant.

Trusted NVIDIA telemetry found the original GPU UUID busy with a foreign
compute process. GPU 0, `GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`, was idle
and reports the same NVIDIA GeForce RTX 4080 SUPER class and 32,760 MiB memory.
Use this device for recovery and all SSM pricing/main work. This resource-only
adjustment preserves target, source, data, method, dtype, TF32 and XLA policy;
cross-device runtime differences will not be attributed to the policy repair.
The admission check still runs immediately before every launch.

The sequence launcher is `scripts/run_hmc_c1_recovery_then_ssm.py`, frozen
under `artifacts/hmc-c1-recovery-ssm-2026-09-28/prepared-ssm-r1/source/`.
The fresh numerical output is its sibling `sequence-r1/`. The detached parent
service is `bayesfilter-hmc-recovery-ssm-20260928-r1.service`; its recovery
subunit adds `-recovery`. The exact systemd command is archived beside the
sequence before execution. Numerical verification of source, memory growth
and identity remains required on each completed child.
