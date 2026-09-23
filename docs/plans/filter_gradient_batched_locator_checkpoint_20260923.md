# Reusable batched locator and current DZ5 deadline checkpoint

The internal batched locator now accepts starts and scale as runtime operands
under one stable XLA signature. It resets its19 resources on each invocation.
The existing TFP search, exact incumbent, replay, eligibility and accounting
arithmetic are unchanged. A standalone invocation lock covers completion;
enclosing controllers must hold their own lock while using the compiled callable.
The public `locate_batched_local_center` wrapper is unchanged pending GPU
qualification, consumer checks and complete costs.

CPU03276--03282 and03284 pass all eight complete-record cases against pinned
d6a568384: batches1/3, rotated quadratic, nonquadratic, flat ties and invalid
rows. Each includes changed starts/scales and return to the original inputs,
exact target order/counts, one trace, unchanged HLO and actual callback/graph/
owner collection.03283 preserves a diagnostic JSON failure on nonfinite trial
positions;03284 uses explicit nan/inf labels for reporting and passes the
unchanged raw numerical comparison. No runtime repair was needed for that case.

CPU03285 additionally passes an enclosing XLA recurrence: an entirely invalid
attempt followed by a valid attempt, across changing inputs and return to the
original input. Complete records and callback order/counts match the original,
both inner and outer trace once, HLO is stable and all measured Python owners
are collected. This establishes the intended reset/composition mechanics on
CPU. It does not establish the GPU path. Two bounded GPU preflights declined
before worker launch because other campaigns heavily used both non-desktop
devices. Desktop devices were protected. The original int32 resource placement
remains to be checked on GPU before any compatibility repair.

The locator unit has consumed11/20 workers and248.566129/3600 charged seconds,
including all129 passing policy checks in03288. The full new module is guarded; the partial campaign
guard now covers228 sources and1333 exact exceptions, with no new exception.

The actual CDF proposal caller is
`MacroFinance-dz5-neutra/scripts/run_dz5_cdf_proposal.py`, whose `supervise` call
uses the hard wall deadline in `scripts/run_bayesfilter_estimation.py`. This
differs from the earlier inspected progress-only hierarchical supervisor.
The diagnostic tests execute the exact current standard-library function and
record both source hashes and its actual caller anchor. No external source,
frozen CDF campaign or model runtime was changed.

All four current-parent checks pass in03287 (the first three also pass03286):

| Worker | Observed outcome |
| --- | --- |
| Quiet successful work |Completes without a timeout; output preserved |
| Blocked call |Independent deadline terminates and reaps worker |
| Blocked worker with ordinary descendant |Process-group termination leaves neither process running |
| Worker ignoring SIGTERM |Five-second grace is followed by SIGKILL; worker reaped |

The supervision unit closes at2/4 workers and15.964788/120 charged seconds.
This qualifies the current parent's tested mechanics. It does not qualify actual
DZ5 target/transition wiring, completed compiled initializer telemetry, arbitrary
descendant escape, or the older hierarchical supervisor.

Receipt: `artifacts/filter-gradient-repair-20260917/batched-locator-and-dz5-parent-checkpoint-03287.json`.
Individual numbered manifests retain commands, environment, source hashes,
wall time, complete records and observations. The CPU probes explicitly hide
GPUs. No scientific or tuning thresholds changed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep internal locator candidate |All nine CPU checks pass |GPU qualification absent |Original GPU resource placement and downstream integration |Resume eight GPU records plus enclosing check on an available non-desktop GPU |Public/default or GPU readiness |
| Retain current-parent mechanics |Four actual-function checks pass |No tested deadline/cleanup failure |Actual compiled consumer still unrepaired |Use the verified parent in isolated DZ5 integration |Complete E5 closure |
| Preserve public integration gate |Current API source unchanged |Remaining numerical/consumer/cost checks |Default GPU and complete downstream records |Continue planned E5 work |Main merge readiness |

Review: original arithmetic and complete callback traces are the authorities;
component success does not close the public endpoint. Native memory retention
is distinct from Python owner collection. The implementation remains internal
and every failed/declined attempt is preserved. Canonical LEDH rebuilding is
outside this campaign.
