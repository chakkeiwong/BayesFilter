# State-space progress continuation: implementation audit and execution

The [active plan](bayesfilter-hmc-ssm-progress-continuation-plan-2026-10-02.md)
has passed implementation review and focused validation. The previous run's
nine complete main fits remain preserved. The first refreshed queue contains
seven unfinished main fits and five unfinished pilots; 16 original main slots
await matching complete pilot workloads. All 32 slots remain in the report.

## Repair and skeptical implementation audit

The dispatcher supports productive retries until the global deadline and
discovers newly eligible work after every turn. The active state-space entry
point uses this mode, without the old six-hour stage boundary or three-attempt
cap. Each frozen-child continuation receives its recorded prior cost plus the
current quantum, and exactly the next permitted continuation count. Complete
negative assessments remain terminal, and a negative assessment that did not
complete the declared workload cannot price a main fit.

The audit found and addressed a second budget issue: repeated localized
relaunches could otherwise each draw 48 hours while spending the old balance.
All this amendment's receipts now share a campaign ID, and allocation subtracts
their cumulative cost from its 172,800-second ceiling. A conservative 30-second
closeout charge is included once. Manifest/invariant exceptions settle the
enclosing reservation; terminal source/report errors cannot masquerade as
completed workload coverage. Numerical child shutdown remains bounded by the
existing supervisor and systemd control group.

Full source/prepared-input checks and actual final receipt bindings pass. The
numerical source remains commit `de80aaff5812ebfbed551977476c0868551a2c88`,
identity `a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`.
The controller is from the shared dirty checkout, with explicit file hashes
and copied sources recorded per run; its HEAD is `70a6d7e9611a9f859a2519f37f236b61cac35c1e`.
No numerical-source, acceptance, posterior, data, seed, or candidate-selection
policy change is part of this repair.

Remaining risks are workload affordability, another preparation needing more
than its first quantum, and strongly dependent acceptance evidence. The first
two produce explicit incomplete/diagnostic outcomes; the last remains the
separate uncertainty-policy research problem. None is hidden by relaxing a
numerical criterion. K6's missing posterior oracle and K7's approximate-target
interpretation remain explicit. Runtime and completion counts do not rank HMC
candidates or establish statistical calibration.

## Validation and budget

- Broader selection: **147 passed**, including supervision, interval reporting,
  prior recovery regressions, and the new dispatcher/inventory tests.
- Actual linear-Gaussian and nonlinear sigma-point public-pipeline recovery:
  **2 passed**, with GPUs deliberately hidden and unchanged numerical evidence.
- Final affected selection after accounting/master-status changes: **46 passed**,
  overlapping the previous selection and adding one master-status check.
  There are **150 distinct passing checks** across these selections.
- Initial focused failures were confined to new test fixtures: a fixture's
  slot naming differed from the real campaign and tuple/list equality ignored
  JSON normalization. Both fixtures were corrected; failures remain archived.
- The trusted TensorFlow probe passed on host GPU 1, UUID
  `GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`, with memory growth enabled.
  It is the same hardware class as the prior worker. Each actual child must
  still verify GPU/XLA and pre-initialization memory growth.

Exact commands, elapsed times, environment, source hashes and test logs are in
`artifacts/hmc-ssm-progress-continuation-2026-10-02/validation-r1/`.
CPU tests use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, two-thread limits,
`CUDA_VISIBLE_DEVICES=-1`, and explicit diagnostic status. The official LaTeX
tuning chapter, agent reference and regression list describe the repaired
allocator; their scientific tuning procedure is unchanged.

The new GPU ledger carries all prior charges once, adds 172,800 seconds and
records 244,012.317 seconds available before reservation. This amendment can
use at most its new 172,800 seconds cumulatively; the prior unused balance and
separate 1,200-second allowance are not automatically drawn. The CPU ledger
records the separate new 172,800-second grant, measured test wall time and a
labeled conservative 600-second allowance for short local audits/build/reporting.
It records the prior CPU balance separately without drawing it. Both ledgers
are under the new artifact root; the master points to them.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Launch repaired continuation | Dispatcher, accounting, source and real recovery checks pass | No identified implementation veto in checked paths | Full remaining workload cost and shared-device throughput | Run one bounded GPU queue from preserved checkpoints | All 32 fits will finish or pass |
| Preserve completed evidence | Nine final assessments validated | Exact interval miss and partial numerical overlaps retained | Calibration precision and reference accuracy | Include unchanged results in terminal reporting | Completed means statistically calibrated |
| Keep uncertainty research separate | Its experimental rule failed prior calibration | No statistical-default promotion | Dependence and estimand requirements | Continue its separate planned research when scoped | Scheduling repair validates a new acceptance policy |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Engineering/source checks pass before launch; numerical checks remain enforced per child |
| Statistically supported ranking | None |
| Descriptive-only differences | Work cost, candidate counts, completion counts and saved reference agreement |
| Default-readiness | This is a campaign-allocation repair; statistical default readiness remains open |
| Next evidence needed | Completed remaining workloads, terminal source/evidence audit and independent calibration evidence |

Post-review decision: launch is justified within the owner's new allowance.
The strongest alternative explanation for future noncompletion is total search
or preparation cost rather than queue eligibility. A changed/corrupt checkpoint,
missing provenance, or absent durable progress would overturn eligibility for
that job; a completed negative assessment rejects that candidate without
discarding the rest of the campaign. Launch and initial-progress details are
recorded below once observed.

## First launch: diagnosed environment failure

`r1` launched at October 2 04:13:01 Shanghai and closed after 67.995 seconds,
with 97.995 seconds charged including closeout. The queue attempted 12 jobs;
each child rejected checkpoint reconstruction with
`execution device or numerical policy mismatch` before any numerical progress.
The original source binds `CUDA_VISIBLE_DEVICES` exactly. The prelaunch
assumption that a different GPU of the same model was interchangeable was wrong.
The passing readiness probe did not establish checkpoint compatibility.

All 12 immutable evidence sets and saved before/after mutable checkpoints match.
Their prior attempts were resource stops with durable progress. The nine
completed assessments remain unchanged. An explicit, narrow recovery branch
now checks the exact failure, unchanged evidence, prior progress and restored
original GPU selection. Failed receipts and their costs remain preserved.
Pre-dispatch checks reject a different GPU before any child starts. The original
GPU is `GPU-4f1220f9-7ba2-21ad-3f9a-b24c2ca4ce91`; a fresh `r2` continuation
uses that selection under the remaining 172,702.005 seconds of this amendment.

This result invalidates the launch-environment assumption, not the numerical
checkpoint or HMC candidate. The failure is an infrastructure repair trigger,
and gives no new sampler or calibration evidence. The amended plan and
`validation-r1/device-repair-audit.json` preserve this diagnosis. Additional
regressions exercise wrong-device preflight, successful narrow recovery,
changed checkpoint/evidence rejection, unexpected numerical-failure rejection
and cumulative charges.

The repaired selection passes **54 checks**, including the two actual
linear/nonlinear recovery tests and six new device-failure regressions. Across
the broader and affected selections there are **156 distinct passing checks**.
The official chapter rebuilt successfully and its changed page was rendered
and inspected; the isolated build has the expected unresolved cross-chapter
diagnostics reference. Trusted device inspection confirms the original GPU is
available with 808 MiB used of 32,760 MiB; concurrent work is present and the
shared-device policy applies. Remaining uncertainty is total execution cost,
not checkpoint compatibility under the checked original selection. Resume is
justified under the unchanged scientific contract and remaining budget.

## Corrected launch

`r2` launched at **October 2 08:58:00 Shanghai** as
`bayesfilter-hmc-ssm-progress-20261002-r2.service`, with service MainPID 3884415.
The first job is `pilot-K1`, native child attempt 006. Its manifest confirms
the original numerical source, GPU execution, XLA enabled and memory growth
configured before logical-device initialization. The log confirms XLA
compilation, and the allocation records the checked startup-failure repair.
The service is active. Its maximum remaining allocation is 172,702.005 seconds
including shutdown, ending by approximately October 4 08:56:22 Shanghai if
the full allowance is required. Completed work and failed startup costs remain
accounted for; no numerical promotion follows from successful startup.

The saved launch audit confirms new durable numerical progress beyond the
original 147 K1 observations. All earlier immutable numerical evidence, all
nine final main assessments and the coordinator source hashes remain unchanged.
The active reservation plus the failed-launch charge equals exactly 172,800
seconds, so the restart has not renewed the campaign allowance. The trusted
service check confirms active/running status. See `r2/launch-audit.json` for
the timestamp, current observation count and checks.
