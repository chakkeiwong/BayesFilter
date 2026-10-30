# Scoped v7 release execution record

## October 5: persistent health monitoring enabled

The primary campaign remains active. On inspection, `r01-lgssm_qr` and
`r02-lgssm_qr` both completed with all 22 verified members exported/reloaded,
11,968 valid trials apiece, reconstructed checkpoints and no unaccounted work.
`r01-nonlinear` and `r01-funnel_residual` each deferred before TensorFlow import
after their original 60-second readiness cap. The observed competing PIDs
2187809 and 2192303 subsequently exited. `r02-nonlinear` was progressing through
closeout. These are current outcomes, not a terminal reliability verdict.
At the 18:10 UTC refresh, that nonlinear search had completed with all 20
verified members, 10,976 valid trials and a reconstructed checkpoint. The
residual-funnel slot `r02-funnel_residual` was progressing with a fresh checkpoint.
The live report then recorded three complete deliveries, two resource deferrals,
no other failures, and no progress or storage warning.

The prior service enforced bounds but had no independent observer or completion
notification. The new `bayesfilter-hmc-v7-monitor-20261005-r1.service` checks
progress, immutable inputs, completed member/device/memory records, storage and
service status every minute. `monitoring-01/status.md` and `status.json` are its
live reports; `events.jsonl` preserves alerts. The local desktop notification
service accepted the startup and two deferral alerts. No callable chat push
tool is available in this session, so unattended chat messages are not promised.

Fault-injection checks cover invalid source/seed/design, missing members,
incorrect memory/device records, missing terminal results, stale-progress
warnings, live-primary retry refusal, sampled-failure retry refusal, budget
limits, owned-child timeout cleanup, idempotent cost settlement and missing
receipt accounting. The selections establish **77 distinct passes**, including
**27 monitor tests**. The live-input audit passed in 0.217446 enclosing CPU
seconds and left the original campaign untouched. The inspection itself cost
0.104696 seconds on this early two-completion snapshot; later costs may differ.

The monitor may retry each untouched resource deferral once, with identical
configuration/seed and fresh output directory, only after primary termination
and within the original unused reservation. Original outcomes stay authoritative
and unchanged. This repairs missing numerical work as secondary evidence and
cannot turn a primary failure into a release success. The monitor has no generic
code-repair or release authority; other errors are recorded for diagnosis.

The active plan is `confirmation-monitor-plan-2026-10-05.md`. Exact frozen
observer/test hashes, request, command and launch review are under
`monitor-preparation-01/`. The CPU allocation transfer of 3,600 seconds comes
from existing funds; the GPU reservation and numerical source are unchanged.
At termination the observer settles primary and secondary GPU receipts once,
records its active CPU work excluding sleep/GPU intervals, and releases unused
reservations. The terminal scientific audit remains P4/P5 work.

## October 5: independent confirmation launched

The 96-search confirmation is running in `confirmation-run-01` under persistent
service `bayesfilter-hmc-v7-confirmation-20261005-r1.service`. It started at
2026-10-04 16:51:41 UTC. The frozen design interleaves 32 independent searches
per family. Its 96 root seeds were generated before sampling, checked against
recorded roots and saved with complete profiles. Source-23 and the separately
tested supervisor are unchanged; all 1,811 source hashes and 71 corresponding
current runtime files matched at launch. Every numerical profile matches its
audited full development search apart from seed and confirmation labels.

The final skeptical preflight passed in 0.366486 CPU seconds; the trusted device
probe used 0.103582 GPU seconds. Both are charged once. Available disk exceeded
the measured allocated projection with 20% sensitivity. The supervisor budget
is 189,063 seconds, including its 600-second settlement reserve; service cleanup
has at most 10 additional seconds. The full 189,073-second reservation fits
inside the release allocation. The outside 9,278.015433 GPU seconds are unchanged.
The enclosing service cost will be charged at termination without nested debits.

The first QR worker passed readiness and verified incremental GPU allocation
before initialization. Its saved trial records name GPU:0 and XLA. Confirmation
outcomes, independent delivery probability and release status remain pending.
No inference follows from successful startup alone. All failures/timeouts stay
in the original denominator; a malformed harness result stops further sampling.

The exact design, source/environment, seeds, preflight, command and launch status
are in `artifacts/hmc-v7-release-2026-10-02/confirmation-design-01/`. The evidence
contract and numerical/default audit are in
`artifacts/hmc-v7-release-2026-10-02/confirmation-execution-2026-10-05.md`.
Next: finish all 96 outcomes, independently audit raw and retained evidence,
settle cost, and assess the unchanged simultaneous lower-bound criterion before
the P5 scoped support decision. Load and seed-dependent cost remain unresolved
timing risks; no release or default promotion is claimed.

## Prelaunch October 5 checkpoint

Status: release **pending**. Both complete source-23 development price vectors
and independent audits pass. The resource-observed repeat completed in
3,657.258328 GPU seconds, retained all 22/19/17 verified members, and exactly
matched all 31,072 original raw trials, candidate inventories and states.

The repeated 96-slot point forecast is 32.51 GPU hours; the predeclared 20%
reserve, readiness and terminal allowances bring it to 40.78 hours. The
October 5 grant adds 24 GPU hours; 55.10 authorized hours now remain, with
14.32 hours beyond that scenario. It fits inside the release allocation alone.
No additional CPU grant is assumed. Confirmation remains unfrozen and
unlaunched. Both whole vectors and margin sensitivities are retained in
`artifacts/hmc-v7-release-2026-10-02/resource-price-affordability-grant-01`;
the current amendment is `artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-05.json`.

The bounded host-affinity diagnostic completed. All 36 native calls and
1,152 full trial records matched exactly. The quieter cores made warm record
assembly slower in both repetitions for all three families, and native timings
overlapped. No affinity or numerical-runtime change is adopted. The initial
resource deferral and successful retry are charged; all reservations are
released and no GPU run remains active.

P4 now requires the final 96-slot design: freeze independent seeds disjoint
from development, unchanged configurations, tested supervisor, exact budget
and fresh output root; recheck source, GPU and storage readiness, then execute
and audit all slots. The measured funding gap is cleared. Future costs remain
uncertain: the earlier slower vector with the same allowances requires 60.49
hours and exceeds the balance. Preserve the hard budget and classify failures,
timeouts and deferrals on the original denominator. No speculative optimization
or further complete pricing is required before design preparation. P5 remains
the final source/evidence, regression/documentation and local support decision.

The [master](bayesfilter-hmc-repair-master-program-2026-09-16.md) and
[current continuation record](artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-05.md)
contain the current evidence contract, skeptical audit and exact budgets. The
[October 4 record](artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md)
preserves the completed experiments and decision/inference-status tables.
Source-23 remains numerical authority. Defaults
and reporting-only tuning R-hat are unchanged; every verified member remains.
Earlier checkpoints below remain historical evidence.

## Earlier October 4 checkpoint: same-source pricing resumed


The final source/evidence state remains release **pending**. A user-authorized 24 GPU-hour grant (86,400 seconds) resumes the unchanged campaign after the historical affordability stop. The source-23 QR and residual-whitened funnel pricing queue is running in `gpu-price-source23-qr-funnel-grant-02` (session 14466), with a 6,200-second reservation. The first launch rejected a pre-created output directory before GPU initialization; its failure is preserved and a conservative one CPU-second launch allowance is charged. No confirmation has been frozen or launched.

The objective and the remaining QR/funnel prices, independent confirmation and support decision are preserved. The added allocation provides 109,846.965 uncommitted GPU seconds inside the release ledger, plus 9,278.015 previously reconciled authorized seconds outside it. It clears the prior nonlinear-only point forecast screen but does not establish affordability for the complete three-family confirmation. The skeptical pre-run audit and evidence contract are recorded in `artifacts/hmc-v7-release-2026-10-02/continuation-grant-2026-10-04.md`.

The uninterrupted source-23 nonlinear development price completed in
**2,227.336864 GPU seconds**. It explored 45 candidates and 25 repairs, retained
all 19 verified members, reloaded them, reconstructed the checkpoint, and
accounted for all 10,816 valid trial chunks and attempted work. The original
seed `(20261002,2502)`, starts, data, horizons, L grid, M100 cap and criteria
were preserved. No concurrency option or new lock was used.

`serial-price-audit-01/result.json` independently checks every saved chunk
hash, complete issued work identity, source/profile/device/memory-growth
record, member inventory and the enclosing ledger charge. Its 5.031 CPU-second
audit passed. The 71 current HMC runtime/release-driver files still equal
source-23. This evidence closes one full-procedure development price, not
independent delivery reliability or posterior correctness.

The supervisor is implemented with bounded pre-sampling memory-readiness
recovery, explicit wait costs, fixed independent slot/seed inventory, and
unchanged priced numerical profiles. Malformed/missing/wrong-seed results stop
further sampling and preserve all slots; late corruption/source drift also
prevents a delivery verdict. Timeouts remain unsuccessful. Its final focused
reporting/execution/inventory suite passed **146 checks**. Tests use fake workers
and clocks; no confirmation GPU campaign has been launched.

The original three K0 posterior panels/seeds and endpoint-health controls pass
all **12 checks** on immutable source-23 (130.680 CPU seconds). Warmup draws
are archived and excluded; retained streams are separate; mean and variance
agree with the independent reference within the existing four-estimated-MCSE
diagnostic tolerance. Reference JSONs remain under
`serial-posterior-interface-01/pytest-temp/`. This is not SBC or a coverage
claim. The next terminal check exposed a documentation-assembly defect: the
frozen tree lacks the renderer imported by guide tests. Its 5.230-second failed
collection is preserved. The repaired assembler's fresh documentation tree
passes all **161 guide/supervisor/reporting/inventory checks**, while source-23
passes **28 preparation, missed-mode and adversarial checks**. Together with
the 12 posterior/endpoint checks, these are **201 distinct scoped passes**.
All 71 runtime/driver files in the documentation tree match source-23. Numerical
and price evidence remains bound to source-23. A full-selection collection
attempt in the deliberately unbuilt documentation tree failed for the absent
native Sylvester library; its 8.186 seconds are charged. Collection succeeds
in the existing built workspace. Those earlier checks are terminal; current price state is recorded above.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept nonlinear full development price | Complete search, all-member replay, checkpoint/accounting and source/device audit pass | No numerical or artifact failure found in this run | Single original development seed and shared-resource timing | Carry this exact source/scope into affordability review | Independent delivery reliability or runtime guarantee |
| Resume release-pending pricing | The new allocation clears the prior nonlinear-only 71,274.780-second screen; complete QR/funnel prices and margins remain unmeasured | Confirmation remains unfrozen until all three prices and readiness costs are audited | Other-family prices, seed variability and resource margins remain unmeasured | Complete and settle source-23 QR/funnel prices, then recompute affordability | No three-family budget estimate, supported v7 release or default promotion |
| Close scoped interface/assembly checks | 201 distinct checks pass across current-source reference/negative controls and fresh documentation assembly | Missing renderer repaired; incorrect full collection in unbuilt documentation tree corrected | Scope remains bounded to these references and controls | Preserve evidence and proceed through remaining funded P4 gates | No relaxation of statistical release criteria |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Concurrent optimization remains withdrawn; serial complete run has no invalid trial; missing documentation dependency repaired; all negative controls detect their intended failures |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Native/stage/enclosing timings and one-seed price extrapolation |
| Default-readiness | Unchanged and separate from explicitly selected v7 release |
| Next evidence needed | Complete source-consistent three-family costs, affordable frozen 96-slot confirmation, simultaneous delivery bounds and terminal support decision |

Before the new grant, the nonlinear-only point forecast was **19.80 GPU hours** against **9.09 hours** remaining and exceeded the then-authorized balance by **38,549.799 seconds**. That was an early additive affordability screen, not a statistical lower bound on future runtime. The added grant clears that historical screen. Do not combine historical different-source QR/funnel prices to claim a complete forecast or quietly reduce the denominator; settle the current-source prices first.

Post-run skeptical review: the strongest alternative explanation for the
measured cost is shared GPU/CPU contention; its causal share is unquantified.
The old interrupted source-23 timing is not a second uninterrupted price.
A justified measured cost reduction could overturn the funding screen.
The result supports the serial implementation and supplied target at this
scope; it does not reject the research direction, prove general burn-in,
or supply independent confirmation. Publication and default promotion remain
separate from the scoped local support goal.

The subsequent read-only cost investigation passed in 3.423 CPU seconds.
`serial-cost-components-01/result.json` checks the row/seed inventory of all
338 native batches and partitions the 2,227.337-second price into disjoint
components. Native calls cost 628.468 seconds, serialization 63.842, other
search work 1,002.909, and setup/closeout 532.118. The first runner-shape call
accounts for 9.323 native seconds. No single zero-cost-component counterfactual
brought 32 nonlinear searches under the then-remaining 32,724.981-second allowance.
The checked result nominates no new localized repair; the largest uninstrumented
category alone cannot close the point-forecast gap. This is a descriptive cost
finding, not a hardware-independent lower bound. No GPU run or numerical change
was made; all release criteria and evidence remain intact.

## Earlier October 4 checkpoint: serial implementation restored

The goal remains a **supported, explicitly selected v7 release**; release is
**pending**. The [release plan](bayesfilter-hmc-v7-release-plan-2026-10-02.md)
controls P0--P5 and the [execution record](bayesfilter-hmc-v7-release-execution-2026-10-02.md)
preserves every result and cost. Defaults, all-candidate retention and the
reporting-only role of tuning R-hat remain unchanged.

The optional host-concurrency experiment is **withdrawn**. Unit and saved-data
parity checks passed, but the real source-26 nonlinear search failed during
TensorFlow summary analysis after 62.341 GPU seconds. The low-level cause is
not isolated. The proposed broad lock is not adopted: it serializes almost
all work and adds complexity. Source-27's smoke then failed in its diagnostic
helper before comparison, so it supplies no numerical evidence. Both failures
are preserved and charged. None of these experimental worker configurations
may contribute to release confirmation or silently reload under serial settings.

The working tree's 71 HMC runtime and release-driver files now match checked
**source-23** byte for byte. The unneeded public worker field, codec/CLI extension,
private TensorFlow context handling and lock are removed. Both guides describe
the retained serial procedure. All 57 compatibility/inventory/guide checks pass.
Source-23's prior exact GPU stream parity and 41 frozen-source checks remain
scoped evidence; its original full nonlinear search completed with 19 verified
members and the recovered closeout checked all 10,816 trials and attempted costs.
That interrupted-plus-recovered attempt is not an uninterrupted runtime price.

The next diagnostic reconstructs the exact failed work's 64 saved trials three
times through source-23's serial helper, under a 90-second worker cap. It makes
no new sampling or artifact-authority claim. If this fails, diagnose before any
new search. If it passes, obtain the original full nonlinear/QR/residual-funnel
prices on the same checked source, then freeze an affordable independent design.

| Gate | Remaining work | Required evidence |
| --- | --- | --- |
| P0/P1 implementation and recovery | Final compatibility/source audit | Exact supported runtime, immutable earlier failures, current readers and documented entry points |
| P2 complete procedure and numerical validity | Serial failed-work diagnostic and full current-source prices | Original L grid, starts/data/horizons/repairs; every verified member reloads; checkpoint and charges reconstruct |
| P3 calibration | Carry compatible bounded-trial/numerical evidence | Unchanged error allocations, validity vetoes and separation of reports from admission |
| P4 reliability and cost | Price, freeze and execute independent confirmation | All planned slots counted; simultaneous one-sided 95% per-family lower delivery bounds > .80 |
| P4 posterior interface | Final compatibility review of checked K0/negative controls | Warmup exclusion and posterior convergence/precision/reference assessment remain independent |
| P5 guide and release decision | Final support matrix and document audit | Every required scoped gate passes; supported scope and exclusions explicit |

The live allocation governs launches. Before the current serial diagnostic,
764.279 CPU and 25,685.609 GPU seconds remain inside the release allocation;
250 CPU seconds and 100 GPU seconds are reserved. Outside it, 48,936.184 CPU
and 9,278.015 GPU seconds are reconciled but not transferred. Proposed 32
searches per family remains unfrozen; no denominator, family, grid or criterion
will be reduced to manufacture a release. Existing power calculations remain
sensitivity analyses, not measured delivery probabilities.

Skeptical review rejects treating a passing smoke or saved-record comparison as
proof that native execution is robust. The full-run failure overrides the
favorable preliminary timing. Reusing checked serial code is justified by exact
source comparison and compatibility tests; its full costs and reliability still
need measurement. If a defensible confirmation cannot fit the reconciled budget,
report the precise funding/scope decision, with release still pending.

## Historical source-23 checkpoint: source-23 probe repair checked; full nonlinear price running

Source-23 manifest SHA-256 is
`358b840848d2b4ad07772f3a0bb326c42dde268bb8dd2d9a226df9ad5cd9a72f`.
It differs from source-22 in the Model B inspection graph, new focused tests,
maintained inventory/selection, and a native rebuild from unchanged C++.
The original nonlinear target rebuilt a compiled recurrence on every scalar
start probe. Its eager likelihood now reuses a shape-bound graph with current
theta, data, alpha and observation sigma as explicit inputs. Native HMC traces
the original implementation directly; no tuning policy changed.

The four-start batch alternative was rejected: Gaussian/funnel outputs matched,
QR/nonlinear did not. That diagnostic cost 27.495 GPU seconds. The singleton
graph diagnostic passed exact original-probe and changed-data equality with one
trace, costing 24.190 GPU seconds. Its .068--.082-second warm probe timing is
descriptive, not a complete-search performance claim.

The 12 worktree checks pass (75.662 CPU seconds). Source-23 assembly passed
(53.269 CPU seconds), and all 41 frozen checks pass (156.798 CPU seconds): both
public routes, live target/input/source/geometry mutations, independent QR and
nonlinear references, and partial-row fresh-process recovery on both SSMs.
The maintained selection collects 356 tests; collection is not execution.
A trusted GPU replay passed exact old probe and sample/trace equality for every
one of the 32 original source-22 first-work streams in 19.573 seconds.
No posterior or delivery probability is inferred from those engineering checks.

The full original nonlinear public search is running at
`gpu-price-stable-probe-01` (session 75038) with source-23. Its 3,100-second GPU
reservation includes the original 1,800-second search cap and a 3,000-second
process cap. All verified members must be exported/reloaded, the checkpoint
reconstructed and all attempted work charged. A capped search remains failed
as a complete cost even if it delivers checked members. Original base seeds
are preserved; source-bound trial seeds change with source as designed. The
saved-seed parity diagnostic does not resume or upgrade an old checkpoint.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Adopt the checked probe repair and price its full nonlinear procedure | Exact probe/native-stream controls and 41 frozen checks passed | Direct SSM batch replacement rejected; previous capped searches preserved | Complete downstream cost and independent delivery probability | Settle this price, then current-source QR/funnel costs and confirmation design | Release, speed superiority, stationary acceptance or posterior convergence |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No new probe, raw trace, reference or recovery failure in the adopted repair |
| Statistically supported ranking | None sought or supported |
| Descriptive-only differences | Cold/warm component times; full new-source price pending |
| Default-readiness | Not established; default unchanged |
| Next evidence needed | Complete same-source costs, affordable powered independent confirmation, terminal release audit |

Red team: native compilation can change arithmetic, so saved original trial
samples and traces were compared exactly. Graph reuse can conceal stale data,
so current tensors/constants and changed callbacks were tested. Remaining
complete-search overhead may still make the confirmation plan unaffordable;
a faster probe alone cannot close that gap. Source-21's capped nonlinear run
rejects that execution, not the model or finite-trial tuning direction.

## Earlier checkpoint, October 4: complete funnel, partial nonlinear, checked source-22

The two-family source-21 queue is terminal. Nonlinear produced ten verified
members, all exported/reloaded, but nine candidates remained unfinished at
the 1,800-second search cap. It is partial delivery and an unsuccessful complete
price. The residual-whitened funnel completed with 18 verified members out of 35,
all exported/reloaded. Checkpoint reconstruction and attempted accounting passed
for both. Total enclosing charge is 3,589.238 GPU seconds, counted once.
`source21-settled-audit-01` checked source/configuration/seed identity, GPU/XLA
and memory growth, all 15,136 chunks' complete work metadata, member inventories
and exact charges. It is a file/accounting audit of saved numerical checks,
not another independent run.

The code audit found and repaired a partial work-metadata comparator: missing
required fields and unexpected null fields could escape the comparison. Two
reproduction cases failed before the fix; 77 focused cases passed after it.
Source-22 manifest SHA-256 is
`b5a316e326d2691789949fc34370a9a1a3206d0cf66f5ff7b7e081416f804d0e`.
Its 186 focused tests, 346-test maintained selection collection and two actual
SSM reference checks pass. The fresh C++ build uses unchanged sources. Both
source trees and every failed artifact remain preserved. No scientific
threshold, public tuner, seed or numerical reduction changed.

Four recent regression files are now in the maintained selection and root-cause
inventory. The earlier 326-test collection and 109 inventory/reporting tests
passed; the later 346 includes the 20 new work-field controls. Counts from these
overlapping suites are not independent evidence replications.

Current-source timing localization found 1,463.021 seconds of nonlinear
within-work overhead beyond native HMC/serialization, and only 32.434 seconds
outside work-item calls. The exact saved-trial diagnostic passed for 32 trials.
The exact saved-statistics and R-hat probe also passed; warm calls took about
.02--.03 seconds each. The latter does not support R-hat as the dominant cost.
These are descriptive, instrumented/shared-device timings, not complete prices
or causal estimates of all runtime.

The unchanged exact binomial criterion needs at least 19 all-success searches
per family. Proposed 32 permits one failure each, with joint pass probability
.1406 under independent .95-delivery families and .8829 under .99. The rates
are sensitivity assumptions. No confirmation inventory is frozen or running.
No complete affordable current-source forecast exists.

The actual first-work profile is complete and matched every trial seed,
sample, trace, health record, score, analysis and candidate state between
plain and instrumented processes. It cost 255.311 GPU seconds. Instrumented
work time was 28.562 seconds, versus 15.949 in the inner trial observer. The
source call chain places a binding validation before that observer. Direct
follow-up on the same real binding measured 8.954--9.529 seconds in its probe
within 8.966--9.538-second validations; every stored probe value, score and
status matched. This cost 44.871 enclosing GPU seconds. Those measurements
identify a repeated cost but do not forecast a replacement or attribute every
second of the full-search residual.

All diagnostic workers are terminal. Final balances at this checkpoint are
1,637.480 CPU and 29,502.265 GPU seconds, with 250 CPU seconds reserved for
reconciliation. The next bounded hypothesis compares current native batch
probe evaluation with the exact scalar baseline while preserving mutation
checks. No probe optimization, changed numerical default or confirmation
inventory has been adopted. The separate old-policy SSM service is closed;
its evidence and remaining allocation are not presumed transferable.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve release-pending status and profile actual within-work cost | Funnel complete; nonlinear partial; source-22 checks pass | No invalid native evidence or accounting deficit observed; nonlinear complete-delivery criterion failed | Achievable complete-search cost and independent delivery probability | Validate exact current-target probe repair, then reprice and review confirmation feasibility | Release, stationary acceptance, posterior convergence or default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Exact checks pass in their stated scope; incomplete nonlinear search stays unsuccessful |
| Statistically supported ranking | None sought or supported |
| Descriptive-only differences | Timings and candidate counts; sensitivity power is conditional design arithmetic |
| Default-readiness | Not established; default unchanged |
| Next evidence needed | Complete affordable same-source prices, independent confirmation and terminal release audit |

Red team: instrumented cold calls and current contention can mislead attribution.
Only exact-output comparisons plus complete downstream costs can justify a cost
repair. The 19-replication minimum is not a well-powered plan by itself. A complete
single-seed funnel result cannot rescue the missing nonlinear/full confirmation
evidence. The result rejected the capped nonlinear execution, not the target,
finite-trial method or release direction.

## Earlier checkpoint, October 4: source-21 checked and two-family prices running


The current scoped replay implementation passes all 37 focused frozen-source
checks and both actual QR/nonlinear partial-row process recoveries. Source-21's
manifest SHA-256 is
`e174198d1163725f2fb2b635a76ccb1bfde999e5019c84418da80acb844db851`.
The preceding local suite passed 21 checks, the grouped/seed/reporting suite
passed 100, and three later predecessor controls passed. Source-20's 34 focused
and two recovery checks are intermediate evidence; repeated checks are not
additional independent replications. The final repair phase cost 893.320 CPU
seconds, including both source builds and all reruns.

`complete-health-batch-probe-01` passed exact equality, but its descriptive
savings were modest (.56--.69 scalar versus .38--.45 grouped seconds for the
same 32 records). It cost 19.472 GPU seconds. `complete-health-batch-probe-02`
changed the normalized-return vector and failed the exact comparator, costing
9.162 GPU seconds. Both are diagnostic evidence only. Neither was adopted.

The adopted repair shares only actual raw reconstruction between fresh readers
inside one synchronous closeout. A live tuning cache cannot populate it; keys
bind execution, schedule and current bytes. Reuse also checks each reader's
predecessor inventory. Independent file, source, observation, state and charge
checks remain. Scope exit discards the shared memo, including after exceptions.
The source-19 551.481-second checkpoint stage motivates this repair; it is not
a measured saving or an affordability claim.

The original nonlinear and residual-whitened funnel development cases are now
running sequentially in `gpu-prices-scoped-closeout-01`, using source-21. The
first worker records correct source, original seed, GPU/XLA and verified memory
growth. Search/process caps are 1,800/3,000 seconds; 6,200 enclosing GPU seconds
are reserved for the queue. The session is 45882. No confirmation is running.

Current charges are 19,498.931 CPU and 26,589.945 GPU seconds, leaving 2,103.355
CPU and 33,410.055 GPU seconds including reservations. Only 27,210.055 GPU
seconds are uncommitted. The separate old-policy SSM service is still marked
running; none of its reservation has been transferred.

A concurrent read-through found and repaired an independent-replication input
gap in the reporting helper: the actual model seed must agree with the frozen
slot, and the complete serialized search must match the declared profile.
Relabeled copies, changed acceptance bands/error allocations/evidence counts,
modified epsilon proposals and changed search controls are rejected. All 104
combined reporting/pricing checks pass. A CLI check still recognizes the actual
complete source-19 QR price and refuses a three-family forecast. These reporting
changes are outside the running frozen tree and cost 2.286 CPU seconds.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute the two missing full-family cases | New source passes engineering/recovery controls | Arithmetic batching rejected; historical incomplete prices remain failed | Other-family delivery and complete current-source costs | Settle the two cases and reassess affordable confirmation | Release, default readiness or speed superiority |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Final focused/recovery checks pass; other-family complete-procedure results pending |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Probe timings, old full-price timings and member counts |
| Default-readiness | Not established; default unchanged |
| Next evidence needed | Complete source-consistent prices, affordable independent confirmation and terminal audit |

Post-repair red team: cached arithmetic is valid only with the same current
inputs, including predecessor inventory. Negative controls now cover that
otherwise hidden dependency. Complete statistical reliability remains the
weakest evidence class; repeated engineering tests do not supply it.

## Earlier October 4 checkpoint: complete QR price


Source-19 completed the original seed/grid/M100 QR procedure in 2,318.380
enclosing GPU seconds. All 22 verified candidates out of 42 were exported and
reloaded; independent checkpoint reconstruction agreed. The 12,096 complete
valid trials account for 3,290,112 transitions and 40,282,112 gradient-work
units, with no attempted work outside complete trials. GPU/XLA and memory
growth checks passed. No release process is running.

Recorded native batch-call shares sum to 299.685 seconds; native calls plus
per-row tensor serialization sum to 365.095. These are host spans, not isolated
GPU-kernel measurements. Of 1,094.586 search seconds after preparation, 729.490
remain unattributed. Export cost 65.744 seconds, reload 576.700, checkpoint
reconstruction 551.481, and accounting 10.358. Inclusive measurements must not
be added twice. Shared-GPU utilization alone cannot explain this cost.

The exact-denominator confirmation analyzer passes 82 combined tests. Independent
SciPy beta references cover all success counts for n=32. Under the declared
simultaneous lower-bound requirement, 31/32 passes and 30/32 fails. The actual
QR price is recognized, but the missing other families prevent a full forecast.
The QR-only extrapolation is 74,188.171 GPU seconds, already above the remaining
33,438.688. No confirmation has been frozen or launched.

Both final diagnostic receipts have now been charged once. CPU charge is
18,603.325 of 21,600; GPU charge is 26,561.312 of 60,000. Balances are
2,996.675 CPU and 33,438.688 GPU seconds, with only the existing 250 CPU-second
reconciliation reservation active. The independent old-policy SSM ledger is
unchanged. The cap was combined search plus closeout, not an independent
closeout stopwatch; actual complete work remained within that cap.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept complete QR development evidence; keep release pending | Full search and all-member replay/reconstruction passed | Prior incomplete searches remain failed; no new evidence veto | Other-family delivery and affordable reliability confirmation | Audit and reduce measured host/replay cost, preserving exact evidence | Reliability, release or default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current QR complete-procedure checks pass; other-family full prices missing |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Single-run timings, candidate counts and forecast |
| Default-readiness | Not established; current default unchanged |
| Next evidence needed | Complete three-family prices, frozen affordable independent confirmation, terminal audit |

Post-run red team: a complete development seed does not establish delivery
probability. Timing attribution is incomplete and device sharing was observed;
the strongest alternative explanation for slow execution is a combination of
host processing and contention. An exact-preserving repair needs measured
component savings and a new complete price, not a reduced scientific design.

## Earlier October 4 checkpoint (superseded)

The original-source saved-member reload finished successfully: all 14 members
passed the public loader's existing checks in 510.927 enclosing GPU seconds
(505.182 loader seconds). The profile records 9,056 trial reconstructions;
trial assembly took 298.163 inclusive seconds, of which health analysis took
231.721. Chunk validation took 77.116, charge-registry construction 42.106,
and file loading/checksums 22.395. These inclusive component timings overlap.
The source-15 search remains incomplete with eight pending work items; this
reload does not repair that search or establish its checkpoint closeout.

Source-19 contains the emitted-analysis reuse repair. Its 44 focused frozen
checks pass, including both public multi-member routes, changed/failed evidence,
attempted-seed accounting and the mechanics branch. Two original failures are
preserved: the new mutation test named the wrong method, and the first repair
assumed the mechanics runtime had an exact-score analysis cache. Both were
corrected; the two focused reruns pass. Both QR/nonlinear fresh-process recovery tests pass (193.245 enclosing CPU
seconds). Source-18 remains withdrawn. One source-19 full-grid QR price is now
running in `gpu-price-emission-01`, with verified growth and original base seed.
The reservation is 3,100 GPU seconds: 1,800 search plus 1,200 closeout and
launcher allowance. This is development pricing; confirmation is unlaunched.

Source-16 passed 27 focused public/recovery/accounting checks and one separate
nonlinear recovery check; all 9,056 actual charge streams reconstruct identically.
The seed-registry diagnostic measured 20.344 versus .133 seconds descriptively.
Source-17's fused health checks passed 136 CPU checks, 46 trusted GPU checks
plus 32 exact saved-trace comparisons, and 152 focused frozen-source checks.
The source-16 180-second and source-17 150-second capped suites have no terminal
JUnit reports and remain incomplete. Frozen evidence keeps its original scope.

Both source-16 and source-17 reconstructed all 64 saved trial payloads exactly;
the descriptive profiled times were 5.586 and 5.528 seconds. Existing graph
composition also preserved every returned tensor but yielded only modest
savings. Neither measurement supplies a complete search price. Source-18's
manifest bundle was withdrawn and restored to source-17. Both earlier invented
summary probes were retracted as adoption/exact-parity evidence. Their original
receipts and audit dispositions are preserved.

The opening reconciliation is `reconciliation-2026-10-04-01.json`; subsequent
enclosing receipts are in `allocation.json`. Current totals are 18,595.174 CPU
and 24,242.931 GPU seconds, leaving 3,004.826 CPU and 35,757.069 GPU seconds.
Balances include the live 3,100 GPU reservation and 250 CPU reconciliation
reservation. Failed, capped and withdrawn attempts are charged once. CPU ceiling
is 21,600 under the existing amendment; GPU remains 60,000. The separate
old-policy SSM campaign remains active and keeps its own reservation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Price the checked source-19 repair | Complete search not yet passed; old saved replay passes | Original incomplete prices remain failed; placement change rejected | Full-work cost and delivery after checked reuse | Settle the running full-grid QR price, then refresh affordability | Release readiness or runtime superiority |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Complete-procedure prices failed; failed/incomplete suites remain recorded |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Saved-call and profile timings; partial delivered-member counts |
| Default-readiness | Not established; existing default unchanged |
| Next evidence needed | Complete family prices and affordable independent replication, then terminal audit |

The guide and chapter remain aligned. The existing 34 documentation/pricing
checks, 646-page book build and inspection of pages 421/428 retain their scope.

## Historical checkpoints

**October 3, 15:48 UTC:** release remains pending. The source-13 QR batch32
price ended incomplete at 1,980.571 enclosing GPU seconds. It returned from
tuning at 1,807.679 seconds and exported seven verified members by 1,955.855;
the full search, reload and checkpoint reconstruction did not finish. A separate
trusted GPU profile then reloaded all seven members successfully in 271.859
enclosing seconds. The original failure and every cost remain preserved.

The actual 100-record writer profile took 24.941 CPU seconds. Identical full
and incremental checkpoint bytes cost 4.025--4.807 versus .138--.139 seconds
per save; repeated history hashing dominates full saves. The repaired driver
groups charge/returned-row saves and reuses unchanged history during internal
batched dispatch, with full validation on every exit, explicit save, export and
reload. An initial flag-restoration defect accidentally made grouped saves
full, causing excessive cost; its incomplete/failed checks are preserved.
After repair, all 25 boundary/mutation controls pass. Source-14 builds and its
focused frozen public-route/recovery/accounting suite is running.

The case-subset seed bug is repaired with 18 pricing tests: each development
family keeps its original seed regardless of queue order. No confirmation is
launched or frozen. Next: settle source-14 checks, reconcile costs, and price
the original QR search with a measured allowance for complete closeout before
deciding whether additional family pricing can fund independent replication.

The following dated checkpoints are history, not current run status.

**October 3, 14:20 UTC:** source-13 built successfully and all 50 focused
frozen-source checks passed (44.005 + 663.560 CPU seconds). Both public
multi-member routes, all six batch recoveries and repair/verification recovery
passed. The original-seed QR six-L/M100 price is now running in
`gpu-price-batched-01`, with batch size 32, verified growth and actual GPU/XLA.
Its manifest binds source hash `337af410a8ac32a5fbfabf4ebc17b33134727c975cddda3572afd22dfd11d2e7`.
Source-bound science remains frozen. The complete search plus every checked
export/reload and checkpoint reconstruction is the price's pass condition.

Charged totals are 14,025.723 CPU worker seconds and 18,712.320 GPU seconds.
Remaining balances are 3,974.277 CPU and 41,287.680 GPU seconds, including the
2,000-second live price reservation. Unused native-repair reservations have
been released. Confirmation remains unfrozen and release pending.

The following dated checkpoints preserve prior state.

**October 3, 14:06 UTC:** final native RNG-loop parity passes on CPU and GPU.
All 32 original sequential-TFP streams agree exactly on Gaussian, QR, nonlinear
and residual-funnel targets. `native-batch-loop-01` passed 17 CPU checks;
`native-batch-gpu-04` passed all four maximum-size GPU checks in 140.253 seconds.
The two failed/partially saved batch mutation controls pass after correcting a
test's tuple/list comparison (`batch-failure-charges-02`, 6.484 CPU seconds).
The failed assertion and its 6.696-second cost remain preserved. Source-13
assembly is running before final actual recovery/accounting/harness checks.

The single original-seed QR full-search price remains conditional on those
checks. No confirmation is launched or frozen. Reconciled charges are
13,318.158 CPU worker seconds and 18,712.320 GPU seconds; remaining release
balances are 4,681.842 CPU and 41,287.680 GPU seconds, including reservations.
The old-policy SSM allocation remains separate. Release remains pending.

**Previous checkpoint, October 3, 13:43 UTC:** source-08's three-model queue ended with all
searches incomplete in 5,825.951 enclosing GPU seconds. QR/funnel timed out at
1,980.425/1,980.586 seconds after exporting two/eight members; their closeout
was incomplete. Nonlinear returned partial search with zero exports after
1,864.935 seconds. The full queue is charged once; none is a complete price.

Source-09 remains checked by 212 public/recovery/posterior/accounting tests,
35 health tests and exact saved-trace GPU health parity. Its guide/reference
updates passed 15 checks and the latest book build/render review. Native small
calls still prevent an affordable complete-work price.

The 107.093-second native capacity probe passed 18 calls on QR/nonlinear.
Descriptive warm cost per bank at 32 banks was .041--.044 seconds for QR and
.026--.027 for nonlinear, versus 1.386--1.430 and .677--.695 at one bank. Those
shape-dependent streams are not eligible tuning evidence. The subsequent
stream-preserving runner uses the original per-trial TFP seeds and arithmetic.
Its 13 two-trial parity/boundary tests pass on CPU (83.872 seconds) and in the
verified GPU/XLA run (164.155 seconds), covering four models at L=3 and L=25.
The earlier purported GPU run actually inherited pytest's CPU device hiding;
its label is retracted and its cost retained as CPU mechanics only.

The optional trial-batch driver and stage timing are implemented. Default
batch size 1 preserves old payloads. Integration/recovery and maximum-size
CPU checks are running; source-11 assembly is running. Next: complete those
checks, verify maximum-size GPU parity, freeze a complete pricing attempt,
and assess affordability before confirmation. Proposed 32 seeds/family remain
unfrozen and no release claim is made. `allocation.json` controls the unchanged
18,000 CPU / 60,000 GPU second release ceilings.

The following paragraphs preserve preceding checkpoint detail.

The v4 GPU queue has ended: all three slots are incomplete, with 5,857.215
enclosing GPU seconds. Source-06's CPU QR reference reached refinement and six
verified candidates, but timed out after 1,080.439 seconds during closeout;
only the first member was exported, and no terminal result exists. Preserve
both attempts. The saved first member independently reloads correctly; profiling
took 194.595 seconds and identified repeated JSON normalization/hashing and
tensor decoding. Confirmation has not launched.

The host-cost repair is implemented and checked: strict JSON hashing for
normalized evidence, call-scoped decoded-tensor reuse, and grouped retained
export/reload using the existing shared-bundle/member formats. All members
undergo numerical validation and later operations revalidate. Source-07 passed
78 grouped-I/O/health/accounting/posterior/recovery checks in 580.148 enclosing
seconds. Source-08 adds strict hashing on decoded checkpoint readback; its
147 legacy/checkpoint/SSM-recovery/pricing checks passed in 661.046 seconds.
The two initial focused failures were test assertions (JSON list/tuple
equivalence and nested telemetry call counting); their corrected checks pass.
The 15 documentation contracts pass; the rebuilt book's changed pages 418,
421 and 428 have been visually inspected.

Next: price frozen source-08 on the original three development seeds with the
repaired endpoint-reporting profile and grouped replay. Reserve 6,000 GPU
seconds within the unchanged release ceiling. Count only a complete search
with checked export/reload of every verified member as a complete price. No
confirmation has launched. The October 3 prelaunch audit confirms the full
six-L/M100 workload, exact panels/seeds, independent verification and all-member
predicate; shared GPU load remains an explanatory cost risk. This run measures
the repaired procedure; it is not a paired speed comparison with earlier
sources or evidence of reliability across seeds. Keep all earlier failures.

After reconciliation, charges are 9,844.152 CPU worker seconds and 11,749.527
GPU seconds. CPU balance is 8,155.848 seconds; GPU balance before this new
reservation is 48,250.473 seconds. `allocation.json` is the budget authority;
the separate old-policy SSM reservation remains untouched. No source-bound
checkpoint is resumed under changed source.

The bounded native-graph diagnostic passed exact sample/trace parity on QR at
L=3 and L=25 (four seed pairs at each L). Static graphs took 0.538--0.623 seconds
per warm L=3 call and 2.282--2.445 seconds at L=25; the corresponding dynamic
calls took 0.746--0.827 and 2.722--5.032 seconds. These small shared-GPU samples
are descriptive only. Static mode compiled two graphs; dynamic mode compiled
one. The result does not establish sufficient full-search affordability and
does not change the live configuration. Peak TensorFlow allocator use was
545,792 bytes. Two startup harness failures (missing binding arguments and a
model without the assumed `source_paths` method) preceded the corrected run;
all three attempts, totaling 76.932 GPU seconds, remain charged within the
original 240-second diagnostic allocation. Results are in `native-graph-03/`.

## Evidence and repairs so far

* The initial focused set passed 64 checks, including independent 80-digit
  inversion references at 64, 1,024 and 16,384 trials. Seven corresponding
  trusted GPU/XLA checks passed; every interval tensor was explicitly placed
  on GPU. The GPU run took 5.978 seconds. This validates the named numerical
  envelope, not a formal bound on every floating-point platform/input.
* The real windowed preparation test reaches the v7 controller and six-L
  pilot configuration. Its single trial deliberately cannot qualify a member.
  A missing seed in the first test fixture was repaired; nine focused checks
  then passed. The v7 policy's hashed historical status field is unchanged.
* The first full six-L/M100 Gaussian price stopped at its 900-second numerical
  cap, with 931.272 enclosing CPU seconds. A snapshot showed 418 native chunks
  taking 7.499 seconds within 237.641 enclosing seconds. The separately saved
  profile located repeated full-history JSON serialization and hashing.
* Checkpoint writes now use compact JSON and an equivalent strict JSON-native
  checksum. Internal within-work saves reuse unchanged persisted evidence;
  explicit saves, completed-work, error/pause, export and reload boundaries
  still validate live evidence. Charges remain durable before native calls.
* Actual fresh-process interruptions during repair measurement and verification
  reproduced uninterrupted samples, scores, decisions and two retained members.
  The interrupted attempt consumed additional budget without adding a score.
  Both cases passed in the 31-pass checkpoint-repair set. Its one failure was a
  new assertion comparing different elapsed-time snapshots; it was corrected
  to compare full and incremental serialization of identical state.
* The second frozen Gaussian price also stopped at its unchanged numerical
  ceiling: 956.882 enclosing seconds, 1,504 completed trials, 11 candidates and
  no fresh verified export. Five members had passed measurement. This is an
  incomplete full search, not positive delivery and not evidence against HMC.
* The next profile/trace found redundant per-trial health details in controller
  observations and duplicate analysis inside retained-member validation.
  New v2 observation summaries aggregate health counts, while all detailed raw
  trials remain in immutable numerical evidence. Historical v1 analysis is
  reconstructed in its original representation. Within a retained validation
  call each record is recomputed once; no analysis cache survives that call.
  A test comparing decoded lists with new tuples was normalized to the existing
  JSON wire semantics. The subsequent fresh-tree regression checks these fixes.
* Fresh-source assembly begins with Git and records explicit HMC/validation
  overlays, excluding unrelated q20/learned-NeuTra dirty changes. The first
  assembly passed 143 checks but failed 16 SSM imports because the native
  Sylvester library was missing. The assembler now includes the tracked CMake
  inputs and builds the library in the new tree. Both fresh builds succeeded;
  the manifest records the compiled library alongside sources and fixtures.

The source-03 regression passed all 223 checks. `clean-checks-03` covers checkpoint/accounting/codec,
repair and actual SSM process recovery, repeated analytic posterior handoff,
adversarial inputs, legacy execution/artifacts and full-procedure regressions.
Source-04 later passed 187 checks and failed the original K0 replication 2
because of a chance endpoint return. Source-05's 25 passing checks changed the
panels/seeds; they cannot close that failure. The original panels and seeds are
restored. `original-panel-replay-01` reproduces the failure and preserves the
raw trial; `endpoint-checks-01` passes all 38 checks with the explicit
endpoint-reporting policy, including the original three posterior checks and
frozen-path/cycle/divergence/nonfinite controls. The respective enclosing CPU
times are 51.977 and 193.678 seconds. No library default or historical result
has been changed. Source-06 assembly/native build succeeded, and its 30 focused
checks pass (465.610 CPU seconds), including the three original panels and
both actual process-recovery boundaries. The 15 documentation contracts pass;
the 645-page official book builds, and changed PDF pages 418 and 421 were
visually inspected. The scoped guide still describes v7 as experimental.

The price-reporting audit found and repaired a separate interpretation error:
one exported member from an incomplete search is useful partial delivery, not
a complete-search price. The runner now requires controller completion,
reconstruction and every verified export before returning price success. All
11 focused positive/negative cases pass. The frozen running runner is preserved;
its eventual outputs will be audited under the corrected full-search predicate.

## Historical pricing and next-decision checkpoint

The first `gpu-prices-01` attempt used frozen `source-03`. All three slots ran:
QR LGSSM and nonlinear returned partial-budget outcomes with no verified member
after 1,927.675 and 1,978.377 seconds; the residual-whitened funnel process timed
out after 1,980.277 seconds. The queue's genuine terminal receipt records
5,886.334 seconds and no unstarted cases. An earlier administrative closeout
incorrectly inferred termination from sandbox process visibility; its
`closeout.json` is explicitly retracted. Preserved checkpoints remain development
evidence. This is an incomplete v3 price, not v4 release evidence.

The completed second price used frozen source-05 containing controller policy v4. It
declares the same QR LGSSM, nonlinear SSM and supplied residual-whitened funnel
cases, six-L grid, M100 cap, candidate-specific repair, refinement and
all-member exports. The queue retains its 6,000-second ceiling and 1,800-second
per-model numerical cap. Memory growth, GPU/XLA placement and all outcomes are
recorded before interpreting delivery. This source predates the explicit endpoint
health repair, so it prices scheduling and old-health behavior only. A complete
price of the intended release profile is required before any
confirmation denominator is frozen. The independent old-policy SSM service and
its reservation remain unchanged.

Source-05's QR and funnel slots hit their 1,980-second process ceilings without
terminal results. Nonlinear returned a partial-budget result after 1,896.655
seconds with no verified export. All three slots started; the genuine terminal
queue result records 5,857.215 seconds. The source-06 CPU reference also remains
incomplete as described above. These are cost/delivery failures, not evidence
against HMC or authority to substitute easier seeds. Its six verified members
do not make the incomplete full search a usable confirmation price.

After pricing, inspect every candidate/work state, all-survivor exports,
native device records, attempted-work accounting and full enclosing costs.
Only then freeze an affordable multi-seed confirmation design. Preserve all
failed prices and planned denominators. Insufficient funding or delivery
keeps the corresponding release gate open; it cannot be repaired by changing
a failed threshold or substituting the earlier single-pair results.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue scoped release work | Source-15 focused frozen checks pass; original-seed QR price running | Full-search/closeout cost remains a promotion veto; failed attempts preserved | Complete full-search cost and independent-seed delivery | Audit this price's complete procedure and affordability before confirmation | Release readiness or default promotion |
| Repair host persistence/validation cost | Profiled repairs and recovery/mutation regressions pass | Raw evidence and full boundary validation preserved | Native execution remains substantial; static/dynamic timing is descriptive | Use complete prices and their cost breakdown to select any further repair | GPU speedup, sampler superiority or changed statistical power |
| Preserve separate posterior assessment | Original three analytic panels/seeds and positive/negative controls pass | A posterior veto cannot erase tuning or certify another member | Target scope and adequacy of reference diagnostics | Preserve checked handoff; report remaining posterior limitations | SBC, broad coverage or arbitrary-model posterior correctness |

| Inference status | Current conclusion |
| --- | --- |
| Hard veto screen | All previous full-search prices, including batch32 QR, are incomplete. Seven QR member reloads are valid scoped evidence; they cannot turn a partial search into a complete price. Source-15 focused checks pass; its full-search price remains live. |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | Native/enclosing timing ratios, profile times and single-model prices |
| Default readiness | Unchanged and separate from explicit v7 release |
| Next evidence | Completed full searches, funded independent replication and terminal scoped regression/guide audit |

The strongest alternative explanation for a failed long search is a combination
of conservative evidence requirements, geometry/trajectory behavior and shared
device load, beyond the measured host overhead. Compact files alone do not
settle those possibilities. A fully priced run must show where time and work
went; a successful short fixture cannot overturn the full-search failures.

`allocation.json` reconciles completed receipts separately from active
reservations. Failed attempts remain charged. Never charge nested native/build
receipts twice, or draw from the old-policy service's reserved allocation.

## October 4 host-analysis experiment decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Withdraw concurrent analysis and its untested lock | Saved-data equality passed; actual complete-search delivery failed | Native TensorFlow reduction error vetoes this optimization | Low-level TensorFlow cause is not isolated | Use checked serial source and measure complete procedure | No rejection of the tuner, model or GPU platform |
| Continue serial procedure | Exact failed work:64 trials, three identical reconstructions, all health-valid | No invalid saved tensor or trial found in this diagnostic | Diagnostic replay is shorter than a complete native search | Finish uninterrupted same-source development prices | No delivery probability or posterior claim |
| Keep release pending | No completed independent confirmation | Missing reliability/cost evidence prevents support decision | Budget feasibility and independent delivery remain unmeasured | Price all three original families, freeze complete design, execute or report precise shortfall | No default promotion or public release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Concurrent native search failed; proposed broad lock and subsequent smoke do not repair it |
| Statistically supported ranking | None |
| Descriptive-only differences | Three alternating warm timings nominated concurrency; native failure overrides that nomination |
| Default-readiness | Unchanged; experimental v7 is not promoted |
| Next evidence needed | Complete current-source prices, independent fixed-denominator delivery, final compatibility and guide audit |

Post-run skeptical review: the strongest alternative explanation is a TensorFlow
error not exclusive to concurrent callers; the serial failed-work replay does
not disprove it. A serial native failure would trigger further localization.
The weakest evidence was assuming32 saved trial replays covered the full
procedure's shapes, conditional branches and compilation history. The failed
experiment is preserved, while the implementation restores the previously
checked serial route. The driver for independent confirmation must preserve all
planned slots and separate resource/algorithmic/harness dispositions; a
development-price loop or analysis report alone is not that execution evidence.
