# October 31 terminal result: completed, convergence not established

Attempt-04 finished cleanly at 11:46:54 Asia/Shanghai on October 31, 2026,
with 10,000 retained draws per chain, four chains, 40,000 total. The supervisor
returned zero without a termination request or hard kill. The stop is the
predeclared 10,000-draw maximum, not forecast affordability, ESS or budget
exhaustion. All cumulative saved tensors and the final checkpoint were verified;
compute charges are reconciled once. Evidence is in
artifacts/q20-walnutpie-estimation-20261029/attempt-04-terminal-verification.json.

Maximum R-hat is 1.0816781 versus the required 1.01. It was 1.0723291 at
8,192 draws; these descriptive diagnostics do not show steady convergence.
The maximum mean MCSE/SD among the first twelve monitored quantities is
0.1542893 versus 0.05; negative-region-probability MCSE is 0.0469229 versus
0.025. Mixing and separate precision screens fail. ESS remains diagnostic
only. No posterior estimate is promoted.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve inconclusive completed estimate | R-hat and MCSE fail | No recorded terminal source/state failure; promotion screens fail | Persistent chain differences and correlated sampling | Inspect chain-level mixing before choosing further continuation or repair | Convergence, exhaustive coverage, or rejection of WalnutPie as a method |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Mixing/precision promotion screens fail; execution completes cleanly |
| Statistically supported ranking | None; single-method estimation |
| Descriptive-only differences | Maximum R-hat 1.1721 at 4,096, 1.0723 at 8,192, 1.0817 at 10,000 |
| Default-readiness | Not established |
| Next evidence needed | Chain-specific mixing and precision investigation; an explicit plan for any longer run |

Strongest alternative explanation: finite-run mixing is insufficient rather
than a demonstrated kernel defect. The saved results do not distinguish these
explanations. A checked implementation discrepancy or additional valid
convergence evidence would change the next decision. The current numerical
run is complete; it is not automatically extended beyond its declared maximum.

## Historical execution records

# October 30 continuation: forecast repaired, attempt-04 running

The owner supplied 48 additional GPU-device hours. Attempt-04 launched on
GPU 2 at 17:04:55 Asia/Shanghai, October 30, from 4,304 retained draws per
chain. All 22 restored sampler tensors are byte-identical to attempt-03, and
the complete draw/diagnostic history is preserved. There was no retuning.
All 729 source checks, memory growth and adapter parity passed.

The forecast-based early stop is removed. Forecasts remain advisory; the
actual 42-hour supervisor cap and CPU limits are enforced. The cap ends
November 1 at 11:04:55 Asia/Shanghai and fits the existing CPU allocation.
ESS remains diagnostic only. R-hat and separate MCSE criteria are unchanged.
The next planned cumulative assessment is 8,192 draws per chain, then at
most 10,000. The last completed 4,096-draw assessment is still inconclusive.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Resume saved frozen chains | Engineering recovery checks pass; scientific screen still fails at 4,096 | No startup source/state/memory failure | Persistent chain differences | Continue to full-stream 8,192 assessment within actual budget | Posterior convergence or method rejection |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Startup engineering checks pass; previous mixing/precision screen fails |
| Statistically supported ranking | None; single-method estimation |
| Descriptive-only differences | No new assessment yet |
| Default-readiness | Not established |
| Next evidence needed | Full-stream cumulative mixing/precision assessment |

60 focused CPU regressions passed. Exact command, balances and PIDs are in
artifacts/q20-walnutpie-estimation-20261029/attempt-04-launch.json.
State verification: attempt-04-resume-verification.json in the same directory.
Review: bayesfilter-q20-walnutpie-forecast-continuation-2026-10-30.md.

## Historical result records

# October 30 terminal update: stopped, convergence not established

Attempt-03 exited cleanly at about 08:28 Asia/Shanghai on October 30.
Neither supervisor nor worker remains running. It preserved 4,304 retained
draws per chain (17,216 total), and the final state checkpoint passed all
22 tensor checksums. The last scheduled assessment used 4,096 draws per
chain: maximum R-hat 1.1721164, with R-hat and separate MCSE screens failing.
ESS is diagnostic only and did not cause this stop.

The automatic stop was budget_insufficient_for_next_assessment. Its forecast
for reaching 8,192 draws per chain was 29.535 hours including the 25% reserve,
versus 26.637 hours left on the run clock at that time. This was a predictive
affordability stop, not a crash or exhaustion of the allocated GPU hours.
The forecast uses the slowest of the recent four chunks and is uncertain.
No posterior estimate is promoted. State and completed draws remain resumable.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve incomplete estimate | R-hat and MCSE fail at 4,096 | No recorded target/state failure; runtime forecast stops continuation | Persistent chain differences and variable cost | Review chain diagnostics and forecast before continuation | Convergence or rejection of WalnutPie as a method |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Convergence/precision screening fails; process exits cleanly |
| Statistically supported ranking | None; single-method estimation |
| Descriptive-only differences | R-hat decreased from 1.202 to 1.172 |
| Default-readiness | Not established |
| Next evidence needed | Chain-level investigation and cumulative assessment after justified continuation |

Machine-readable terminal evidence:
artifacts/q20-walnutpie-estimation-20261029/ess-correction-20261030/terminal-status-20261030.json.

## Historical records (running statements below are superseded)

# October 30 correction: ESS diagnostic-only; attempt-03 running

The owner clarified that ESS is diagnostic, not a minimum threshold. The
agent-added bulk/tail ESS >=400 screen was wrong for the requested procedure
and has been removed. R-hat <=1.01 and the separate existing MCSE conditions
remain unchanged. The new assessment policy is
q20_walnutpie_rhat_mcse_ess_diagnostic_v1.

56 focused tests passed. Attempt-02 stopped cleanly without a hard kill.
All 641 referenced tensors passed checksum verification. Attempt-03 resumed
from 2,432 retained draws per chain, preserving the exact sampler state, RNG,
frozen settings and all committed draw/diagnostic history. First resumed
state tensors are byte-identical to the predecessor checkpoint. No retuning
or restart from the initial positions occurred.

Launched October 30 at 03:01:10 Asia/Shanghai on GPU 2. Supervisor 1015304,
worker 1015305. Cap 115677 seconds, within the original deadline and budget.
All 729 source checks, memory growth and adapter parity passed again.
Live status: artifacts/q20-walnutpie-estimation-20261029/attempt-03-supervisor.json.
Correction and review: bayesfilter-q20-walnutpie-ess-diagnostic-correction-2026-10-30.md.
Evidence: artifacts/q20-walnutpie-estimation-20261029/ess-correction-20261030/.

At the latest completed 2,048-draw assessment, R-hat 1.2019 still fails
independently of ESS, and the unchanged MCSE checks fail too. Existing
reports retain their original booleans; the correction is recorded separately.
No posterior convergence, statistical ranking or default-readiness is inferred.
The next cumulative assessment remains 4,096 draws per chain.

## Historical records (superseded live status below)

# Repaired WalnutPie attempt-02 launched

Updated October 29, 2026: the owner authorized repair, review and launch.
The fresh attempt began at 11:10:08 UTC on GPU 2 with four batched chains,
identity mass and author macro-quality target 0.80. The master has a 40-hour
hard cap within the existing study budget. Attempt-01 remains preserved.

Master repairs remove stale implicit device/time defaults, check study and
shared balances before spawning, prevent duplicate supervisors, and preserve
status/accounting when a child exits immediately or during signalling.
The focused regression suite passed 38 tests. Engineering evidence and
source hashes are in artifacts/q20-walnutpie-estimation-20261029/master-relaunch-review.json.

Startup verified all 729 frozen source files, memory growth and the intended
controller/configuration. Adapter and observer parity passed, including exact draw/state parity. The
initial checkpoint is durable and checksum verified; adaptation is running.

| Decision | Primary criterion | Veto checks | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Launch authorized fresh attempt | Engineering tests pass | Source and memory checks pass | Runtime and posterior mixing | Complete startup parity and adaptation | Posterior convergence |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | No startup veto observed; numerical startup checks still running |
| Statistically supported ranking | None; this is a single-method estimation run |
| Descriptive-only differences | No repaired-run posterior result yet |
| Default-readiness | Not established or requested |
| Next evidence needed | Parity, adaptation handoff and cumulative retained diagnostics |

## Earlier attempt and repair record

The historical stopped/running entries below refer to attempt-01 and the
completed repair interval, not the currently running attempt-02.

# Fixed-mass WalnutPie q20 estimation: execution record

CURRENT STATUS: stopped at the owner's request on October 29. The controller
is repaired and engineering-validated; the scientific campaign has not been
restarted. Read bayesfilter-q20-walnutpie-controller-repair-results-2026-10-29.md
for the current verdict. All following launch/audit entries describe the old
attempt chronologically and are not current running instructions.


Tuning-target correction, October 29: the pinned author default is 0.80 for
base-macro exp(-abs(deltaH)); this campaign overrides it to 0.70 and adds its
own 0.65--0.75 qualification rule. These are not author tuning requirements.
See bayesfilter-q20-walnutpie-tuning-target-audit-2026-10-29.md. The core uses
the author statistic, but the complete controller is not an author-procedure
replication. The audit has not changed the running attempt.

October 29 sample-retention review: the fixed 128/128/2,000 exclusions lack
target-specific justification. All completed draw chunks are archived. See
bayesfilter-q20-walnutpie-sample-retention-audit-2026-10-29.md for the
mathematical distinction between adaptation, initialization bias and unnecessary
exclusion. This review has not changed the live controller. Its latest saved
pricing also exceeds the remaining wall cap; the original 32-hour planning
estimate must not be reported as current.

The reviewed campaign has launched, not completed. The frozen target source
closure (729 files) and GPU memory-growth policy were verified at startup.
Nine focused CPU tests passed; the phase-boundary checkpoint regression also
passed after the final recovery repair. These are engineering checks, not
evidence of posterior convergence.

Plan: bayesfilter-q20-walnutpie-estimation-2026-10-29.md.
Run: docs/plans/artifacts/q20-walnutpie-estimation-20261029/attempt-01/.
Supervisor status and log sit beside that directory. The machine-readable
controller, progress, tuning assessments and retained assessments are the
current execution authority. This note must not be read as a terminal result.

Settings: frozen step-3584 two-chart NeuTra, unchanged rectangular QR SR-UKF
approximate posterior, four chains and four parameters, FP64/TF32-off, packed
GPU/XLA execution, supplied identity mass without adaptation. The deterministic
WalnutPie source performs step and microstep adaptation. The local tuning target
is 0.70 for base-macro exp(-abs(deltaH)), with [0.65,0.75] qualification and
fresh fixed-kernel verification; it is not endpoint MH acceptance.

The controller automatically proceeds through adaptive preparation, frozen
measurement, fresh verification, discarded fixed-kernel equilibration and
cumulative estimation. No earlier HMC or tuning draws enter the estimate.
It saves complete state after phase decisions. Every retained assessment
includes latent coordinates, physical categorical draws, conditional physical
means and conditional negative-region probability. Weighted chart summaries
preserve both sign regions instead of reporting only a misleading pooled mean.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Execute reviewed bounded study | Engineering checks pass | No failed launch prerequisite observed | Actual tuning, equilibration and retained precision pending | Monitor first numerical checkpoints, then automatic reviewed ladder | Successful tuning or posterior estimation |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Engineering checks passed; runtime checks continue |
| Statistically supported ranking | None; no comparison campaign |
| Descriptive-only differences | Prior placement and continuing-chunk timings |
| Default readiness | No default change |
| Next evidence needed | Frozen verification, modern R-hat/ESS/MCSE, mode-specific estimates and reference review |

## Review limitations

The acceptance intervals are working block-MCSE intervals with block-size
sensitivity, not calibrated sequential coverage. Passing them cannot establish
mixing. Both trained charts represent known regions; their existence does not
prove all posterior modes were found. The target is the declared UKF
approximation, not an exact state-space likelihood. Those limitations remain
even if the computational estimation screens pass.


## Actual q20 startup qualification

The worker passed value/score parity with maximum errors exactly zero and
passed row-permutation isolation. The actual q20 GPU observer test matched the
ordinary sampler trajectory exactly (maximum draw difference zero), with
identical work/RNG/state checks; counted base macros were [6,9,8,14]. Those
parity draws were discarded. Supplied identity mass passed initialization;
the upstream step search returned approximately 0.6835451778053224 for all
four chains. This remains a starting step, not a verified tuned result.

Adaptive preparation began after 364.59 seconds of source verification,
compilation and qualification. The supervisor remains active. No retained
posterior estimates or convergence claims are available at this stage.

## First resumed chunk verified

Attempt-04 reached 4,320 retained draws per chain (17,280 total).
All 26 tensors in the new checkpoint, draw chunk and decoded quantities
passed checksum checks. All four chains moved during this chunk.
The supervisor remains active, PID 1065302; worker PID 1065304.
Evidence: artifacts/q20-walnutpie-estimation-20261029/attempt-04-first-chunk-verification.json.
This verifies execution and persistence only; the next scientific assessment
is still pending. The last completed convergence screen remains inconclusive.

## October 31: cumulative R-hat report every 500 draws

The owner-requested cumulative table uses 500, 1,000, ..., 10,000 retained draws
per chain across the four saved chains. No suffix or 500-draw moving window
was selected. The 4,096/8,192/10,000 checks match saved assessments exactly.
Report and all individual quantities:
artifacts/q20-walnutpie-estimation-20261029/rhat-every-500-20261031-01/report.md,
rhat.json and rhat-all-quantities.csv. Overall R-hat remains above 1.01 at all
reported checkpoints; these are descriptive diagnostics with no new sampler
run or promotion decision. Recorded CPU cost is 15.448946 core-seconds.
