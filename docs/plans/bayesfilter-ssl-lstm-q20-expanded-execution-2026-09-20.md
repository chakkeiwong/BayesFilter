# q20 execution with the additional CPU and GPU allowance

Last refreshed: September 21, 2026. Training is complete and the existing master
is running `tune-neutra-beta1`. The owner has extended the calendar deadline to
September 25 at 18:00 Asia/Shanghai. The
[deadline continuation plan](bayesfilter-ssl-lstm-q20-deadline-continuation-2026-09-21.md)
supersedes the eight-hour stop as the final tuning allocation; the current
attempt finishes under its original bounds before checkpointed continuation.
The original observations below remain historical records of their timestamps.
The [active operations plan](bayesfilter-q20-master-operations-repair-2026-09-21.md)
now supplies bounded checkpoint recovery, phase decisions and fixed approved
`status`/`ensure` commands. The original numerical worker remains active.

The owner authorized another 48 CPU hours and 48 GPU hours and requested that
the executable master run. Continue the existing q20/T30, four-parameter,
float64 UKF-approximate posterior estimation. Plain NeuTra HMC runs first; a
completed candidate failure may lead to the tempered NeuTra ensemble. Preserve
the current training trial criteria, identity latent mass, complete public
tuning cohort, independent verification, sequential posterior checks and
independent reference assessment. The scientific evidence contract and numerical
provenance are those in the estimation-reset and staged-budget repair plans.

## Budget and realistic scope

The settled predecessor has 77,017.92183943938 campaign seconds (21.3939 hours),
including 1,258.3540581594789 diagnostic seconds. Add 172,800 GPU seconds to
give 249,817.92183943938 campaign seconds (69.3939 hours). The separate 172,800
CPU seconds are reserved for CPU-only diagnostics/repairs and are not added to
the GPU campaign ceiling. Use the preceding campaign's wall-time convention:
the current master runs one GPU worker at a time and conservatively charges its
whole supervised lifetime, including host setup and analysis, to that allowance.
No three-GPU speedup or 48-hour elapsed allowance on each GPU is assumed.

Allocate at most another 7,200 seconds of the new GPU allowance to qualification
and pricing: this is the existing six diagnostic attempts times the existing
1,200-second ceiling. The diagnostic sublimit becomes 8,458.354058159479 seconds;
it remains included in campaign spending. No timing or accuracy threshold is
changed. A small external readiness probe is charged separately before launch.
Dedicated CPU work must record actual elapsed time and concurrency against its
separate reserve; no such long CPU job is launched here.

Saved measurements give about 4.86 hours for the training floor and a 9.71-hour
initial training allocation. Tuning is capped at eight hours per scope. At that
training cap and tuning cap, roughly 48.26 hours remain for sampling after the
first reference assessment, posterior analysis and shared two-hour repair hold
(before fresh startup/qualification/pricing). The earliest posterior check costs
about 3.9 hours at measured L=3 or 24 hours at measured L=25, before the engineering
safety reserve. This is sufficient to attempt the whole plain-NeuTra procedure
with meaningful sampling time. It does not guarantee convergence, successful
training/tuning, all maximum chain lengths, or a fully exhausted ensemble fallback.

## Execution, review and evidence contract

Use the already tested isolated source
`/tmp/BayesFilter-q20-staged-budget-20260920-r2`; all 490 recorded source files
still match their saved hashes. The preceding repair has 80 distinct passing
latest checks, including a supervised end-to-end known-target smoke and partial
checkpoint replay. No numerical source changes require another test ladder.
Keep the protocol byte-for-byte unchanged so the saved eight-map checkpoint
retains its config identity and 4,096 historical optimizer updates. Current-scope
credit remains unchanged. Import it into a fresh versioned campaign root.

Skeptical pre-execution audit: the stale numbers inside the saved protocol are
not active spending authority; the explicit expanded allowance file is. CPU
hours cannot be spent as GPU hours. No old all-cap reservation, unqualified-map
timing, short-chain acceptance, or CPU fixture result can establish convergence.
The complete tuning cohort may remain inconclusive under its eight-hour cap,
and longer trajectories can still exhaust sampling funds. Those outcomes pause
or reject only under the existing contract. Target/source corruption and missing
required evidence remain continuation vetoes. Budget pauses retain checkpoints
and never cause scientific rejection or promotion. This audit passes for launch.

The independent integration reference is the posterior comparator, not another
method. Promotion requires one permitted method to pass sequential numerical
health, movement/energy, R-hat, ESS, MCSE, starting-group and qualified-reference
checks. Training loss and measured runtimes are explanatory or trial nomination
only. Do not infer exact nonlinear-likelihood correctness, global mode coverage,
perfect whitening, or method superiority. The first valid estimate ends the
campaign. Localized infrastructure repair/retry is authorized within the same
scientific contract and remaining budget; source/checkpoint identity and attempt
limits remain enforced and every attempt consumes the balance.

Launch the existing master through a transient user service, with a whole-service
deadline no greater than the funded campaign time and its existing per-worker
timeouts. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before import; every GPU worker
verifies growth before initialization and selects a fresh idle device among all
three GPUs. Preserve the measured environment: tfgpu Python, two TF intra/inter-op
threads, one OpenBLAS thread, and the existing GPU/XLA path. GPU 1 was idle in the
trusted prelaunch inventory; workers select again on entry.

Output root:
`docs/plans/artifacts/ssl-lstm-q20-expanded-execution-2026-09-20/`.
Preserve `allowance.json`, `cpu-allowance.json`, unchanged `protocol.json`, exact
launch command/environment/source record, `console.log`, and `campaign-01/`
containing the master's manifests, charged attempts, checkpoints and result.
The master owns stage progression after launch. Observe current qualification
and worker health, record the live service/PID, and leave the service running
under its deadlines when returning a launch-status report. That report is not a
terminal scientific result. The next substantive result note must state the
decision, vetoes, uncertainty, remaining allowance and justified continuation.

## Launch observation

The trusted launch succeeded at September 20, 17:22:12 CST. Transient service
`bayesfilter-q20-expanded-20260920-0921.service` owns coordinator PID 93403.
The readiness stage completed and the master entered beta-one qualification on
host GPU 1. TensorFlow 2.20.0 verified memory growth before logical device
initialization; XLA compilation was observed. All eight saved maps imported.
The master is running under its 69.39-hour total allowance and worker deadlines.
Live attempt accounting is in `campaign-01/campaign.json`; `live-status.json`
records a timestamped observation, including unsettled in-flight wall time.
No posterior conclusion follows from this launch observation.

The subsequent startup observation confirmed beta-one qualification completed
and the master advanced automatically to `price-neutra`, still on GPU 1 with
verified memory growth. The expanded service remains active. A sandbox-only
service-status read was denied access to the user bus; the trusted read succeeded.
This affected only observation and did not interrupt the campaign.

## September 21 master refresh

At 00:14 Asia/Shanghai, the campaign remains in `running:tune-neutra-beta1`.
The latest service observation confirms the original coordinator is active;
the worker manifest records GPU 1, XLA, TensorFlow 2.20.0 and verified memory
growth. All 490 discovered execution-source files match the campaign snapshot,
and all 19 workspace `q20_*.py` modules match the running copy. The training
receipt, final cohort checkpoint and selected map export match their recorded
checksums. No execution-source change or restart is indicated. Independent
workspace changes to shared HMC code are not incorporated into this live run.

Training used 17,429.35 seconds (4.84 hours). All 12 direct maps reached 512
current-scope updates and passed the existing learning and map-reliability
checks for an HMC trial. The fixed declared order selected
`direct-w16-lr0.0005-r0`. These checks do not establish complete whitening or
posterior convergence. Historical updates remain separately recorded.

Tuning started September 20 at 22:26:48. About 1.79 of its eight allocated
hours have elapsed. The first five pilots, at L = 3, 5, 9, 13 and 18 with
epsilon = 0.01, recorded mean acceptance between 0.99763 and 0.99866. Each
returned `repair_step_higher`, with valid recorded evidence and no recorded
hard-health, promotion-veto or engineering-invalidity reasons. The existing
1.5-fold repair rule has queued epsilon = 0.015 children. The L = 25 initial
pilot is still incomplete; no kernel has fresh verification yet. These
observations trigger step-size repair, not rejection of NeuTra or a conclusion
about posterior mixing. No candidate ranking is supported or used.

The live checkpoint's `partial_budget` and current work's `interrupted` fields
are resumable snapshot states, not evidence that the running worker timed out.
Use service state, worker completion and supervisor accounting together.

The measured allowance after the readiness-probe charge is 249,816.92 seconds
(69.39 hours). Settled work is 18,272.59 seconds (5.08 hours); including observed
in-flight tuning leaves about **62.53 campaign hours**. The included diagnostic
sublimit has about 2.12 hours left. The separate 48 CPU-hour reserve has no
dedicated campaign workload charged to it. These are timestamped observations;
the coordinator settles actual worker lifetimes.

The next executable actions are already in the master:

1. Finish the declared tuning cohort, bounded repairs and fresh verification.
   Tuning must complete before a verified member can be selected by the fixed
   lexicographic rule. Identity mass remains fixed in transport coordinates.
2. Freeze that member and run four batched chains through sequential warm-up
   and retained sampling. Existing health, movement, energy, R-hat, ESS and
   precision requirements remain in force; warm-up is excluded from estimates.
3. After the sequential posterior checks pass, run the independent integration
   reference and final posterior assessment. The first valid estimate ends the
   campaign. A completed candidate failure may advance to the tempered ensemble;
   a budget pause or infrastructure interruption does not authorize that switch.
4. If tuning remains incomplete at its cumulative cap, preserve the checkpoint
   and inspect its evidence and allocation. The supervisor deadline is about
   **September 21, 06:26:48 Asia/Shanghai**. The master does not automatically
   renew this scope merely because total campaign funds remain.

Refresh audit: the running source, receipts, budget and permitted continuation
agree. The initial step-size repair is already implemented. Sampling remains
conditional on a complete tuning result with verified members; pilot acceptance
and training progress are not substituted for posterior evidence. No new
numerical choice, comparator, compute allowance or promotion rule is introduced
by this documentation refresh.

| Decision item | Current evidence |
| --- | --- |
| Decision | Continue the existing supervised tuning run. |
| Primary criterion | Stable posterior estimate pending; sampling has not started. |
| Veto status | No recorded hard veto in the five completed pilots; initial epsilon requests upward repair. |
| Main uncertainty | Whether tuning finds and verifies a kernel within its eight-hour cap, and whether subsequent posterior checks pass. |
| Next justified action | Complete tuning, then sequential sampling and qualified-reference assessment. |
| Not concluded | Complete whitening, convergence, method superiority or production readiness. |

The versioned observation is
[`master-refresh-20260921-001412.json`](artifacts/ssl-lstm-q20-expanded-execution-2026-09-20/master-refresh-20260921-001412.json).
`live-status.json` was also refreshed. This is a status audit using saved
artifacts and source checks, not a new numerical experiment; no test rerun is
needed for these documentation changes.
