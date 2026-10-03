# q20 master refresh and continuation

September 18 successor: the owner requested continuation. See
`bayesfilter-ssl-lstm-q20-training-continuation-2026-09-18.md` and its artifact
directory. Eight actual GPU gradient/Adam checks passed; a budgeted extension
to 512 updates is running. That successor campaign owns all subsequent charges;
do not spend this predecessor's settled allowance again.

Status of this completed phase: `TRAINING_CALIBRATION_COMPLETE`; finished 2026-09-17 04:53:37
Asia/Shanghai. All-device selector repaired; 17 focused tests passed. The r3
master service has exited successfully; no q20 worker remains running.
The prior source's GPU diagnostic passed; its capacity pause is preserved below.
Continuation remains authorized by the owner's September 17 request.
This is the q20 campaign, not the concurrent general HMC validation campaign.

## Question, current gates and evidence

Can the executable master resume the repaired q20 work with fresh GPU capacity
checks, the settled allowance and preserved failure/accounting evidence?
The current HMC implementation has passed its focused CPU/XLA consumer checks;
q20 GPU qualification and the paired accepted-status-reuse comparison remain
pending. Full training and posterior readiness remain unestablished. This phase
continues the owner's explicitly requested q20 engineering/diagnostic work;
it does not issue a plain-NeuTra paper-replication or enhancement-gate pass.

The engineering comparator is the current master and the existing standalone
status-reuse diagnostic. Acceptance requires the same diagnostic to execute
under master deadlines/accounting, busy capacity to produce a resumable
WAITING_FOR_GPU result before TensorFlow import, and failed workers to retain
their actual failure status. No completed training/checkpoint receipt may be
created from a capacity failure. Completed stages must replay without charge.

The numerical comparison is unchanged from the September 16 efficiency roadmap:
same public batched runner, with versus without the status wrapper at tracing;
q20/T30 strict FP64 target; four chains; L=3, epsilon .01; two transitions and
three paired calls (first compile/execute plus two warm calls). Qualification
uses the existing four-transition check at beta .5 and 1. Exact discrete parity,
floating rtol 1e-9/atol 1e-10, finite/proposal health and the existing XLA checks
are hard candidate vetoes. Timing differences are descriptive only. Resource
contention, corrupted evidence, source drift and budget exhaustion stop the
affected run. Failed optimization candidates trigger repair, not rejection of
NeuTra. No posterior, convergence, ranking or full-training-affordability claim
can follow from this diagnostic.

## Repair and execution

1. Add actual-launch GPU readiness to numerical workers, before TensorFlow
   import; preserve the installed probe receipt and recheck compute contention.
   A coordinator check alone cannot reserve a device across scheduling delays.
2. Preserve worker resource-failure receipts through the supervisor. Charge
   their time but do not consume numerical retry counts. Return to the user with
   resumable state on busy capacity; no polling loop or displacement of jobs.
3. Expose the existing status-reuse comparison as the master's `diagnose` stage,
   using the same source snapshot, process deadline, artifact checks and budget
   ledger as other stages. Accept the master's actual protocol in the script.
4. Run focused resource/accounting regressions and the actual tiny CPU master
   calibration/resume test. Copy the tested source to an isolated checkout for
   numerical execution; preserve all concurrent unrelated edits in main.
5. Execute `diagnose` in a fresh campaign root with the latest settled allowance.
   If it passes and capacity remains available, execute `calibrate` in the same
   campaign, using current measured prices for admission. Calibration completes
   only its already declared first-root/first-rung inventory. The full cohort
   remains subject to its separate complete forecast and evidence requirements.

Artifacts: `docs/plans/artifacts/ssl-lstm-q20-master-refresh-2026-09-17/`.
Starting allowance:
`artifacts/ssl-lstm-q20-status-reuse-2026-09-16/settled-allowance.json`,
120851.41110222784 campaign seconds including 39874.95812729796 diagnostic
seconds. This is not a renewed allocation. Reserve at most 1800 diagnostic
seconds for refresh/checks/comparison: 480 for tests, 1200 for the GPU attempt
including readiness/termination, and 120 for bounded setup/accounting. Charge
actual supervised commands and explicitly settle the setup envelope. A
subsequent calibration uses its current quote within the remaining campaign
balance, never a new allowance. Stop for unavailable capacity and preserve the
next exact command. Every launch gets a new attempt directory.

Commands, from the isolated source checkout:

```text
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_q20_gpu_runtime.py tests/test_q20_campaign_runtime.py tests/test_q20_master_program.py::test_calibration_master_prices_only_training_and_preserves_partial_cohort
env TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py diagnose --config <protocol.json> --budget-record <execution-allowance.json> --output-dir <campaign>
```

GPU work requires trusted execution. CPU tests intentionally hide GPU devices.
The master records commands, hashes, Git commit, timing and receipts. No sampler
or training mathematics changes, so MathDevMCP/literature reinspection would not
answer the scheduler question; the existing mathematical audit remains in the
efficiency roadmap.

## Numerical provenance and skeptical audit

The target, sampler diagnostic counts/tolerances and training protocol are
inherited hypotheses/checks from the linked roadmap and parameter ledger, not
newly calibrated scientific defaults. The 60-second probe timeout and
480/1200/120-second allocations are convenience engineering caps chosen to fit
the existing allowance; they are not runtime predictions. Resource observations
must be made inside the worker after approval delay. Timing is unusable if
another GPU process appears. Missing timings are not zero.

Audit passed after identifying three flaws in a direct relaunch: stale capacity
selection, loss of structured nonzero-exit receipts, and a pending comparison
outside master accounting. This plan repairs those first. The baseline remains
batched, same seeds/target/health checks are preserved, no smoke is promoted,
and partial pricing cannot authorize the full campaign. The master still has
serial stages; this repair must not be described as multi-GPU scheduling or
completion of the remaining performance roadmap.

## Prelaunch result and active command

All 16 focused tests passed in 17.625660 supervised seconds with GPU intentionally
hidden, including the real calibration worker and exact cached resume. The
isolated source did not change. Syntax and whitespace checks passed. The tests
exercise resource receipts, pre-import checks, late contention, nonzero-exit
failure preservation, checkpoint recovery after a capacity wait, cumulative
accounting and repeated master resume without spending numerical retry counts
on unavailable capacity.

The active protocol sets the diagnostic attempt cap to 1200 seconds and one
numerical attempt per stage for this bounded continuation (no automatic rerun
of a timed-out comparison). Six total diagnostic stages allow the explicit
comparison, training pricing, two beta qualifications, complete pricing and one
reserved diagnostic stage; this is an execution inventory limit, not extra
compute. All scientific settings are unchanged. The source is isolated at
`/tmp/BayesFilter-q20-master-refresh-20260917`, detached from Git commit
`d86dadf68ea57772642c6802990f46a3c6a04c30`, with current working-tree sources
copied and hashed before testing.

After the test debit and the declared 120-second setup/accounting envelope,
`execution-allowance.json` gives 120713.78544263585 campaign seconds including
39737.33246770598 diagnostic seconds. The master will debit its actual attempts
from that balance. Actual launch command:

```text
env TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py diagnose --config /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-refresh-2026-09-17/protocol.json --budget-record /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-refresh-2026-09-17/execution-allowance.json --output-dir /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-master-refresh-2026-09-17/campaign
```

## GPU comparison result and calibration continuation

The refreshed master completed `status-reuse` on an idle policy-permitted host
GPU 1 (RTX 4080 SUPER), TensorFlow 2.20.0, TFP 0.25.0, verified memory growth,
FP64, TF32 enabled and XLA. Both beta .5 and beta 1 qualifications passed through
the shared one-step and public batched runner. The paired runs matched exactly:
maximum observed floating difference zero, exact discrete fields, all declared
health checks passed, one trace per graph and no Python callbacks. Source
inventory stayed unchanged. All recorded contention checks were empty.

| Execution | Without accepted-status reuse | With accepted-status reuse |
| --- | ---: | ---: |
| First compile/execute, seconds | 35.608536 | 29.630500 |
| Warm repeat 1, seconds | 16.694739 | 13.634799 |
| Warm repeat 2, seconds | 16.529367 | 13.550502 |

The two warm means were 16.612053 versus 13.592650 seconds, an observed 18.18%
reduction for this exact two-transition/L3 diagnostic. This is descriptive,
without a supported general runtime ranking. It does not extrapolate to L25,
long chains, learned charts, training, or the complete campaign. TensorFlow
reported allocator peak 268822272 bytes; it is not process reservation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the repaired master execution path and retain status reuse | Focused lifecycle tests and actual paired q20 GPU checks passed | No health, parity, resource or source veto | General runtime and downstream posterior behavior | Continue already declared training calibration | Posterior or full-campaign readiness |
| Begin training pricing/calibration | Current measured reservation must fit remaining allowance | Missing price or over-budget quote blocks training | Learning and loss variance remain uncalibrated | Price actual batch/width/beta scopes, run existing partial cohort if funded | Training floor/architecture is scientifically established |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | All declared paired numerical/health/XLA checks passed |
| Statistically supported ranking | None |
| Descriptive-only differences | The three elapsed-time pairs above |
| Default-readiness | Engineering implementation checked; no posterior promotion |
| Next evidence needed | Calibration learning/variance, complete downstream forecast, frozen-map tuning and untouched posterior checks |

The attempt consumed 454.6068283240311 supervised seconds, already charged by
`campaign/campaign.json`. Balance before calibration: 120259.17861431182 campaign
seconds including 39282.72563938195 diagnostic seconds. Preserve this campaign
and its isolated source unchanged; run the same command with mode `calibrate`
and the same config/output directory (no new allowance). The coordinator will
price training under a 1200-second diagnostic cap, then admit only the measured
calibration reserve, bounded by the existing 28800-second arm cap. Calibration
is ordinary campaign work; pricing is diagnostic work included in campaign time.
An independent systemd user service may keep this authorized master alive after
the interactive turn; its workers retain the same external deadlines and budget
accounting. This is process supervision, not extra compute authorization.

Post-run red-team: repeated starts and only two warm replicates limit timing
interpretation. The strict factor/score kernel remains expensive, status is not
fused into the integrator, and independent campaign stages remain serial. These
checks close this bounded comparison; they do not complete the efficiency
roadmap or establish that the full campaign fits its allowance.

## Live execution handoff

At 2026-09-17 02:54 Asia/Shanghai, started
`bayesfilter-q20-calibration-20260917-r1.service`. Its master reached
`running:price-training` and wrote its first actual price row. The worker selected
GPU 1 and verified memory growth before initialization. `continuation.json`
records the exact service command and source root. The service limit is 31260
seconds: two 1200-second readiness/pricing envelopes plus the existing
28800-second calibration arm limit and a 60-second coordinator/teardown reserve.
This outer bound does not replace the tighter per-stage budget checks.

The live [campaign ledger](artifacts/ssl-lstm-q20-master-refresh-2026-09-17/campaign/campaign.json)
is now the budget authority. Do not start a new campaign from the September 16
settled allowance or this phase's prelaunch allowance while this campaign is
active. Settle any live attempt before issuing a successor allowance. Before
training pricing, the diagnostic and readiness workers consumed 457.611505
seconds, leaving 120256.173937 seconds campaign and 39279.720962 seconds
diagnostic, **before the running worker's eventual debit**.

The master prices all required calibration scopes, checks the actual remaining
balance, then launches the existing first-root/first-rung calibration. Completion
requires reviewing its learning/variance result before extending training.
Pricing or calibration failure must be classified from the worker receipt; it
does not establish failure of NeuTra. Full training and the downstream campaign
still need their complete forecast. The code remains an isolated, hashed copy
of the refreshed main working tree; unrelated concurrent work is preserved.

Status command:

```text
systemctl --user status bayesfilter-q20-calibration-20260917-r1.service
```

Master state and stage artifacts are under
`artifacts/ssl-lstm-q20-master-refresh-2026-09-17/campaign/`. The master result
file is the last completed invocation until the active invocation finishes;
use `campaign.json` for live state, not the previous diagnostic's `result.json`.

## Capacity pause observed at 03:00 Asia/Shanghai

Training pricing wrote its four scope measurements, but its terminal capacity
check found another compute process, PID 2369570, on GPU 1. The worker returned
`waiting_for_gpu`; the master preserved the measurements without admitting them
as uncontended pricing and did not start calibration. The service exited
normally. A fresh installed NVIDIA probe at 19:00:19 UTC confirmed
`no_idle_policy_permitted_gpu`. This read-only check is covered by the previously
charged setup/accounting envelope.

The failed capacity attempt consumed 186.554836 seconds. The campaign ledger
now records 644.166341 seconds spent, leaving 120069.619101 campaign seconds
(33.352672 hours), including 39093.166126 diagnostic seconds (10.859213 hours).
No worker remains owned by this campaign. Resume the same `calibrate` command
from the unchanged isolated source when capacity is available; the resource
wait did not consume its numerical retry count. No automatic GPU polling service
was installed.

For the owner's duration question, the earlier uncontended planning estimate
was approximately 1.381 hours raw calibration work and 2.762 reserved hours,
excluding startup/pricing. These remain rough planning figures; the interrupted
current prices do not replace them. GPU waiting has no known duration, and the
complete downstream campaign is still unpriced.

## All-device selection correction (owner request, September 17)

The owner explicitly directed use of the machine's three GPUs rather than
waiting for GPU 1. Inspection found the installed readiness helper hard-codes
only devices 0 and 1. It also treats every NVIDIA compute-list PID as a research
workload. On this machine PID 6866 is `/usr/NX/bin/nxnode.bin` and PID 6247 is
`/usr/libexec/gnome-remote-desktop-daemon`; neither is another research job.
The two transient Python research workers have since exited.

Repair the repository's q20 selector to consider all inventoried GPUs and
separately record those two exact desktop executable paths. Unknown processes
and numerical workers continue to count as competing work. Prefer a device with
no other processes, then lower observed utilization/memory use; this ordering
is a scheduling convenience, not a performance claim. Do not impose GPU 1 or
carry the coordinator's selection past a worker scheduling delay. Workers
select afresh, set visibility before TensorFlow import, and verify the existing
memory-growth policy before numerical work. Desktop coexistence is recorded;
timings on such a device are descriptive of that load, not dedicated-device
benchmarks. Ordinary training can coexist with desktop services. No process is
terminated and no global probe or approval configuration is changed.

Skeptical audit: three visible GPUs do not imply three idle GPUs, small memory
use does not imply no computation, and a desktop process may still affect wall
time. Actual numerical PIDs therefore remain a resource wait, exact desktop
identities are recorded rather than inferred from low utilization, and the
per-stage checks remain. Test selection of GPU 2 when 0/1 have jobs, desktop
coexistence, unknown-process rejection, new contention and worker reselection.
No math, target, training count, dtype, tuner or promotion criterion changes;
previous HMC comparisons remain scoped to their original source. Calibration
uses fresh prices and source receipts, not transferred numerical admission.

Reserve at most 300 seconds for focused CPU tests plus a 60-second
setup/readiness/accounting envelope, charged to the existing balance. After
tests, carry forward the previous campaign's settled remainder into a fresh
source-bound campaign and execute `calibrate`. Its previously declared pricing
and training limits remain. Preserve the previous campaign and isolated source.

All 17 focused CPU tests passed in 17.036909 supervised seconds, including
real calibration/resume and GPU 2 selection with GPUs 0/1 occupied. No source
changed during testing. The replacement source snapshot is
`/tmp/BayesFilter-q20-all-gpu-20260917`; changed files were copied independently
so the previous execution source and receipts remain unchanged. These source
changes affect scheduling only, with no scientific numerical changes.

After the test charge and 60-second setup/readiness/accounting envelope,
`all-gpu-allowance.json` carries 119992.582192 campaign seconds including
39016.129217 diagnostic seconds from the inactive predecessor campaign. The new
[live campaign](artifacts/ssl-lstm-q20-master-refresh-2026-09-17/campaign-all-gpu/campaign.json)
is the current budget authority; do not spend either predecessor balance again.
`all-gpu-protocol.json` preserves the same scientific protocol and time caps.

The replacement master checked all three devices. The transient numerical
workers had exited: GPU 0 carried the NX desktop process, GPU 2 carried the
GNOME desktop process, and GPU 1 had no other processes. It selected GPU 1 for
that reason, not a fixed preference. The actual pricing worker verified memory
growth and reached `running:price-training`. All three devices remain eligible
for fresh selection at future worker starts. A later genuine numerical workload
can still invalidate a timing attempt; the master does not preempt other jobs.

Exact launch is recorded in `all-gpu-continuation.json`. The user service is
`bayesfilter-q20-calibration-20260917-r2.service`, with the same 31260-second
outer supervision bound and tighter stage/budget bounds. It will proceed from
current pricing to calibration only if its measured quote fits the remaining
allowance. This remains a single-worker master; selecting among three GPUs does
not claim parallel execution across three GPUs.

### Same-source retry after the second capacity collision

The all-device r2 worker also encountered a new numerical process (GPU 1 PID
2378514) during pricing. It preserved the four scope prices but did not admit
the timing result or begin calibration. Charge its 183.553241 seconds and the
0.502615-second coordinator probe to the existing ledger. The process had exited
by the next live check; GPUs 0 and 2 carried only the recorded desktop services.

On the owner's next timing query, resumed the same source, config and campaign
as `bayesfilter-q20-calibration-20260917-r3.service`, with no new allowance and no
source changes. `all-gpu-continuation-r2.json` preserves the prior service
command; `all-gpu-continuation.json` records r3. The master reached
`running:price-training` again. Prior attempts are resource failures, not
scientific failures; their costs remain spent. The rough 1.381-hour raw/
2.762-hour reserved calibration estimate remains provisional until this
uncontended pricing attempt completes. The full posterior campaign is excluded
from that duration.

The r3 pricing attempt completed and the master started calibration at
03:26:40 Asia/Shanghai. Its current measured forecast is 1.411053 hours raw
calibration work (about 85 minutes), with 2.830320 hours reserved including
worker overhead. The corresponding planning finish is about 04:51, with the
reservation extending to about 06:16. These are work-cost estimates, not
statistical runtime guarantees. The live ledger and `calibration-eta.json`
record the start and estimate; the full posterior campaign is still excluded.

## Terminal calibration review

Calibration completed in 5216.885194 supervised seconds (86.948 minutes),
within its 10189.153544-second reservation and close to the 84.663-minute raw
forecast. Eight first-root configurations completed 128 updates each, batch 32:
widths 16/32, learning rates .0005/.001, direct beta 1 and continuation beta .5.
All 1024 updates record valid finite targets, and all eight exported-map
numerical parity checks passed. GPU 1 used verified memory growth, TensorFlow
2.20.0, FP64 with TF32 enabled and XLA. The terminal capacity check recorded no
other compute or desktop processes on that device. Source hashes are unchanged;
all 69 stage artifact hashes and the cohort checkpoint's internal hash pass.

Every assessment reports `continue_training`, `learning_observed=true`, and
`development_eligible=false`. The paired mean loss changes after training are
negative for every configuration; first-rung comparisons use 768 rows and their
descriptive intervals exclude zero. These are single-root learning diagnostics,
not an architecture/LR ranking, posterior check or transport-whitening result.
The cohort is incomplete, and the 128-update maps must not be promoted.

Two parameter hypotheses need examination before extending the protocol:

- All 1024 optimizer updates invoked the inherited gradient-norm cap of 10.
  Per-configuration median unclipped norms range from 156.67 to 609.72.
  Learning occurred, so this does not prove clipping is wrong; it does establish
  that clipping shapes every update rather than only exceptional gradients.
  Objective scaling and the interaction with Adam need a bounded comparison
  before asserting the cap is appropriately calibrated. Do not remove it from
  the running algorithm on this evidence alone.
- The descriptive loss-change interval half-widths are 8.62–36.31, versus the
  inherited .02 fine-precision threshold. Constant-variance row extrapolations
  give 143 million–2.53 billion rows; these are explanatory warnings, not a
  sampling recommendation. All current continue-training decisions resolved
  with the first bank. Later incremental differences can have different
  variance; neither current baseline variance nor loss scale calibrates the
  eventual plateau criterion or posterior accuracy requirement.

The first-bank full-cohort floor is forecast at 20.327 raw hours / 40.655 reserved
hours from a fresh start, before downstream HMC/reference work. Credit for the
completed calibration and future precision decisions must be included when
repricing remaining work. The full campaign remains unpriced and cannot be
admitted using this partial forecast. Do not launch the full grid or enlarge
validation banks merely because this calibration completed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Calibration complete | Eight declared scopes, 128 updates each, valid saved results | No recorded numerical, map-parity, source or artifact failure | One training root; no downstream sampling | Preserve checkpoint for exact continuation | Completed training or posterior validity |
| Keep all eight configurations as development candidates | Each shows descriptive learning and requests continuation | No configuration rejected by this run | Cross-root variability and later learning behavior | Inspect clipping/precision hypotheses and price further work | Any candidate is superior |
| Review cost and numerical choices before extending | Current calibration answered learning/variance/cost question | Full campaign lacks a complete affordable forecast | Remaining floor, validation demand, HMC/reference costs | Reprice with completed-work credit; conduct smallest planned discriminating repair | More budget, weaker checks, or millions of validation rows are authorized |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Update validity and map numerical checks passed; no posterior screen performed |
| Statistically supported ranking | None; all eight remain viable under this diagnostic |
| Descriptive-only differences | Paired loss changes, gradient norms, and wall times |
| Default-readiness | Not established; no development/posterior-qualified map |
| Next evidence needed | Clipping/scaling assessment, decision-specific precision, multi-root continuation, frozen-map tuning and posterior/reference checks |

Post-run red-team: reduction in reverse-KL training loss can coexist with missed
modes and poor HMC mixing. First-rung baseline loss variance need not describe
near-plateau incremental variance. Clipping frequency alone is not evidence of
harm, and one successful resource check does not bound future contention.
These limitations block stronger conclusions, not the research direction.

The concise result is `artifacts/ssl-lstm-q20-master-refresh-2026-09-17/calibration-review.json`.
All attempts are settled. Remaining allowance in that directory's
`settled-allowance.json` is 114407.593864 campaign seconds (31.779887 hours),
including 38648.026082 diagnostic seconds (10.735563 hours). No allowance was
renewed. The successful result, service disposition and next action are recorded
in `all-gpu-continuation.json`.
