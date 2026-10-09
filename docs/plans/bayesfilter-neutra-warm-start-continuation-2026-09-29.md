# Authorized continuation of the simpler-model NeuTra campaign

Execution closeout: the declared queue completed at 18:39:16 Asia/Shanghai on
2026-09-29. All 111 base jobs, 15 capped repairs and six capacity/duration
controls completed. The terminal scientific review, failed candidates and
remaining budget are in
`bayesfilter-neutra-warm-start-continuation-results-2026-09-29.md`; exact
aggregation is `campaign-r1/terminal-review-r1.json` under the artifact root.
No campaign process remains active. Completion of this bounded plan does not
establish reliable training on every fixture.

The owner granted "48 hours more" and requested continued execution. This
amends the existing master plan and resumes `campaign-r1`; it does not reset
past costs, numerical failures, or the uncertain orphan charge. Interpret the
increment conservatively as 48 total compute-hours, allocated as 32 CPU
core-hours and 16 GPU process-hours. The approximately 2:1 split follows the
observed campaign cost ratio. The cumulative ceilings become 122,400 CPU
core-seconds and 64,800 GPU process-seconds. CPU and GPU charges remain separate;
neither resource is double-authorized for an additional 48 hours.

The inherited evidence contract, targets, reference banks, canonical IAF,
calibration, confirmation seeds, numerical/posterior criteria and nonclaims are
in `bayesfilter-neutra-warm-start-master-2026-09-29.md`. This allocation funds
the previously planned repairs and remaining three-seed simpler-target study.
It does not extend the scientific target to q20 or change the IAF architecture.

## Research intent and execution order

The main question remains whether approximate forward initialization yields a
useful transport whose coverage survives classic RKL and supports accurate,
precise frozen-map HMC. Plain RKL is the baseline; exact-example forward fitting
is the diagnostic control. A low score residual is explanatory only. Missing
reference basins, numerical invalidity and failed posterior checks block
candidate promotion. They trigger the specified repair rather than rejecting
all warm-start mechanisms. Invalid target/reference evidence, unrecoverable
artifacts or resource exhaustion stop the affected continuation.

1. Resume the saved ordinary-mixture Gabrié/oracle pre-RKL maps, test the
   declared lower-learning-rate/doubled-batch repair, and qualify viable maps.
2. Qualify the existing ordinary-mixture waste-free candidate and re-evaluate
   warped-oracle qualification with the already corrected binary-region test.
3. Execute the instrumented two-stage/full-inner-loop AFT diagnostic before
   repeating expensive AFT or CRAFT jobs. Repair only an evidenced implementation
   or performance defect; preserve source update/weight ordering and perform
   focused equivalence checks before renewed method-quality runs.
4. Complete discovered-mode initialization and CRAFT controls, then remaining
   target/arm/seed jobs. Expected candidate failures continue to their planned
   repairs; only the declared continuation vetoes stop dependent work.
5. Refresh the result and inference tables, pending work, measured costs and
   remaining allocation. No descriptive method ranking becomes a scientific
   ranking without the predeclared replication/uncertainty evidence.

Every attempt uses a new directory under the existing campaign. Existing
attempt caps, per-job limits, trusted GPU access and memory-growth requirements
remain active. Timed-out scientific work is incomplete evidence. Per-job limits
may be revised from measured full-loop pricing within this allocation, with the
reason recorded; a larger global allowance alone does not justify a slow loop.

## Skeptical review before resumption

The plan survives review with these explicit qualifications:

- Earlier successes are one-seed results on known analytic fixtures. Several
  numerical settings were nominated with exact/reference data; this is not an
  unknown-posterior training protocol or evidence transferable to q20.
- The RKL repair isolates optimizer schedule/noise using the exact saved warm
  map. It is not another independent warm-start replicate and may still fail.
- The original AFT price diagnostic did not exercise all updates. Revision 2
  does, and records each operation and trace count. Its explicit synchronization
  changes timing overhead; use it to localize cost, not rank throughput.
- AFT/CRAFT bridge kernels and stage-gradient clipping remain hypotheses;
  calibrate or diagnose those if they explain an observed quality failure.
- The lost CPU counter remains conservatively charged at its launch allowance.
  New funding resolves the continuation boundary without certifying the old
  accounting estimate. No completed result is erased or relabeled.
- GPU assignment must be checked against other active work before launch.
  Keep batch-native TF/XLA and incremental allocation. No package mutation is
  part of resumption.

The immediate exact commands are the saved continuation queue, starting with:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_master.py run \
  --output docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1 \
  --through qualify --targets mixture --arms gabrie,oracle --seeds 11
```

The amended configuration and authorization record are saved before launch.
Execution findings will be appended below and to the campaign summary.

## First repairs and discriminating warm-training extension

Both saved-map RKL repairs completed and still collapsed the ordinary mixture.
They retained some minority mass at 256 updates and lost it by 1,024. Their
recorded clipping fraction was zero. Thus the lower learning rate/larger batch
does not repair these particular warm-map starts at the declared terminal rung.

The oracle and Gabrié forward histories stay near FKL .95 from updates 256
through 2,048; waste-free decreases from .945 to .246 in the last interval.
For the widely separated mixture, the forward KL of its moment-matched
Gaussian is approximately .5 log(1+100 w(1-w))-H(w)=.936, w=1/3. This neglects
the small component overlap and is an explanatory reference, not proof that
the maps are Gaussian. It motivates testing insufficient nonlinear learning
before changing the architecture or RKL objective.

The next targeted arm extends the exact saved oracle warm checkpoint, including
Adam state, from 2,048 to 4,096 and 8,192 lifetime updates. These are geometric
budget extensions, not convergence thresholds. It uses the same teacher bank,
canonical width, learning rate, batch size and clipping setting. Saved teacher
SMC clouds may use the same continuation mechanism; adaptive Gabrié walkers
must first be persisted before that arm can resume honestly. Do not invent or
silently restart its walker history. At the declared warm endpoint, run the
unchanged classic RKL ladder, preserve before/after maps and 1,000-point probes,
and qualify viable final maps under fresh tuning output paths.

Skeptical review: this intervention changes training duration only for the warm
phase; preserving Adam distinguishes continuation from a fresh optimizer
experiment. The 2,048 parent is the comparator. Primary evidence remains
downstream posterior qualification. Heldout FKL, nonlinear fit, gradients and
clipping are explanatory/repair signals; finite and region checks can veto
promotion. A transient loss decrease does not establish convergence or
superiority. Fresh directory and parent hashes preserve provenance; these
are continuations of seed 11, not new replications. Budget remains the amended
global allocation with the existing 240-second worker cap until measured costs
justify otherwise. Unit checks must demonstrate resumed optimizer equivalence
and reject mismatched state before this extension runs.

The optimizer-resume equivalence check passed as part of 39 focused tests.
The 8,192-update oracle continuation stayed at heldout FKL .948 and collapsed
during RKL again. More duration alone did not repair that exact checkpoint.
The corrected warped-oracle HMC run passed at 2,000 warm-up and 2,000 retained
draws per chain. Waste-free ordinary-mixture HMC reached 10,000 retained per
chain with region ESS 236 and region MCSE/SD .071, missing the unchanged 400
and .03 requirements. Its doubled-budget retraining also missed coarse coverage.

## AFT nested-kernel diagnostic

Pricing revision 2 localized repeat cost to `evaluate` (approximately 1.5 wall
and 4--5 CPU seconds/update). Validation and optimizer repeat calls take about
.001--.002 seconds. TensorFlow reports one trace for each; no ordinary Python
retracing is observed. The captured stack is inside TensorFlow execution of
the gradient function. These observations do not yet identify backend cause.

Test the hypothesis that differentiating the separately compiled target inside
the compiled AFT step causes repeated compilation/dispatch work. Expose the same
analytic log-density tensor kernel for composition inside an outer TF/XLA
function and compare nested versus composed calls on identical maps, particles,
weights and seeds. Require FP64 loss/gradient/state/weight equivalence within
1e-10 absolute plus relative tolerance, a conservative deterministic arithmetic
screen. Measure three calls (cold, two repeat) for gradient and mutation, plus
trace counts and actual device. No flow update or target equation changes.
The 120-second diagnostic cap and global budget apply. If the composed version
passes equivalence and removes the repeat cost, use it in AFT/CRAFT and repeat
full-loop pricing before scientific retries. Otherwise preserve the result and
continue diagnosis. Speed is descriptive engineering evidence, not method ranking.

The fixed-input/no-update comparison passed at normalized errors below 3e-17
for gradients and 2e-16 for particle advancement. Both versions repeat in
milliseconds, so it did not reproduce the observed slowdown. Diagnostic
revision 2 interleaves validation and optimizer mutation as the actual AFT
loop does, using identical updates in each version and preserving the same
equivalence criterion. This corrects the first comparison's missing causal
condition rather than treating its fast repetitions as a performance repair.

## Capacity protocol repair

The 256-update calibration nominated width 8 before either width had shown
substantial nonlinear learning. The 8,192-update continuation has now shown
that this exact width-8 seed-11 optimizer trajectory remains on its plateau.
Test width 16, already in the reviewed capacity grid, with the same teacher,
seed, learning rate, batch, architecture family and 8,192-update warm endpoint.
Preserve all 256/1,024/2,048/4,096/8,192 rungs; recalibrate its gradient clipping
with the existing objective-specific pilot. Its RKL endpoint remains 2,048.
Compare to the width-8 8,192-update control, account for all compute, and qualify
any viable final map. Width 16 is a candidate configuration, not a promoted
default or an architecture replacement. Repeat on the declared independent
seeds if the first run is valid; continuous single-seed differences remain
descriptive. Record the override explicitly in the job identity and manifest.

The interleaved composition diagnostic reproduced the slowdown and passed
equivalence: nested repeat gradients took 1.513 seconds versus .0030/.0022
seconds composed, with normalized gradient error 6.8e-17 and particle/weight
error 3.4e-16. Both still had one TensorFlow trace. This supports removing
the nested compilation boundary for these analytic target kernels; it does
not establish which internal backend cache/compiler mechanism is responsible.
The numerical body is unchanged and remains inside the outer compiled step.
Adopt composed target bodies for AFT/CRAFT, run the focused regression and
full two-stage pricing as revision 3, then retry incomplete teachers using
new attempt directories under the existing caps.

Full-loop pricing after the repair completed all 64 gradient evaluations and
six population advances in 24.93 process seconds (81.29 CPU seconds). Gradients
totaled 2.28 seconds including startup; all trace counts remained one. The
previous nested version hit its 120-second cap before completing. Forty focused
tests passed, including optimizer-resume equivalence and target composition
across optimizer updates. The width-16/8,192-update oracle candidate is now
executing. Full AFT/CRAFT attempts may resume under their unchanged 240-second
cap; no larger per-job budget was required by this performance repair.

## Durable execution queue

The repaired AFT mixture and warped-mixture teachers completed in about
64/66 seconds, and both final maps passed coarse coverage. Neither is promoted
from that screen alone. Complete the active seed-11 training queue, then execute
the saved AFT-mixture qualification, the remaining seed-11 qualifications and
bounded repairs, the width-8/16 8,192-update oracle controls on seeds 23 and 37,
and the remaining baseline matrix on seeds 23 and 37. This is the previously
authorized three-seed scope plus the documented capacity intervention.

Use a lightweight sequential queue around the same master, with durable
per-command status and one local queue lock. It waits for the active master
at a worker boundary, invokes the actual master for every phase, and delegates
all scientific decisions, retries and CPU/GPU charging to that master. Exit 3
stops at the campaign budget; nonzero infrastructure failures stop dependent
commands for repair. Candidate rejection with preserved valid evidence proceeds
to the master's already declared repairs and next candidates. A systemd user
unit keeps this local authorized work alive across UI turns. No new target,
external service, environment mutation, mass adaptation or promotion criterion
is introduced. The queue is not a model agent and performs no open-ended edits.

Pre-launch review: the queue must not create a new budget root, duplicate a live
master, overwrite prior attempt results, treat a zero process exit as posterior
success, or skip its state update after interruption. It uses the existing
campaign config and atomic progress files; focused orchestration tests check
success/resume and failure behavior. The user budget authorization already
covers these phases; no additional approval statement is required.

The final queue audit found two orchestration defects before launch. The
`checks_require_repair` status did not match the command-line failure suffix,
so a failed check could return zero to the queue. Also, the five-repairs-per-seed
limit was applied to each filtered invocation instead of the cumulative ledger.
Repair the exit classification and count distinct repair jobs already attempted
for that seed before allocating new ones. Existing incomplete repairs retain
their per-job retry limit. Focused regression checks must cover failed-check
exit propagation and remaining repair slots after a filtered resume. This
changes no numerical method, scientific screen or authorized resource ceiling.
All 37 eligible seed-11 base training jobs are complete; no master or worker
was active during this review. Host GPU 1 was idle in the trusted device check.

The repaired master passed 44 focused tests, including both orchestration
regressions, in `campaign-r1/attempts/checks-1c51fe5a277b-r1/`. That attempt
charged 35.996 wall seconds and 54.537 CPU core-seconds to the existing ledger.
The durable service started at 14:51:44 Asia/Shanghai on 2026-09-29 with:

```bash
systemd-run --user --unit=neutra-warm-start-continuation-20260929 \
  --description='BayesFilter simpler-model NeuTra continuation' \
  --property=WorkingDirectory=/home/ubuntu/python/BayesFilter \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  /home/ubuntu/python/BayesFilter/scripts/continue_neutra_warm_start_campaign.py \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1
```

The first GPU worker recorded verified memory growth before initialization.
The assigned host GPU is 1; its isolated TensorFlow logical name is GPU:0.
Initial AFT-mixture qualification completed without a verified pair and
automatically proceeded to its declared wider search. The service's inotify
watch warning did not prevent startup or artifact writing; the filesystem had
1.1 TB available. No system limits or unrelated jobs were changed.

The wider AFT-mixture search found a verified L=18 kernel, but warm-up reached
10,000 per chain without readiness (latest binary-region R-hat 1.487 versus
1.05). Its numerical health passed and no retained draws were produced. The
automatic 68.58-second training repair then lost minority coverage during RKL.
The queue advanced to the remaining seed-11 qualifications without treating
either candidate failure as a continuation veto. The detailed interim result
and decision tables are in
`bayesfilter-neutra-warm-start-continuation-results-2026-09-29.md`.
