# Simpler-model NeuTra warm-start master

The current October 1 follow-up uses
[`run_neutra_geometry_repair.py`](../../scripts/run_neutra_geometry_repair.py)
under the [reviewed geometry and selection plan](../plans/bayesfilter-neutra-geometry-selection-plan-2026-10-01.md).
The preceding [gap-closure cycle](../plans/bayesfilter-neutra-gap-closure-results-2026-10-01.md)
completed three previously unresolved posterior screens. This follow-up checks
their score tails, preserves eligible RKL checkpoints for downstream selection,
and adds an explicitly versioned mixture shape profile. Earlier passing results
retain their original scope.
Read the current campaign's `repair-next-phase.json` for its resume command;
the state and complete attempt history remain in the same funded campaign.
Historical queue descriptions below preserve their original scope.

New refinement outputs include `checkpoint-candidates.json`: each shape-passing
checkpoint has its own frozen map, assessment and map-bound 1000-point probe.
The qualification consumer tries eligible RKL checkpoints in reverse update
order, followed by the warm map. It tunes each map separately and selects on
the shared sequential posterior checks. Forward KL remains a nomination metric,
not a substitute for that downstream assessment. The final reference is opened
once after selecting both map and HMC member; final failure ends the attempt.
The historical `moments_regions_shape_v3` profile added named mixture CDF,
valley and unwarped second-coordinate events. Terminal review found its additive
reference-SD tolerance could accept almost complete omission of the rare valley.
The prospective `moments_regions_shape_v4` profile requires both outcomes in each
retained chain and removes that additive allowance for binary probabilities.
Neither profile establishes exhaustive coverage or precise relative estimation
of rare probabilities. The v4 repair was checked on saved draws and exact iid
controls; those checks do not turn the prior v3 run into a fresh v4 confirmation.

The repaired stages restore map, Adam and walkers together, bind continuation
to unchanged teacher particles and weights, retain clipping/progress diagnostics,
and use a measured-cost continuation ladder. Finite failed terminal fits retain
the standard 1000-point diagnostic. HMC starts come from the frozen map, with
effective search settings saved before tuning. The optional calibrated temporal
screen applies only to this campaign; all sequential posterior checks remain.
Teacher overlap, mutation, training quality and downstream HMC outcomes are
reported separately. A passed benchmark screen is not q20 readiness.

Entry point: `scripts/run_neutra_warm_start_master.py`. The reviewed first
campaign is in `docs/plans/bayesfilter-neutra-warm-start-master-2026-09-29.md`.
Its execution record is
`docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/result.md`.

The master tests Gaussian, unequal mixture, warped mixture, author-code wiggle
and 10D funnel targets. Each uses the shared configured IAF. Warm-start arms
are exact examples where available, Gabrié sampling/forward training, ordinary
and waste-free SMC, AFT and CRAFT. Plain RKL supplies a control. A separate
Gabrié arm uses discovered representatives. Every warm map is preserved before
classic RKL, assessed after RKL, and checked with the standard 1,000-point probe.

Run commands from the repository root using the existing `tfgpu` interpreter:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_master.py status \
  --output docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1

/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_master.py run \
  --output docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1
```

GPU execution needs the environment's trusted/elevated tool permission. The
parent selects the configured GPU and enables memory growth before the worker
imports TensorFlow. It never stops other GPU workloads. The implementation
records its actual FP64 dtype; enabling TF32 does not turn FP64 arithmetic into
TF32. The `plan` command creates a fresh configuration but does not allocate
additional compute or authorize a new scientific direction.

`--through checks|prepare|calibrate|price|train|repair|qualify` selects the last phase.
`--targets`, `--arms` and `--seeds` accept comma-separated declared entries.
Omitting filters requests the configured matrix. Resume skips completed jobs,
retains failed attempts, recovers an existing worker, and charges cumulative
CPU/GPU usage. Local locking prevents a second master from launching duplicate
jobs in the same campaign. An attempt cap or exhausted allowance is recorded
as incomplete work. A new directory alone does not replenish the budget.
Unknown final CPU counters reserve the interrupted worker's recorded launch
allowance. Status and summaries expose the descriptive estimate separately
from conservative capacity available for a new launch. A positive estimate
does not authorize a launch when the conservative remainder is exhausted.

The default final phase uses the supported fixed-transport HMC tuner and its
verified member's sequential controller. Four chains run as one native batch;
the mass is identity in latent coordinates. Warm-up is archived and excluded
from retained estimates. Physical moments, region probabilities, modern R-hat,
bulk/tail ESS, Monte Carlo precision and independent references determine the
declared posterior check. A failed coarse training screen or tuning search
triggers the recorded bounded repair, without relaxing scientific criteria.

`config.json` describes the full intended matrix. `state.json` records the
currently requested subset and attempts. `summary.json` and `result.md` collect
available results; `manifest.json`, source copies, serialized maps, optimizer
checkpoints, particle clouds, diagnostics and chain chunks live under unique
attempt paths. `requested_matrix_attempted` means the requested work has been
attempted, not that every candidate passed or all configured seeds were run.
The first campaign does not authorize q20 or promote a production default.

The historical first allocation ended at the CPU accounting boundary, with
25 completed seed-11 training runs and five configurations passing the declared
posterior checks. Its preserved review is
`docs/plans/bayesfilter-neutra-warm-start-master-results-2026-09-29.md`.

Continuation update: the owner subsequently authorized 48 additional hours.
`docs/plans/bayesfilter-neutra-warm-start-continuation-2026-09-29.md` records
the active allocation (32 CPU core-hours, 16 GPU process-hours), repairs and
execution. The previous one-hour proposal is historical. The master now also
accepts `--warm-parent <completed attempt> --warm-total-updates 8192` to resume
a fixed-teacher warm optimizer, and `--capacity-width 16 --warm-total-updates
8192` for a separately identified candidate within the reviewed width grid.
Both require one target, arm and seed and preserve the unchanged RKL and
downstream criteria. Adaptive Gabrié continuation requires archived walkers;
it is not silently reconstructed from mode representatives.

The durable continuation entry point is
`scripts/continue_neutra_warm_start_campaign.py --output <existing campaign>`.
It shares the master ledger, waits for any active master, and records its ordered
commands in `continuation-queue-state.json`. The local service
`neutra-warm-start-continuation-20260929.service` was launched on 2026-09-29.
Inspect it with `systemctl --user status
neutra-warm-start-continuation-20260929.service` or read the campaign state.
Do not launch a competing master while that queue is active. Its command order
is AFT-mixture qualification, the remaining seed-11 qualifications/repairs,
the reviewed oracle capacity controls, and seeds 23/37. Completed commands
are skipped on queue resume. Failed checks stop dependent work; the five-repair
limit for each seed is cumulative across filtered invocations.
The queue completed at 18:39:16 Asia/Shanghai on 2026-09-29 with 111 base
training runs, 15 capped repairs and six duration/capacity controls. No worker
remains active. Fifty-six distinct maps passed the declared HMC checks;
reliable multimodal training remains unresolved. The terminal review and
remaining allocation are in
`docs/plans/bayesfilter-neutra-warm-start-continuation-results-2026-09-29.md`.
Resuming the same completed queue skips its completed commands; it does not
execute the proposed next scientific repair or replenish repair allowances.

September 30 audit: `docs/plans/bayesfilter-neutra-warm-start-pipeline-audit-2026-09-30.md`
records reproduced calibration/resume defects and unresolved teacher validation,
stage selection and final-reference separation. The completed matrix is a
bounded diagnostic study, not a validated full initialization procedure. No
runtime repair or new campaign was executed by this audit.

The owner then authorized the executable repair plan:
`docs/plans/bayesfilter-neutra-warm-start-repair-master-2026-09-30.md`.
Its entry point is `scripts/run_neutra_warm_start_repair_master.py`, using the
same funded ledger and the repaired original worker. The fixed launcher is
`bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_warm_start_repair_campaign.sh start`
(replace `start` with `status` to inspect it). Its exact prefix has a persistent
local approval. The service is `neutra-warm-start-repair-20260930.service`.
Current decisions and the next executable phase are in `repair-master-state.json`
and `repair-next-phase.json` under the existing campaign directory. Successful
warm maps remain eligible if RKL damages their fit. Independent teacher checks
and fresh final references are separate from IAF fit diagnostics. Inspect the
repair master record for active work rather than resuming the historical queue.
