# q20 checkpoint recovery and estimation execution

Active September 20 update: the owner approved replacing the fine-plateau
training gate with bounded validation for HMC trial admission. See
`bayesfilter-ssl-lstm-q20-training-admission-repair-2026-09-20.md` for the repair,
current code/tests, new forecasts and carry-forward checkpoint. The reservations
below preserve the earlier maximum-bank policy and are historical cost decisions.

The owner requested repair of the two outstanding resume gaps, a thorough master
review, and execution. The September 19 estimation question and all numerical
acceptance rules remain in force: obtain a validated q20/T30 UKF posterior
estimate using plain NeuTra HMC, then the tempered NeuTra ensemble if needed.
Identity mass in frozen transport coordinates is required. No classical sampler
comparison, mass adaptation, or method ranking is part of this execution.

## Executable phases

1. Inspect the original v2 protocol and eight 512-update checkpoints against
   their preserved source tree. Require checksums, compatible target/data,
   architecture, optimizer, seeds, temperatures, and training/validation rules.
   Only the documented v3 estimation reset and strict-to-cached factor backend
   transition are supported. Do not import old validation caches or exports.
2. In a bounded GPU worker, restore every map and Adam/random state. Rebind the
   restored map to the current bridge through the repository checkpoint codec,
   check frozen-map forward/inverse/logdet/score parity, and check strict/cached
   target value/score/status agreement on deterministic map points. Preserve all
   original evidence. Record reuse as a warm start, not exact numerical replay.
   Preserve total Adam iteration, weights, random position and history; reset
   current-scope update credit, assessments, plateau counts and exports. Current
   training must satisfy its own unchanged floor and assessments. The original
   initial map is reevaluated as the learning baseline; the migrated map is the
   first incremental baseline. A method attempts only its own training schedule.
3. Qualify the current GPU/XLA enclosing transition and actual batched public
   HMC runner for the temperatures needed by the current method. A receipt for
   an old source tree or different temperature cannot substitute for this phase.
4. Measure actual NeuTra training, validation, transformed four-chain HMC,
   posterior analysis and independent reference costs. Price the actual
   multi-chart mixture if the ensemble is reached. Preserve compile and steady
   timings separately. Check the remaining allowance before starting long work.
5. Continue assessed training, frozen-map tuning with fresh verification,
   sequential posterior estimation, and independent reference assessment. Stop
   at the first validated estimate. A candidate failure may lead to the second
   method; infrastructure failure triggers a localized repair and fresh attempt.

The CLI will accept the old protocol explicitly for migration, and expose
`migrate` as an inspectable phase. `campaign` runs these phases automatically.
Completed phases are reused only with matching inputs and preserved artifacts.

## Evidence contract and default audit

| Diagnostic | Role and implication |
| --- | --- |
| Checksum, config, source and state restoration | Continuation veto on corruption or incompatible scientific scope. |
| Strict/cached value, score, status and frozen-map parity | Migration/engineering pass criterion; failure requires repair, never posterior promotion. |
| Current GPU memory-growth, XLA and batched public-runner checks | Execution prerequisite; no convergence claim. |
| Measured method costs and remaining allowance | Budget stop under the recorded conservative reservation policy; not proof of a runtime lower bound. |
| Assessed training and fresh frozen-kernel verification | Candidate promotion veto; diagnostics alone do not establish posterior accuracy. |
| Sequential R-hat, ESS, precision, numerical health, start consistency and integration agreement | Existing estimate acceptance criteria/vetoes, unchanged. |

The baseline for backend agreement is the strict TensorFlow implementation of
the same UKF target. The posterior comparator is independent integration of that
same target. These checks do not establish an exact nonlinear likelihood, global
mode coverage, perfect whitening, or a statistically superior method.

No scientific number is newly calibrated here. Architecture, learning rates,
training floors/caps, validation counts and tolerances, seed policy, epsilon/L
search, posterior criteria and factor-two cost reserve are inherited hypotheses
from the parameter ledger and September 19 plan. The smallest migration bank
uses the existing 32 reliability rows and existing 1e-9/1e-10 relative/absolute
tolerances. Historical optimizer memory is a warm-start hypothesis; fresh loss
assessment and downstream kernel/posterior checks can reject it. Historical
updates earn no current-scope training-floor credit. This deliberately avoids
claiming exact continuation across a backend change.

## Review, budget and execution records

Skeptical pre-execution audit: a config-hash rewrite alone would incorrectly
promote old evidence, so migration must reconstruct and test numerical state and
clear old admission. A fixture pass cannot qualify q20/GPU. Source snapshots must
be isolated from concurrent repository edits. Historical classical forecasts
cannot price NeuTra. Worst-cap reservations are conservative engineering policy,
not measured minimum costs; report them as such. Reference failure stops for
reference repair rather than launching another sampler. The revised plan passes
these checks; focused executable tests must verify the new call chain before
GPU execution. The final code review and results are recorded below.

Carry-forward allowance: 79,899.92701190927 campaign seconds, including
4,140.359230629365 diagnostic seconds, from the September 19 repair settlement.
There is no new allowance. CPU verification and all GPU attempts consume both
applicable balances. Focused CPU tests have a 600-second process ceiling, an
engineering convenience bound. Each diagnostic worker retains the inherited
1,200-second cap, further bounded by remaining diagnostic time; long phases use
measured reservations and the external supervisor. Source/config incompatibility,
invalid target/state, missing required evidence, exhausted time/attempt limits,
or no available GPU pauses execution with a recorded reason. Preserve every
attempt and never overwrite an earlier output root.

Output root: `docs/plans/artifacts/ssl-lstm-q20-resume-execution-2026-09-20/`.
The verification record preserves exact commands, environment, wall time,
source snapshot and allowance settlement. Campaign worker manifests preserve
seeds, target identity, GPU inventory, memory growth, XLA, command, plan and
result paths. The terminal note will distinguish engineering readiness,
numerical/sampler evidence, and scientific conclusions.

## Completed master review

The review followed CLI -> coordinator -> supervised worker -> migration,
qualification, pricing, training, public fixed-transport tuner, retained runner,
sequential assessment and independent integration. It also inspected cost credit,
completed-stage replay, checkpoint selection, source checks, cumulative deadlines,
memory-growth initialization and candidate/infrastructure failure routing.

Three repairs were necessary: the v2 importer had no migration route; migration
adds a second initial validation map that must be priced; and training's internal
deadline previously failed to subtract worker initialization. The added phase
reconstructs maps through the existing codec, proves numerical state preservation,
clears old admission, tests strict/cached parity, and preserves explicit parent
hashes. The learned and original baseline maps both receive fresh evaluation.

CPU verification covered 40 distinct cases across two overlapping invocations.
The initial run had 33 passes and two test-expectation failures in 122.87 seconds:
one expected a particular error string rather than the earlier target guard;
one incorrectly expected a tiny unqualified reference to pass. Both assertions
were corrected to the scientifically appropriate behavior. The follow-up passed
seven checks in 59.41 seconds, including both corrected cases, GPU-qualification
receipt logic on CPU fixtures and CLI validation. The supervised migration test
replays without repeating completed work, then proceeds through plain NeuTra
training/tuning/sampling to the expected reference-repair stop. The existing
complete passing-estimate fixture also passed. Same-target next-update testing
verifies exact map/Adam/RNG preservation; it does not claim strict/cached training
trajectories remain bitwise identical. No unresolved failure remains in these
focused checks. No external reviewer was used.

Charge 183 seconds (each pytest duration rounded up): 79,716.92701190927 campaign
seconds remain, including 3,957.359230629365 diagnostic seconds. The launch uses
a separate source snapshot at `/tmp/BayesFilter-q20-estimation-execution-20260920-r1`;
all 489 numerical source entries match the reviewed tree. The original v2 source
tree matches the original checkpoint exactly. Long-run affordability, actual q20
GPU migration, current-source qualification and convergence remain execution
questions. Proceed with the bounded master; do not infer their answers from CPU
fixtures or inflate the allowance.

## Execution evidence

The trusted master launched on September 20 and selected idle host GPU 2 (RTX
4080 SUPER), with TensorFlow memory growth verified before initialization.
Checkpoint migration completed for all eight maps. All 4,096 historical optimizer
updates remain represented in the preserved weights, Adam state and history;
current-scope update credit is zero. Every map passed frozen codec parity and
strict/cached value, score and status agreement. Maximum observed target-value
difference was zero and score difference was 7.105427357601002e-15 on the declared
32-point map banks. These are local engineering checks, not posterior evidence.
The master then entered current-source beta-one qualification. Detailed receipts
and attempt accounting are in `campaign-01/campaign.json` under the output root.

Beta-one GPU/XLA qualification passed with one trace for each enclosing graph
and the public four-chain runner. Actual measured training costs then gave a
28.19-hour floor-with-validation-cap reservation, versus roughly 22.08 campaign
hours remaining. The first-bank floor scenario was 9.46 hours; neither forecast
is a calibrated runtime bound or a convergence guarantee.

This exposed another coordinator defect: supplying a checkpoint bypassed the
early training-budget decision even after remaining-work credit was calculated.
The master continued expensive downstream timing despite an already unaffordable
training reservation. The owned first master was interrupted cleanly; its actual
time and partial pricing evidence are preserved. Repair the guard to use checked
checkpoint credit and the remaining allowance, and return before HMC timing when
the reservation already fails. This does not change the scientific plan or any
threshold. Regression tests cover both permitted methods and prevent downstream
entry on this branch. The next source snapshot will import the already migrated
v3 checkpoint through the existing coordinator-only import, preserving all eight
maps and avoiding a second migration run. Current-source qualification remains
required. Proceed with a fresh campaign root and the settled remaining balance.

The localized repair passed 17 focused checks (30.63 seconds), four checks on
the isolated execution tree (30.84 seconds), and an interrupt-settlement check
(0.16 seconds). Counts overlap; the 63-second rounded charge is preserved in
`repair-02.json`. The second campaign imported the migrated v3 checkpoint without
changing map/Adam/RNG state. A competing workload appeared on GPU 1 during the
first qualification attempt, causing a resource pause; its 136.05 seconds were
charged. The same campaign resumed and completed uncontended qualification at
beta one and beta 0.5, then both training cost decisions. No retry renewed budget.

## Terminal result and reset memo

Status: `ESTIMATION_UNDER_BUDGETED`. The executable migration and qualification
gaps are repaired and exercised on real q20 GPU work. The conservative training
reservation stops both permitted methods before serious training or HMC tuning.
Complete downstream method costs remain unmeasured: measuring those cannot make
an already excessive training reservation fit the same fixed budget policy.

| Method and scope | Measured raw training floor plus validation cap | Factor-two reservation | Coverage |
| --- | ---: | ---: | --- |
| Plain NeuTra | 14.12 h | 28.23 h | Both widths, all 12 direct candidates |
| Tempered ensemble | 13.64 h | 27.28 h | Width 16 only, both positive temperatures; width 32 remains unpriced |

The plain raw estimate comprises 4.01 optimizer hours, 10.07 validation hours and
0.04 setup hours. Thus the dominant reservation is heldout evaluation at the
declared maximum bank size, multiplied by the inherited factor two. The
first-bank plain scenario reserves 9.48 hours. These scenarios do not establish
which validation size will resolve decisions or how much training/tuning/sampling
will actually be needed. A 28.23-hour reservation is not a mathematical lower
bound on runtime. No claim that NeuTra or the ensemble is scientifically invalid
follows from this stop.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept checked migration | Eight maps restored; exact map/Adam/RNG preservation and backend/map parity passed | No migration veto | Learned-map adequacy under current assessment | Continue from campaign-02 imported checkpoint when affordable | Convergence or whitening |
| Accept current GPU/XLA graph qualification | Beta 1 and 0.5 public four-chain runners passed, one trace each | Resource contention retried and resolved | Learned-kernel stability at tuned epsilon/L | Fresh fixed-map tuning after assessed training | A tuned kernel or validated posterior |
| Stop this invocation at budget policy | Both measured training reservations exceed remaining balance | Cost reservation prevents long launch | Adaptive validation demand and downstream cost | Review validation allocation and conservative reservation using measured costs; retain all scientific acceptance criteria | That more total time is necessarily required, or either method failed scientifically |

The final attempt is fully settled. Remaining campaign balance is
78,552.22873334838 seconds (21.8201 hours), including 2,792.66095206848 diagnostic
seconds (46.54 minutes). Nothing is left running by this master. No new serious
training, tuning or posterior estimation was launched; only bounded pricing
updates and qualification trajectories ran. Keep `campaign-01` and `campaign-02`
and the two source snapshots. The active migrated checkpoint is
`campaign-02/training-import/cohort-00000.json` under this plan's output root.
It preserves historical updates and requires fresh current-scope assessment.

Terminal artifact review verified all completed-stage receipt and artifact
checksums, both qualification temperature inventories, passing GPU endpoints,
source-import changes, partial-versus-complete pricing inventory, and settlement
of every attempt. Strongest alternative explanation for the cost stop: the
maximum-bank reservation considerably overestimates actual adaptive work. A
reviewed allocation that funds the unchanged scientific procedure within the
measured remaining budget could overturn that stop. The weakest evidence is
the two-update timing sample and unmeasured downstream learned-map behavior.
The appropriate next repair is cost allocation/validation scheduling, not
weakening posterior checks or claiming that a smoke result is an estimate.

## Superseding active execution state, September 20

The subsequent [training-admission repair](bayesfilter-ssl-lstm-q20-training-admission-repair-2026-09-20.md)
bounded validation and measured the downstream path. The
[staged-budget repair](bayesfilter-ssl-lstm-q20-staged-budget-repair-2026-09-20.md)
now replaces the all-caps affordability gate with actual remaining-balance
allocations, complete-cohort trial handoff, and checkpointed resource pauses.
Its settled allowance, preserved execution source, imported checkpoint and
prepared command are the active continuation state. The balances and blocker
descriptions above remain historical. No posterior estimate has been produced.
