# q20 master bootstrap initialization repair

Status: `MASTER_BOOTSTRAP_REPAIR_INCOMPLETE`, terminal September 19 02:50:39
Asia/Shanghai. Both initializers nominated finite startup pairs, but both
preparations timed out during the third bootstrap round; mass adaptation was
not reached. The service is inactive and all attempts are settled. See the
terminal assessment below; the intermediate live observations remain historical.
Authorization: the owner requested a master repair phase, thorough review and
execution. Use the existing settled campaign allowance: 97538.033422 campaign
seconds, including 21778.465641 diagnostic seconds. The previous refresh has
9716.335424 seconds remaining in its original 14400-second envelope. This
repair does not renew that envelope or either allowance.

## Question and evidence contract

Can finite initialization of the classical HMC step prevent the observed
bootstrap failure while retaining the same q20/T30 target, prior-scale affine
coordinates, starts, exact scores, XLA, and subsequent preparation checks?
The comparator is the preserved failed preparation at beta .5 and beta1 in
`ssl-lstm-q20-master-resume-2026-09-18/campaign`, including the exact beta .5
replay with 30/32 nonfinite log acceptances and zero movement.

| Role | Criterion |
| --- | --- |
| Engineering pass | Shared preparation initializer is exercised by both q20 pricing and classical tuning; stable XLA signatures, source/seed records, exact restart accounting and complete failure reporting |
| Initialization pass | All momentum probes have finite retained and proposed state/value/score/log acceptance and valid target status, and their mean acceptance probability reaches the existing lower repair threshold |
| Preparation pass | Fresh bootstrap and existing operational mass adaptation pass with a real metric update; initialization alone does not pass this criterion |
| Promotion veto | Any invalid proposal blocks that candidate; incomplete mass adaptation, absent verified tuning or posterior checks block later promotion |
| Continuation veto | Invalid initial/retained target, runtime/program exception, inconsistent sources/starts, corrupt evidence or exhausted budget |
| Repair trigger | Invalid rejected proposal or very low finite acceptance during initialization causes a smaller-step probe; later bootstrap/mass failure remains preserved for diagnosis |
| Explanatory | Acceptance, energy/log-acceptance magnitude, timings, first bad proposal and compilation costs |
| Not established | Posterior whitening, convergence, mode coverage, transport quality, method ranking, full-campaign affordability or default readiness |

## Implementation and review

Add an optional repository-owned initializer to shared operational preparation.
Keep the numerical target and existing bootstrap/mass algorithms. Both q20
classical consumers explicitly select the initializer. The initializer builds
the existing bootstrap affine adapter and the reviewed TensorFlow/TFP HMC
kernel. It checks the initial target and batches four momentum probes from the
same start. Epsilon is a runtime tensor; cache a graph for each encountered
leapfrog count. Each graph has a stable explicit input signature and defaults
to XLA. No NumPy, pfor, scalar target loop or Python callback enters the kernel.

At each step use the bootstrap's declared `ceil(tau/epsilon)` rule with its
existing leapfrog cap, so the probe tests the trajectory the bootstrap will
consume. A failed proposal is recorded and cannot seed sampling. The search
only decreases epsilon; all retained-state invalidity and execution errors
remain fatal. It records the initial formula, each probe, chosen pair and
original geometry hash in the new geometry provenance. A fresh bootstrap uses
its original independent seed stream. The helper issues no tuning authority.
The first failed proposal includes position, score, value and health flags,
using explicit JSON markers for nonfinite values.

Expose a `repair` mode in the existing master CLI. Require the predecessor
campaign, validate its settled ledger and preserve references to its training
checkpoint and prices. Use a fresh isolated copy of its numerical source plus
only the reviewed repair changes. Requalify XLA at both temperatures because
the source inventory changes. Run the repaired preparation at both temperatures
in fresh output directories. Retain incomplete bootstrap/mass progress at the
deadline. Successful preparation adds only its measured cost; absent mixture
and metric-transition prices remain absent. Do not repeat training or launch
unaffordable posterior work.

Review the call chain from CLI through stage supervision, source checks,
preparation, initializer, bootstrap and mass adaptation, plus the ordinary
classical tuning consumer. Verify old failures are never promoted; finite
retained states cannot hide invalid proposals; retries share budgets and reuse
completed stages; arbitrary source changes reject import. Focused CPU tests
cover healthy and unstable targets, invalid initial state, stable tracing,
fresh bootstrap dispatch, consumer wiring and master cache/budget behavior.
One bounded CPU XLA check verifies compilation of the new probe kernel.

## Numerical choices and budget

The initial epsilon, prior scale4, geometry scale .5, trajectory formula and
L cap25 come from the failed preparation and remain hypotheses. The prior
geometry does not measure posterior curvature. The shrink factor2 comes from
the existing preparation repair factor and provides geometric exploration,
not a stability proof. At most20 probe rounds is inherited from the existing
reasonable-epsilon search; it spans a factor2^19 in starting step and is a
finite engineering limit. Four simultaneous momentum probes match the existing
operational initializer's probe count; they cannot establish tail safety.
The acceptance floor is the existing preparation repair-band lower endpoint
(.55 for this protocol), used only to nominate a startup step. A fresh full
bootstrap and downstream tuning retain their original criteria.

Use the existing preparation seeds, with a separately hashed initializer
stream so bootstrap verification never reuses its probe momenta. Preserve the
historical pricing start seeds exactly. Both price and serious tuning record
the initializer policy and progress. No posterior threshold changes.

New artifact root:
`docs/plans/artifacts/ssl-lstm-q20-bootstrap-repair-2026-09-18/`.
CPU verification is capped at900 seconds and charged by measured wall time;
setup/accounting gets the already stated 180-second envelope as a new repair
debit. Subtract both from the predecessor's campaign, diagnostic and remaining
refresh phase balances. Fresh qualification has up to1200 seconds per beta
(previous actual about180 seconds). Each preparation has up to4000 seconds,
further restricted by the single remaining phase balance. Its cooperative
deadline leaves100 seconds for checkpoint/cleanup inside the external cap.
These caps are engineering allocations, not expected completion times; the
existing complete mass-adaptation work may not fit. Every interruption or
failed attempt is charged, and completed stages are not charged again.

## Skeptical audit and pre-mortem

Audit passed after rejecting two inadequate approaches: changing only the
pricing wrapper would leave the actual tuner broken, and probing a fixed short
trajectory then recomputing a longer L could hand bootstrap an untested pair.
The proposed shared initializer addresses both. It does not merely change an
arbitrary epsilon constant or relax a numerical veto. The generic reasonable
epsilon helper already supports the search idea, but currently constructs
per-step kernels without stable explicit signatures and does not preserve the
first failed proposal. Reuse the same reviewed TFP transition, with a batched
probe and stable graph cache, rather than importing those limitations.

Main contains concurrent unrelated HMC extraction/validation work. Executing
all of main would change the comparator. Preserve that work and prepare a new
isolated source from the exact previous run. Existing training and timing
artifacts keep their original source identity and are carried as historical
completed work; they are not relabeled as newly produced evidence.

A smaller step can make arithmetic finite while leaving poor mixing or very
slow adaptation. Only the full unchanged preparation/tuning/posterior checks
can establish their respective claims. Failure at all funded small steps
triggers scrutiny of the first invalid state, score and target numerics. A
deadline during otherwise finite warmup is a compute limitation, not a method
rejection. The plan must record which of these occurred.

## Implementation audit and execution handoff (September 19)

The CLI now exposes `repair --previous-campaign ... --previous-source-root ...`.
Its coordinator checks the settled predecessor and original remaining phase
allocation, credits the preserved training/timing stages without rerunning them,
requalifies both temperatures, and runs repaired classical preparation. Original
price records retain their source identity. Preparation prices carry the new
source identity separately. Mixture-dispatch and metric-specific transition
prices remain missing and prevent full-campaign admission.

Both `price_preparation` and the classical branch of `tune_scope` call shared
preparation with `initialize_bootstrap=True`. The shared prefix invokes
`initialize_bootstrap_step`, then the existing fresh bootstrap and windowed mass
adaptation. Invalid initial/retained state stops immediately. Proposal failures
shrink the startup hypothesis; runtime exceptions are not converted into low
acceptance. Structured exceptions captured by the existing bootstrap also stop
the repair scope, rather than being treated as an ordinary failed candidate.

The execution source is `/tmp/BayesFilter-q20-bootstrap-repair-20260918`, copied
from `/tmp/BayesFilter-q20-master-resume-20260918`. Only nine reviewed source
paths changed: the new initializer/coordinator, the shared preparation and its
compatibility wrapper, the numerical dependency inventory, both q20 consumers,
master dispatch and CLI. The frozen import structure is retained; concurrent
main HMC extraction and validation changes are not part of this numerical run.
`prepare_execution_source.py` records that bounded transfer.

Executable checks cover finite Gaussian startup, CPU XLA, nonfinite proposal
shrinkage, first-failure serialization, fatal initial/retained/runtime errors,
the exact epsilon/L handoff to fresh bootstrap, both real q20 call sites,
source isolation, stale evidence rejection, phase accounting and completed-stage
reuse. The existing classical end-to-end tuning/posterior fixture and supervisor
and bootstrap regressions passed (34 tests). Main's initializer/coordinator,
reporting and extraction tests passed (30 tests). Final isolated-source repair
verification is recorded in `verification.json`; its cumulative measured time
includes two setup commands that exited before test collection because a test
file had not yet been copied or was absent from the frozen source. No numerical
test failed in those setup commands.

Review found no target, prior mass, posterior threshold, training-state or
scientific promotion change. It specifically checked that startup nomination
cannot bypass bootstrap/mass checks, retries cannot reset the inherited phase
balance, and failed preparation cannot produce a completed price. The strongest
remaining risk is a locally finite startup followed by poor exploration or
expensive adaptation. A successful startup does not resolve that risk. The
4000-second preparation cap may expose only partial warmup; all available
progress must be preserved and classified honestly.

`integration.json` records exact source inventories and review scope;
`execution-allowance.json` deducts cumulative CPU verification and the declared
180-second setup allocation from both the settled campaign/diagnostic balances
and the remaining refresh envelope. `launch.json` records the exact trusted
systemd command. GPU workers use memory growth, batch-native TF/TFP and XLA.
The master uses any available device among the three GPUs and retains its
existing external timeout and process-group cleanup.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute bounded repair phase | Focused implementation/call-chain tests passed | No source, test or accounting veto | Actual q20 startup and complete mass adaptation | Requalify both betas and run repaired preparation | Production readiness, convergence or full campaign affordability |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Old bootstrap proposals were nonfinite; those attempts stay rejected |
| Statistically supported ranking | None; this repair does not compare methods |
| Descriptive-only differences | Subsequent epsilon, acceptance and runtime observations |
| Default-readiness | Not established; opt-in shared repair selected by q20 consumers |
| Next evidence needed | Finite q20 startup, fresh bootstrap, full mass adaptation and later verified tuning/posterior checks |

## Live execution checkpoint

Service: `bayesfilter-q20-bootstrap-repair-20260919-r1.service`.
Original hard service deadline: September 19 02:55:50 Asia/Shanghai. RuntimeMaxSec 9540
includes the predeclared coordinator cleanup allowance. Numerical workers share
9479.416981 seconds remaining in the original refresh phase; each qualification
has a 1200-second cap and each preparation a 4000-second cap, further restricted
by the remaining shared balance. Those are caps, not predicted completion times.

CPU verification, including unsuccessful collection/setup invocations, consumed
56.918444 seconds. Together with the 180-second setup debit, execution started
with 97301.114978 campaign seconds (27.0281h), including 21541.547197 diagnostic
seconds (5.9838h). Worker charges are in `campaign/campaign.json`; any running
attempt's cap remains an outstanding hold until settlement. The predecessor
ledger and all prior results are unchanged.

Use the exact command in `launch.json` for unchanged-scope continuation. Do not
edit the live execution source or reset the allowance. Completed stages are
checksummed and reused; a failed preparation retains its progress and prior
attempt, and any retry shares the remaining per-stage and phase allocation.
The next observer should inspect the latest attempt's
`worker/data/preparation_progress.json` before drawing a numerical conclusion.
Initialization success alone is not completed preparation.

### Local interface repair and retry

Both original qualifications passed (beta .5: 187.555169 seconds; beta1:
185.552228 seconds). Preparation failed after 24.508366 worker seconds, before
any HMC proposal, because the initializer used the combined value/score
diagnostics as public HMC telemetry. The q20 combined result contains a lone
`min_innovation_eigenvalue`; its existing public telemetry method deliberately
omits the incomplete optional pair. The initializer now uses that existing
telemetry boundary and separately retains the richer raw diagnostics. The
target, value, score, prior geometry, acceptance floor and sampling checks are
unchanged. New graph and CPU XLA regressions reproduce this precise schema
difference and verify that the raw eigenvalue is preserved.

The first failure is an implementation/interface failure. It does not test the
startup step hypothesis, and it neither weakens nor rescues NeuTra. Original
source and artifacts remain intact. The retry uses
`/tmp/BayesFilter-q20-bootstrap-repair-20260919-r2` and
`ssl-lstm-q20-bootstrap-repair-2026-09-18/retry-01/`.

The retry allowance carries the first ledger by checksum, deducts its actual
397.615763 worker seconds, and charges cumulative CPU verification (including
the original checks). The 180-second setup allocation is charged once. Each
stage also inherits its earlier cost: preparation at beta .5 has at most
3975.491634 seconds left, and neither XLA qualification gets a renewed
1200-second allowance. A focused regression verifies both phase and per-stage
accounting. Sources changed, so qualification is repeated before the new
initializer runs. The fresh receipt and exact launch command are under
`retry-01/`; no prior numerical evidence is relabeled.

The corrected retry started at 00:31:33 Asia/Shanghai under
`bayesfilter-q20-bootstrap-repair-20260919-r2.service`; its hard deadline is
03:03:42. After 70.417536 cumulative CPU verification seconds, the original
180-second setup debit and all prior worker costs, its phase balance was
9068.302126 seconds. The overall balance was 26.9139 campaign hours, including
5.8696 diagnostic hours. The old run's allowance and verification record remain
unchanged. The schema regression suite passed 24 tests; the subsequent focused
accounting suite passed 10 tests after adding inheritance of per-stage caps.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retry after interface repair | Public-status schema regression passed with and without CPU XLA | Previous interface exception fixed; target has not yet been probed by the new initializer | Actual q20 startup and mass adaptation | Requalify changed sources, then execute the same preparation question | The old failure was numerical evidence against HMC or NeuTra |

### Live numerical observation, approximately 00:40 Asia/Shanghai

The corrected-source qualifications passed at both betas (192.053582 and
186.048449 worker seconds). The beta .5 pricing start record matches the
original failed preparation exactly, and the initial state/value/score/status
checks pass. `retry-01/matched-starts-and-initial-target.json` preserves that
comparison and the raw versus public status fields.

The initializer has executed real four-row GPU/XLA momentum probes. The first
two pairs, epsilon .3535533906 with L5 and epsilon .1767766953 with L9, produced
invalid proposals while retained states remained valid. The first invalid
proposal, physical and latent state, value, score, flags and seed are preserved.
At epsilon .08838834765 with L18, all proposals were finite, but the descriptive
mean acceptance probability was about 7.12e-72. This fails the declared startup
nomination floor, so the controller continues shrinking epsilon. L changes by
the predeclared trajectory rule; these observations are not a comparison of
epsilons at fixed L.

This confirms the numerical repair path is executing and preserving failure
evidence. It has not yet nominated a startup pair or passed fresh bootstrap or
mass adaptation. Four local momentum probes cannot establish global stability,
whitening, convergence or method superiority. The active service and ledger
continue under the recorded caps; the terminal preparation result remains
pending.

## Terminal assessment, September 19

The corrected run finished at 02:50:39 Asia/Shanghai with status
`MASTER_BOOTSTRAP_REPAIR_INCOMPLETE`. The service is inactive and no matching
q20 repair worker remains. Its normal coordinator exit means accounting and
reporting completed, not that numerical preparation passed.

The initializers nominated epsilon 0.011048543456039808 at beta .5 and epsilon
0.022097086912079615 at beta1, both with L25. Initial/retained states and the
selected proposal checks were valid. The respective four-probe mean acceptance
probabilities were 0.9974152816 and 0.7748416360, descriptive startup checks only.
Each encountered L graph traced once. Nomination took approximately 280 and
219 seconds, including the initialization/progress prefix.

| Temperature | Completed fresh bootstrap rounds | Numerical hard veto in completed rounds | Acceptance disposition | Completed round time | Terminal stage |
| --- | --- | --- | --- | --- | --- |
| beta .5 | 2 | None reported; runtime finite | Both above the bootstrap band, triggering epsilon repair | 1528.3 and 1482.6 seconds | Timeout during round index2 |
| beta1 | 2 | None reported; runtime finite | First below, then above the band, triggering epsilon repair | 1326.2 and 1390.0 seconds | Timeout during round index2 |

The original immediate nonfinite bootstrap failure did not recur in these
completed rounds. That supports the narrower startup repair result. Neither
bootstrap reached a passing terminal result, and neither mass adaptation nor
later tuning/posterior work started. The observed bottleneck is the cost of
repeated bootstrap acceptance repair: about 22-25.5 minutes per completed round.
There is no evidence here that NeuTra failed to whiten the posterior; these
preparations used prior-scale affine coordinates without a learned map.

The external supervisor terminated both workers at their stage deadlines
(returncode 124), after 3971.074916 and 3995.458444 worker seconds. The last worker
progress files still say `running` because termination occurred inside a
compiled call. Those are partial snapshots. The settled campaign ledger and
`supervisor.json` files are authoritative for terminal status. Completed-round
progress is preserved, but no complete preparation handoff exists.

The corrected run consumed 8344.635391 seconds. Including its predecessor run,
cumulative verification and setup, the original refresh envelope has
723.666734 seconds left. Both preparation stage allowances are exhausted for
launch purposes (less than 5 seconds each). Restarting the same command cannot
fund another preparation round. The overall settled balance is 88545.364732
campaign seconds (24.5959h), including 12785.796950 diagnostic seconds (3.5516h).
No renewed allocation was assumed. The ledger checksum and exact sum of worker
charges were verified when recording this assessment.

Evidence: `retry-01/campaign/result.json`, `settled-allowance.json`, both attempt
`supervisor.json` files and their `worker/data/preparation_progress.json` files.
The reviewed plan, commands, environment, source inventory, seeds and device
records remain in the preceding manifest and attempt artifacts. Training and
timing evidence was preserved, and the three unpriced categories remain
classical preparation, multi-chart mixture dispatch and classical metric-specific
transition cost.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept finite startup as limited repair evidence | Both initializers nominated a pair; two subsequent rounds were finite at each beta | No reported numerical veto in completed rounds | Behavior beyond the observed starts/rounds | Preserve exact startup/probe records | Global stability, whitening or convergence |
| Keep full preparation incomplete | Bootstrap never passed; no mass update | Stage timeout and exhausted stage allowance | Final bootstrap setting, mass adaptation and full cost | Review bootstrap repair cost, progress/resume boundaries and feasible allocation before another run | Production readiness or method rejection |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Historical nonfinite settings remain rejected; new completed bootstrap rounds reported no hard veto; timeouts leave later checks unavailable |
| Statistically supported ranking | None |
| Descriptive-only differences | Startup acceptance, selected epsilons and observed round runtimes |
| Default-readiness | Not established; full preparation is incomplete |
| Next evidence needed | Passing bootstrap, completed mass adaptation, separately verified tuning and posterior checks |

Post-run review: a finite initial proposal is only local evidence, and the
opposite acceptance dispositions in beta1's longer rounds show why it cannot
replace fresh bootstrap. The observed timeout neither invalidates the target
nor rejects the HMC/NeuTra research direction. The weakest evidence is the small
set of local probes and incomplete bootstrap trajectories. Any later numerical
veto would overturn a broader stability inference; none is asserted here.

## Why the preparation workers timed out

Read-only call-chain audit, September 19, against the exact executed source
`/tmp/BayesFilter-q20-bootstrap-repair-20260919-r2`:

1. `q20_master_repair.py:263` supplies a 4000-second cumulative preparation
   allowance per beta. The beta .5 retry had 3975.491634 seconds after charging
   the prior interface failure. `q20_campaign_runtime.py::Campaign.execute`
   invokes GNU timeout with TERM at cap minus the 5-second cleanup grace. Both
   exits were 124 at those deadlines. The overall campaign balance was not the
   binding limit, and the systemd lifetime limit did not terminate these runs.
2. `hmc_kernel_tuning.py::_public_bootstrap_config` selects 32 screening
   results and 8 discarded transitions. The smaller initialized epsilon drives
   the existing `ceil(trajectory/epsilon)` rule to its L25 cap. Each round
   therefore performs 40 sequential HMC transitions, or 1000 leapfrog steps,
   before returning to the controller. The logged round times are consistent
   with substantial execution work: the second rounds reused the compiled
   runner and still cost 1482.6 and 1390.0 seconds. Compilation does not explain
   the repeated long rounds. The earlier four-chain physical timing had already
   measured roughly 37-41 seconds per L25 transition; its different topology
   prevents treating it as an exact bootstrap forecast, but it was a clear
   warning that a complete preparation would not cheaply fit this allocation.
3. `_classify_bootstrap_screen` requires realized acceptance in [.65,.75],
   measured from 32 retained accept/reject indicators. Finiteness alone does not
   pass this bootstrap. The two beta .5 rounds were above the band; beta1 was
   below, then above. `_repair_step_size` changes epsilon and starts another
   fresh full round from the original affine zero state. The initializer's
   four-probe mean acceptance probability is a different statistic and cannot
   substitute for this screen. Acceptance here is a startup tuning rule, not
   convergence evidence. Its necessity and cost before mass adaptation still
   require review; the repair did not evaluate a replacement rule.
4. `HMCPreparationProgress.phase` only asks whether the deadline has already
   passed. It does not use a measured round duration to check whether the next
   round fits. At round 3 start, at most 679 seconds remained for beta .5 and
   1060 for beta1 under the external deadlines (actual time was smaller because
   worker initialization precedes the preparation clock). Their preceding
   rounds had cost 1483 and 1390 seconds. Nevertheless the controller entered
   another compiled call. There is no internal checkpoint/deadline callback
   inside that call, so the 100-second cleanup margin could not produce a
   graceful stop or save its unfinished chain state.

The bootstrap still uses one four-parameter chain: its adapter start is shape
`(4,)`, passed directly to the reusable runner. The four momentum probes in the
new initializer and the four-chain qualification are separate execution lanes.
They do not make the later bootstrap a four-chain batch or distribute it across
three GPUs. Successive transitions in one chain remain sequential even under
XLA.

This is an incomplete orchestration repair and an inadequate completion budget,
not evidence of a stalled GPU or exhausted overall campaign. The plan explicitly
allowed a partial outcome, but that did not justify starting a predictably
unfinishable third round. The agent repaired unsafe initialization while leaving
expensive pre-adaptation acceptance refinement, absent round-cost admission and
absent numerical bootstrap checkpoints in place. The next repair should address
those mechanisms and derive allocations from measured work before launching;
merely raising a timeout would leave these defects intact. No new numerical run
was launched for this audit.
