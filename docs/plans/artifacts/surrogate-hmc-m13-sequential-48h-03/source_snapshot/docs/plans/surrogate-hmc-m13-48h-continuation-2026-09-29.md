# M13 surrogate-HMC continuation — additional 48 hours, 2026-09-29

## Active authorization and objective

The owner said "you have 48 more hours. can we continue?" Authorize continued
local GPU research under the existing surrogate-HMC objective. This allocation
adds at most172800 GPU-worker seconds and172800 measured CPU-worker seconds.
It expires2026-09-30T16:27:39Z (2026-10-01 00:27:39 HKT), 48 hours after the
authorization was checked. Use existing hardware and serial GPU workers.
Old allocations and unspent retained reserves remain separately accounted.
No new permission is needed for repairs/retries within these bounds.

Goal: qualify surrogate HMC on the existing frozen M13/data1704/T20 target.
Canonical analytical DCTR supplies proposal forces; frozen4096x8 full-support
trajectory importance likelihood plus the same prior supplies MH endpoints.
State4/observation2/parameter2, FP32/TF32/GPU/XLA force and FP64 endpoint.
Data3701 remains untouched. LG1 is already accepted; this phase does not close
all remaining models. Claude review remains deferred.

## Evidence contract and skeptical audit

The prior attempt was numerically healthy but never passed warmup readiness.
Its nomination chose the first acceptable trajectory: L=1, epsilon=.0697635.
Code skipped L=2,4 after that acceptance-only screen. This policy is a weak
exploration hypothesis, not evidence that L=1 is adequate. Do not spend48 hours
blindly repeating it or call additional draws a repair by themselves.

Primary promotion remains warmup>=2000/latest1000 modern Rhat<=1.05, then
cumulative verification>=1000/Rhat<=1.01, each capped10000; retained sampling
uses calibrated looks1000/2000/4000/6000/8000/10000, Rhat<=1.01, bulk/tail
ESS>=400, and the existing MCSE-aware reference margins with the tighter
six-look simultaneous critical value. The original two-look results stay historical. Health chunks250; all warmup and verification
discarded from posterior estimates. Only the public tune_hmc_kernel authority
with a typed endpoint-corrected force binding can issue the handoff.

Vetoes: nonfinite states/energy, endpoint invalidity, fresh force fallback,
divergences, no chain movement, corrupted evidence, source/binding drift, and
resource/deadline caps stop the affected phase. Early Rhat failures extend
rather than stop. Failed mixing candidates can trigger a prospective repair.
Acceptance, jump distances, autocorrelation, force comparisons, short screens,
and elapsed time explain/nominate only; they cannot establish convergence.

Comparators: saved L=1 surrogate HMC; same-target RWM as a cheap mixing control;
the independent refined posterior grid as accuracy authority; prior Gaussian
geometry as a tuning baseline. Interior/tail retained mean/SD/CDF checks remain
unchanged; no tuning on their final comparison. No method ranking without
predeclared uncertainty evidence. No exact-likelihood or all-model success claim.

## Phases and bounded budget

1. Archive diagnosis: checksum-validate saved6250/chain warmup, nomination and
   state boundaries. Report per-chain acceptance, actual displacement in
   adapted coordinates, lag correlations, and the candidate selection record.
   CPU-only diagnostic exception, no new inference; allowance300CPU seconds.
2. Refreshed repair plan: use the diagnosis to specify the smallest target-
   specific tuning repair and focused regressions. Audit force/endpoint units
   if movement evidence suggests a mismatch. Any probe is descriptive only.
   Implementation, tests and bounded GPU diagnosis together cap6 hours.
3. Repaired public tuning: at most two fresh scientific attempts total, each
   preceded by a recorded configuration and budget. At most32 hours in total,
   reduced by actual earlier charges and the deadline. Independent fresh
   verification still owns admission; nomination metrics cannot replace it.
4. Qualified-only retained assessment: reserve10 hours, with cumulative calibrated
   checks. Refresh the master's next-phase plan after every phase/chunk.
   Infrastructure retries consume the same allocation. Stop for a true budget
   veto, not merely an expected early Rhat failure. Record unresolved models.

These are upper phase envelopes, not a promise each maximum fits. Their sum is
bounded by48 hours including diagnosis and tests. Refresh exact reservations
before each scientific launch. A new directory never resets charges or attempts.

## Defaults and assumptions

| Choice | Provenance / justification | Risk and smallest diagnostic | Status |
| --- | --- | --- | --- |
| Existing target, data and force controls | Preserves the question already authorized | Approximate endpoint; frozen reference comparison remains required | Frozen comparison scope |
| Existing public tuner | Registry supports typed position fields with exact endpoint correction | First-passing L policy can select diffusive moves; inspect archive and actual call chain | Baseline to repair |
| Existing mass and starts | Own-target discarded RWM warmup and adaptation | Poor geometry; per-coordinate jump/lag diagnostics | Warm start |
| Longer trajectories | Plausible exploration repair, not yet selected | Step adaptation can cancel the gain; bounded actual-target screen | Hypothesis |
| Readiness/verification/retained thresholds | Existing BayesFilter protocol and calibrated comparison | Expensive false starts; archive every chunk and enforce caps | Unchanged promotion criteria |
| Additional48 hours, same GPUs | Explicit owner authorization | Compute may still be insufficient; hard receipts/deadline | Hard upper bound |

Pre-mortem: a high acceptance rate could conceal tiny moves; a good short screen
could fail fresh chains; longer trajectories could merely shrink epsilon and
cost more. Diagnose each before interpreting an expensive failure as evidence
against surrogate HMC. Evidence is stored under new versioned
docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-* directories with command,
commit/source snapshot, environment, device, seeds, CPU/wall charges and results.

## Phase 1 finding and prospective phase 2 screen (2026-09-29)

Archive validation passed. Four-chain warmup acceptance was82.75%,93.78%,
87.02%,88.54%, but adapted-coordinate mean squared jumps were only.00368-.00452.
Lag1 correlations were.993-.998; lag100 correlations were.494-.867.
Both longer nomination candidates were marked unevaluated. These support
tiny-move/high-autocorrelation diagnosis; they do not establish its sole cause.

Before modifying the public tuner, run a diagnostic screen with the saved affine
geometry. Seven predeclared (L, epsilon multiplier) arms relative to the saved
epsilon are (1,1),(1,2),(1,4),(1,8),(2,2),(2,4),(4,2). All use the same existing
force/endpoint. Two32-transition replications use archived warmup states at4750
and6250 as paired starting states. Common random numbers across arms are
intentional for descriptive comparison; new seeds are disjoint from future
public tuning/verification/retention. These are discarded exploratory draws.
The arms test whether larger steps or more integration steps actually produce
larger accepted moves, or whether reduced acceptance cancels the gain.

Cap the entire screen at10800GPU/10800CPU seconds inside the6-hour diagnosis/
repair envelope. Archive every chunk, acceptance, health, displacement, and
wall time. Emit a checkpoint after each arm. Candidate numerical failure vetoes
that diagnostic arm; endpoint corruption or source mismatch invalidates the
screen. No short-chain Rhat or acceptance becomes a promotion criterion.
No statistical superiority ranking is inferred from two small replications.
Use these findings to refresh the public tuning repair prospectively.

Also evaluate the force and central differences of the *same endpoint* at the
four first-replication starts, with h=.001 and.0005. This is a diagnostic of
force mismatch only: derivatives of that finite endpoint do not become the
canonical analytical DCTR score or a replacement claim-bearing force.

## Superseded uniform-step hypothesis and continuing budget rules

Historical proposal, suspended before execution after the completed diagnostic:
the scoped candidate was a larger
step obtained through the existing public tuner. It does not alter the force,
endpoint, metric regularization, maximum trajectory grid, or sequential
convergence controller. The first hypothesis changes the dual-averaging
acceptance target from0.70 to0.50 and the nomination mean band to[0.35,0.65].
The second, used only after a healthy inconclusive first attempt, uses target0.35
and mean band[0.20,0.50]. Both use nomination per-chain band[0.05,1] and at most
two factor2 step repairs. These are scoped calibration hypotheses, not general
HMC recommendations or relaxed posterior criteria.

The lower adaptation targets test whether trading acceptance for larger moves
repairs the measured autocorrelation. The finite upper nomination bands prevent
an extremely conservative step from passing solely because acceptance is high.
The5 percent per-chain floor rejects nearly stationary nomination chains;
all-chain movement and modern Rhat remain mandatory in independent fresh chains.
Short nomination bands can be noisy or miss tail behavior; the diagnostic's two
starting groups and the unchanged fresh protocol are the early and downstream
checks. No statistical ranking follows from the short probe. If its completed
results contradict the proposed mechanism, revise these hypotheses before launch.

Seeds are (20260929,3101)/(20260929,3102) for attempt1 nomination/fresh checks
and (20260929,3201)/(20260929,3202) for attempt2. Probe seeds2100/2101 are separate.
The existing retained seed has not been used because no prior attempt reached
retention. Neither old warmup nor calibration draws become retained samples.

The first attempt is capped at20 hours of CPU/GPU allowance; both attempts
together at32 hours, subject to remaining allocation and deadline. Unspent
first-attempt allowance remains available to the second within the32-hour total.
At least10 hours must remain for a second attempt, after reserving10 hours for
retention. A phase budget exhaustion can trigger the second planned hypothesis;
allocation or deadline exhaustion cannot. Numerical/endpoint/health failures
remain continuation vetoes and never automatically authorize a mixing retry.

Implementation audit so far:26 orchestration tests passed, including receipt
carry-forward, separation from old allocations, deadline and retained reserve,
health-veto stopping, and automatic failed-first/qualified-second/retained
handoff. The exact planned start command matches the installed narrow allow
rule. No new permission rule is needed. Final consumer wiring, focused full
tests and GPU qualification remain to be checked before research launch.

The local mechanism is explicit in adapted coordinates. For one leapfrog step,
q' = q + epsilon*p - epsilon^2*f(q)/2. Thus the leading proposed displacement
variance is epsilon^2 when momentum has unit covariance. Doubling epsilon
predicts four times that leading variance, before acceptance. Also
Delta H = epsilon*p dot (gradient U - f) + O(epsilon^2): expand U(q') to first
order and use p' = p - epsilon*f(q) + O(epsilon^2) in kinetic energy. A mismatch
between the proposal field and endpoint gradient can therefore make acceptance
decrease even when the integrator is numerically healthy. This derivation
motivates measuring accepted movement jointly with acceptance; it does not
identify an optimal target acceptance or establish convergence.

The minimum second-attempt allowance was corrected during skeptical review: a
4-hour reservation cannot plausibly cover768 nomination transitions plus the
minimum2000 warmup and1000 verification transitions. The diagnostic baseline
measures about8.7 seconds per four-chain transition after compilation; adding
up to256 step-repair transitions places the minimum near10 hours. This is a
feasibility screen, not a runtime guarantee. The guard still enforces exact
remaining CPU/GPU charges and deadline, and later trajectory candidates cost more.


## Prospective fixed step-mixture diagnostic, 17:33 UTC

The proposed .50/.35 target-acceptance repair is now suspended. L1 at four and
eight times the saved step stalled the same chain in the second starting group.
L2 at twice the saved step also stalled that chain, with expected acceptance
9.63e-7. All other numerical checks passed. Larger uniform trajectory length
therefore does not resolve the observed region-specific mismatch.

Evaluate an optional fixed, state-independent mixture of step sizes, with
equal probabilities and independent choices for each chain and transition.
Diagnostic hypotheses use multipliers (1/8,1) and (1/4,1), with base step eight
times the saved step and L=1. These ratios come from the measured small-step
escape and large-step movement probes; they are hypotheses, not defaults.
Use the same two archived starts and 32 discarded transitions per replication.
Seeds (20260929,2200) and (20260929,2201) are separate from prior probes.
Cap this additional GPU diagnostic at1800 CPU/GPU seconds, still within the
6-hour diagnosis/repair envelope. Preserve all health and per-chain movement
diagnostics. Zero movement vetoes a diagnostic arm. No short run establishes
convergence, superiority, or posterior correctness.

For each fixed scale s, run the existing general neural-force kernel in
z=q/s, pulling the proposal force back as f_z(z)=s*f_q(s*z), then map back.
This is precisely the existing leapfrog with epsilon_s=s*epsilon in q.
The diagonal chart determinant is constant within the transition and cancels
in its endpoint energy difference. Conditional on s, the same reversible,
volume-preserving proposal and MH correction apply. If K_s preserves pi and
w_s are fixed, integral pi(dx) sum_s w_s K_s(x,A)
= sum_s w_s pi(A) = pi(A). No state-dependent weights or runtime retuning
are introduced. Random choices use disjoint stateless seeds from momenta/MH.
The force and endpoint functions themselves remain unchanged.

Implement this scoped candidate by wrapping the existing general kernel; do
not duplicate its leapfrog or MH calculation or edit the active probe's
shared sources. The public tune_hmc_kernel authority and its verified native
TensorFlow consumer remain required. The optional wrapper supports that
native lane only; the legacy host runner must reject it explicitly. It is not
a new repository default or a claim of support in every execution lane.

Skeptical pre-run audit: the main implementation risks are wrong force pullback,
missing endpoint mapping, state-dependent scale selection, shared chain
randomness, and accidental loss of health/provenance fields. CPU regressions
must check component parity against the existing kernel at the corresponding
epsilon, stateless replay, independent per-chain choices, exact identity when
disabled, binding identity, public-tuner/retained wiring, and rejection of the
unsupported host lane. GPU/XLA parity and the actual M13 probe are required.
The mixing diagnostic can nominate a candidate only; fresh warmup, verification,
ESS and MCSE-aware posterior checks remain the promotion criteria.


## Prospective retained sampling extension within the same campaign

The additional allocation should not inherit a terminal count of2000 when the
BayesFilter staged policy permits up to10000. Before any new retained samples,
qualify the same ten-function MCSE comparison at prospective cumulative looks
1000,2000,4000,6000,8000,10000. This changes the sampling ceiling and multiplicity
correction, not the practical margins, Rhat/ESS requirements, target or reference.
The existing two-look artifacts remain unchanged and retain their old scope.

Use the existing stationary reference-law coverage experiment (256 ensembles,
four chains, refresh probabilities1,.5,.2,.05) at the six looks with a new seed
(20260929,2710). The joint one-sided95% coverage lower bound must still be>=.95
in every condition. Six-look Bonferroni correction is applied before seeing
retained draws; reference allowances remain frozen. Cap this CPU-only diagnostic
at1800 measured CPU seconds and1200 wall seconds, within the6-hour development
envelope. It can run beside the serial GPU diagnostic. A failure blocks use of
the new interval scope, not the existing two-look scope or warmup diagnosis.

Skeptical audit: using the old critical value at more looks would invalidate
the predeclared family error allocation. Wire an explicit supported look policy
through coverage, assessment, and the retained consumer, and reject mismatched
calibration or off-schedule assessments. New retained runs may stop early only
for the existing equivalence/health criteria or resource limits. A10000 ceiling
is an upper bound, not a promise that10 hours of reserved compute can reach it.
The cheap reference-law experiment is a calibration diagnostic, not evidence
that an arbitrary M13 HMC chain is stationary.

The archived MCSE graph also enforces its old count list. Preserve that
reference-bound version and use an explicitly versioned six-look estimator
whose numerical body is identical; only its allowed-count guard changes.
An AST-body equality check and numerical parity at1000/2000 establish that
relation. New10000-draw shape/finite checks and the six-look coverage experiment
are required. This is an assessment-version extension, not a claim that the old
two-look interface already supported longer chains. Two early calibration
launches stopped at source/count guards before producing coverage evidence;
their costs are charged and their logs are preserved.

## Six-look assessment qualification (2026-09-29)

The prospective extension completed under a CPU-only guard at
artifacts/surrogate-hmc-m13-sequential-48h-development-01/coverage-extended-04.
All four reference controls passed the simultaneous coverage requirement.
At refresh probabilities1,.5,.2,.05, the256-ensemble success counts were
255,256,255,254; one-sided95% lower bounds .98160,.98837,.98160,.97561.
The last control deliberately creates strong serial dependence. These are
coverage-calibration results, not M13 posterior convergence evidence.

The frozen reference and its original assessment source remain byte-identical.
The new staged-assessment module copies the original numerical estimator with
only the allowed-count guard extended; AST and bitwise1000/2000 parity tests
passed, as did a10000-draw finite/shape check. The retention consumer passed a
real public-tuner/archive/continuation test through4000 cumulative draws.
New master prerequisites require the qualified six-look method, completed
calibration receipt, matching reference, and matching estimator source hashes.

Three failed calibration harness attempts were preserved and charged:
a correctly rejected reference-source change (restored), the original estimator
guard rejecting extended counts (explicit version added), and a caller passing
a string to a Path-only writer (invocation repaired). None supplied scientific
evidence; coverage-extended-04 is the completed authority. No thresholds were
changed in response to the coverage observations.

## Active public-tuning configuration after the completed mixture probe

Both fixed laws (.125,1) and (.25,1), chosen independently with probability.5
per chain and transition, passed both32-transition starting-state probes.
All finite/endpoint/energy/movement/fallback checks passed. The GPU/XLA
constant-chart step matched the unchanged general HMC kernel at its equivalent
step size, with source snapshots unchanged throughout the probe.

Attempt1 uses (.125,1): its smaller component preserves the small escape scale
that moved every chain in the original probe. If this healthy candidate remains
inconclusive under the planned limits, attempt2 uses (.25,1), with fresh seeds
and independent tuning. This order follows the escape-step rationale; it is not
a statistical ranking of32-draw jump distances. At the second starting group,
the most difficult chain had expected acceptance.1666 and.0739 respectively.
Its persistence remains the main risk. These probes provide no Rhat evidence.

Both attempts retain the original dual-averaging target.70, base-step warm start
.2, mass windows128/256, adaptation256, nomination128, maximum trajectory grid
L1/2/4, and existing prior-shrunk metric protection. The scoped nomination mean
band is[.50,.85], per-chain band[.05,1], with at most two factor2 step repairs.
The lower mean bound retains the previous nomination floor; the upper bound
prevents extremely conservative proposals from passing solely on acceptance.
The per-chain floor allows the deliberately mixed difficult chain to nominate;
all-chain movement, fresh readiness and posterior diagnostics retain their full
vetoes. These bands are calibration hypotheses, not universal HMC guidance.
The prior uniform-step target.50/.35 proposal is superseded and will not run.

Default audit: fixed equal weights avoid learning a state-dependent mixture;
two predeclared component ratios are viable hypotheses from the discarded
probe, not selected posterior criteria. The base epsilon must be independently
adapted on the same target. Changing the state-independent mixture changes
proposal efficiency only; both components use the identical MH endpoint.
The temporary constant chart uses the existing force and transition kernel,
and its constant Jacobian cancels in MH differences. Per-component parity and
public tuner -> artifact reload -> retained runner tests check that call chain.
Wrapper identity binds the actual underlying force/endpoint source closure,
the mixture law and wrapper source. Legacy (1,) is exact identity.

Preflight verifies the actual selected binding equals the completed GPU probe,
then revalidates the frozen posterior reference before long tuning. Retention
reconstructs the identical mixture from the public tuning report and reloads the
public manifest against that binding; a different mixture cannot reuse it.

Skeptical audit: the same-target RWM remains a comparator, and independent
refined-grid agreement remains the accuracy test. Short-probe acceptance and
jump sizes nominate only; no target, holdout, precision margin or convergence
threshold changes. Mixed small steps might still leave a difficult chain stuck,
so every250-draw health check and the warmup/verification maxima remain active.
A healthy phase-budget exhaustion can use the second planned attempt under the
remaining allocation; it is not mislabeled a research-direction rejection.
Full focused checks and actual M13 preflight must pass before scientific launch.

Implementation audit and exact launch command: surrogate-hmc-m13-48h-development-review-2026-09-29.md.


Infrastructure repair before first scientific launch: master48h-01 stopped
before any worker because it passed the campaign48-hour ceiling to a shared
guard whose maximum is24 hours per worker. The master now caps each worker at
24 hours while retaining the total48-hour allocation, first20-hour tuning cap,
aggregate32-hour tuning cap and10-hour minimum retention reserve. A regression
exercises actual reserve_job for every phase. No scientific attempt was used.
The new versioned root is surrogate-hmc-m13-sequential-48h-02, with the same
start command and budgets except that output suffix. The prior record remains.

## Energy-veto repair, 2026-09-29 03:20 UTC

Attempt1 in sequential-48h-02 stopped after250 warmup/chain, with no verification
or posterior draws. Archive analysis health-veto-analysis-02 checksum-replays the
only fresh failure: chain1, transition56, rejected deltaH1211.0069 > bound1000.
All accepted energy errors were<=4.139; states, endpoints, movement, divergences
and force-fallback checks passed. Nomination maximum|log acceptance ratio|158.979
passed the same numerical bound retrospectively, but nomination did not enforce
that bound prospectively. Adding it would not have caught this particular event.

The current kernel remains rejected. This is finite proposal instability in a
candidate, not evidence of target corruption or rejection of surrogate HMC.
Automatic blind attempt2 remains forbidden. A prospective energy-step repair
may use the second and last scientific attempt after the following screen.
This replaces the original second-mixture hypothesis; it changes no target,
force, data, hardware, promotion threshold, attempt cap or total allocation.

Screen the SAME mixture(1/8,1) and saved geometry at half the failed base step:
0.949153975512196/2=0.474576987756098. This factor is the already declared
bounded factor2 tuning repair, now prompted by fresh energy evidence. Use two
discarded64-transition GPU/XLA replications: original initial states/seed,
and the four states just before the failure with seed(20260929,4101).
All original health checks, including energy<=1000 and all-chain movement, must
pass. Cap the screen at1800 CPU/GPU seconds. Its report preserves tensors,
checksums, binding, hardware, memory policy, source hashes and wall time.
The original250-draw archive is the comparator; no posterior comparison is used
to select the repair. Short healthy chains only nominate this repair.

If the screen passes, add an optional nomination step ceiling and energy bound
to the existing public TensorFlow tuner. Apply the ceiling after adaptation and
during bounded verification repair; enforce the existing energy bound in the
nomination checks too. Neither changes the force, endpoint or HMC integration.
The ceiling is a scoped hypothesis, not a repository default; absent settings
must preserve the original route. CPU regression checks must exercise cap,
energy rejection/repair, config roundtrip and the real public consumer.

The fresh second scientific attempt uses the same mixture, independently
adapted geometry/epsilon with that ceiling, and already reserved independent
attempt2 seeds. Warmup starts from the declared dispersed starts, never by
continuing the rejected chain. Full warmup, verification, retained/reference
criteria and the ten-hour retention reserve remain unchanged. Another health
veto rejects this last candidate; no third scientific attempt is authorized.

Skeptical audit: a short screen can miss rare energy excursions; the full fresh
guard remains required. Smaller steps may restore stability but mix too slowly;
readiness and retained ESS remain the decision criteria. No convergence, speed
ranking or posterior claim follows from the diagnostic. No acceptance-band or
energy-threshold relaxation is allowed. This is a repair within the master,
with explicit phase refresh, rather than a new research direction.

### Final-source repair screen and source review

The optional native nomination guards change a dependency hash in the
repository-issued runner binding, even though fixed-kernel integration does
not change. The repair must therefore carry a newly issued binding and a
completed GPU screen under those final sources. Old probe identity cannot be
restamped. Probe03 repeats the same two64-transition checks and records the
old/new binding difference, requiring unchanged endpoint/force identities and
unchanged integration/fixed-kernel/affine code by executable AST comparison.
The only allowed runner-dependency change is the reviewed native tuner.

Reserve at most3000 GPU/wall seconds and3600 measured CPU seconds for this
final-source screen, inside the existing6-hour development envelope and total
allocation. The first original-source replication took950 wall seconds, so the
earlier1800-second two-replication cap may end before its second archive is
written. An incomplete probe is diagnostic evidence only and cannot admit the
repair. Preserve and charge every such attempt. Probe03 must complete both
replications; another incomplete screen is a resource failure, never a pass.

Skeptical audit: adding an energy check does not prevent a previously unseen
outlier, and halving epsilon can worsen mixing. Keep the declared energy1000,
warmup, modern R-hat, ESS, posterior-reference and no-fallback checks unchanged.
The real public tuner and reloaded retained runner must be tested with the
optional guards enabled. The master must inspect attempt2 during resumed
preflight and require the completed final-source screen before sampling starts.
No inference is made from a high acceptance rate or the short screen alone.

The diagnostic freezes attempt1's affine geometry to isolate the step change.
Attempt2 adapts its geometry independently, so the diagnostic does not certify
attempt2's energy behavior. The ceiling is a hypothesis in normalized affine
coordinates; a different fitted covariance can still produce a health failure.
Only the fresh public verification and subsequent retained assessment can
qualify the newly fitted kernel.

The detached development coordinator
artifacts/surrogate-hmc-m13-sequential-48h-development-01/continue_energy_repair_01.py
runs probe03 within those reservations and automatically starts the existing
master only after a completed healthy screen and unchanged reviewed sources.
It preserves phase status in energy-repair-continuation.json. The exact master
command, run from the repository root, is:
```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_recovery.py start --allocation 48h --cpu-budget 172800 --gpu-budget 172800 --output /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-03
```
The master then owns tests, GPU qualification, attempt2 preflight, full tuning,
and qualified retained sampling, refreshing next-phase-plan.json between phases.
Screen failure blocks the full attempt; no silent numerical threshold change,
extra scientific attempt, or renewal of the48-hour deadline is permitted.

### Saved control-selection compatibility under the nomination repair

The complete test run correctly rejected the saved control selection after its
native-tuner dependency hash changed. The control consumer calls only the
native affine transform, fixed transition loop and tensor writer; the six
modified/new nomination definitions are unreachable from it. Preserve the
original selection, hashes and all control evidence. Permit only the reviewed
old/new native hash pair after comparing every remaining module statement and
definition by AST, then recompute the saved selection decision. Reject any
other force, endpoint, selection, loop or source change. Test both actual
control consumers against the saved and current transition loops on identical
CPU/XLA Gaussian inputs. This compatibility proof avoids falsely labeling the
new sources as the old ones; it does not change the selected LEDH controls.

Pre-launch review, 2026-09-29: focused compatibility checks passed17/17,
including both real control consumers under old/current loops. The full master
regression had171 passes and one dependency-compatibility failure, now repaired
and covered by these17 checks. The master repeats its entire test phase before
full tuning. A prior targeted repair suite had46 passes plus two test-fixture
failures; both corrected cases then passed. None is posterior evidence.
The coordinator mechanics check confirms failed screens cannot launch tuning.
Current cumulative charge: CPU24429.854637/GPU23015.114224 seconds; remaining
148370.145363/149784.885776 seconds respectively, also bounded by the original
2026-09-30T16:27:39Z deadline. Exactly one full scientific attempt remains.
