# NeuTra scientific campaign recovery

## Active checkpoint: October 6 revised criterion; random tests complete

The owner explicitly authorized repairing the forward criterion and continuing.
Active decision rule: `bayesfilter_neutra_forward_warm_start_then_final_v2`.
Forward suitability requires reloaded finite calculations, a finite complete
1,000-point probe, the existing identity-Gaussian gross-fit guard and mass-error
screen, and each component retaining at least half its reference mass. Forward
feature z is explanatory. The final RKL full accuracy screen is unchanged.
The factor-two coverage bound is a documented heuristic for these synthetic
targets, not an HMC/q20 guarantee. Plan and skeptical review:
`bayesfilter-neutra-forward-warm-start-criterion-repair-2026-10-06.md`.

Focused verification passed 41 criterion/controller tests and five numerical
call-chain tests. A test-fixture tuple/list mismatch was fixed; costs preserved.
All six calibration and six fixed cases pass the revised pair rule, keeping
the selected `smc/native_0`, 16,384 forward updates and .0001 / 256 reverse
recipe frozen. The old forward failures and controller decisions are preserved;
their reclassification is retrospective. See
`forward-reverse-criterion-revision-v2.json` and `*-legacy-v1.json` in the campaign.

The master is COMPLETE and IDLE after execution from `source-fa00123930f45395`
using `bash scripts/run_neutra_scientific_campaign.sh forward-reverse`.
All 12 randomized geometry/seed cases passed the teacher, warm-start and final
RKL screens prospectively under v2. All random forward maps also passed the old
full diagnostic. The minimum forward component-mass ratio was .9182; final
feature maxima ranged from 1.831 to 3.979. No old numerical work was repeated,
no random case was retuned, all training was finite and no update was clipped.
Tool sessions 42078 (master) and 26512 (read-only monitor) ended successfully.
Read `state.json` and
`forward-reverse-state.json` under
`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/` for live state.
Remaining: 111,947.28 GPU-process / 104,304.11 CPU-core seconds (31.10 / 28.97
hours). Random-stage cost: 5,905.96 GPU-process / 6,994.74 CPU-core seconds.
All prior attempts and the criterion-test fixture failure remain charged.

The r2 terminal audit passed 936 artifact hashes, 1,439 source hashes and all
78 finite complete probes across 48 workers. The 24 pre-revision result hashes
still match. The frozen controller, run against a copy of the completed state,
launched zero workers and preserved live state. Both local/shared accounting
match each completed worker exactly once; no charge is pending. Saved checks:
`forward-reverse-terminal-audit-r2.json`, its adjacent `.py`, and
`forward-reverse-terminal-accounting-r2.json`. Preserve the previous r1 audit.

Result note: `bayesfilter-neutra-naf-forward-reverse-results-2026-10-06.md`.
The authorized queue is finished; a resume must stay idle. Passing these
two-dimensional density/coverage screens does not establish q20, TF32 or HMC
readiness. Final 1,000-point probe median residuals range from .0967 to .3885,
but p99 residuals range from 10.38 to 67.65. Localized large score residuals
remain; downstream sampling is the next separate validation question, not
permission to call the maps uniformly whitened. No method ranking is supported.

## Historical checkpoint: October 6 fixed-stage results and prerequisite review

The master has finished 24 workers (six calibration and six fixed-target
teacher/fit pairs) and is idle at `forward_reverse_fixed_screen_failed`.
**All six final RKL maps passed**, but two forward-only checkpoints failed the
feature screen (unwarped seed 51 z=5.555; warped seed 53 z=7.786; threshold 5).
The explicit rule requiring BOTH checkpoints to pass blocks randomized tests.
Do not describe this as RKL failure, missing mode mass, or a numerical crash.
All teachers passed, every update was finite, and no clipping occurred.

Selected recipe: `smc/native_0`, 16,384 forward updates, then 256 reverse updates
at .0001. The terminal audit passed 528 artifact hashes and 54 complete finite
1,000-point probes. See
`bayesfilter-neutra-naf-forward-reverse-results-2026-10-06.md` and campaign
`forward-reverse-terminal-audit-r1.json`. Randomized geometries are untouched.
Remaining: 117,853.24 GPU-process / 111,393.73 CPU-core seconds
(32.74 / 30.94 hours).

Next: review whether the intermediate forward feature gate is appropriate for
an approximate warm start that RKL is intended to correct. Changing the declared
criterion after these outcomes must be explicit and cannot retroactively make
these holdouts untouched. Current `resume` preserves this stop; do not relaunch
the same numerical work or silently change a threshold. This is a scientific
prerequisite decision, not a request for an infrastructure retry. The six final
maps remain viable under exploratory screens; no HMC/q20 claim follows.

## Historical checkpoint: October 6 NAF forward/reverse campaign executing

The owner added another 24 GPU-process / 24 CPU-core hours and authorized review
and execution. The program review passed; it also made the existing .25-nat identity-map
cross-entropy-difference screen explicit in the plan. New cumulative local caps are 178,800 / 184,800
seconds, with all previous attempts charged. Allocation event:
`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/allocation-extension-2026-10-06.json`.

The master is RUNNING from `source-d44adeb70522aa32` via
`bash scripts/run_neutra_scientific_campaign.sh forward-reverse` (tool session
87531 at launch). Its first native SMC teacher (two-mode development geometry,
seed 601) passed both independent-bank screens and took 19.60 wall / 51.06
CPU-core seconds. The first full GPU fit completed in 730.81 wall / 814.85 CPU-core seconds;
its forward map and five of six reverse endpoints passed. The .0003 / 1,024
endpoint failed the feature screen and remains rejected. The second two-mode
fit (seed 607) is active on GPU 2. Subsequent GPU fit ceilings are 1,097 wall /
1,223 CPU-core seconds, from measured cost times 1.5. Do not start a competing master. Read `state.json` and
`forward-reverse-state.json` under the campaign root for current worker/state.
The budget before this first full pair was 124,430.72 GPU-process seconds and
119,234.95 CPU-core seconds. Monitor phase outcomes and repair infrastructure
failures within the unchanged contract and budget. Result note:
`bayesfilter-neutra-naf-forward-reverse-results-2026-10-06.md`.

## Historical checkpoint: October 6 readiness before execution

The scalar-nonlinearity study is complete: all 87 workers and terminal audit
passed execution/integrity checks; 30/32 nonlinear confirmation fits passed
their distribution screen, and all four directional density contrasts passed
the declared sign test. See
`bayesfilter-neutra-nonlinearity-attribution-results-2026-10-06.md` for the
scientific interpretation and the two rejected fits. The monograph now contains
the derivations, study results and limitations and has compiled successfully.

NAF (`huang_dsf` / `author_cmade`) is now the owner-designated default for this
training study. The active plan is
`bayesfilter-neutra-naf-forward-reverse-master-2026-10-06.md`; readiness and
validation are recorded in
`bayesfilter-neutra-naf-forward-reverse-readiness-2026-10-06.md`.
The master is `scripts/run_neutra_scientific_campaign_master.py`, reached through
`bash scripts/run_neutra_scientific_campaign.sh forward-reverse`. `resume` follows
this new phase after the successful smoke; it no longer re-enters the old IAF
sequence. No full native-teacher calibration or final campaign has launched.

Next: price/calibrate native CPU teachers, GPU NAF forward KL and pure reverse
KL on development targets with seeds 601/607/613; freeze a passing recipe;
require all fixed warped/unwarped cases to pass; then expose reserved random
two/three-mode geometries 1103/1104/2103/2104. SMC and AIS are implemented;
FAB/Gabrié/AFT/CRAFT remain explicitly deferred for the gaps in the active plan.
Do not confuse this scoped procedure with an all-six-method comparison or q20
validation. The run retains the FP64 diagnostic exception.

Validation: the initial 42-check suite passed, the native CPU tracing repair
passed seven checks, and the final recovery/controller suite passed 22 checks
(overlapping checks, not 71 distinct tests). The first native smoke failed during
concurrent derivative tracing; tracing the shared immutable kernel before the
CPU threads repaired it. The retry passed native CPU generation, batched
GPU/XLA forward/reverse updates, checkpoint reload and three complete finite
1,000-point probes. Every attempt and subsequent check is preserved and charged.

Current master state is `forward_reverse_prepared`, with no active worker.
Remaining: 38,030.72 GPU-process seconds (10.56 hours) and 32,834.95 CPU-core
seconds (9.12 hours). This is a ceiling, not a full-campaign runtime estimate.
The first complete calibration pair sets measured reservations for later stages.
Use the existing shared lock/accountant and fresh output directories under
`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/`.
GPU 2 was used for the smoke; inspect availability before a new launch and
preserve unrelated GPU 0/1 workloads. Preserve the shared dirty worktree on
`preserve/shared-main-before-fab-20260926`; no commit or push was requested.

## Historical checkpoint: October 5 scalar-nonlinearity attribution

The owner-authorized attribution campaign is RUNNING on GPU 2.
Plan: `bayesfilter-neutra-nonlinearity-attribution-plan-2026-10-05.md`.
Command: `bash scripts/run_neutra_scientific_campaign.sh attribution`.
Campaign root remains `artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/`.
This is the current task; native-teacher transfer remains later work.

The shared core now has an explicitly diagnostic affine DSF readout with the
same cMADE arrays/head sizes, plus a diagnostic Hoffman-style conditioner with
asymmetric component initialization. Ordinary IAF/NAF default semantics and
existing artifact hashes are preserved. The experiments are FP64 GPU/XLA
references with verified memory growth; external sample generation runs on CPU.

Completed: 136 full checks, 18 expanded derivative/controller checks, and 13
queue/controller checks. A finite-difference check needed tighter reference
inverse precision and a step sweep; no training tolerance was changed. All
failed test attempts are preserved as pending charges for the next shared-lock
settlement. Gaussian affine/nonlinear fits both pass. The one-dimensional
mixture has nonlinear KL 0.0009864, 99% conditional MC upper limit 0.0013204,
below the mathematically proven Gaussian lower bound 0.4581454; affine fit KL
0.45949. This proves the restricted architecture obstruction and gives a
checked finite-NAF example, not a multivariate necessity theorem.

Four two-mode pilots are complete. KL benefits (affine KL minus nonlinear KL)
are .022291/.015860 for cMADE seeds 409/419 and .020625/.013753 for the
diagnostic Hoffman conditioner. All four affine fits pass the shape screen.
Three nonlinear fits pass; cMADE seed 409 fails (z=5.378). Initial density
differences slightly favored the affine controls, by .00008--.00072 nats.
Preserve the distinction: lower density loss is not map promotion. These are
pilot findings, not independent confirmation. Three-mode pilots are now active.
Eight independent confirmation pairs per target/conditioner, canonical capacity
and rate controls, longer affine training and conditional fresh-target transfer
are queued. They have not yet completed.

Controller PID 2414670 is running the initial immutable source
`source-80e6c9addb002ff7`; session 91712. A bounded supervisor, PID 2464868,
session 94229, waits for that pilot controller to finish before resuming the
reviewed queue from live source. It corrects the next reservation using full
pilot timings and adds all sensitivity/longer-training controls. It never
interrupts an active numerical worker and stops on a failed worker for
inspection. Read `attribution-supervisor.json`, `attribution-state.json` and
`state.json` for live truth; do not start a competing master.

At this checkpoint 77,551.36 GPU-process / 79,240.09 CPU-core seconds remain,
before the active job and pending follow-up tests. Earlier costs are retained.
The supervisor will settle pending tests under the same shared lock. GPU 0/1
have unrelated workloads; GPU 2 shares only the remote desktop process at launch.
Do not stop unrelated jobs. No commit/push request; preserve shared dirty work.

## Historical checkpoint: completed development stage before attribution

Owner authorized the revised source-fit remedy and added 24 GPU-process / 24
CPU-core hours. Cumulative limits 92,400 / 98,400 seconds; all prior charges
remain. Branch `preserve/shared-main-before-fab-20260926`; preserve shared dirty
work. No commit/push request. Campaign root:
`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/`.

At that checkpoint no worker was active. The six-fit NAF comparison and five-map downstream validation
completed, followed by terminal audit and an actual no-relaunch resume. Five
final fits pass and all five pass shared sequential HMC, precision and fresh
reference; three-mode seed 11 failed shape screening. All 98,304 training
updates finite, three clipped; twelve standard 1,000-point probes finite.
Score tails and finite extreme proposal errors remain. Canonical IAF is not
replaced: this is the separately configured optional NAF and exact-teacher
scope. Previous 42 canonical IAF endpoints all failed shape checks.

Downstream selected kernels (two-mode seeds 11/37/73; three-mode 37/73) required
2,000 / 2,000 / 6,000 / 4,000 / 4,000 retained draws per chain after 2,000 warm-up.
Three-mode seed 73 first kernel failed nonfinite proposals in warm-up; the next
independently verified kernel passed. It produced no invalid retained estimate.
Full checks: 131 passed; terminal resume/controller checks: 17 passed. Current
source is later than completed immutable `source-a67ea2ecb0523cba` only for
plan/route clarification and no-relaunch terminal reporting.

Repairs preserved: scalar NAF inverse internal half-tolerance stopping
(unchanged final tolerance); exact failure replay and finite differences;
responsibility tail-ESS rounding defect repaired locally with stable log-odds
for ordered diagnostics. Original mode probabilities remain in precision and
reference checks. Mathematical justification and positive/negative exact iid
controls are in `bayesfilter-neutra-development-hmc-2026-10-05.md`.

Evidence: `development-terminal-audit-r1.json`, `development-hmc-result.json`,
`representation-terminal-audit-r1.json`; read the current summary at the top of
`bayesfilter-neutra-source-fit-results-2026-10-05.md` for decisions and limits.
All pending charges are settled exactly once. Current remaining allocation:
81927.755904 GPU-process / 84631.889878 CPU-core seconds.

NEXT: specify and price the native-teacher transfer phase under the authorized
proposal. Do not claim the whole campaign is done. The existing native-bank
harness has unresolved FAB alpha-two prerequisites, partial Gabrié/AFT/CRAFT
source equivalence, and GPU external sample generation that needs a policy-
compliant multicore CPU lane. Keep native sampling quality distinct from
student fitting and corrected HMC. Frozen unseen targets 1103/1104/2103/2104,
q20 transfer, TF32/default promotion and comparative ranking remain untested.
No mandatory permission is outstanding for authorized local repair/research;
write the next concrete evidence contract before new serious jobs.

The same wrapper remains approved. `resume` now audits completed results
without numerical relaunch; it will not invent the next research protocol.
`bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh status`
reports current state. GPU runs require trusted permissions, GPU 1 and memory
growth. GPUs 0/2 belong to unrelated work. There is no persistent external AI
supervisor between turns; the current phase was actively monitored to completion.

## Previous checkpoint: selective remedy revision

The current task is the owner's request to adopt reasonable parts of Claude's
review. The revised proposal is
`bayesfilter-neutra-source-and-fit-remedy-plan-2026-10-04.md`; its assessment is
`docs/reviews/bayesfilter-neutra-fit-remedy-review-assessment-2026-10-04.md`.
The next scientific task is P0: classify the eight earlier cases from their raw
results and price the complete first matched fitting comparison. Then repair
the objective-branch controller (P2) and conduct a small exact-teacher study
(P3). A bounded original-author baseline (P1) is separate; full adaptive-port
reproduction need not delay independent learner diagnosis. Persistent poor
density fitting can trigger a representation comparison (P4).

Only documentation was revised. No new training, implementation, scientific
threshold, backend/default change or compute allocation follows from this update.
The last measured remaining balance is 3,895.501185 GPU-process seconds and
8,660.105809 CPU-core seconds; the full remedy has not yet been priced. The
original review and mathematical assessment are preserved. Pre-edit proposal
and memo copies, the revision audit and verification are under
`artifacts/neutra-remedy-review-incorporation-2026-10-05/r1/`.

## Existing campaign and evidence history

The active program is `scripts/run_neutra_scientific_campaign_master.py`, invoked
through the exact wrapper `bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh`.
Verbs are `check`, `preflight`, `price`, `run`, `resume`, `status`; the twelve
direct/shell forms are installed in the user rules. Managed reviewer permissions
still apply. GPU verbs require trusted execution. There is no new token gate.

The cumulative allocation remains 6,000 GPU-process and 12,000 CPU-core seconds
across all scientific revisions. Never reset this counter when starting a repair.
The older warm-start shared ledger is also charged. Preserve all prior artifacts.
The active root is `docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r3`.
Its `state.json`, `matrix.json`, `next-phase.json`, `pricing.json` and terminal
audit are the recovery authorities. A terminal resume audits without relaunching
workers; a new scientific repair must name its new output root and carry costs.

R1 ran an invalid uncertainty screen and a smaller training recipe than its plan.
Its negative method/map labels must not be used as algorithm rejection evidence.
See `campaign-r1/terminal-scientific-audit-r1.json`. Its full costs are retained.

R2 fixed the independent-population uncertainty calculation, fixed frozen-map
reload/evaluation, implemented forward training followed by shared-core RKL,
and required the same complete recipe to pass both development targets.
All four exact-teacher calibration banks passed, but all student endpoints
failed the local distribution screen. No clipping occurred. RKL also degraded
component mass agreement for the larger recipe. This is a valid rejection of
these short recipes; no native teacher or final generalization cell ran in r2.
Its terminal artifact audit passed. Status is correctly under-calibrated.

The r3 repair tests width 32/4,096 and width 64/8,192 forward updates, with
heldout learning histories at geometric rungs, then 256 RKL updates. It retains
the same canonical architecture, numerical screens, GPU, reference arithmetic
exception and cumulative budget. Final random targets 1103/1104 and 2103/2104
were not observed in r2. They must remain outside calibration.

R3 is now terminal: both rungs ran on both development targets; all exact
teacher banks passed, all four final maps failed. The terminal audit and
audit-only resume both passed, with no worker relaunched by resume. The final
source gate passed 75 tests. Total charged cost across r1/r2/r3 is 2,104.4988
GPU-process and 3,265.1359 CPU-core seconds, leaving 3,895.5012 / 8,734.8641.
Current results and decision tables are in
`bayesfilter-neutra-scientific-campaign-results-2026-10-04.md`. No final random
target was evaluated under the corrected protocol. Further research needs an
explicit numerical-plan extension; do not treat the terminal state as success
or quietly relaunch the rejected recipe.

FAB is blocked by unresolved nonlinear alpha-two tail applicability; a row for
FAB is not evidence it executed. The Gabrié arm is a fixed-map global-MH/MALA
control, not a full adaptive-controller reproduction. AFT/CRAFT use the existing
local stage-mechanics adaptations. No result from this campaign establishes
upstream feature equivalence, HMC readiness, posterior correctness or q20 transfer.

## October 4 root-cause investigation

The user requested a code-and-math explanation after the terminal campaign.
See `bayesfilter-neutra-scientific-root-causes-2026-10-04.md` and its diagnostic
plan. All eight saved r3 forward/final checkpoints were checked using the exact
execution source, CPU-only FP64 reference evaluation, analytic mixture moments
and 32,768 new common latent draws. Density, inverse, log determinant, target
score, transformed score and parameter derivatives passed the local checks.
No model was trained or changed.

The forward maps have excess probability outside all component ellipses: 8.39%
for width-64 two-mode and 21.62% for width-64 three-mode, against an exact target
upper bound of 1.83%. In the wider three-mode run, the RKL finish reduced the
second component responsibility mass from 21.57% to 2.24%, versus true 24.96%.
An exact augmented-distribution KL decomposition shows that the conditional
shape-cost reduction exceeded the wrong-weight penalty. Thus RKL reduced its
actual objective while worsening coverage. This does not establish that RKL
always fails. All forward endpoints already had shape errors; simply undoing
RKL is insufficient.

The controller switches objectives unconditionally after fixed update counts.
The wider recipe retains three stages and confounds width with number of
updates. Learning histories do not establish a settled forward optimum;
optimization versus finite family remains unresolved. There was no gradient
clipping or observed strong cap saturation in these corrected runs. A future
repair should use matched calibration and protect coverage across the objective
switch, without declaring the IAF family impossible or teacher methods failed.

Artifacts: `artifacts/neutra-scientific-2026-10-04/root-cause-r1/` contains all
completed numerical checks and a preserved plotting-dependency failure;
`root-cause-r2/` reuses them and renders the saved map densities with the already
installed base-environment Matplotlib. No package installation occurred. Both
attempts are charged to campaign-r3 and the shared older ledger. The additional
74.758258 CPU-core seconds leave 3,895.501185 GPU-process and 8,660.105809
CPU-core seconds. The campaign remains terminal and under-calibrated.

## Original-code evidence audit

The subsequent user question asked whether the original literature methods have
the same fitting failures. See
`bayesfilter-neutra-upstream-training-evidence-audit-2026-10-04.md`.
No matched complete original-versus-port training comparison on the current
mixtures was found. FAB's original JAX operations did execute in the September
26 equivalence work: the FP64 stress comparison passed 1,804 checks, while the
final compiled-callback TF32 comparison failed seven declared gradient checks.
These bounded tests do not establish eventual training quality. Gabrié's
original sampler primitives executed in controlled fixtures, but its original
RealNVP training controller was not reproduced in the inspected records.
AFT/CRAFT have source and local mechanics checks, not complete original-controller
training parity on these targets.

Gabrié Appendix G.1 reports a residual bridge even in its successful mixture
example; it also demonstrates failures from missing initial modes and nonmixing
local-only sampling. Its architecture, sampling cadence and training procedure
differ materially from our common IAF recipe. Corrected sampler success and
accurate raw-flow fitting are distinct claims. The current exact-teacher failure
rejects our fitting recipe, not six unexecuted native methods. Derivative parity
does not eliminate architecture, optimization or controller-reproduction gaps.
The discriminating comparison must separate original training, matched port,
IAF substitution and the forward-to-RKL objective switch. This source/evidence
audit launched no numerical workers; the recorded campaign budget is unchanged.

The follow-up literature assessment is
`bayesfilter-neutra-literature-remedies-2026-10-04.md`. It checks combined
example/energy objectives against Noé's paper and original notebook, and
separates representation, fresh-particle, rare-region and sampling-correction
remedies. Do not present a combined loss as an untried cure: the October 2
controlled campaign already ran joint and joint-continuation arms without
reaching fresh posterior confirmation. That historical local result does not
establish a matched reproduction of the author's complete successful method.

## Monograph, remedy proposal and Claude handoff

The user next requested documentation and a proposal for Claude's substantive
review. The existing controlled-training chapter is preserved, with an added
`docs/chapters/ch26f_neutra_literature_remedies.tex` include covering the exact-
teacher results, responsibility-KL decomposition, original-code evidence and
source-grounded remedies. The proposed phases are in
`bayesfilter-neutra-source-and-fit-remedy-plan-2026-10-04.md`. The handoff is
`docs/memos/neutra-fit-remedy-claude-handoff-2026-10-04.md`; it requests full
diagnosis, skeptical plan review and a stronger alternative when justified.
At that handoff, Claude's review was pending; it is now recorded below.
No new numerical campaign was launched for the documentation and handoff.

The focused document and full monograph build under
`artifacts/neutra-literature-remedy-update-2026-10-04/r1/`. A protected chapter
baseline, compile logs and rendered-page inspection accompany the update.
The proposal still needs measured pricing and concrete final accuracy bands
before research execution; existing coarse screens are not promoted to
posterior correctness. It does not assume a new compute allocation.

Claude's review has now been received at
`docs/reviews/bayesfilter-neutra-fit-remedy-claude-review-2026-10-04.md`.
The response assessment is
`docs/reviews/bayesfilter-neutra-fit-remedy-review-assessment-2026-10-04.md`.
It accepts measured pricing, detailed October 2 diagnosis and a smaller initial
fitting study, but corrects the review's regional-KL algebra and oscillatory-KL
proof, rejects the circular representation trigger, and flags unmeasured and
inconsistent runtime/budget estimates. The fixed-physical geometry objective
must not be conflated with a moving-target-score objective requiring a target
Hessian. That assessment preserved both the review and the then-current plan;
the October 5 revision now incorporates its justified changes into the proposal.
The original review is unchanged. Its new thresholds and allocation request
have not been adopted, and no new experiment was run for either the assessment
or this revision.
