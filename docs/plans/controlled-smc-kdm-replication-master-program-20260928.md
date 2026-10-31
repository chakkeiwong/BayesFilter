# Authoritative master: controlled SMC and KDM/IWSG replication

## Active program after main synchronization — 23 October 2026

Continue on `ledh-refactor-with-policy-fix` in
`.localresources/worktrees/ledh-iapf-execution-20260922`. The LEDH source,
tests, paper and compact evidence were committed as `9bc3e7ec7`, merged into
main as `90515228c`, and brought back to LEDH. The merge retained main's
batched Gaussian/transport optimizations and the guarded guide implementation.
There were no textual conflicts. Focused CPU regressions passed (97 before
integration and 56 on the merged batch path), and the oracle commit hook
passed. These are engineering checks, not fresh N1008 efficacy results.
See the [integration record](controlled-smc-kdm-main-sync-and-refresh-20261023.md).

The active research question is whether freshly fitted controlled-SMC-derived
and SIR neural-KDM proposals improve the terminal moments of the actual
corrected LEDH consumer at N>=1008. The
[reviewed restart plan](controlled-smc-kdm-n1008-restart-20261023.md) replaces
all older execution instructions below. Its next step is to remove hard-coded
N64 scope assumptions from the campaign callers and propagate a single explicit
scope through training, evaluation, continuation, reset design and manifests.
The source audit also found that the current neural trainer sums row gradients
inside a TensorFlow loop: main's separate batched optimization does not prove
that this guide consumer uses a batch-native target.

After the source/configuration repair, check a real N1008 training update and
analytical-score consumer before sizing a new campaign. Use fresh simulation
labels, calibration, initialization and heldouts; compare plain LEDH, a
current-observation proposal, a simple quadratic future-message proposal,
controlled-SMC-derived guidance and neural KDM. Train actual terminal first
and raw-second moments, not a model score or a guide-only proxy. Paired heldout
loss and conditional heuristic comparisons answer the moment question; score
accuracy requires a separate qualified reference and downstream evaluation.

Retain OT(1,5), the shared Contract-E/full-correction path and its validity
checks. Do not restart the completed slowdown investigation or remove guards
as a shortcut. No trained-guide efficacy or universal speed claim is currently
supported. This refresh spends no new GPU time and does not reuse historical
campaign allowances. The restart plan specifies the preparation, evidence
requirements and budget-sizing step; it is not a launch of the old N64 commands.

## Owner reset: previous 64-particle results disqualified — 23 October 2026

All previous N64 results in this investigation are disqualified for scientific
and performance decisions, including the conclusions in older sections below.
Do not reuse their timing, training selections, moments or scores as evidence.
The minimum is now N1008. Historical files remain only for provenance.
The [fresh N1008 runtime comparison](ledh-sir-n1008-runtime-20261023.md) is complete:
576 timed calls and36 warm calls, all valid; current/pre-merge and current/checked
parity pass. For one analytical score direction, current/pre-merge mean ratios
are1.2133 plain,1.3585 conditional and0.7282 physical-mixture, with pointwise95%
intervals excluding1 in the same direction for both datasets. Uniform speed
recovery is rejected. This is FP64/T5/single-cloud timing with guide=None, not
native parameter-batch or learned-guide efficacy evidence. See the
[fresh result](artifacts/ledh-sir-n1008-runtime-20261023/result.md).
No N64 result establishes whether the guides work. Future efficacy work must
train/evaluate anew with N>=1008.
Completed follow-up: [trace the N1008 slowdown](ledh-sir-n1008-slowdown-trace-20261023.md)
through the compiled consumer and isolated diagnostic cost interventions.
Production safeguards and OT(1,5) remain unchanged; diagnostic variants are
ineligible as algorithm candidates. The trace is complete: the merged
determinant and innovation validity checks are live inside the 120-substep SIR
flow recurrence per call, which explains the plain and beta=1 conditional
slowdown. The physical beta=.9 arm already needed the determinant and benefits
from the current 9x9 low-rank determinant versus the pre-merge 18x18 QR route.
See the [trace result](artifacts/ledh-sir-n1008-slowdown-trace-20261023/result.md).
The isolated intervention timings are not used for guard-specific percentages
because the diagnostic workers shared the target GPU. A future repair must
preserve rejection semantics and optimize the checked path rather than remove
its safeguards.

## Historical chronology — superseded execution instructions

The sections below preserve the investigation history. Their N64 performance
and scientific conclusions are disqualified, and their next-action statements
do not govern the active restart. Read the current program and restart plan
above before executing anything.

## Guard-placement repair complete; overall speed criterion fails — 22 October 2026

The [reviewed execution plan](ledh-sir-guard-placement-repair-20261022.md)
now implements bounded batching of the unchanged checked determinant in the
shared analytical flow. Sequential state/tangent protection, chronological
prefix checks and absorbing rejection remain. No unchecked PSD shortcut is
used. CPU checks: 61 focused tests plus 84 consumer tests passed, one skipped.
GPU/XLA: all 61 focused tests passed. The only test repair preserves the nested
diagnostic output in a recorder; numerical assertions remain unchanged.
All 2928 frozen SIR replay rows are bitwise identical to the retained checked
baseline, including raw moments, analytical scores and exact validity flags.
Both the eight-block comparison and independent fixed 48-block confirmation
failed the overall speed criterion. In the latter, plain and physical-mixture
pointwise pooled intervals favor batching; conditional-guide intervals include
one, with no individual regression veto. The earlier regression is preserved
and attempts are not pooled. Retain the checked shared engineering refactor;
make no uniform speed-promotion claim. All planned tests and replay are complete.
See the [terminal result](artifacts/ledh-sir-guard-placement-repair-20261022/result.md)
for the decision, uncertainty, source and budget. No process is running or further
timing launch planned. Earlier scientific vetoes remain. The next performance
question is which conditional-endpoint cost limits the benefit; no exclusive
cost attribution has yet established that answer.

## Guard placement and repair design — 22 October 2026

The [design](ledh-sir-guard-placement-repair-design-20261022.md) distinguishes
the old boundary checks from the added substep coverage. It proposes hoisting
invariants, preserving local protection and absorbing failure flags, and removing
unused determinant work only with an explicit equivalence argument. A conditional
invertibility derivation is recorded, including its unverified covariance premise.
No runtime change or new GPU attempt follows from this source review.

## Completed deeper dependency trace — 22 October 2026

The [source/HLO trace](artifacts/ledh-sir-remaining-overhead-trace-20261022/result.md)
confirms why the retained repair remains costlier than pre-merge in the plain
and conditional consumers: determinant validation is on the state recurrence's
critical path, and its derivative solve is on the tangent recurrence's path.
The nine-pivot body runs 120 times per measured SIR call; seven compiled
innovation-validation fusions remain even in the determinant ablation. These
are checked dependencies and static counts, not exclusive timing shares.
Auxiliary states/tangents were already present before the merge. Source identity
and consumer-wiring assertions pass. No runtime change or GPU run occurred.
Complete cost attribution and the first floating operation behind the plain
score discrepancy remain unresolved; prior numerical and timing verdicts stand.

## Completed bounded merge repair — 22 October 2026

The owner requests tracing and repairing all identified slowdown and merge
defects in the actual SIR consumer. Follow the reviewed
[plan](ledh-sir-merge-repair-20261022.md) and concise
[checkpoint](controlled-smc-kdm-checkpoint.md). The
[source audit](artifacts/ledh-sir-merge-repair-20261022/audit-findings.md)
identifies newly retained determinant work, an unreachable alternative batch
optimization, dropped safety-only activation and reset marginal checks, and
ill-conditioned moment sensitivity. The two dropped checks are restored;
an equivalent observation-space determinant reduces factorization order18 to9.
Final-candidate timing passes all six declared criteria against the previous
runtime repair; 123 distinct tests and all2928 frozen replay rows pass, including
raw moments and analytical scores. Direct pre-merge timing still shows slower
plain nonzero and conditional calls, and plain numerical equivalence still fails.
The completed diagnostic ablation identifies material determinant-validation cost;
its weakened checks are ineligible for runtime use. A further checked Cholesky
determinant candidate passes parity but fails retention because three timing
criteria are inconclusive. Its evidence is preserved; the guarded candidate02
repair is restored and matches the frozen tested source. No task job is running.
See the [terminal result](artifacts/ledh-sir-merge-repair-20261022/result.md).
Complete speed recovery and broader scientific improvement are not established.
The next repair must distinguish numerical conditioning from a changed formula
with an independent reference, while any further speed change must retain the
validity contract. The original guide-quality veto remains; do not start new
training or promote scientific conclusions from this engineering comparison.

## Completed runtime repair — 21 October 2026

The [repair plan](ledh-sir-runtime-repair-20261021.md) is complete on
`ledh-refactor-with-policy-fix`. Checked same-matrix elimination replaces QR
in the shared analytical flow while preserving validity checks, OT (1,5),
data, controls, fitted parameters and analytical scores. Eight alternating
blocks on four regimes give approximately10–18% less runtime than merged QR;
all six predeclared endpoint/direction interval criteria pass. Complete replay
passes all2496 output rows and432 nonzero score checks at the unchanged
abs1e-8 + rel1e-8 tolerance, including raw moments, with exact flags.
Tests total92 distinct passes and1 inherited skip; all980 captured current
package files match the tested snapshot. The
[terminal report](artifacts/ledh-sir-runtime-repair-20261021/result.md) records
the proof, call-chain checks, intervals and limitations. Independent Claude
review was unavailable; skeptical executor review and executable checks pass.

This accepts a scoped implementation optimization. It does not prove complete
pre-merge speed recovery, resolve the earlier plain-arm pre-merge numerical
discrepancy, or improve guide/score accuracy. The guide's existing heuristic
promotion veto remains unchanged in every reported regime. Repair uses1821.33
GPU-process seconds; parent total6964.23 of19800, with12835.77 remaining.
All processes exited; no authorized repair stage remains. Changes remain
uncommitted and unrelated dirty work is preserved. Broader claims require
fresh scoped evidence.

## Completed slowdown diagnosis — 21 October 2026

The [runtime follow-up](artifacts/ledh-sir-postmerge-runtime-20261021/attempt-01/result.md)
checks actual optimized consumer HLO and eight alternating timing blocks.
New flow-validity dependencies retain QR determinant/tangent work in plain and
conditional endpoints that XLA previously removed. Physical mixtures already
needed it. The guided consumer does not call the newly optimized native batch
factory. Observed zero-direction ratios are1.822/1.878 for plain/conditional
and.940 for the physical control; these are descriptive one-dataset timings.
All108 calls reproduce their own archived results exactly; source/input checks,
validity and single tracing pass. No guard or numerical setting was changed.
The follow-up uses412.984804 GPU-process seconds; combined replay/diagnosis uses
5142.898765 of19800, leaving14657.101235 at that checkpoint. All processes
exited. The repair above preserves validity checks while removing redundant
factorization work.
This explains extra computation; it does not repair plain numerical parity or
establish a general speed ratio. Earlier sequential timing caveats remain valid.

## Completed frozen merge replay — 21 October 2026

The [paired replay plan](ledh-sir-postmerge-replay-20261021.md) is complete on
`ledh-refactor-with-policy-fix`. Both source versions ran2496 holdout rows and432
nonzero analytical directional checks; the old version exactly reproduces its
archive, with no invalid rows in either version. Eight non-plain particle arms
pass numerical parity. Plain LEDH exceeds the fixed tolerance: maximum loss
change4.282215e-6, value change1.252784e-4 and derivative change.02394952.
No tolerance was relaxed. The frozen moment results retain the previous negative
conclusion and every heuristic promotion veto. This is exposed-data replay,
not fresh training or score-accuracy evidence.

Plain and conditional warm evaluations were2.249 and2.360 times longer in this
sequential replay; changing machine load prevents a causal performance claim.
All10 reporting integrity tests pass after correcting the saved regime label.
Four launches used4729.913960 of19800 GPU-process seconds; all have exited and
the fifth continuation was unnecessary. The
[terminal report](artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/result.md)
records exact provenance, complete tables, budget and inference limits.
The later runtime investigation above completes the proposed interleaved check;
focused numerical attribution remains open. No step remains in this replay;
future guide promotion still needs independent untouched moment-risk evidence.

## Completed training recovery — 21 October 2026

The user's renewed execution request is complete under the
[reviewed continuation plan](ledh-sir-neural-kdm-continuation-20261021.md).
The prior512-update timeout lost accepted neural/Adam state because the runner
saved only after a complete rung and validation. A thin recovery runner now
inherits the actual training loop, preserves state every8updates, and marks
recovery records unselectable.43 CPU checks pass, including exact resumed Adam,
parameter and next-seed parity. Numerical package source remains unchanged.
Attempt21 completed256 and attempt22 completed512. All234 overlapping updates
exactly reproduce the original interrupted run, and all76 terminal inspection
checks pass. Validation loss is.671551 at256,.682223 at384 and.707064 at512;
these reused validation measurements establish no independent improvement.
Cumulative process charge21785seconds leaves1981.197895057seconds. Both
initializations completed128; the other initialization's optional512 rung
requires at least3872.684seconds before validation and is under-budgeted.
No process remains live. No exposed holdout was reused or candidate promoted;
the previous negative comparison remains unchanged. The
[continuation result](artifacts/ledh-sir-neural-kdm-20261020/attempt-20-recovery/result.md)
records the terminal review, budget and next evidence needed. The active
checkpoint identifies the remaining research question.

## Completed bounded repair and comparison — 21 October 2026

The [staged repair plan](ledh-guidance-repair-plan-20261020.md) governs execution.
Phase A is complete: actual validity flags, strict paired-result validation,
retained conditioning diagnostics and full package-source capture passed 59
focused tests and the GPU/XLA smoke. Healthy values/scores are unchanged; the
original 544-row conclusions remain unchanged. See the
[repair result](artifacts/ledh-guidance-repair-20261020/attempt-01/result.md).
Phase B is complete for the user-selected **SIR-specific neural KDM proposal**,
an explicit adaptation of the replicated three-state bearings method. The
[neural plan](ledh-sir-neural-kdm-adaptation-20261020.md) specifies its density,
conditioning, full-consumer moment loss, derivatives and fresh-data protocol.
LaTeX propositions and a rebuilt PDF are available; MathDevMCP is inconclusive
on probability formalization, while direct derivations and consumer tests pass.
Phase C engineering passes three fresh GPU/XLA minibatch updates, with all neural
tensors updated. The nested-gradient constant-specialization defect is repaired;
warm timing is recorded by training phase, including the later slowdown.
A scoped CUDA command-buffer workaround
avoids the independently localized WSL driver segfault without disabling XLA.
Attempt12 completed the16-configuration screen. After its supervisor exited143,
the numerical child completed unchanged; attempt14 completed both128-update
continuations. Its optional512 process timed out after362 valid updates without
a completed checkpoint. Attempt16 selected from completed32/128 checkpoints and
finished64 fresh paired datasets:1472 valid rows, unchanged source. The trained
loss is.770637 versus.774066 untrained,.765211 plain and.533512 recursive Gaussian.
The primary paired intervals include zero; the10% criterion and heuristic veto
fail. See [neural result](artifacts/ledh-sir-neural-kdm-20261020/attempt-01/neural-holdout-result.md).
Fresh restricted cSMC/RBF comparisons completed in attempt19:1024 valid rows,
unchanged source, adapted losses.863626/.872854. Neither adapter's paired
learning-gain interval excludes zero; all four restricted arms lose to recursive
Gaussian in every required regime. The combined analysis contains2496 rows and
does not change the failed neural primary decision. The exploratory neural-versus-
restricted intervals favor the neural fit, but do not establish a learning gain
or compare the full original methods. See the
[terminal result](artifacts/ledh-sir-neural-kdm-20261020/attempt-01/result.md).
At that comparison's close, attempt19 had charged3685seconds and
4891.197895057 remained; the completed continuation above gives current totals.
All previous failed charges remain debited. No candidate/default is promoted.
Broader integration is incomplete; this candidate fails to establish heldout
moment benefit, and score accuracy remains unevaluated. See the checkpoint for the exact next
action; the old restricted adapters are not full neural training.

## Additional implementation audit — 20 October 2026

The [bounded consumer audit](artifacts/ledh-guidance-implementation-audit-20261020/result.md)
confirms two additional defects: the physical branch leaves guide-specific
validity flags true for an invalid guide, and its result assembler accepts
missing, duplicated or unpaired particle replications as valid comparisons.
The overall numerical rejection still works. The saved544-row holdout is
complete and paired; these defects do not explain that result.23 focused CPU
tests pass. Computed mixture-conditioning margins are discarded, and the
eight-file provenance check omits executed dependencies. The eight recorded
hashes match, but do not establish full executable source identity.

Actual neural KDM integration, full cSMC twisting, flexible message/policy
training, adaptive proposal allocation and downstream score improvement remain
incomplete or unevaluated. The implemented restricted moment pilot must not be
described as completion of that larger proposal. No additional arithmetic error
was established in the inspected physical density or analytical derivatives.
Audit complete; runtime unchanged; no active process or new GPU charge.
Next repair should close diagnostic/evidence defects before another training
campaign. Remaining budget23826.197895057 GPU-process seconds.

## Completed training diagnosis — 20 October 2026

The [code/math diagnosis](artifacts/ledh-moment-training-failure-diagnosis-20261020/result.md)
completes the next question from the physical-proposal pilot below. Its112
traces and7 exact consumer-parity checks confirm a correct simulated-moment
objective and effective adapter controls. The executed learner has only3/12
parameters; the kernel arm approximates a cSMC message rather than calling the
replicated KDM learner. Fixed beta=.5 allocates about31/64 final draws to a flow
component with negligible weight in the checked runs. Small-minibatch descent
does not establish generalization. Reset moment discrepancies are at roundoff,
not an explanation for the failed training. No OT retuning is justified here.

The consumer corrects to the filtering target at each time; the ideal full
controlled-SMC twisting argument therefore does not prove its moment benefit.
The result derives this distinction and records remaining causal uncertainty.
No algorithm/default changed or new candidate was selected. A next repair must
address learner capacity, proposal allocation/flow quality and training-signal
reliability on fresh partitions; beta=1 is a conditional-only comparator, not
LEDH success. Diagnosis complete, no active GPU process. Campaign remaining
23826.197895057 GPU-process seconds after419.199960844 charged to this stage.

## Completed physical-proposal pilot — 20 October 2026

The [reviewed flow-aware plan](ledh-flow-aware-proposal-repair-20261020.md)
supersedes the next actions in the historical sections below. The optional
physical-output proposal is integrated through the shared LEDH executor, with
full mixture-density correction and total analytical derivatives. Beta=1
cancels the affine flow and is a comparator; beta=0.5 retains the LEDH density
component. UKF, GenUT, Contract E and diagnostic OT(1,5) remain unchanged.

The43 distinct focused CPU tests pass. GPU integration repairs the known failed fixture
for all four physical arms;12/12 healthy derivative checks pass with maximum
relative error8.46e-8. This establishes scoped engineering evidence, not moment
or score superiority. Fresh training and32 independent datasets/two particle
replications are complete; all holdout methods are valid. Both trained guides
fail the recursive-Gaussian comparison in both observation regimes, and have
higher pooled loss than the conditional-only proposal. Neither establishes a
gain over its own untrained adapter or the no-future half-mixture.

The [terminal result](artifacts/ledh-flow-aware-proposal-repair-20261020/result.md)
separates successful numerical repair from failed candidate promotion. The
mathematical objective remains correct; score improvement was not evaluated.
The manuscript now contains the full derivation and independent negative
result. Math/code audit and executable call-chain checks are complete;
MathDevMCP matrix checks are inconclusive, not proof certificates. The98-page
PDF builds cleanly and the new mathematics/results have been visually reviewed.
Campaign remaining24245.397855901 GPU seconds at this pilot's completion,
before the diagnosis recorded above. Component allocation and pre/post-reset
moments have now been investigated; this holdout cannot tune a next candidate.
All experimental phases of this pilot are complete.

## Completed collapse investigation — 20 October 2026

The [reviewed collapse repair](ledh-moment-collapse-repair-20261020.md) fixes
the training endpoint's ignored shape-availability flag and exposes complete
covariance loss at its actual scale. 112 focused tests pass. GPU replay leaves
both baseline state traces unchanged; six healthy analytical/FD checks pass.
The existing moment-safety switch does not repair the failed time4 cloud.

The [terminal result](artifacts/ledh-moment-collapse-repair-20261020/result.md)
traces the upstream mechanism to a flow based on large predictive UKF P acting
on conditional Q proposals. The density correction is correct, but the
proposal is poorly matched: failed-step ESS1 versus29.04 for the exact
conditional reference on the same fixed cloud. This is not evidence for
replacing the UKF lifecycle. An exact affine preimage repair passes numerical
identity checks, but has not been integrated into the full filter.

Terminal call-chain review corrected an overbroad capacity claim: the negative
required information eigenvalue excludes the trained quadratic adapter only.
The kernel arm uses mixture-product moments and can expand covariance; its
ability to realize this preimage is not checked. A regression now protects
that distinction. The updated manuscript is94 pages, Section15.10 and
Proposition45; final builds and changed-page inspection pass.

Next: review and implement an optional flow-aware proposal through the shared
consumer, including total analytical derivatives of map and proposal density.
Retain UKF, Contract E and OT(1,5). Do not resume the old failed holdout or
claim efficacy from fixed-cloud identities. Scope-specific calibration and
fresh independent moment evaluation follow numerical consumer validation.
This supersedes the next actions in the historical sections below.

## Active restoration repair — 20 October 2026

The [reviewed repair plan](ledh-moment-restoration-repair-20261020.md) governs
the current action. Exact replay isolates rounding loss when tiny particle
offsets are added to a large mean, followed by Cholesky of an exactly zero
covariance. The shared restoration now retains offsets through centering;
OT settings and the covariance ridge are unchanged. Local non-harm and
derivative tests pass; final GPU endpoints are finite and 12/12 score
finite-difference checks pass. All 100 focused tests pass. The materialized
time4 covariance is nevertheless zero and standardized shape diagnostics
remain undefined, now explicitly flagged. The [terminal repair result](artifacts/ledh-moment-restoration-repair-20261020/result.md)
accepts the narrow factorization repair, not full moment fidelity. Next:
review the remaining representation limit before a fresh numerical scope
and independent efficacy comparison. The prior heldout comparison remains
incomplete and cannot support a guide ranking.

## Active simulated moment training — 20 October 2026

The owner requested implementation, plan review and execution of simulated
moment training. The [reviewed plan](ledh-simulated-moment-training-20261020.md)
governs the completed training stage; the repair section above governs the
current action and supersedes the proposed next study below.
Continuous cSMC and positive-RBF guide coefficients are optimized against
terminal first/second-moment loss through the existing shared LEDH executor.
Score labels are explanatory only. All 35 focused tests and both actual GPU
consumer/analytical-derivative smoke checks pass. All training runs completed,
but campaign-01 stopped on a reproduced nonfinite unguided baseline at the
seventeenth heldout dataset. No heldout ranking is justified. The
[terminal result](artifacts/ledh-moment-training-20261020/result.md) separates
successful implementation from the incomplete scientific comparison. The
subsequent numerical repair is complete with the limitations recorded above.
OT stays `(1,5)`. The RBF adapter is KDM-inspired, not original neural KDM
retraining. See the [active checkpoint](controlled-smc-kdm-checkpoint.md) and
[execution record](artifacts/ledh-moment-training-20261020/execution.md).

## Failure review — 20 October 2026

Implementation and supervised pilot committed as `4abbaf9f1` on
`ledh-refactor-with-policy-fix`. The [thorough review](artifacts/ledh-supervised-guidance-20261019/failure-review/review.md)
finds a correct risk identity but finite-bank selection rather than coefficient
training for score risk. A physical future guide applied before the LEDH flow
becomes `h(F^{-1}(x))` after it, not `h(x)`; an exact Gaussian counterexample
shows this can hurt even with a perfect message. The existing proposal density
correction remains correct. Its causal importance for SIR is unmeasured.
No guide is promoted and no new GPU run was launched. This supersedes the next
action below: first plan/test a guide composed with the actual shared affine
flow, including its analytical tangents, before a larger fresh supervised
comparison. Keep OT `(1,5)`. See the active checkpoint for scope and budget.

## Completed supervised guidance pilot — 19 October 2026

Replication remains closed. [Simulation-supervised guide selection](ledh-supervised-guidance-20261019.md)
has executed on the full shared analytical LEDH executor with disjoint
simulated datasets. The new proofs, 34 tests and final 96/96 valid T5 rows
support the bounded implementation, not statistical improvement. Unguided
has the lowest validation loss; no guide is promoted. See the
[result and audit](artifacts/ledh-supervised-guidance-20261019/result.md).
OT remains `(1,5)`. Next: fresh scope-specific calibration and more independent
datasets. The finite RBF bank is not original neural KDM retraining or
canonical admission. Earlier diagnostic-consumer results below remain history.


## Earlier program handoff — 19 October 2026

Replication remains closed. Continue the original score question through the
[corrected SIR guidance execution](ledh-sir-guidance-execution-20261019.md),
holding OT at `(1,5)`. Fit SIR-specific backward messages and compare their
analytical score consumers with the unguided and simple-proposal arms.
The positive-kernel adapter is KDM-inspired, not original KDM training; the
current three-density consumer remains noncanonical diagnostic evidence.
This handoff supersedes earlier next actions. The owner now authorizes twelve
hours total for this campaign; bounded allocations and repair rules remain in
the plan. October 18 is closed.
Use the [active checkpoint](controlled-smc-kdm-checkpoint.md) for recovery.
The [completed SIR result](artifacts/ledh-sir-guidance-20261019/result.md)
shows conditional harm for both equal-mixture guides on both datasets.
All 96 keys are evaluated, with one failed global comparator. The subsequent
[independent-beta repair](ledh-sir-beta-repair-20261019.md) uses fresh seeds;
replication remains closed and OT remains (1,5).
Six independent beta fits pass their optimization/validation checks and all
112 held-out rows are valid. The [completed result](artifacts/ledh-sir-beta-repair-20261019/result.md)
supports two matched-beta gains on data 26100832 under its conditional
bootstrap analysis, but neither guide meets the full cross-dataset criterion
or passes the componentwise simple-method screen. Lower pilot objectives are
not score improvement. GPU spending is 9,325.73/43,200 seconds; no campaign
process remains. Next resolve SIR proposal/reset versus message error and
the canonical consumer call chain, under a fresh bounded attribution plan.
Do not reopen replication, sweep OT, or promote the diagnostic consumer.

## Active-program handoff — 18 October 2026

Replication remains closed at the bounded scope below. The owner's instruction
to refresh and continue resumes the [original score master](younis-kdm-score-master-program-2026-09-14.md),
now updated with the [completed message/consumer comparison](artifacts/ledh-message-consumer-20261018/result.md).
The 2,688 fresh GPU/XLA rows, 16,128 trace checks and 48 derivative checks pass.
None of 36 simultaneous contrasts establishes a gain or harm. All four
product-consumer candidates fail the descriptive heuristic promotion screen;
all remain numerically viable. This local kernel experiment is not original
KDM training or canonical LEDH admission.

There is no numerical continuation veto. The
[paired particle-count repair](artifacts/ledh-particle-count-stress-20261018/result.md)
passes its audit of 768 rows, 4,608 trace checks and 16 derivative checks.
All 14 simultaneous contrasts remain unresolved; both product arms still
lose descriptively to Gaussian-future global in the second score coordinate
at N64. The [final bounded follow-up](artifacts/ledh-particle-count-followup-20261018/result.md)
also passes: 1,536 rows, 9,216 trace checks and 16 derivative checks on fresh
nested N64/N128 clouds with twelve replicates per fit. Four of 14 simultaneous
contrasts support conditional gains over each product's own N64 arm and over
unguided local at N128; none establishes improvement over the strong global
heuristics. Both products still fail heuristic promotion. The next repair
must separate the current-observation Gaussian/map approximation from the
moment-restoring reset, not assume that more message fitting will suffice.
MathDev verifies scoped scalar identities; its full matrix-proof audit remains
parser-inconclusive. The campaign closes at 894.367691905/1,200 GPU-process
seconds, 4.615978229/180 CPU reference/analysis seconds and three of three
launches. No process is running. The launch limit is a real resource boundary,
not a numerical continuation veto; unused time does not authorize attempt four.
Earlier budgets remain closed. Use the
[active checkpoint](controlled-smc-kdm-checkpoint.md) for process and next-action
status. Phase completion and candidate rejection trigger automatic audit,
repair and refresh within the remaining authorized budget.
This is algorithm-improvement work, not a reopened replication ladder; the
dated replication actions below remain historical.

## Final campaign summary and repository handoff — 14 October 2026

The bounded replication phase is complete: controlled SMC retains its accepted
source/failure-fidelity scope; released 25-particle bearings MDPF/A-MDPF passed
the corrected 283-check closure audit. Exact original-paper reproduction,
population-quality equivalence, original iAPF completion, and Task3 remain
outside this completion. No additional training is scheduled.

The authoritative task branch is ledh-refactor-with-policy-fix in the separate
ledh-iapf-execution-20260922 worktree. The owner-authorized
[synchronization plan](replication-branch-synchronization-20261014.md) records
the migration, commits, main integration, checks, and remote synchronization.
Source code and compact results are versioned; bulk campaign evidence stays at
its recorded local paths. Main's newer LEDH algorithm must be preserved.


Updated 14 October 2026: the bounded released-source replication phase is
COMPLETE. Controlled-SMC remains accepted on its declared tested-input and
failure-fidelity scope. The released 25-particle bearings KDM/IWSG MDPF/A-MDPF
implementation is now ACCEPTED_BOUNDED_RELEASED_SOURCE_REPLICATION on the
tested source operations and consumers.

The corrected closure audit passed 283 evidence/provenance/wiring checks across
seven operation groups. All eight numerical runtime files match the verified
implementation. Saved numerical evidence covers initialization, data/masks,
architecture, density/IWSG, proposal/weight/bandwidth heads, native draws,
pretraining, full loss/TBPTT, Adam, controller, transfers and native metrics.
No new training or GPU run was needed. See the
[terminal result](controlled-smc-kdm-source-closure-result-20261014.md),
[coverage matrix](artifacts/controlled-smc-kdm-replication-20260928/source-closure-20261014/attempt03/coverage.json)
and [closure plan](controlled-smc-kdm-source-closure-plan-20261014.md).

This is a source-implementation verdict, not population trained-quality
equivalence. All unfavorable checkpoints and residual NLL/RMSE gaps remain
recorded. Exact Appendix B and other original-paper experiment reconstruction
remain separate, incomplete work. Original iAPF and E0/Task3 remain deferred.

## Completed closure and next action, 14 October 2026

1. Source/paper and consumer coverage: COMPLETE for the selected released
   bearings implementation; exact original-paper coverage is separately listed.
2. Current runtime binding and saved numerical evidence: PASS. Three changed
   surrounding harnesses were reviewed explicitly; runtime hashes match.
3. Actual compiled consumer-to-kernel wiring and native endpoint coverage: PASS.
4. Prior discrepancy classification: COMPLETE for the investigated cases;
   restored-state, resampling-branch and angular-input explanations remain
   bounded to their tested cases.
5. Terminal review and documentation: COMPLETE. Stop the source-closure phase.

No automatic training ladder follows. Reopen replication only for a named
missing source operation/consumer, a reproducible common-input mismatch, an
additional port invalid outcome or a changed numerical implementation.
A finite independent-training score gap alone is explanatory evidence, not an
automatic failure or a proof of equivalence. Original-paper reconstruction,
population-quality qualification and algorithm improvement each require their
own explicit question, scope and budget.

The first two closure audit outputs are INVALID for acceptance: their reporter
did not enforce the contract. Their provisional verdicts were withdrawn and the
outputs preserved. Corrected attempt03 passed 14 auditor failure-case tests and
computes its verdict from the checked evidence. Saved numerical experiments
were not changed.

Latest allocation record before closure: 205463.582169 CPU / 71779.382403 GPU
seconds. Terminal audit: 1.515 CPU-only wall seconds, zero GPU. Conservatively
reserving the full 600-second closure cap leaves at least 204863.582169 CPU
seconds and unchanged GPU time; no new allocation is assumed.

## Completed unit02 initialization intervention

<!-- UNIT02_INITIALIZATION_BEGIN -->
Updated 2026-10-13T17:00:10.315779+00:00. Unit02 initialization intervention: **COMPLETED**.

All declared stages completed; interpretation remains unit-conditional. Validation NLL gap: 0.513438 -> 0.098476 nat. T150 NLL gap: 0.762433 -> 0.157799 nat.

Attempt01 completed 270 MDPF epochs (best174), 21330 updates and 70512
source-controller checks. Its endpoint checker incorrectly required four
models instead of this intervention's three. Attempt02 repaired that scope
check and reused the frozen checkpoint: 472 endpoint checks and six author
metric assertions over three saved clouds passed, with exact baseline replay and no retracing.

Run: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt02.
The recovery used 40.596 GPU / 4.991 CPU seconds. Both attempts together used
5155.750 GPU / 95.732 CPU seconds, within 21600 GPU / 3600 CPU seconds.
The phase is closed; training was not repeated.

Plan: [unit02 initialization](controlled-smc-kdm-unit02-evaluation-repair-plan-20261014.md). Result: [unit02 result](controlled-smc-kdm-unit02-evaluation-repair-result-20261014.md). This intervention is complete; released-source closure is decided above, while whole-paper reproduction remains incomplete.
<!-- UNIT02_INITIALIZATION_END -->

## Latest completed endpoint extension

The [recent-checkpoint result](controlled-smc-kdm-recent-endpoint-result-20261013.md)
closes native endpoint coverage for all nine recent fixed models: 18 complete
GPU/XLA executions, 1,385 runtime/provenance checks, 18 author metric assertions over nine saved clouds and
89 terminal checks passed. No invalidity, retracing or new translation defect.
Unit01's controlled initialization change reduces the observed native T=150
evaluation NLL gap from 0.777971 to 0.056944 nat (92.68%), supporting
the separate 95.26% full-validation result. The completed unit02 intervention
likewise supports initialization sensitivity, while leaving a 0.157799 nat
T150 NLL gap. Neither result establishes that all residual differences are
intrinsic source variation or that an additional translation defect exists.

The recent endpoint phase used 21.116640 CPU / 215.938134 GPU seconds, within
its 300 CPU / 900 GPU second cap and the same active allocation.

## Target and current verdict

Replicate released author behavior. Intrinsic author failures are acceptable.
On matched author-valid inputs, the port must add no invalid outcomes; numerical
behavior must be no materially worse where an identifiable comparison exists.
Do not turn a local diagnostic discrepancy into a replication failure when the
author shows the same behavior. Paper-wide reproduction, original iAPF and
E0/Task 3 remain separate. A local arithmetic check cannot substitute for
independent trained-quality evidence.

Controlled SMC has bounded tested-input implementation and failure fidelity, not population no-worse certification. One hundred one exact saved source Gram/RHS replays pass. Q2 contains 13,120 paired trials in 41 cells, 14 author-intrinsic failures, five TF failures confined to author-failed inputs, and zero added TF invalid outcomes on author-valid inputs. The conditional Octave normal-equation solve on ill-conditioned designs explains selected coefficient discrepancies relative to centered TF QR. The former 1% likelihood and 5% variance margins were wrong planning and are withdrawn. Source variance from about 4.18e-5 to 4057 and effective counts down to 1.45 make a continuous-outcome bound unresolved; a wide interval or source-self envelope is not proof of no worse. Preserve the bounded result and do not repair intrinsic source failures.

KDM/IWSG: bounded released-source replication is accepted by the 14 October
closure result. Source-grounded Adam repair, complete native training,
same-input gradients/updates, source controller, native cloud law, transfers
and filtering/metrics checks provide the evidence. The pre-repair checkpoint
remains ineligible. Three independent development pairs and the unfavorable
MDPF scores are retained as descriptive evidence. The old 0.01 nat NLL / 1%
RMSE population statement remains unestablished and is not a source-replication
completion criterion. No statistically supported independent-training ranking
is claimed.

Latest checked findings:
- Fresh constructors: 324 comparisons; both FP64 trajectories agree. Unit01
  FP32 has feature drift beginning at index 59 and a gradient flag at index 67.
  Restoring the author state clears index 59 and index 67; the latter gradient
  difference falls from 2.60975e-5 to 1.11759e-8 across 340 exact replay checks.
- Crossed starts: 64 paired updates in both precisions; all 140 adapter checks
  pass. The unit01 TF-start discrepancy at update2 persists on identical state.
- That saved discrepancy is now causally localized. One FP32 ULP in the CDF
  changes one ancestor. Forcing only that first choice removes the loss gap and
  reduces the gradient error from 0.03084 to 2.38e-7. A negative source-only
  perturbation reproduces the TF branch: whole gradient arrays then agree to
  2.98e-7. This is shared resampling sensitivity, not a demonstrated defect in
  the translated IWSG gradient.
- The diagnostic reporting bug is repaired: crossed-start capture now records
  the actual consumed ancestor/CDF/uniform inside XLA. Both saved-case launches
  pass 2,358 checks; the repaired shared sampler reproduces the old result.
  Old recomputed ancestor/CDF fields remain provisional.
- Identical-history controller replay passes 3,278 checks. The common author
  evaluator still sees worse TF-trained final checkpoints on the 64-row panel:
  NLL increases 0.741/0.656 nat, RMSE increases 31.84%/45.50%. This rules out a
  TF-only evaluation bug as the sole explanation of that panel. It does not
  distinguish training variation from checkpoint selection.
- Terminal audit: 155 checks pass. Stable exercised signatures trace at most
  once; GPU memory growth and XLA execution are verified. No algorithmic
  numerical default changed.

## Completed replication investigation, 13 October 2026

The following records the executed investigation. Its next-step language is
historical; the 14 October closure verdict and stop condition above govern.


The skeptical review found wrong planning in treating an expensive population
noninferiority proof as the required next source-replication step. It also
found the previous next action too vague to execute. The approved replication
target remains author behavior, including intrinsic author failures. The old
0.01 nat/1% population claim is a separate unestablished claim; neither lack of
precision nor a local replay pass establishes replication completion.

1. Audit the actual author pretrained-module loader against all saved TF
   starting parameters, constants, optimizer state and rates. Compare both
   saved pretrained measurement models on exactly shared author clouds for
   units01/02. Use the source method and unchanged data.
2. Verify the current runtime with the source-stage checker, then reproduce
   the first two saved unit01 TF MDPF epochs through the actual epoch driver
   on the same RTX 5080. Identical row/seed schedules, controls and close
   numerical outputs are required before interpreting an intervention.
3. Train unit01 MDPF with only the initial measurement-module weights replaced
   by the author's pretrained weights. Preserve the original global stage
   index, TF random schedule, Adam, data, loss, TBPTT and stopping rule. Do not
   repeat weight pretraining or A-MDPF.
4. Evaluate its best checkpoint, the old TF best and author best with the
   author's complete native validation method. Record whether the isolated
   change explains any observed checkpoint gap. One controlled trajectory is
   diagnostic; no population ranking or replication completion follows.
5. COMPLETE. Saved-pretraining loss/gradient/Adam, partial-batch validation and
   actual source checkpoint-selection tests passed. Do not treat different
   histories under different random streams as a controller error.
6. COMPLETE. Native two-level pretraining clouds pass source/analytic moment
   checks through both actual consumers. No new repair is indicated. Do not
   repeat the completed generic or composed sampler tests.
7. COMPLETE. MDPF/A-MDPF endpoint and failure-accounting inventory is in the
   cloud-law result. Training, transfer, controller and metric call chains have
   executable evidence within stated scopes. Recent fixed checkpoints now have the completed step8 coverage.
8. COMPLETE. Recent native endpoint coverage passed for nine models and two
   seeds, with pinned author metric replay. See the recent-endpoint result.
9. COMPLETE: unit02 control replay, source-prescribed MDPF training, native
   validation and T150 evaluation. See the
   [evaluation recovery result](controlled-smc-kdm-unit02-evaluation-repair-result-20261014.md).
   Only the model-scope checker required repair; no numerical algorithm changed.
10. NEXT: consolidate the released MDPF/A-MDPF call-chain coverage, updating
    the cloud-law inventory with recent endpoint and both initialization
    interventions. Name any specific unchecked source operation before further
    testing. For residual NLL differences, first inspect the saved shared-start
    histories and replay evidence; if a consequential unmatched operation is
    identified, declare one bounded same-input test. A residual finite gap alone
    is not a newly invented replication threshold. Record Appendix B driver
    availability and original experiment coverage separately. Do not repeat
    completed sampler/controller tests, automatically launch independent
    training, or reopen accepted controlled-SMC fidelity. Original iAPF and
    Task3 remain deferred.

Plan, budgets, stop conditions and commands are in
[stage-isolation plan](controlled-smc-kdm-stage-isolation-plan-20261013.md).
Phase caps: 16200 CPU wall seconds / 14400 active GPU seconds; three original
attempts plus at most two evaluation-only recovery attempts under the same
caps, as explicitly amended in the plan. One recovery completed. Total charged:
11250.946277 CPU / 11090.917908 GPU seconds; phase remaining:
4949.053723 CPU / 3309.082092 GPU seconds. No extra allocation is assumed. The prior 1%/5% cSMC criteria remain
withdrawn; bounded cSMC fidelity stays accepted. Original iAPF and Task3 remain
outside this repair.

Completed evidence retained: full-state pretraining replay clears the selected
gradient flag; the trained small-angle calculation explains98.52% of the
selected discrepancy; native full-validation checkpoint differences persist;
the current independent-unit precision design is unaffordable. None supplies
an undiscovered code defect or justifies changing the author's algorithm.

The completed boundary phase used74.797690 CPU /70.825340 GPU seconds, under
its separate2400-second-per-resource diagnostic cap and the same active grant.
Allocation immediately before unit02 was 205559.313788 CPU / 76935.132129 GPU
seconds. After both unit02 attempts, 205463.582169 CPU / 71779.382403 GPU
seconds remain. The stage-isolation costs above refer only to that completed
earlier phase. No new allocation is assumed.

The [cloud-law plan](controlled-smc-kdm-cloud-law-plan-20261013.md) capped its
completed diagnostic at 1,200 seconds per resource. Actual charges were
27.096454 CPU / 27.096385
GPU seconds. One attempt completed; that diagnostic phase is closed.

## Exact scope and identities

| Track | Selected replication | Boundaries |
|---|---|---|
| Controlled SMC | Heng, Bishop, Deligiannidis and Doucet, *Controlled Sequential Monte Carlo*, Annals of Statistics 2020: neuroscience Figure 4 right, Lorenz Figure 6 and Table 1 cSMC/APF, manuscript Appendix F multimodal experiment | Source baseline, paper-result reproduction and TensorFlow portability have separate verdicts. Figure 4 left/FFBS, Figure 5/PMMH, static-model examples and the separate 1000-repetition iAPF comparison are outside this selected scope. |
| KDM/IWSG | Younis and Sudderth, *Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes*, NeurIPS 2023: released 25-particle bearings MDPF/A-MDPF training/filtering; exact Appendix B comparison retained as separate paper-reproduction scope | Released-source implementation accepted on tested operations/consumers. KDM is the density representation; IWSG is the training estimator. Independent-training population equality and exact original-paper coverage are not established. |
| Original iAPF | Preserved earlier work | Incomplete and deferred. A controlled-SMC result cannot complete the original iAPF replication. |
| E0 enhancements | Wider precision than the author baseline, new projections, floors, damping, trust/KL controls and other changed numerical routes | Downstream, separately named and separately budgeted. No E0 result closes C/P/K replication. |

Controlled SMC learns a backward twisting policy and refines it
multiplicatively. The neuroscience/Lorenz source fits quadratic log policies;
the multimodal source fits a positive RBF expansion with NNLS and knot
retention. Substituting one for the other changes the method.
KDM represents a multimodal filtering density and uses the specified IWSG
sampling/weighting/gradient procedure for learning.

Source-prescribed safeguards remain part of replication. Correcting our
translation or solver-compatibility error while preserving the same objective,
feasible set, update and stopping semantics is replication work. FP64 matching
the author's double arithmetic is also replication/reference work; precision
beyond that baseline is an enhancement. An equivalent algebraic reformulation
must pass endpoint output/invariant checks before acceptance. A changed
objective, projection, regularization or stopping rule is E0.

## Source authority and remaining specification gaps

Controlled-SMC repository:
.localresources/code/heng-controlled-smc,
commit 2696f6d611aae172d10102cb7bfbfd314bb6b88f.
Paper: .localresources/papers/heng2020-controlled-smc-published.pdf;
manuscript/supplement:
.localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.pdf.
Read published Sections 4.2–4.4,6.1–6.2 and the relevant supplement alongside
the [source map](../reference/controlled-smc-replication-source-map-20260929.md)
and [remaining-source audit](../reference/controlled-smc-remaining-source-audit-20260929.md).

C0 verified 52 selected source/data files against Git blobs and preserved their
relative paths. It recorded installed Octave/statistics dependencies and paper
hashes. The source experiment matrix is:

| Experiment | Author baseline | Readiness and limitation |
|---|---|---|
| Lorenz Figure 6 | cSMC_VaryF.m / APF_VaryF.m; d8,T100,40 forcing values,100 repetitions; cSMC N512, up to 4 adaptive refinements; original APF particle-count mapping | Attempt145: Lorenz cSMC stopped at grid36/repetition38 after 3537 cases; original source covariance rejected by Octave. APF blocked by the incomplete cSMC refinement matrix. Supplied SimulatedData.mat is locked. Original MATLAB/Octave numerical equivalence remains untested. |
| Neuroscience Figure 4 right | NormConst_cSMC.m / NormConst_BPF.m; supplied 3000 observations;20 variances,100 repetitions; cSMC N128,I3 and BPF N5529 | Attempt145: neuroscience cSMC and BPF each completed 2000/2000 cases. Memory isolation passed all four consumer checks; source settings remain unchanged. Existing 4000-estimate TensorFlow results remain port evidence, not unchanged-source results. |
| Lorenz Table 1 | Repeat_Dim8/16/32/64.m; three noise regimes,100 repetitions, original N and one refinement | Generative-design rerun is possible. Original unseeded datasets are unavailable; exact dataset/point-value reproduction is unresolved. Needs a separate costed specification. |
| Multimodal Appendix F/Table 2 | NNLS/RBF source, N512,I1,M16 and original BPF comparators | Named Repeat.m is missing; supplied data have 101 observations while F.2 says 100. Source bug/corrected mathematics and reconstructed driver must be separate, explicit claims. Full run remains blocked. |

Lorenz source omits the Gaussian observation normalization constant. Preserve
that source convention in direct source comparisons, and report a separately
identified normalized-likelihood view. A constant leaves variance(logZ)
unchanged but changes mean(logZ) and hence its relative variance. Do not silently
mix these statistics.

KDM repository: .localresources/code/younis-mdpf-neurips-2023,
commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
Paper: .localresources/papers/younis-sudderth-2023-long-range-tracking.pdf.
The [K0 source preparation plan](controlled-smc-kdm-k0-source-config-20261003.md)
anchors Appendix B equations 20–22 and the actual bearings configs/code.
The author's five generated config sets for 11 trials now exist in a verified
snapshot. Component pretraining and MDPF-before-A-MDPF dependencies still apply;
NLL and MSE training stages are distinct; the22 active evaluation tasks all load NLL checkpoints, including tasks labelled MSE. See the20261006 wiring audit. Appendix B's exact numerical
initialization/schedule and driver remain unrecovered. The nonlinear Cappe toy
dataset is wrong for Appendix B's linear/bimodal model. Do not substitute it.

## Phases, dependencies and actual readiness

| Phase | Completion condition | Current evidence / next work |
|---|---|---|
| G0: scope and review | Source, port and enhancement claims separated; old evidence preserved; executor follows new order | This review passes the restructuring contract. Historical repair launches are blocked. |
| C0: source lock | Every selected source experiment is executable or has its exact missing input named | Metadata complete at attempt139; Table 1 exact-data and multimodal gaps remain visible. Metadata completion does not make every source experiment executable. |
| C1: original-source baseline | Audited outputs and intrinsic failures for the declared comparison cases, with correct settings, runtime, warnings and costs | Attempts 140/141 cover twelve cost-pilot cases. Attempt 144 verified source/RNG identity after the attempt 143 interruption. Attempt 145 is terminal: both neuroscience matrices 2000/2000; Lorenz 3537/4000 with preserved author covariance failure, APF blocked. C2 full-grid analysis still lacks complete Lorenz matrices. Table 1/multimodal remain separate work. |
| C2: paper-result analysis | Paper-level numerical findings assessed separately with declared uncertainty; successful full-figure reproduction is not a port prerequisite | Implemented as c2_source_results; 24 focused software checks pass. BLOCKED by the Lorenz failure; the ineligible queue has been retired. It requires all 12000 source cases, and rejects partial inputs both before queueing and during execution. Criteria remain frozen in the full-source plan. |
| C1 failure replay | Exact saved-case reproduction and covariance/precision capture | COMPLETE, attempt 146: the unchanged and observation-only source replays reproduce the same failure. Accept it as baseline behavior; no author-method repair is required. The saved policy has finite, complex coefficients. |
| P0: TensorFlow translation | Actual consumer call chain matches source values/updates/invariants and accounts for intrinsic failures | Source controls pass; 101 saved Gram/RHS source replays are exact. Conditional solver and rounding differences are localized. Strict prediction differences remain, but exact coefficients are not a prerequisite. Frozen centered-QR Q2 is the relevant downstream evidence. |
| P1: TensorFlow implementation/failure fidelity | Matched source operations and tested outcomes; no added port failures | ACCEPTED on bounded tested inputs. Q2 has zero additional TF failures; the historical 1% likelihood/5% variance gates are wrong planning and withdrawn. Population continuous-outcome noninferiority remains unestablished, separately from replication acceptance. |
| K0: released bearings source specification | Model, initialization, estimator, training stages and data anchored to author source/paper | COMPLETE for released 25-particle bearings implementation. Exact Appendix B and original checkpoint/experiment reconstruction remain K-paper below. |
| K1: reference/learning implementation | Source values, gradients, updates and native sampling checked through actual consumers | PASS. Current runtime hashes, saved stage/cloud/boundary assertions and executable call chains verified in attempt03. Selected ancestry/angular sensitivities remain case-specific explanations. |
| K2a: author training | Execute and preserve the declared source development units | COMPLETE: three source development units audited/evaluated. Population variability and original multi-trial reproduction remain unestablished. |
| K2b: TensorFlow training and native consumers | Source-matched operations, valid consumer wiring and same-output metric parity | COMPLETE for bounded implementation evidence. Three repaired pairs, unit01/unit02 interventions and native MDPF/A-MDPF endpoints checked. Residual finite trained-score gaps remain descriptive; no population equality claimed. |
| Q0/Q1a: complete-run controls and calibration | Native source/RNG unchanged, actual GPU controller checked, independent pilot outcomes/cost retained | COMPLETE: 96 pairs, one matched failure, no extra TF failure; calibration only. |
| Q1b: full-grid source calibration | Exact 40-point grid plus generating parameter, source moments/cost/failures | COMPLETE: 1,309/1,312 valid; 41 pristine controls; 2,074.90 CPU seconds. |
| Q2: frozen untouched comparison | Full grid, fixed candidate/seeds/n/inference, wrong-planning 6 October margins | COMPLETE: 13,120 pairs; zero additional TF failures; historical terminal NOT_QUALIFIED because only 5/41 cells met adequacy and the withdrawn margins. This verdict does not establish a port failure under the active rule. |
| Q3: source-only feasibility and possible fresh comparison | Test source tails, precision, cost and a separately justified no-worse bound before freezing any new downstream qualification | Source-only feasibility has not produced an independently justified, resolvable degradation bound. The old margin-driven scout is retired. Preserve bounded implementation fidelity and report stochastic no-worse unresolved; any fresh comparison needs a defensible bound and untouched cases. |
| J: replication review | Bounded source-implementation verdicts separated from paper/statistical claims | COMPLETE: bounded controlled-SMC and released bearings KDM replication accepted. Corrected terminal closure: 283 checks, seven groups, no failed checks. Reopen only on a named implementation/coverage issue. |
| K-paper: original experiments | Exact Appendix B driver/settings, original checkpoints and full declared experiment scope recovered or faithfully reconstructed | NOT_ESTABLISHED. Generic toy code exists, but exact Appendix B mapping is unreconstructed. This limits whole-paper reproduction; it is not an unchecked operation in the accepted bearings implementation. |
| E0: Task 3 robustness | Affected source and port replication accepted, then a separate enhancement question and budget | Deferred. A failed or unavailable replication does not automatically enable E0. |

Dependencies are per experiment:
G0→C0→C1 baseline accounting→P0→P1; C1 also feeds the separate C2 paper analysis.
G0→K0→K1→K2a→K2b.
C1 accounting accepts preserved intrinsic source failures as baseline outcomes;
it does not require a successful full matrix. Available audited source cases
can feed their port immediately. K work progresses independently of C.
J assesses no-worse translation using P1/K2 evidence and reports C2 separately.
Missing experiments remain unevaluated, but author failures alone do not block
acceptance of a faithful translation within its declared scope.

## Qualification and interpretation

Source execution, source-result reproduction and port numerical quality are
three different questions. Frozen source settings and full case accounting
establish execution identity. They do not establish the paper's variance or
learning claims. Paper-level comparisons need the paper's repetition/stage
scope and uncertainty; original random draws and unavailable uncertainty in
published plots remain limitations.

The port must compute the same declared quantities. Check policy evaluation,
proposal moments/densities, normalizers, weighting/resampling and, for K,
learning gradients and updates at the actual consumer endpoints. Coefficient
identity alone is inappropriate for numerically nonunique fits. Conversely,
a small training residual cannot excuse wrong downstream outputs.

A source-valid fixed case requires the source's relevant domain/covariance/
density checks, not merely a finite final likelihood. The port must have no
additional invalid outcomes on those cases. Repeated-run comparisons must
report all failure rates and matched downstream errors. P0 operation-specific bounds need mathematical scale/conditioning justification.
P1 controlled-SMC margins frozen on 6 October were wrong planning and were
withdrawn on 11 October. Source-only work must first determine whether the
downstream summaries and any independently justified no-worse bound can be
tested within the budget. Source repeatability alone does not define an
acceptable bound. A new untouched stochastic qualification remains unfinished.

| Diagnostic role | Consequence |
|---|---|
| Wrong source, target, data, artifact or reference | Stop the affected experiment and repair validity. |
| Faithfully reproduced intrinsic author warning/failure | Accept as reference behavior for the matched case; not a replication veto. Preserve its runtime and inputs; do not repair the author method. |
| Port mismatch on valid source inputs | Blocks the port claim and triggers source-preserving localization. |
| Failure to support a stochastic ranking | Report inconclusive comparison; no silent equivalence claim. |
| Runtime, ESS, condition number, coefficient drift, trace count | Explanatory unless a phase explicitly gives another role. |
| Poor performance against a heuristic | Vetoes practical-quality/superiority/default claims; faithful reproduction of that outcome remains a replication result. |

Keep the original paper's APF/BPF/learning comparators in scope. Additional
EKF/UKF/persistence/constant-motion adversaries belong to later practical-quality
claims; they must not become new prerequisites for reproducing the original
baseline. Exact Kalman/mixture-Kalman controls remain correctness references,
not replacements for the selected experiments.

## Defaults, runtime and safety

Every phase freezes source/model/data, particle count, horizons, optimizer and
training stages, seeds, dtype, runtime, solver and output statistic with
provenance, failure risk and an early diagnostic. Unknown settings are unknown;
a reconstruction hypothesis cannot silently become an exact-author setting.

Repeated BayesFilter kernels use TensorFlow/TFP with stable signatures and XLA
under the project policy. GPU runs need trusted access and verified memory
growth. No pfor or NumPy numerical runtime substitution. Original Octave and
author PyTorch execution are explicitly labeled independent references.
Retain source warnings and pure observability; numerical safeguards that change
accepted results require their own E0 evaluation after replication.
Correct source-prescribed safeguards remain part of the baseline.

## Executable closure and completed command history

The terminal audit command actually executed is:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/audit_kdm_source_coverage.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/source-closure-20261014/attempt03

It exited successfully. The output directory is immutable evidence; do not
rerun into it. The auditor uses the standard library, explicitly hides GPUs
and checks saved numerical results. No new permission rule was needed for
this invocation. Closure is complete, so no new execution is queued.

The dispatcher commands and allocations below describe completed historical
stages. Their old qualification prerequisites do not override the current
bounded replication verdict above.

Dispatcher: scripts/run_controlled_smc_kdm.py. Output root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/.
New attempts preserve snapshots with relative paths, manifests, exact commands,
data/source hashes, seeds, hardware, wall charges and result summaries.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py progress
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py report --phase c1_source_full

| Recorded command phase | Specification | Readiness |
|---|---|---|
| c0_source_lock | [C0](controlled-smc-kdm-c0-source-lock-20261003.md) | Executed, metadata PASS |
| c1_source_cost_pilot | [C1 pilot](controlled-smc-kdm-c1-source-cost-pilot-20261003.md) | Executed, 12 cases accounted for; one bounded timeout retry |
| k0_source_config | [K0 preparation](controlled-smc-kdm-k0-source-config-20261003.md) | Executed, metadata PASS; no training |
| c1_source_full | [C1 full source](controlled-smc-kdm-c1-full-source-20261003.md) | Terminal: both neuroscience matrices completed2000 cases; Lorenz stopped after3537 cases with an original-source covariance failure |
| c1_source_isolation_check | [memory repair](controlled-smc-kdm-source-memory-repair-20261004.md) | Attempt144 PASS; all four source consumers and RNG states checked |
| c1_source_failure_replay | [frozen failure](controlled-smc-lorenz-source-failure-replay-20261004.md) | COMPLETE: attempt146 reproduced the exact source failure in4.23 CPU seconds |
| c2_source_results | [C2 analysis](controlled-smc-kdm-source-continuation-repair-20261004.md) | BLOCKED by incomplete Lorenz matrices; executable and software-checked |
| p0_perturb_source / p0_perturb_tf | [perturbation](controlled-smc-perturbation-experiment-20261006.md) | Complete:174 source/348 unique TF trials; no-worse qualification unfinished |
| p0_offset_localization | [localization](controlled-smc-offset-localization-20261006.md) | Attempt156 complete: identical-target and six policy-substitution checks passed; diagnostic only |
| p0_tf_solver_compatibility | [solver compatibility](controlled-smc-tf-solver-compatibility-20261006.md) | Attempts157–159 terminal; tightened controls pass, saved-system compatibility fails; no autonomous promotion |
| p0_tf_solver_cpu_reference | [solver compatibility](controlled-smc-tf-solver-compatibility-20261006.md) | Attempt160 complete; CPU graph also fails compatibility; explicit non-default reference |

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py report --phase p0_offset_localization

The full source reservation comes from the measured four-driver pilot:
conservative estimate 67.9 hours, approved cap 72 hours. This covers Figure 6
and Figure 4 right only. It does not fund Table 1, multimodal or KDM training.
The owner approved 72 additional CPU wall hours on 4 October 2026. The
allocation is recorded explicitly in the ledger. That grant was CPU-only;
the historical GPU balance is not a current grant. Consult the current balance
at the top of this file and the live ledger. No resource conversion
was made. Local repairs/retries may continue within the approved scientific
scope and total budget. The bounded supervisor rejects unfunded or duplicate
numerical launches.

The source, diagnostic and single-unit complete-training phases have executable
supervisors. The most recent root-cause supervisor has completed:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py --phase root-causes --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-03

This records the executed command; do not rerun into the occupied directory.
Its exact approved prefix covers this diagnostic; no broad Python or shell
rule is needed. The current repaired KDM development pilot has its own reviewed cap and exact
command above. Fresh paired multi-unit qualification still needs a separate
uncertainty design. This roadmap is not an unattended end-to-end qualification
program. Preserve historical evidence.

## Preserved source execution and diagnostic history

Historical statuses and next-step statements below are preserved for provenance.
They do not override the 14 October bounded released-source closure.


[Repair review and execution result](controlled-smc-kdm-source-memory-repair-result-20261004.md).

Preserved source finding,5 October: attempt145 is terminal. Neuroscience cSMC
and BPF each completed2000 cases. Lorenz retained3537 completed cases and its
source covariance failure at forcing7.8846153846/repetition38; APF remains
blocked by missing iteration counts. Attempt146 reproduced the exact failure
twice. Attempt147 captured six cases/22 stages with source results and RNG
unchanged. Attempt148 passed all22 valid forward comparisons, but the failed
policy was not passed end to end because it contains finite, complex
coefficients. The later CPU input smoke confirms rejection before tracing;
failed numerical trajectory parity remains untested. The precision guard
rejected its indefinite matrix. Full C2,
P0/P1 and KDM remain incomplete. No source ridge, projection, changed seed or
E0 repair is authorized by this result. Subsequent KDM execution is recorded above.
Detailed result: controlled-smc-p0-forward-result-20261005.md.

Attempt 143 retained finite-real outputs for 1989 Lorenz cSMC cases before its
Octave child returned -9. The kernel recorded an Octave OOM kill at about
38.8 GiB RSS; the old manifest did not record that child PID. Exact nested-call
retention remains a runtime hypothesis. Preserved partial rows do not complete
C1 and do not establish a paper or TensorFlow failure.

The repair runs one original case per Octave process and restores both random
streams and accumulated matrices. It keeps the original grids, repetitions,
N/I/T, source numerical functions and seeds. All four source consumers must pass
exact uninterrupted-versus-isolated output/RNG checks before the retry. The
full retry also compares each available interrupted-prefix case exactly.
An 8 GiB address-space guard bounds accidental resource exhaustion.

The old analysis queue expired after 11000 seconds and had no C2 executable.
It now follows the actual predecessor deadline, prevents duplicate launches,
reserves 600 CPU seconds for analysis, and rejects failed/incomplete source
inputs. C2 independently verifies native MAT matrices, CSV rows, source hashes,
settings and APF refinement pairing before the frozen bootstrap analysis.
Execution completion, numerical validity, statistical findings and replication
completion remain separate verdicts.

Current numbered attempt ceiling is 160, with 160 used. The supplementary
diagnostic ledger includes the circular GPU comparisons (37.76449583284557
seconds) and CPU localization (5.536095131188631 seconds). The current active allocation is maintained in the live checkpoint and dispatcher;
older balances in this historical section are superseded.
Routine engineering checks are recorded separately. The explicit 6 October grant is additive; other historical balances
are not additional grants.

The exact-matrix compatibility and subsequent DGELSD arithmetic audits are
complete. Forced source branches are insufficient; replacing only the source
divide-and-conquer bidiagonal solve changes policy predictions on the selected
systems. Outer scaling, diagonal flooring and splitting do not trigger there.
A compatible TensorFlow fit remains unresolved. Do not repeat the old request
to inspect these already-checked mechanisms. A changed rank floor, ridge or
damping would alter the baseline and is not authorized by this replication task.

P0 output/invariant bounds must be justified by conditioning. The historical
P1 cSMC margins were wrong planning; the active acceptance reset requires
source-only feasibility and an independently justified, resolvable bound
before any fresh stochastic no-worse decision.
Equivalent-algebra QR remains an unqualified candidate under the existing
scope. Continue KDM source
specification and training as separately budgeted work. Existing authorization
covers localized repairs inside the remaining campaign budget; a new attempt
allocation is bookkeeping, not a new compute grant or human approval gate.
The whole replication is not an executable unattended program until these
remaining specifications and runners are implemented.

## Archived evidence

The pre-restructure master is
artifacts/controlled-smc-kdm-replication-20260928/master-restructure-20261003-01/master-before.md.
The full pre-audit revision, with its detailed historical links, is
artifacts/controlled-smc-kdm-replication-20260928/master-audit-20261003-01/master-before.md.
It retains native six-case source evidence, prior scalar/multivariate parity,
the 4000-estimate TensorFlow Figure 4 run, RBF diagnostics and the repair sequence
through attempt 138. These retain their original limited meanings.
No historical diagnostic PASS is retroactively promoted to replication.

The approved launch, permission rules, final checks and live progress are
recorded in [the source-run execution note](controlled-smc-kdm-source-execution-20261004.md).
Use its exact absolute command forms to avoid shell-prefix mismatches.

## Completed owner-requested perturbation test, 6 October 2026

Completed the bounded full-recursion random-input experiment in
[controlled-smc-perturbation-experiment-20261006.md](controlled-smc-perturbation-experiment-20261006.md).
Phases p0_perturb_source and p0_perturb_tf measured 1e-8 sensitivity and autonomous
port discrepancies. Keep strict equality as localization only; perturbation
magnitudes are calibration evidence, not an automatic tolerance or replication
completion. No numerical policy changes or new compute allocation.


## 10 October localization result

The bounded gap-closure execution completed under the active plan. GPU preflight, memory-growth verification, source controls, and artifact accounting passed. The [localization result](controlled-smc-kdm-gap-closure-result-20261010.md) is authoritative for this stage.

For cSMC, all 16 source-policy forward controls passed. The first divergence is the fitted quadratic at the first fit stage: both source-normal-equation and centered-QR arms were finite, but 0/16 strict policy screens and 0/16 downstream screens passed. This localizes the diagnostic mismatch to ill-conditioned regression/solver semantics. Octave's backslash has a conditional singular-matrix branch; the raw TF normal-equation arm is not a complete source reproduction. This is a localization finding, not a rejection of the cSMC port or a new Q2 qualification result. The copied graph emitted warnings while per-wrapper trace counts were one; the subsequent root-cause phase repaired redundant wrappers and preserved all controls.

For KDM, the coupled source sequence passed all 64 native/replay controls. TF FP64 passed all 1,808 checks; TF FP32 failed 47 MDPF checks and two A-MDPF checks, first at a proposal gradient and accumulating by update 3. This supports float32 arithmetic sensitivity under evolving Adam state. It is not a population ranking or a proof of downstream degradation. K1 remains conditional on a predeclared matched multi-unit design and budget.

The allow-list now has four exact diagnostic prefixes, including the reviewed gap-closure supervisor. Exact-prefix and negative arbitrary-Python parser checks passed. Current-session hot reload is not assumed. No broad Python or shell rule was added.

## 10 October root-cause repair and terminal review

The [execution plan](controlled-smc-kdm-root-causes-20261010.md),
[result](controlled-smc-kdm-root-causes-result-20261010.md) and
[terminal review](artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/terminal-review.json)
are complete. Adam scalar repair reduces isolated FP32 parameter error from
1.19e-7 to 7.45e-9. Checkpoint v2 records the policy; legacy v1 optimizer states
cannot be silently resumed. Residual gradient localization and complete-training
noninferiority remain open. Acceptance and replication-only scope are unchanged.

Three attempts used 11.4206 CPU and 340.8969 GPU wall seconds, including the
preserved Octave export failure and serialization-only retry. Confirmation
verified memory growth, GPU/XLA, single traces, 29 source/input hashes and four
checkpoint cases. Next: bounded layer/adjoint localization and the validated
source-preserving paired-training smoke; no numerical run remains active.
