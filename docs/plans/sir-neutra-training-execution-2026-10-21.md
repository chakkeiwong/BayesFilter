# SIR NeuTra training and surrogate-HMC execution

## Active question and authority

Does a target-specific canonical IAF make the existing SIR surrogate-HMC
sampler useful? The target remains the T=50 SIR data and literature prior.
Training uses the finite LEDH log likelihood plus that prior; Metropolis
accept/reject uses the existing independent fixed-design endpoint plus the
same prior. These are different numerical likelihoods. Training loss cannot
establish posterior accuracy or convergence of the sampler.

The owner requested prerequisite completion, thorough review, and execution.
The allowance is **16 GPU hours for prerequisites/training and 24 additional
GPU hours for downstream HMC**, explicitly confirmed in this conversation.
Charge compilation, monitoring, failed attempts, and retries to their phase.
CPU engineering checks are separate. Do not overwrite previous evidence.

Owner correction, 2026-10-22: obtain one working NeuTra method; a comparison
with dense-affine HMC is not requested. The dense-affine attempt is stopped
and its elapsed time remains charged. Reuse completed training and frozen-map
checks. Give the entire remaining HMC allowance to frozen NeuTra, with the
existing adaptive warmup and cumulative sampling rules. No comparison run or
agreement with another sampler is a prerequisite for this objective.

This is the active executable plan. The technical/source audit remains in
[the parent plan](surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md)
and [integration audit](surrogate-hmc-sir-neutra-integration-audit-2026-10-20.md).
Their old unimplemented/awaiting-budget statements are historical. The
two-update optimization result is engineering evidence, not a trained map.

## Evidence contract and research intent

| Item | Predeclared rule |
|---|---|
| Candidate | Author IAF, three stages, two ELU hidden layers, widths 3/6/9, shared core, cap 2 with free bias outside the cap |
| Comparator | No second sampler; assess NeuTra against its cumulative R-hat convergence criterion |
| Primary criterion | Frozen-map adaptive HMC completes valid retained sampling with cumulative modern R-hat <=1.01 in all three parameter-chart coordinates |
| Promotion veto | Wrong density/score pairing, broken map/change-of-variables, unsupported scope, nonfinite accepted state, missing required validity/R-hat diagnostics, or failure of the declared R-hat criterion; ESS and MCSE cannot veto completion |
| Continuation veto | Broken target/math or corrupted artifacts, source changes during a run, resource conflict, or exhausted allowance; failed candidate alone is not a campaign veto |
| Repair trigger | Invalid initialization, rejected optimizer updates, failed tuning or no warmup readiness; try the next predeclared candidate/repair while budget remains |
| Explanatory diagnostics | Bulk/tail ESS, mean MCSE/SD, loss, geometry, endpoint discrepancy, acceptance, finite energy errors, FD step-size sensitivity, precision differences, tuning-stage R-hat and timing |
| Forbidden inference | A low ELBO, finite smoke, or smaller gradient discrepancy proves posterior coverage; the fixed numerical endpoint is the exact SIR posterior; one seed proves superiority |

The already-completed identity/diagonal/NeuTra transition diagnostics remain
historical evidence from the earlier comparison plan. No further heuristic or
dense-affine run is scheduled. Comparative superiority and a new repository
default are outside this owner's narrowed objective. Convergence here refers
to the existing fixed numerical endpoint, not the exact SIR posterior.

## Prerequisites and implementation audit

1. Freeze current sources, data, prior, particle design, seed policy and output
   root. GPU0 only; FP32 with TF32 off, N=4032, K=2016, T=50; memory growth
   verified before initialization. HMC coordinates/map evaluation remain FP64;
   the expensive analytical LEDH force remains FP32.
2. Recalibrate the actual FP32 filter scopes with fresh calibration/validation
   simulations disjoint from the fixed inference observations. The old SIR
   selection used mixed precision and is only a warm start. Keep owner-selected
   epsilon=1 and five Sinkhorn iterations; calibrate defensive mixture weights.
   Validate native B=4 training and B=2 HMC with the *same* frozen controls and
   weights. Preserve precision discrepancies as warnings; nonfinite/incomplete
   trajectories remain invalid. No arbitrary percentage score-change veto.
3. Check the actual SIR training consumer, including chart and prior, at
   center/dispersed/tail positions. Report finite differences over several
   FP32-resolvable step sizes; do not use one tiny difference as an oracle.
   Evaluate candidate initialization scales 1, .35, .1, .02 on independent
   Gaussian batches. Choose the widest completely finite tested initialization,
   explicitly as a hypothesis rather than a posterior scale estimate.
4. Use native B=4 shards of logical B=8, one batch reduction/gradient clip/Adam
   update. Existing whole/shard and exact-resume tests remain prerequisites;
   add executable tests for master seed advancement, checkpoint recovery,
   budget accounting, phase progression and scope binding. No sample-wise target
   loop, pfor, Python scalar target or NumPy runtime path.
5. Independently monitor post-update target evaluations; checkpoint optimizer,
   parameters, accepted step, attempted step, next seed, schedule position and
   charged time. Invalid updates consume their seed and time. Never silently
   drop invalid rows or reuse a rejected batch's random seed.
6. Reevaluate the selected map after FP32-to-FP64 conversion; verify inverse,
   log determinant, analytic pullback and same-endpoint HMC wiring. The existing
   two-update conversion test is a prerequisite, not a substitute for this map.

## Training protocol and assumption audit

Widths 3/6/9 and learning rates .001/.01 form six target-specific candidates.
These are hypotheses grounded in the audited author family, not promoted
defaults. Use stateless latent Gaussian draws in the GPU graph. No external
replay dataset is needed for reverse KL. Keep train, monitor, selection and
final heldout seed families separate. Training objective is standard pathwise
reverse KL with the analytical target score; no differentiation through the
filter and no score substituted from the independent MH endpoint.

Run 64 accepted updates per candidate; nominate up to two for 256 updates.
Loss on an independent common selection bank nominates candidates only. Then
run three fresh initialization/noise seeds for the nominated configuration to
512 updates each, subject to the total 16-hour cap. This is a budget ladder,
not an assertion that 512 updates suffice. Report independent monitoring at
each rung, learning curves and replicated uncertainty. Completing all three
fits is `REPLICATED_UPDATE_BUDGET_COMPLETED`; convergence remains
`NOT_ESTABLISHED_BY_UPDATE_COUNT`. Fewer completed fits are `TRAINING_INCOMPLETE`.
A frozen map may still receive an explicitly exploratory downstream test;
continuing loss trends must be reported as unresolved fitting, not convergence.

Use a declared learning-rate schedule: base rate through update 127, half rate
128–383, quarter rate thereafter. This is a local schedule hypothesis, not
the original author's schedule. Adam (.9,.999,1e-7) is the existing optimizer
baseline. Clip norm 1 is a hypothesis: report raw norms and clipped fraction,
and compare the first update with clip 10 and unclipped on the same batch.
Persistent clipping or invalid proposals explains failure and triggers the
lower-rate arm; clipping is not evidence of convergence. The author cap c=2
is frozen architectural policy; record activation saturation, free biases,
log determinants and Jacobian conditioning rather than retuning it silently.

Outer center zero is a warm-start hypothesis in the existing scientific chart.
The initialization bank challenges its scale; old nonconverged warmup draws
are not labeled posterior draws. Reverse KL can miss modes. Monitor separate
central/dispersed/tail behavior and use independent overdispersed HMC starts.

Reserve at least 90 minutes of the training allowance for final checks and
serialization. At the measured ~16 seconds/update, 2304 planned updates cost
~10.2 hours before compilation and monitoring. Actual timing governs remaining
rungs; incomplete replication stays an explicit limitation.

## Downstream surrogate-HMC protocol

Freeze transport before HMC. No online map or filter retuning. Tune step size
for NeuTra toward acceptance .70; acceptance is not convergence evidence.
Use the existing shared sequential controller: four chains, at least 2000
warmup transitions, recent 1000 at modern R-hat <=1.05, max 10000; retain
sampling cumulatively to R-hat <=1.01 or 20000 draws/chain or the time cap.
For downstream assessment require valid completed sampling and cumulative
parameter-chart R-hat <=1.01. Report pooled bulk/tail ESS and mean MCSE/posterior
SD as precision diagnostics only, per the owner's 2026-10-23 clarification.
They cannot reject convergence, trigger additional sampling, or make a completed
run incomplete. The sampler stops at its declared R-hat/sample/time rule.
Archive all warmup but exclude it from inference. No extra validation chains.
Large **finite** energy errors remain warnings, never a run stopper.

Allocate all remaining HMC time to frozen NeuTra, including its own kernel
tuning, adaptive warmup and retained sampling. Preserve the charged time of
the stopped dense-affine attempt and completed mechanics diagnostics. The
master must not require a dense-affine result or launch a comparison retry.
If actual transition cost prevents convergence, report under-budgeted
posterior evidence rather than a failed mathematical method.

Skeptical audit of this correction: removing a comparator does not change the
sampling target, map, prior, force, MH endpoint, tuning rules or diagnostic
thresholds. A trained map alone still cannot pass. Missing/nonfinite diagnostic
components or incomplete retained sampling must fail assessment. Regression
checks will verify that the live consumer receives the frozen NeuTra payload,
the master launches only that method, all remaining HMC time is available,
prior charges survive continuation, and a qualifying NeuTra result needs no
second sampler. No speed, superiority, exact-posterior or default claim follows.
This audit passes for the narrowed objective; the owner's instruction directly
authorizes the change without increasing the total budget.

Implementation check: all 24 focused master tests pass in 20.35 seconds with
GPU intentionally hidden; `git diff --check` passes. The new tests exercise the
actual worker-to-public-inference call with the frozen map for fresh and resumed
HMC, reject a comparator job, preserve prior HMC charges, and assess convergence
without a second result. The stopped comparator used 6.072536 hours; 17.674803
of the original HMC allowance remain. Prior training and freeze results are
reused. This is an execution/assessment repair, not new sampling evidence.

## Execution, review and artifacts

Master: `scripts/run_sir_neutra_training.py`; current default output root
`docs/plans/artifacts/sir-neutra-training-20261022-neutra-only01`. The previous
`docs/plans/artifacts/sir-neutra-training-20261022-repair01` preserves the map
checks and stopped comparator. The original root
`docs/plans/artifacts/sir-neutra-training-20261021-01` preserves training evidence. Each worker attempt has
a fresh directory and log. `campaign.json` records budgets, attempts and
progress; `next-phase.json` is refreshed after each result, including repairs.
Every worker records git/source hashes, exact command, environment, GPU/memory
policy, seeds, data/target identity, elapsed time and result location.

Command (trusted GPU execution):

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_sir_neutra_training.py run
```

Status uses the same script's `status` command. Repeating `run` resumes saved
work, charging previous and failed attempts. A local infrastructure retry is
authorized within the same target, method and allowance; it does not require
new scientific approval. Source changes require a recorded repair and fresh
affected scope/checkpoint, never silent reuse.

Skeptical review found and corrected three material gaps: stale mixed-precision
tuning, a two-update fixture presented as the only executable training stage,
and no downstream allowance. The revised plan also distinguishes training
nomination from downstream evidence, budgets replication and retries, and
preserves finite-energy warning semantics. Before launch, the code-level audit
must check seed/resume, exact scope consumption, time caps, and controller
wiring. Record its test results here.

Code audit, 2026-10-21: 71 focused CPU diagnostic tests pass in their latest
executions. These cover the real SIR native adapter and nonlinear pullback,
public fresh/resumed HMC, shared-controller wiring/route discovery, exact Adam
and random-stream continuation, rejected-batch stream advancement, budget
recovery, batch-specific tuning identities, precision-warning versus invalid-
trajectory behavior, and downstream assessment. The audit repaired an XLA
monitor slice-shape error and cached-stage recovery under exhausted budgets.
The shared NeuTra entry point now dispatches the existing public callback
controller without changing its sampling or replay algorithm. CPU runs
deliberately hid GPU devices. Full logs and preparation summary are saved in
the campaign's `preparation/` directory. No remaining known engineering blocker
prevents the bounded GPU prerequisite run; actual initialization, numerical
health, and learning remain empirical questions.

Pre-mortem: a finite, low-loss map may miss a mode; a healthy LEDH training
density may poorly match the MH endpoint; an arbitrary initialization may
fail before learning; compiler cost may consume the budget. Separate these
using dispersed banks, same-point endpoint differences, initialization/gradient
telemetry, and per-attempt timing. None is resolved merely by successful exit.

Terminal result must include the decision/inference-status tables, the strongest
alternative explanation, missing evidence and exact next justified action.
All ranking claims remain conditional on uncertainty and posterior validity.

## Assessment correction — 2026-10-23 owner directive

The owner clarified that ESS is diagnostic, not a convergence criterion.
Completion therefore requires the existing sampler completion/health result
and finite three-coordinate modern R-hat <=1.01. Report bulk/tail ESS and
mean MCSE/SD as precision diagnostics only; neither can veto completion,
trigger further sampling, or create another validation phase. This supersedes
the earlier precision-based qualification and its saved historical incomplete label.

Bounded repair: change only master assessment/reporting, add a CPU-only saved
result reassessment command, and reclassify the completed run. Preserve the old
assessment in a versioned record, all raw outputs, runtime source provenance,
attempts and budget charges. No training, tuning or HMC worker is launched.
The baseline is the identical saved sampler result, not another method.

Pass criterion: low/missing precision diagnostics do not change convergence
classification, while missing/nonfinite/above-threshold R-hat or incomplete
sampling still prevents completion. Regression tests must also show that
reassessment preserves raw results and charges and refuses an active campaign.
Record the revised assessment under the active output's `reassessments/`
directory and refresh the current report, phase and checkpoint. Budget: focused
CPU tests and standard-library reporting only, at most five CPU minutes and
zero additional GPU time. Failure of these checks triggers repair, not relaunch.

Skeptical audit passes: this fixes a mismatch between the agreed stopping rule
and the reporting criterion. It does not alter the target, draws or diagnostics,
tune a scientific method on its outcome, waive sampler validity, infer exact
posterior accuracy, or establish convergence from ESS. R-hat <=1.01 means the
declared convergence diagnostic passed; it is not a mathematical proof.

Executed: all 35 focused CPU-only tests passed in 15.05 seconds. The command
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_sir_neutra_training.py reassess`
reclassified the saved run as `NEUTRA_POSTERIOR_CRITERIA_MET` without importing
TensorFlow or launching a worker. Prior assessment, runtime provenance and
budget accounting are preserved in
[reassessment 001](artifacts/sir-neutra-training-20261022-neutra-only01/reassessments/001.json).

## Execution checkpoint

Updated 2026-10-23 after terminal assessment. The NeuTra-only master and worker
finished normally at 03:57:44 Hong Kong. Scope tuning, training, freeze, kernel
tuning, adaptive warm-up and retained sampling have completed. The four chains
have 2,000 warm-up and 2,000 retained transitions each. Warm-up is excluded from
inference; all 8,000 retained draws are inference draws.

The cumulative sampler passed its declared stopping rule. Maximum modern
R-hat is 1.009824 in latent NeuTra coordinates and 1.009603 in the SIR parameter
chart. Parameter-chart bulk ESS is [412.28, 595.13, 475.48], tail ESS is
[199.29, 1129.09, 769.04], and mean MCSE/SD is [0.05317, 0.04134, 0.04599].
The master now records `NEUTRA_POSTERIOR_CRITERIA_MET`, matching the sampler's
`VERIFIED` status under the agreed R-hat criterion. ESS and MCSE describe
precision only. The previous ESS >=400 qualification was a reporting error
and has been removed; its original assessment is archived for provenance.

Acceptance is 75.90–79.30% by chain (77.70% pooled), with zero recorded
divergences and zero force fallbacks. Numerical, endpoint and movement checks
pass. Maximum absolute retained energy error is 6.44571, explanatory only.
The tested target remains the independent fixed bootstrap 4x4032 approximate
posterior with the scientific SIR prior; exact-posterior agreement is unchecked.

All prior attempts remain charged. Remaining budgets are 8.4580 HMC hours and
8.5678 training hours, from the approved 24 and 16 hours respectively. No worker
remains active. No target, map, R-hat threshold or compute allocation changed.
Only master assessment/reporting changed after the worker finished; the run's
original source hashes and outputs remain preserved.

The selected canonical IAF is width 6, two hidden layers per stage, initial LR
.01, scale .35 and seed 201. Three replicas completed 512 accepted updates each;
that update count alone does not establish training convergence. GPU geometry
checks passed (inverse error 8.88e-16, log-determinant error zero). The master
previously passed all 24 focused CPU tests. The assessment repair now passes
35 focused CPU tests, including a real CLI reassessment and preservation of
saved draws, provenance and charges. No additional GPU computation was used.

The earlier 1,000-draw CPU observation remains preserved in the linked note.
It excluded warm-up, verified tensor checksums, restored NeuTra z from the
additional affine HMC coordinates, and checked the actual affine/R-hat call
chain. Its covariance eigenvalues [0.9342, 1.2375, 1.5761] and maximum absolute
correlation 0.0850 are provisional second-moment diagnostics, not a final
whitening or Gaussianity claim.

Recovery action: report that SIR NeuTra passed the declared convergence
diagnostic. Report ESS and MCSE separately without a pass/fail threshold.
There is no pending automatic phase or required precision extension in this
master run. Do not rerun training or sampling merely to refresh status.

Evidence: [terminal result](artifacts/sir-neutra-training-20261022-neutra-only01/final-result-20261023.md),
[structured assessment](artifacts/sir-neutra-training-20261022-neutra-only01/hmc-result.json),
and [earlier bounded observation](artifacts/sir-neutra-training-20261022-neutra-only01/observations-20261023-01/result.md).
