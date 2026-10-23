# Recoverable SIR neural training continuation

21 October 2026. User request: review the plan and execute, on
`ledh-refactor-with-policy-fix`. The completed independent comparison remains
negative; this stage must not revise it or reuse its exposed holdout.

Terminal status: the recovery repair and nominated initialization's512-update
continuation are complete.43 CPU checks and all76 terminal inspection checks
pass. Cumulative charge21785seconds leaves1981.197895057seconds. The other
initialization's optional512 rung does not fit that remainder. See the
[terminal result](artifacts/ledh-sir-neural-kdm-20261020/attempt-20-recovery/result.md).
The sections below preserve the pre-run audit and its measured timing amendment.

## Question and skeptical review

Can the original validation-nominated K2/width16/beta.9/Adam.001/seed263044001
candidate continue from its exact 128-update neural/Adam state without losing
accepted updates when the process budget ends? The optional512 rung lost its
234 accepted updates because checkpoints were written only after final
validation. Training rows alone do not reconstruct Adam state.

The original full512 continuation cannot safely be promised: its last20 updates
had a median27.97seconds, so384 remaining updates alone project10740seconds,
more than the4891.197895057seconds remaining. Do not relaunch that command.
The selected128 checkpoint, not the lost362-update state, is the resume source.

Repair checkpoint observability without changing any numerical package source,
loss, network, optimizer, random stream or scientific baseline. A thin runner
must call the existing `NeuralKDMStudy.fit` and its actual GPU/XLA update; it
must not copy the training kernel or relax source-hash checks. Recovery states
are not validated candidates and must be marked unselectable. Preserve complete
parameters, Adam slots/iteration, data index, configuration and source hashes.

The audited execution is a training-only continuation, not a new method
comparison. Complete an intermediate192-update checkpoint, then256 only if
measured time conservatively fits. These are bounded intermediate steps toward
the originally proposed512 rung, not evidence that training is adequate.
Freeze the choice of this seed/configuration from the original validation
nomination; do not choose it using the final losses or heuristic comparisons.
At this initial review, the full512 rung remains incomplete; both initializations
have already completed128. The later timing amendment below governs further work.

## Evidence contract and intent ledger

- Baseline: exact saved128-update parameters, Adam state, data index and package
  hashes from attempt14; previous independent evaluation stays immutable.
- Engineering pass: recovery resumes the same next update/seed and preserves all
  tensors; invalid, incompatible or incomplete state rejects. CPU tests check
  interrupted versus uninterrupted continuation and actual-consumer wiring.
- Training pass: only finite accepted updates, all six parameter gradients
  reported, unchanged package source and complete validation at finished rungs.
- Continuation veto: source drift, numerical invalidity, corrupted checkpoint,
  wrong data index, device/memory-policy failure or exhausted total budget.
- Repair trigger: interrupted runtime with a valid recovery checkpoint. One
  localized retry is allowed only under the same total cap and contract.
- Explanatory only: training/validation loss and runtime; none can establish
  independent moment improvement, true-model score accuracy or default readiness.
- Promotion: forbidden in this stage. All existing heuristic vetoes remain.
  The constructed prediction/observation-only/recursive-Gaussian/conditional-only
  adversaries and observation/latent regimes from the original plan must be
  evaluated on new untouched data in a later comparison, never used for tuning.

## Defaults and pre-mortem

All numerical settings are inherited diagnostic comparability controls, not
universal defaults: SIR18,T5,N64,FP64,GPU/XLA,TF32off,OT(1,5),balance32,K=N.
Endpoint validity, source parity and actual gradients remain their early checks.
The learning rate, architecture and Adam coefficients are the previously
screened hypothesis; validation regression is possible and must be reported,
not hidden by selecting on final results. Recovery every8 updates matches the
existing progress cadence; it limits loss after hard termination without
altering accepted numerical operations. Final/timeout saves retain the latest
accepted state. CPU-only recovery fixtures intentionally hide CUDA.

A successful run can still mislead: restored counters alone do not prove slot
parity, validation loss is not independent performance, and one initialization
cannot establish robust learning. Tests must compare every restored tensor and
the next update, not just filenames. GPU evidence must record memory growth and
the existing scoped command-buffer workaround. No new numerical safeguards.

## Execution and budget

1. Implement the thin recovery runner and focused CPU regression checks. Record
   exact test command and outcome under a fresh `attempt-20-recovery` directory.
2. Verify GPU availability with escalation. Start one GPU process, at most4000
   seconds including compilation/validation; reserve the remaining891.197895057
   seconds for engineering probes, accounting and a bounded localized retry.
3. Continue128 to192; project192 to256 using recent upper-tail update costs,
   measured validation time and setup reserve. Do not start an unaffordable rung.
4. Save terminal status, commands, environment, seeds, checkpoint lineage, source
   identity, valid updates, elapsed/charged/remaining time and limitations.
   Refresh the master/checkpoint; do not reopen or overwrite prior evidence.

The reviewed recovery regressions pass33 CPU checks; details are recorded in
`artifacts/ledh-sir-neural-kdm-20261020/attempt-20-recovery/review.md`.
The GPU inventory probe is charged1second, leaving4890.197895057. Exact child
command, executed with an external3990-second TERM timeout and10-second grace:

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b XLA_FLAGS=--xla_gpu_enable_command_buffer= PYTHONPATH=. /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/continue_sir_neural_kdm.py --output docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-21-continuation --resume docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-14-recovered/rung128-seed263044001/checkpoint-0128.json --budget-seconds 3950 --campaign-remaining-seconds 4890.197895057
```

A budget
stop with a valid recovery checkpoint is successful recovery engineering, not
completed training or a negative scientific result. This audit passes for the
bounded recovery/continuation question only; a new independent claim run needs
fresh evaluation partitions and an adequate separately recorded budget.

## Audited continuation after the 256-update result

Attempt21 completes192 and256 with128 finite accepted updates. Every replayed
loss, gradient norm, block loss and optimizer flag equals the corresponding
original interrupted512 row exactly. The validated256 checkpoint and its recovery
copy have identical neural/Adam state. All834 package hashes remain unchanged.
Validation losses are.6825563973 at192 and.6715514501 at256; these are descriptive
optimization diagnostics, not new independent performance evidence.

The complete external process took1105.86seconds from the recorded launch second
to the exit-status timestamp. Charge1107seconds, including startup and a rounding
guard, rather than the child1048seconds that excluded imports. Together with the
inventory probe, cumulative campaign charge is19983seconds, leaving3783.197895057.
See attempt21's `outer-accounting.json`; the child result is not authoritative
for total campaign accounting.

The original optional512 rung is now affordable as an attempt. The last20 updates
have median5.94seconds,90th percentile6.91seconds and maximum7.04seconds. A
conservative projection for256 more updates, two60-second validations and
180seconds startup is `1.25*(256*6.91+120)+180=2541.2`seconds. This is below a
3500-second internal cap and3700-second external cap. The earlier slow
27.97-second median is a real risk; it is not assumed impossible. Periodic
recovery and a384-update validation checkpoint bound lost work if it recurs.
The384-to512 step must independently pass the same measured-cost check.

Continue this same validation-nominated seed from256 to384, then512 if affordable,
under a new versioned attempt22. The existing scientific contract, dataset streams,
numerical package, method, hardware and total budget are unchanged. The thin
runner only gains an explicit target and intermediate-rung schedule; rerun the
focused recovery checks before launch. Never reopen the old holdout, infer
independent improvement, or promote this candidate. A second initialization's512
rung remains conditional on the remaining budget, not silently completed.

The skeptical extension audit passes: this advances the original optional budget
ladder, uses validation-nominated configuration rather than exposed final results,
preserves exact state and source, and cannot exceed the remaining total cap.
Recorded full command (external3690-second TERM timeout plus10-second KILL grace):

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b XLA_FLAGS=--xla_gpu_enable_command_buffer= PYTHONPATH=. timeout --signal=TERM --kill-after=10s 3690s /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/continue_sir_neural_kdm.py --output docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-22-continuation --resume docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-21-continuation/checkpoint-0256.json --target-updates 512 --budget-seconds 3500 --campaign-remaining-seconds 3783.197895057
```

## Terminal execution and review

Attempt22 completes384 and512, with validation losses.6822232422518024 and
.7070644728464445. All256 new updates are valid/applied and all106 overlapping
rows exactly reproduce the original interrupted run. Combined with attempt21,
all234 overlapping rows match. All834 package hashes and saved-state checks pass;
the terminal inspector reports30/30 and46/46 checks. No holdout is reopened.

Attempt22's inclusive external duration is1800.376363765seconds, charged1802
with the existing rounding guard. Campaign charge becomes21785 of
23766.197895057seconds;1981.197895057remain. The second initialization needs at
least3872.684384622seconds under the same conservative projection, before
validation. Its optional512 rung is therefore under-budgeted. The required128
stage is complete for both initializations; no later claim run is authorized by
this training-only continuation.

The engineering decision passes. The validation pattern is descriptive, with
loss falling at256 then rising through512; it does not establish independent
moment benefit, overfitting, convergence or score accuracy. The previous failed
10% criterion and heuristic promotion veto remain unchanged. The linked result
records decision and inference-status tables, exact manifests, budget and the
post-run red-team review. No numerical continuation veto or direction rejection
has occurred; the next optional rung cannot fit the remaining budget.
