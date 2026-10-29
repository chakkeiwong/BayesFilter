# Surrogate-HMC sequential recovery master

## Goal and scope

Test surrogate HMC on the existing M13 data1704/T20/state4/observation2/parameter2
scope. The canonical analytical DCTR field proposes moves; the separately frozen
full-support 4096x8 trajectory-importance endpoint plus the same prior defines the
MH target. Do not claim the field is the exact endpoint gradient. FP32/TF32/GPU/XLA
force and the FP64 endpoint remain unchanged. Data3701 remains untouched.
LG1 and previous runs remain historical evidence, never silently relabeled.

## Research intent and evidence contract

Question: does a properly warmed, cumulatively checked surrogate-HMC kernel
produce a qualified sample from the frozen M13 target?
Candidate: the same force/endpoint with the existing public tuner and repaired
phase control, not a newly trained force.
Expected failure: first-checkpoint R-hat may fail while later checks pass.
Primary tuning criterion: fresh cumulative modern R-hat <=1.01 after separately
passing warm-up readiness. Retained promotion additionally requires the existing
bulk/tail ESS >=400 and two-look MCSE-aware comparison to the qualified reference.
Promotion vetoes: failed mixing/ESS/reference comparison at the declared maximum.
Continuation vetoes: invalid endpoint, nonfinite states/energy, divergences,
no chain movement, force fallback in fresh phases, source/binding drift, corrupted
archives, or a hard resource cap. R-hat failure before the maximum is NOT a
continuation veto. Retuning a rejected candidate is a separate prospective phase.
Explanatory only: acceptance, first-look R-hat, runtime, and nomination scores.
Heuristic comparators: existing same-target RWM (mixing control), independent
refined posterior grid (accuracy authority with recorded finite-box limitations),
and prior-centered Gaussian geometry (initial tuning hypothesis, not a sampler
accuracy authority). Preserve existing reference, coverage and RWM checks.
Evaluate interior and tail behavior through the existing mean/SD/CDF functions;
do not select a candidate on the reference comparison.
Passing tuning does not establish posterior correctness, efficiency superiority,
exact-likelihood inference, all-model success, or production readiness.

## Implementation and skeptical audit

1. Add sequential readiness and verification inside the typed public tuner.
   Warm-up minimum2000/chain, latest1000 window, modern R-hat<=1.05.
   Check every1000, archive/check health in250-transition chunks, max10000.
   Verification starts only after readiness; cumulative checkpoints1000,...,10000,
   R-hat<=1.01. Reuse chain state and frozen kernel; use unique deterministic seeds.
   Stop healthy failures only at the declared draw/resource maximum.
2. Issue a new verified-artifact schema. Reload must recompute sequence health,
   latest-window readiness and cumulative verification from saved tensors.
   Old fixed-length artifacts cannot acquire sequential authority by new labels.
3. Keep retained sampling's calibrated looks(1000,2000) and existing maximum2000.
   Do not expand looks without redoing the uncertainty procedure and budget.
4. Repair shared launcher supervision: independent worker guard, parent-loss
   detection, worker CPU/wall limits, durable heartbeat and final usage receipt.
   Recover partial runs from those receipts; never charge a missing receipt as0.
5. Provide one executable recovery master with preflight, tests, GPU smoke,
   public tuning, conditional retained sampling, status and reconciliation.
   After each phase, atomically save status, scientific decision, charges,
   remaining budget and a refreshed next-phase plan. Launch the next phase only
   from that refreshed state. Expected checkpoint failures extend automatically.
6. Add focused regressions for pass-after-extension, never-ready warmup,
   cumulative-versus-latest verification, health/fallback stop, caps, seed/state
   continuation, archive tampering, public wiring, parent loss, accounting and
   next-phase transitions. Run existing relevant regressions.

Audit findings BEFORE execution: the old verifier's fixed500/1000 phases could
misclassify unfinished warmup as kernel failure; parent-only watchdogs could lose
both enforcement and receipts. These are implementation/evidence defects, not
evidence against surrogate HMC. A plain exact-gradient/NeuTra runner cannot
receive this arbitrary field. The repair retains the typed endpoint/force binding.
Finite health is not convergence; tuning R-hat is not posterior admission.
The retained comparison already stages correctly and must not be needlessly
rewritten. Expanding its two calibrated looks would invalidate its stated
uncertainty procedure. Versioned output and ordinary hashes suffice.

## Defaults, assumptions and limitations

| Choice | Provenance and justification | Failure and early diagnostic | Status |
| --- | --- | --- | --- |
| Warmup2000/latest1000/1.05, max10000 | Existing shared BayesFilter sequential policy | Slow mixing; inspect every window and cap | Repository protocol, not guarantee |
| Verification1000 increments/max10000/1.01 | Shared cumulative sampling protocol | Slow mixing; preserve failed looks | Repository protocol, not guarantee |
| Health chunks250 | Existing retained consumer; limits loss and archives progress | Compilation overhead; graph reuse smoke | Operational hypothesis |
| Existing metric/epsilon/L tuning | Current public tuner and target-specific plan | Poor geometry or exploration; fresh gate | Existing hypotheses, not retuned on reference |
| Existing repaired starts | Discarded RWM warmup only | Residual initialization sensitivity; readiness | Recorded hypothesis |
| Retained cap2000/two looks | Existing budget and calibrated comparison | Inconclusive precision; stop honestly | Frozen existing comparison |
| GPU growth/FP32 force/FP64 endpoint/XLA | Owner policy and unchanged target | Device or numerical mismatch; preflight/parity | Existing execution policy |

## Budget and authorization

The user explicitly requested implementation, review, allow-list repair and
execution. Routine repair/tests proceed. One prior attempt lacks a recoverable
CPU receipt; its observed21927.65-second tuning time and22407.85-second artifact
span do NOT reconstruct CPU usage. Preserve that unknown, and reconcile any
actual receipts found without manufacturing a successful completion.

Proposed NEW separately tracked recovery allocation: at most60000 CPU-worker
seconds and60000 GPU-seconds, including tests, smokes, tuning and retained work.
This incremental allocation requires explicit budget approval because the old
remaining CPU balance cannot be established. It must not be presented as falling
within the old198000CPU/226800GPU cap. Await that decision for serious execution.
Development/smokes are routine authorized work and their measured charges are
recorded. Maximum one fresh scientific nomination/verification attempt in this
allocation; infrastructure retries only from preserved progress and remaining
allocation. Reserve12000CPU/12000GPU for the retained phase. Budget caps take
precedence over the nominal10000-draw maximum. No guarantee those maxima fit.
GPU time is elapsed allocated worker time; CPU is measured resource usage.
Every phase refresh must recompute the reservation before launch.

Output root: docs/plans/artifacts/surrogate-hmc-m13-sequential-recovery-01.
New launches use unused versioned roots; no overwrite of prior evidence.
Program: scripts/run_surrogate_hmc_recovery.py (absolute interpreter and script
paths, direct argv, no environment assignments or shell wrappers).
Allow-list adds only that program's prefix and the fixed rules installer.
Environment is configured inside the launcher before TensorFlow import.
Required outputs: run-manifest.json, accounting.json, phase-history.json,
next-phase-plan.json/Markdown, status.json, command logs, verification tensors,
retained assessment and result.md. Refresh at each phase and chunk checkpoint.

## Completion and stop semantics

Implementation complete means focused tests and call-chain checks pass and the
new master actually executes preflight/tests/smoke and authorized research phases.
Scientific completion requires qualified retained assessment. If the draw or
compute cap is reached, record the candidate as inconclusive/rejected for
promotion, preserve the cause and next justified action, and do not pretend the
research direction is rejected. Ask for expanded scope/compute only at that
actual boundary. Review is local; Claude remains deferred as requested.

## Reviewed execution details (2026-09-28)

The CPU allowance counts measured worker, child-worker and watchdog CPU; the
supervisor's small CPU cost is reported separately. GPU charge is elapsed
allocated worker time. The new allocation carries forward all development-test
and check-run receipts. A new output directory cannot reset that allocation or
launch a second fresh scientific attempt. Warm-up/verification and retained
chunk boundaries forecast both CPU and wall cost; independent watchdogs enforce
the ceilings when a forecast misses.

Review found and repaired: fixed warm-up; fixed first verification look; missing
fresh-seed history in retained handoff; missing reports when readiness never
passes; parent-owned watchdog loss; and resettable cross-run accounting. Existing
consumer source paths now request at least2000 warm-up draws for new v2 runs.
Historical saved results remain v1, without retroactive claims.

Run the mechanical validation through the same master:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/scripts/run_surrogate_hmc_recovery.py check --output /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/artifacts/surrogate-hmc-sequential-check-01

After the requested additional allocation is approved, the same absolute prefix
with action start, fresh output surrogate-hmc-m13-sequential-recovery-01,
and --cpu-budget 60000 --gpu-budget 60000 starts a detached supervisor. It
executes checks, Gaussian GPU/XLA qualification, public M13 tuning, then retained
assessment if verification qualifies. The status action with the same output
directory reports progress. No manually launched next phase is necessary.
A terminal candidate failure refreshes the reason and next justified action;
it does not manufacture a passing posterior or new tuning scope.
