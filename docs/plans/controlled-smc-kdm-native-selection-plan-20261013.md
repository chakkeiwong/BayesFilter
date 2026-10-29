# Native validation selection diagnostic

13 October 2026. Authorized continuation of the replication master. This is a
bounded independent-author CPU reference diagnostic, not a new training ladder.

## Question and evidence contract

Does validation randomness at the native 1,000-row size materially change
selection between nearby checkpoints? Do the two saved unfavorable TF-trained
checkpoints still have worse validation NLL when the author evaluates all rows?
The comparator is the pinned author's own validation epoch and frozen source
and TF checkpoints for paired units 01 and 02. The claimed quantity is the
author's batch-weighted KDE NLL, with 25 particles, 17 time points, and batches
256/256/256/232. It is not heldout test quality or population noninferiority.

Run eight independent validation repetitions per checkpoint in each of two
arms: independent initial RNG states across checkpoints; shared initial RNG
state across checkpoints. The latter is NOT guaranteed common realized draws:
rejection sampling can desynchronize random streams. Record final RNG hashes.
All comparisons are conditional on these checkpoints and these validation data.
Repeated evaluations are nested observations, never independent training units.

To obtain real intermediate states, restart the author training epoch from the
saved full unit01 TF-best state, including Adam moments, step and current rates.
Preserve the initial state and states after each of two complete 5,000-row
epochs. Use native shuffled batches of 64 (last batch 8), gradient clipping100,
truncated BPTT4, model modes and optimizer updates. Rates remain frozen at the
saved values for this two-epoch counterfactual. This is not a reconstruction of
the historical checkpoint sequence or a completed native training protocol.
Do not combine an author best model with terminal optimizer state.

Primary diagnostic: per-checkpoint NLL distributions and paired differences;
nearby-checkpoint winner frequencies and selection regret assessed on the last
four draws after selection using each of the first four draws. Report SDs and
all raw repetitions. No threshold here promotes the replication. The unchanged
quality criteria are 0.01 nat NLL and 1% RMSE on independent training units.
RMSE is not evaluated by this selection-loss experiment.

Validity/continuation vetoes: changed source hashes; incorrect data/checkpoint
identity; missing optimizer state; changed training batch count; nonfinite
loss/parameters/gradients; repeatability or call-chain control failure; exceeded
time/attempt budget. A finite unfavorable result is evidence, not a veto.
Infrastructure failures trigger a localized repair and fresh output directory.
Seed sharing, small-window ordering and noise size are explanatory diagnostics.
Neither stable nor unstable selection proves the cause of historical training
differences. No numerical algorithm, clipping rule, or acceptance margin changes.

## Defaults, assumptions and skeptical review

| Choice | Provenance and reason | Failure mode / earliest check | Role |
|---|---|---|---|
| Native N/T/data/batches/loss | Pinned bearings configuration and saved source data | Wrong endpoint or weighting; compare actual validation epoch with manual calls to the same forward method and batch sizes | Baseline |
| Saved TF-best full state | Exact stored moments/step/rates, matching prior replay | Cold or mismatched restart; exact tensor restoration checks | Diagnostic start |
| Two epochs, fixed rates | Bounded local sensitivity window, no tuning | Cannot represent historical selection; explicitly preserve limitation and all states | Convenience hypothesis |
| Eight draws, first4/last4 split | Small diagnostic panel within 30min cap | Noisy tail/selection estimates; raw values, no population claim | Diagnostic design |
| Shared seed arm | Tests practical RNG coupling without changing native sampler | Draw desynchronization; record terminal RNG hashes and label correctly | Hypothesis |
| Independent seeds | Seed2026101300 plus documented unit/mode/checkpoint/draw offsets | Accidental reuse; emit every seed, preserve per-call repeat control | Reviewed diagnostic design |
| CPU author reference | Pinned PyTorch source environment, GPU intentionally hidden | Not a TF/GPU runtime or speed claim | Explicit reference exception |

Research intent: distinguish native-size validation uncertainty from persistent
checkpoint quality differences. A matching author evaluator/metric cannot by
itself establish faithful training. Baseline ladder and heuristic adversaries
are inapplicable to this measurement-only question: no proposed method is being
ranked or promoted; the unchanged native author evaluator is the exact baseline.

Pre-mortem: a run can succeed yet mislead if a short nearby window is called the
historical selection process, if eight draws become eight training seeds, or if
shared RNG state is called exact common variates. These interpretations are
forbidden. A failed source epoch may be intrinsic source behavior; preserve it
and distinguish that outcome from a harness failure before repair.

Skeptical audit PASS before implementation/execution: correct author comparator,
native validation size and weighting, exact checkpoint identities, preserved
partial batches and optimizer state, bounded counterfactual rather than fake
historical replay, no proxy promotion, explicit CPU exception, finite budget
and stop conditions. The master wording about common draws is corrected to
shared initial RNG state with possible sampler desynchronization.

## Execution and artifacts

New supervisor: `docs/benchmarks/run_kdm_native_selection.py`.
Worker: `docs/benchmarks/reference_kdm_native_selection.py`.
Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_native_selection.py`.
The supervisor launches the pinned author interpreter
`/home/chakwong/anaconda3/envs/tf-gpu2/bin/python`, with
`CUDA_VISIBLE_DEVICES=-1`, OMP threads 1 and its environment library path.

Phase root: `docs/plans/artifacts/controlled-smc-kdm-replication-20260928/native-selection-20261013`.
Two attempts maximum, 1,800s per attempt, 3,600 CPU seconds total, zero GPU.
Use fresh attempt directories; charge all wall time to the campaign ledger.
Save Git/source/data/checkpoint identities, complete commands, source snapshots,
parameters/moments at each window checkpoint, RNG states/hashes, batch results,
progress, raw metrics, environment and wall time. Result:
`docs/plans/controlled-smc-kdm-native-selection-result-20261013.md`.

Afterward, audit validity and report the remaining independent-training-unit
precision problem. Do not launch an unaffordable full ladder or relax criteria.

## Terminal precision feasibility calculation

After the native selection diagnostic, recompute the existing paired-calibration
summary from its three independent units and compare the unchanged illustration
with the remaining ledger budget. This calculation uses existing evidence only.
For each of four endpoints, alpha=.05/4, power=.8 at true equality and paired SD
s give n=ceil(((z_(1-alpha)+z_.8)*s/margin)^2), with margin=.01 for NLL and
log(1.01) for log RMSE ratio. This follows from the normal approximation
SE=s/sqrt(n): the noninferiority upper bound lies below the margin with 80%
probability at zero true difference. It assumes normal paired means and treats
the very uncertain n=3 SD as known; report only as a planning illustration.
It does not validate a sampling distribution or give a definitive required n.

Use actual full source and TF pair runtimes from unit01 and unit02 manifests.
Charge a new pair both its source and TF CPU wall time and its TF GPU wall time.
Compute optimistic and slower affordable pair counts using these observed
costs, with no extrapolated speedup, no reuse of development units as untouched
qualification, and no budget increase. Compute the paired SD that would be
needed for an80%-power design with the optimistic affordable pair count.
Artifact: phase-root/precision-feasibility.json, including input hashes.

Decision: a large mismatch establishes that the currently proposed empirical
precision design is unjustified within budget; it does not prove every possible
design impossible. Freeze the algorithm and preserve acceptance criteria.
A cheaper design needs demonstrated variance reduction or an explicit owner
decision about qualification scope. Do not spend the budget on a ladder whose
expected precision cannot answer the criterion.
Audit PASS: accepted margins and four-endpoint adjustment preserved; training
unit is the unit of inference; evaluation draws cannot increase its count;
no inference of intrinsic-paper failure or population ranking.

## Terminal status

CLOSED after attempt01, 354.036491439 CPU seconds and zero GPU.
96 full-validation repetitions and two native-batch training epochs completed;
2638 worker checks and 479 terminal checks passed.
The post-run precision contract above was added before that calculation and
preserves the original paired-calibration assumptions; the launch snapshot
preserves the pre-run selection plan. No new ladder is justified.
See the result for scientific interpretation, uncertainty and next evidence.
