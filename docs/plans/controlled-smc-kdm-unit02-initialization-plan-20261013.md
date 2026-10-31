# Unit02 MDPF initialization intervention

13 October 2026. Authorized continuation of the replication campaign. The
master remains `controlled-smc-kdm-replication-master-program-20260928.md`.
This phase reuses completed parity evidence and changes no algorithm default.

## Research intent and evidence contract

Question: does the initial measurement model explain the saved unit02 MDPF
quality gap, as it did for unit01? The controlled comparison is between the
saved unit02 TensorFlow training trajectory and a new trajectory that differs
only in its starting measurement-model parameters. The external baseline is
the saved unit02 author-trained checkpoint, evaluated by the pinned author
code at commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.

The paper explicitly pretrains measurement weights before full filter
training (local paper text, Appendix D.2–D.3, lines 1598–1633). The author's
`src/models/kde_particle_filter/kde_particle_filter.py:280` loads the pretrained
modules. The existing source audit executed that loader and reconstructed both
units' starts; all 484 checks and all saved input hashes remain valid. The
runtime verification also still matches every bound replication source file.

| Item | Role and interpretation |
|---|---|
| Controlled initialization substitution | Mechanism under test: independently pretrained measurement functions can lead to different downstream optimization trajectories despite small pretraining-loss differences. |
| Exact two-epoch old-start replay | Continuation veto and harness check: match row order, seeds, commits, controller events, and losses within the existing 1e-5 absolute / 1e-6 relative replay tolerance. |
| Finite source-valid updates, complete rows, source stopping, verified best checkpoint | Required validity criteria. Invalid or incomplete artifacts veto interpretation and trigger localized repair. |
| Final native validation NLL and native T150 NLL/RMSE | Primary descriptive contrast. Report old-TF minus author and intervened-TF minus author separately. Gap reduction in both NLL evaluations supports the initialization explanation for this unit. It is not a newly invented replication tolerance. |
| Loss history, controller events, gradient health, trace counts, runtime | Explanatory diagnostics; a finite unfavorable trajectory is not a reason to stop early or alter the source method. |
| Missing/mismatched input, stale runtime, nonfinite state, resource cap | Continuation vetoes. Preserve partial results; repair harness faults within the same total budget. |
| Remaining unfavorable valid gap | Candidate explanation is incomplete. Next inspect the first consequential divergence using the saved histories; do not silently change the optimizer, source method, or acceptance rule. |

This is a within-unit causal intervention, not a population noninferiority
study. Eight evaluation draws measure conditional evaluation variation; they
are not eight independent training runs. No claim of algorithm superiority,
whole-KDM completion, new default, or author-method repair follows. The cheap
adversary for this question is the unchanged saved TensorFlow start; source
weights are the external reference. Kalman/other heuristic competitions answer
a different question and are not added to source replication.

## Executable phases

1. Reuse the passed stage audit and verification from
   `stage-isolation-20261013/attempt03`. Verify their checksums, every recorded
   input, current bound runtime, and the original unit02 training audit. Require
   the reconstructed state to differ from the original only in the three
   measurement modules, with fresh zero-moment Adam, identical rates/constants,
   and no change to proposal/bandwidth/other modules.
2. Run the original two-epoch control with unit02 data and seed base
   1200000000. Keep MDPF's original global stage index 1. Consume the existing
   `diagnose_kdm_source_epoch_tf.run_epochs` driver and shared runtime.
3. From the audited author-pretrained measurement start, run only MDPF through
   the original stopping rule or 800-epoch source cap. Do not rerun pretraining
   or A-MDPF. Preserve every batch, epoch, controller event, best state and
   final state. Numerical hot paths remain GPU/XLA with stable signatures.
4. Evaluate author best, old TF best, and the new TF best using the author's
   full native validation routine: 1000 sequences, batches 256/256/256/232,
   eight independent draws per checkpoint and unchanged seed formula. Reuse
   the repaired native-YAML constructor, avoiding the historical JSON-key bug.
5. Evaluate those three fixed MDPF models with the native TF endpoint on the
   original unit02 256-by-150 evaluation data, N=25, filter seeds 202610082/83.
   Replay the existing source metric checks. Recomputed source/old-TF metrics
   must reproduce the saved recent-endpoint result on these identical inputs.
6. Check source/input integrity, complete training/evaluation coverage, trace
   counts and exact-once charging; write a result with decision and inference
   tables. Refresh master and checkpoint with the actual result and next action.

## Defaults and assumptions

| Choice | Provenance / status | Risk and early check |
|---|---|---|
| Unit02 and three fixed checkpoints | Existing unresolved development comparison; not selected from new results | A second intervention does not establish population frequency. Preserve negative results. |
| Author measurement start only | Actual pinned loader, previously audited | Accidental changes elsewhere confound attribution; compare every parameter, optimizer field and constant before launch. |
| Data, base1200000000, global index1 | Original unit02 TensorFlow run; frozen baseline | Renumbering the isolated stage changes every random draw; require two-epoch replay before training. |
| Adam, LR schedule, batches, T-BPTT, early stop, cap800 | Pinned author configuration and existing driver; baseline | Tuning could conceal translation error; exact configuration and controller checks remain active. |
| FP32, TF32 off, RTX5080, XLA | Same diagnostic comparison backend as saved trajectory | Different hardware/numerics could invalidate control; exact replay, recorded GPU identity and memory growth check. |
| Eight validation draws, two endpoint seeds | Existing bounded diagnostic protocol | Only conditional variation is measured; report every draw, no population ranking or fixed margin acceptance. |
| No numerical protection added | Source-comparison baseline with finite/domain guards and saved diagnostics | A protection would change the source object; invalidity stops this diagnostic and triggers a separately scoped investigation. |

## Execution, budget and monitoring

Exact launch, with escalated GPU permissions:

`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_unit02_initialization.py`

Read-only preflight adds `--preflight`. The existing approved interpreter
prefix covers these forms; no allow-list expansion, environment mutation or
new approval ceremony is required. The CPU reference uses tf-gpu2 with
CUDA_VISIBLE_DEVICES=-1 before import. GPU uses tftwogpu, pinned RTX5080 UUID,
verified memory growth, FP32 and TF32 off.

Output root:
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/`.
Each attempt uses a fresh directory. Result:
`controlled-smc-kdm-unit02-initialization-result-20261013.md`.

Total phase cap across at most two attempts: **21,600 GPU seconds and 3,600
CPU-reference seconds** (six GPU hours and one CPU-reference hour). GPU-worker
wall time is charged to GPU, consistently with the recent campaign phases.
This is a reservation within the remaining 76,935.132129 GPU and 205,559.313788
CPU seconds, not a new allocation. Prior unit02 MDPF took about one hour;
unit01's intervention took three hours. The six-hour cap allows the full source
cap at the observed slower rate plus checks. No duplicate training after a
successful training stage: evaluation repair must reuse its frozen checkpoint.

The supervisor polls the child, records progress at least every 30 seconds,
enforces per-resource timeouts, terminates its own child on failure, writes a
terminal decision and charges both resources exactly once. A failed replay
stops training. Failed provenance/validity stops interpretation. Successful
training automatically proceeds to both evaluations and the terminal report.

## Skeptical pre-execution audit

Reviewed before implementation. The baseline is the unchanged unit02 source
and TF evidence, not unit01 and not a stronger method. Loss proxies are not
promotion thresholds. The intervention holds the full call chain, source data,
seeds and optimizer fixed; the replay tests that assertion. Original source
stopping prevents endpoint selection by this investigator. Native validation
and T150 evaluation separate training-controller loss from the final consumer.
Existing proofs are reused only after hash checks. Bounds, failure handling,
negative-result preservation and scientific nonclaims are explicit. The plan
passes this audit; code review and focused preflight remain required before
the long launch.

Pre-mortem: apparent improvement could be one favorable trajectory; therefore
the result stays unit-conditional. Apparent failure could be wrong unit paths,
renumbered seeds, stale verification, or JSON-converted source keys; provenance
checks, replay and the already-repaired native constructor detect these early.

## Implementation review and preflight

Before launch, all five diagnostic files parse successfully. The read-only
preflight passes source pin, current runtime, input and dataset checks; exactly
18 changed parameter arrays belong to the measurement modules. The existing
author terminal audit was extended with an explicit isolated-MDPF start mode;
it passes 124520 checks with zero failures on the completed unit01 trajectory.
This tests the actual source scheduler, early stopping, complete batches,
optimizer-step counts, best checkpoint and stable tracing. Diagnostic source
snapshots preserve this change; no numerical runtime implementation changed.

The supervisor reuses valid verification, checks replay before training,
continues automatically through both evaluations, writes final decision/result,
and refreshes the master/checkpoint. Source-mode YAML construction is retained.
Code review finds no baseline substitution, proxy promotion, missing stop
condition, unauthorized new compute or scientific contract change. Launch is
ready under the existing user authorization and sandbox command permissions.
