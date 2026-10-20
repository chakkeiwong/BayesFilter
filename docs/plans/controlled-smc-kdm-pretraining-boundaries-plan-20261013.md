# KDM pretraining boundary investigation

Authorized by the owner's 2026-10-13 instruction to continue. This is a bounded
continuation of source replication, not an algorithm improvement campaign.

## Question and evidence contract

The saved unit01 stage intervention removed 95.26% of the final MDPF NLL gap
when TF started from author-pretrained weights. Does independent pretraining
still contain a translation defect in late optimizer updates, partial batches,
validation aggregation, or best-checkpoint selection?

Comparator: unchanged author WeightModelTrainer forward and Trainer.train
bodies at b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18. Native YAML preserves
integer keys; serialized configuration equality is checked. The author CPU
reference intentionally hides CUDA. TF runs FP32 GPU/XLA, TF32 disabled, with
verified memory growth and explicit stable signatures. Constructor equality
and early matched-state gradients have already been checked; those results
are controls, not reasons to repeat a long trajectory.

1. Replay saved author and TF pretraining validation histories for units01/02
   through the actual author train loop and the TF host controller. Compare
   every scheduler event, learning rate, early-stop state and saved best epoch.
   Also compare the TF replay to its saved events and selected checkpoint.
2. At each unit's TF best pretraining checkpoint, restore identical parameters,
   moments, step and learning rates in author and TF, and compare the first
   full training batch (64), final training batch (8), and all four validation
   batches (256/256/256/232). Include author-best measurement weights as a
   separate parameter-state probe with explicitly fresh Adam moments; these
   are not reconstructed author optimizer histories. Use the same source
   clouds, masks and row ordering in each comparison.
3. Compare features, weights, KDE NLL, raw/regularized gradients, Adam moments
   and updated parameters. Check weighted validation aggregation, exact input
   identity, no validation mutation and one trace per used signature. Preserve
   full paired inputs and source/TF states for any numerical localization.

Primary diagnostic criterion: exact controller/event agreement and finite,
matched-state numerical agreement. The previously used local FP32 discrepancy
screen, abs(delta) <= 2e-5*(1+abs(reference)), is a repair trigger only, not a
replication acceptance margin. A flag triggers a bounded FP64/same-state
localization before assigning a translation error. Hash mismatch, wrong source,
wrong data coverage, invalid state restore, non-finite values or wrong device
invalidate the diagnostic and require a local repair. A valid candidate
discrepancy is not a continuation veto. Stop at the resource/attempt cap or if
the comparator cannot be established.

Passing closes only the checked boundaries. It does not establish independent
training equivalence, population noninferiority, A-MDPF completion, or full KDM
replication. No heuristic-performance ranking is involved: the target is exact
author conformance, and a heuristic cannot substitute for the author comparator.

## Defaults, source anchors, and skeptical audit

- Source random clouds are regenerated during validation as well as training:
  weight_model_trainer.py:32-85,107-168. Independent TF/PyTorch random streams
  can select different checkpoints. This is an explanatory hypothesis; matching
  clouds isolates numerical translation without asserting RNG sequence parity.
- Sparse time selection and weighting follow author source and paper D.1/D.2
  (local text lines1584-1610). Partial batches test the coverage failure mode.
- Exact saved TF Adam states avoid treating a fresh-optimizer diagnostic as a
  late-training test. Author-best probes explicitly have fresh optimizer state.
- Source scheduler/selection/stopping order: trainer.py:613-699 and
  trainer_utils.py EarlyStopping/CustomReduceLROnPlateau. Saved history replay
  tests actual observed ties, reductions and stops, beyond synthetic fixtures.
- Two saved units are diagnostic coverage, not a random-sample population test.
- FP32 screen is inherited from earlier localization, not calibrated as a final
  performance margin; FP64 is a diagnostic exception if a flag needs explanation.

Skeptical review PASS before execution: baseline is original author code;
probes cover the real consumer operations; no proxy is promoted to scientific
acceptance; stochastic streams are controlled explicitly; source failures are
not required to be repaired; all outputs are versioned. No full retraining is
justified by these tests alone. Strongest alternative explanation after a pass
is independent random-cloud/trajectory variation amplified downstream.

## Execution and limits

Launcher: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B
docs/benchmarks/run_kdm_pretraining_boundaries.py` (escalated GPU access).
Each attempt runs the controller reference before the GPU tests and stops on an
invalid harness. At most two attempts, 1,200 wall seconds each; total cap2,400
CPU and2,400 GPU seconds, also bounded by the tighter remaining active campaign
ledger/tranche. This consumes existing authorization; no new 48-hour grant.
GPU charge covers GPU stage wall time; CPU charge conservatively covers total.

Output root: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
pretraining-boundaries-20261013/attemptNN. Preserve commands, source/input hashes,
Git/dirty provenance, device policy, logs, checks, wall time, and ledger charge.
Result: controlled-smc-kdm-pretraining-boundaries-result-20261013.md. Refresh
the master and concise checkpoint after results. A localized harness repair may
use the second attempt under the unchanged contract.
