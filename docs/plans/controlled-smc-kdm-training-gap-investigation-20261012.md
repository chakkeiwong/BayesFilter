# Current MDPF training-gap investigation

12 October 2026. Replication only; author commit
`b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18`. This follows the completed
paired calibration. It does not change KDM acceptance criteria or restart full
training. Controlled-SMC and Task 3 are outside this investigation.

## Question and evidence contract

Why are the two new TensorFlow MDPF checkpoints worse under the common author
evaluator? Distinguish a wrong stage boundary, update mismatch at the newly
visited parameter values, and independent stochastic training trajectories.

1. Reconcile actual saved starting/best checkpoints, optimizer resets, seeds,
   learning-rate events, batch coverage, and training/validation curves. Use
   the existing independent terminal audits and inspect their implementation.
2. At each new unit's actual MDPF start and best checkpoint, use two fixed
   training batches (rows 384:512), source-native realized draws, and two
   consecutive Adam updates. Require exact native/source-replay equality.
   Compare the real TensorFlow update endpoint on the same tensors. Repeat in
   FP64 after promoting the same FP32 inputs to separate arithmetic sensitivity
   from an equation or wiring mismatch. No weights from this diagnostic become
   a training result.
3. Localize any new discrepancy before prescribing a repair. If these checks
   pass, retain stochastic trajectory variation and untested cumulative or
   native-randomness effects as unresolved explanations. A pass does not prove
   that training distributions are equivalent.

Comparator: the pinned author's FullSequenceTrainer forward, KDE loss,
gradient clipping and Adam, with the source architecture and trainable groups.
Primary engineering screen: existing coordinate bounds
`128*2^-23*(1+abs(source))` in FP32, `1e-9*(1+abs(source))` in FP64.
These are localization diagnostics, not replication acceptance margins.
Native/source replay equality, finite gradients, correct state transfer,
parameter coverage and artifact integrity are validity vetoes. A numerical
comparison failure triggers localization; it does not terminate research.
Training-curve differences, angle conditioning and gradient differences are
explanatory only. No population ranking or replication pass follows here.
No heuristic performance ladder applies: this tests translation of an unchanged
source update, not an improvement over the author's algorithm.

## Assumption audit and pre-mortem

| Choice | Provenance and role | Risk and earliest diagnostic |
|---|---|---|
| Actual new start/best states | Completed calibration; fixed diagnostic inputs | Old-state tests miss newly visited regimes; bind hashes of these states |
| Two batches, two updates | Bounded localization convenience, not training evidence | Misses cumulative drift; explicitly leave that unresolved |
| Rows 384:512, seed 202610121 | Fixed development-only probes | Not representative performance evidence; no seed selection or qualification |
| Source CPU / TF GPU-XLA, TF32 off | Existing replication comparison | Precision/backend differences; common tensors and FP64 control |
| Fresh diagnostic Adam | Isolates update law, baseline hypothesis | Does not test the original accumulated moments; stage resets and terminal state are separately audited |
| Native random streams differ in full runs | Explicit existing implementation choice | Paired constructors do not couple training noise or row order; do not attribute outcome differences to implementation alone |

Budget: at most 1,800 CPU seconds and 1,800 GPU seconds, at most three attempts,
charged to the existing campaign (remaining before this work: GPU91,743.507420s,
CPU220,555.559930s). Every attempt uses a fresh directory under
`artifacts/controlled-smc-kdm-replication-20260928/training-gap-20261012/`.
Stop on timeout, corrupt/mismatched source or input, missing gradients, nonfinite
updates, or unavailable GPU. Local harness failures may be repaired within the
same budget. No package changes or full training. Record command, commit,
environment, seeds, source/input hashes, device/memory policy and elapsed time.

Commands: CPU reference uses isolated `/home/chakwong/anaconda3/envs/tf-gpu2/bin/python`
with GPU explicitly hidden inside the worker; GPU comparison uses escalated
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python` and
`docs/benchmarks/diagnose_kdm_neural_precision_tf.py` with this plan and the
result note below. Both run with one framework/inter-op thread, as in calibration.
Reference entry: `docs/benchmarks/reference_kdm_current_training_gap.py`.
Supervisor: `docs/benchmarks/run_kdm_current_training_gap.py`.
Result note: `controlled-smc-kdm-training-gap-result-20261012.md`.

Skeptical review before execution: PASS for this bounded localization question.
The comparison uses actual current states, identical data/draws, a source replay
control and unchanged numerical screens. It does not substitute local parity
for end-to-end replication. The existing 72/72 terminal review and per-unit
audits already check real stage transfers, reset moments and stopping; their
checks have been traced to the saved files. The principal remaining limitation
is that short coupled updates cannot identify a full training-distribution gap.

Execution note: attempt01 stopped before numerical comparison because JSON had
converted the source architecture's integer dimension keys to strings. Repaired
the diagnostic to reconstruct from verified pinned YAML and assert its JSON
representation equals the saved architecture. This is a reference-harness
serialization failure, not evidence against either algorithm. Charged CPU
6.533137s/GPU0.065614s. Attempt02 repeats the unchanged contract in a fresh
directory. Syntax check passed before retry.


Attempt03 extension, reviewed before launch: use the two actual best MDPF
checkpoints with their saved Adam first/second moments, step counts and learning
rates in both libraries. Unit01 step21646 and unit02 step20777 both have rates
5e-6/5e-7. Attempt02 used fresh Adam and original configured rates, which does
not answer this late-state question. Keep exactly the same two batches, draws,
FP32/FP64 comparisons, screens and total 1800s/resource/three-attempt cap.
The source worker restores state immediately after source optimizer creation;
the TF worker uses the real checked load_state_dict and requires exact loaded
state equality. Source native/replay equality remains a validity veto. Source
state promotion to FP64 changes only representation of saved FP32 numbers.
This is a test of the update law at shared saved state, not continuation of the
author's independently trained checkpoint or learned-quality qualification.
Skeptical review: PASS; the new test closes a specific uncovered state dimension,
keeps source arithmetic and scientific criteria unchanged, and does not rerun
full training. Remaining trajectory-level randomness and resampling feedback
are still outside this short fixed-realization probe. Command: previous
supervisor command plus --saved-optimizer. The optional fixture field is
diagnostic-only; absent it, the comparison harness retains its previous path.
