# KDM trained-checkpoint update localization — 8 October 2026

The common-author-evaluator development test found NLL +0.292–0.382 nat and RMSE
+2.25–5.81% for one TF-trained pair of MDPF/A-MDPF checkpoints. Determine whether
the full-sequence loss, gradient or Adam arithmetic diverges at these actual
learned parameters. This is replication diagnosis, not author-code repair.

## Research intent and evidence contract

Candidate: unchanged TensorFlow GPU/XLA BearingsNeuralReplay shared training
kernel. Comparator: pinned author FullSequenceTrainer.do_forward_pass, actual
KDEParticleFilter and torch.optim.Adam, at exactly the same checkpoint, sparse
training batch, initial particles, resampling draws and proposal innovations.
Use both author-trained and TF-trained checkpoints for both variants.
Primary engineering screen: unchanged 128*2^-23*(1+abs(reference)) float32 and
1e-9*(1+abs(reference)) float64 bounds on values, gradients and Adam states;
reuse the independently derived angle condition bound from the October 7 plan.
A mismatch is a repair trigger and promotion veto, not a continuation veto.

First capture one native float32 source update per checkpoint. Replay its exact
tape in float32 and require exact equality of loss, raw/clipped gradients,
parameters, moments and all captured outputs as a harness control. Cast those
same float32 checkpoint/data/draw values to float64 for a precision diagnostic.
Do not regenerate different float64 samples. Native sampling-law differences
remain outside this matched-draw test.

Use the real training batch size 64, T=17, N=25 and mask modulo 4; rows 384–447
of the preserved training dataset, filter seed 202610086. The selected rows and
seed are fixed before results. Zero Adam moments at these parameters are an
explicit diagnostic initialization, not resumed training or a reproduction of
the final optimizer state. Source configured learning rates, epsilon, betas,
clip 100, no weight decay, first loss excluded and BPTT modulo 4 are unchanged.
Float64 is diagnostic only. No training continuation or checkpoint promotion.

A complete exact-input float64 agreement weakens the hypothesis of an algebraic
translation error on these endpoints; it does not prove global training parity
or identify the source of stochastic training quality variation. Failed native
source replay, changed source/checkpoint hashes, missing trainable gradients,
nonfinite source-valid output, unverified GPU growth or exhausted phase budget
invalidates the diagnostic. Local harness faults may be repaired under the same
contract and cap. Checkpoint quality and unchanged numerical failures remain open.

Explanatory outputs: all per-endpoint errors, gradient norms, angle conditioning,
raw gradients, trace counts, placement, peak allocation and compiler cost.
No proxy becomes a qualification criterion. Independent trained models remain
the unit for the accepted NLL/RMSE non-inferiority confidence limits. The exact
source replay is the cheap baseline; no unrelated superiority gate is added
to replication. Situations: both model variants and both learned checkpoint
origins, separated by precision.

## Source/call-chain and assumption review

Pinned source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
FullSequenceTrainer.do_forward_pass, lines 136–241, constructs the initial cloud,
detaches at 5/9/13, skips initial loss and averages sparse NLL. Trainer.train_epoch
lines 437–525 backpropagates, clips and steps the configured optimizers.
KDEParticleFilter.resample_particles lines 706–760 injects IWSG; its forward
lines 984–1125 forms the learned transition, normalized weights and bandwidths.
TF SourceFullTraining._loss calls BearingsNeuralReplay._native_sequence;
_native_sequence and replay _update share _sequence and _apply_update.
Prior native/source-stage checks establish that wiring; this phase checks the
shared operation at completed checkpoints, not the full random-stream law.
Source configs have no batch normalization; actual freeze and trainable
parameter coverage are checked, not inferred from module names.

The principal possible confound is one stochastic training pair. Equal formulas
can produce different trained networks. The test cannot measure that variance.
Other hypotheses are ill-conditioned float32 circular gradients, sampling-law
differences, and cumulative optimizer arithmetic; a first Adam update does not
test all of them. Do not attribute the quality gap to XLA from float32 mismatch
alone. Source paper math and algorithm are unchanged.

Skeptical pre-run audit: wrong baseline avoided by the already audited checkpoint
inventory and direct strict state loading; no MDPF-to-A-MDPF warm-start conversion
replaces the A-MDPF checkpoint. Exact tape replay avoids conflating random streams
with derivatives. Per-checkpoint and dtype results prevent pooling failures away.
Fresh moments are stated as a limited hypothesis, not silently treated as final
training state. No policy/default or acceptance tolerance is altered. Proceed.

## Execution and budget

At most two numerical attempts, 900 CPU seconds including 120 routine seconds,
and 1800 GPU seconds, charged to the current allocation. One source worker and
one TF worker per attempt, sequential. Stop after the fixed eight comparisons.
Fresh directories under controlled-smc-kdm-replication-20260928, prefix
kdm-trained-update-20261008-NN; preserve failed attempts.

Command from /home/chakwong/BayesFilter:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/run_kdm_trained_update.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-trained-update-20261008-01

Supervisor uses tf-gpu2 Python for the independent CPU author worker, hides GPU
before import, and uses tftwogpu for the escalated TF GPU/XLA worker with verified
memory growth and TF32 disabled. No NumPy runtime implementation is introduced.
Record exact commands, input/code hashes, environment, seeds, device settings,
wall time, status and logs. Result:
controlled-smc-kdm-trained-update-localization-result-20261008.md.
