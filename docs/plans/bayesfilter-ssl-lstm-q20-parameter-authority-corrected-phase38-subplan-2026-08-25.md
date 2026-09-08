# Corrected Parameter-Authority Phase 38 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 37 common theta-measure gates pass; held-out residuals remain  
Version: `v2.1-training-measure-bound` (historical diagnostic)
Status: `PASS_CHECKPOINT_SELECTION_AUDIT_REPAIR_TRIGGERED_HISTORICAL_V2_1`
Local cap: 900 s

## Question

For the fixed N=256 M0 theta bank, does selecting a checkpoint using a
predeclared validation score reduce held-out latent moment error relative to
the terminal step, or does the residual persist? This distinguishes a finite
optimization/checkpoint issue from a support or representation issue.

## Mechanism and scope

Use only the existing N=256 M0 bank and the unchanged q=20 target in
`theta in R^4`. Keep the train/validation/audit split, proposal log density,
normalized floored weights, affine construction, architecture arms, seeds,
GPU/XLA policy, and target signature fixed. Run the already implemented 200-step
trace once per identity and train-measure-bound affine arm; use only the
training/validation trace to nominate checkpoints. The audit rows are read
only after selection.

This is a checkpoint/objective diagnostic, not a new target or a claim-bearing
NeuTra training protocol. It does not create a canonical LEDH route.

## Evidence contract

**Primary question:** does a validation-selected checkpoint improve the
untouched audit moment diagnostic under the same empirical measure?

**Comparator:** terminal step 200 of each existing Phase 37 boundary arm.

**Selection rule:** minimize the predeclared scalar

`S_t = validation_loss_t + validation_mean_max_abs_t + validation_cov_offdiag_max_abs_t`

over checkpoints `t in {1, 5, 10, 25, 50, 100, 150, 200}`. This scalar is a
nomination rule only; it is not a theorem of transport quality. Ties choose
the smallest `t`. No audit value may enter selection.

**Hard gates:** existing arm status, finite trace, target/status validity,
measure/signature match, and a complete selected-checkpoint receipt.

**Promotion veto:** any claim of IID whitening, posterior correctness, HMC
readiness, or transport admission. A selected checkpoint that still has large
audit residuals is a candidate failure/repair trigger, not a campaign veto.

**Explanatory diagnostics:** validation score, audit mean/covariance residuals,
loss, ESS, and checkpoint index. No ranking between identity and affine is
claimed from this one bank and one seed.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| N=256 M0 bank | Phase 37 nomination by retained-root count | root count may not mean target coverage | compare audit residuals and support receipts | diagnostic nomination |
| checkpoint set | fixed existing trace checkpoints | optimum may lie between checkpoints | retain full trace; report grid limitation | reviewed hypothesis |
| scalar `S_t` | explicit Phase 38 rule | arbitrary weighting can favor one residual | report all components and audit separately | nomination only |
| terminal step comparator | existing 200-step receipt | terminal may be overfit or undertrained | compare selected vs terminal audit values | comparator |
| affine factor | exact training-measure oracle | does not whiten held-out law | audit residuals and oracle separation | finite conditioning map |

## Pre-mortem

- Selection can improve the 12-row validation set by chance. The untouched
  audit partition is mandatory and remains descriptive because it is small.
- A lower validation score can coexist with poor covariance. Report every
  component, not only `S_t`.
- Existing traces were generated before this subplan was written. They are
  eligible only because the target, arm, split, and trace were frozen and the
  selection rule is applied post hoc without touching audit data; the result
  must say this explicitly.
- A passing finite receipt could be mistaken for whitening. The result must
  retain the no-IID and no-posterior nonclaims.

## Execution

First regenerate each arm in a fresh root with audit moments at the
predeclared checkpoints:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/identity-trace \
  --precondition identity --steps 200 --checkpoint-steps 1 5 10 25 50 100 150 200 \
  --seed 20260825 3811

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase37-support-ladder/n256 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/affine-trace \
  --precondition affine --steps 200 --checkpoint-steps 1 5 10 25 50 100 150 200 \
  --seed 20260825 3811
```

Then use a read-only post-run reporter over:

- `phase37-support-ladder/neutra-identity/result.json`
- `phase37-support-ladder/neutra-affine/result.json`

The Phase 38 trace roots supersede those paths for checkpoint selection; the
Phase 37 roots remain the terminal-step comparator.

This phase is closed as a v2.1 diagnostic. Phase 39 showed that its ordered
validation split can be ancestry-correlated with training rows. Its receipts
remain valid for the split they actually used, but they are not evidence for
the active v2.2 root-group-stratified plan.

Write a unique result root under
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase38-checkpoint-repair/`.
The reporter must record its source hashes, exact selection rule, selected
checkpoint, terminal comparator, audit diagnostics, and a run manifest. It
must refuse to overwrite an existing root.

## Stop and refresh

If both selected checkpoints still have substantial audit residuals, continue
to a separately reviewed objective/support design; do not call the target or
research direction invalid. If a source receipt is malformed, repair the
reporter in a fresh root under the unchanged contract. A true continuation
veto requires loss of the declared theta target/common support or an invalid
source artifact that cannot be repaired.
