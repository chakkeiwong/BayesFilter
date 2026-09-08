# Phase 4 Batch-Native NeuTra Screen Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED`  
Budget cap: `18000 s`  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase4`

## Objective

Test whether valid M0-authoritative particles, and separately labeled M1-M3
auxiliary clouds, can train a batch-native NeuTra transport without confusing a
good predictor or whitening metric with a valid posterior transport.

## Entry gate

At least one authority/auxiliary input bank must have valid target/status,
proposal metadata, and a passing role-specific contract. If M0 is not admitted,
training may use only explicitly diagnostic C0/auxiliary data and may not claim
authority or posterior correctness.

## Pre-execution skeptical audit

The entry gate is satisfied by the Phase 1 contracts and the Phase 2 N=100 M0
candidate. The M0 pilot artifact contains finite target/status rows, proposal
metadata, a protocol hash, and terminal normalized weights. It does not contain
a retained unnormalized importance-weight ledger after resampling. Therefore
the screen will treat `log(normalized_weight)` as the fixed empirical training
measure and will label the result candidate/engineering evidence only; it will
not call the screen an unbiased SMC-U authority or a posterior result.

The two-arm comparison is a calibration/validation exercise, not a superiority
claim. The earliest invalidity checks are device policy before TensorFlow
initialization, static batch rank, finite forward/inverse/Jacobian residuals,
and finite target/status on the untouched audit rows. A failed arm blocks that
arm and triggers the repair note; it does not invalidate M0 or stop Phase 5.

Numeric controls are hypotheses for this screen: train/validation/audit sizes
`60/20/20`, two architecture/learning-rate arms, three updates per arm, and
batch size `60`. They are recorded in the manifest and cannot become defaults
without a longer target-specific tuning campaign.

## Training protocol

- Use GPU by default, set `TF_FORCE_GPU_ALLOW_GROWTH=true` before import, verify
  every visible physical device, and record logical devices, XLA, TF32, dtype,
  batch size, and allocator policy.
- Every optimizer update consumes a batch with more than one row; transport,
  log determinant, target value/score, loss, gradient, and update preserve the
  leading batch dimension. Scalar loops and row-mapped scalar targets are a
  hard veto for claim-bearing training.
- Use a target-specific calibration/validation/audit split, architecture and
  optimizer budget ladder, seed policy, and held-out criteria. Inherited q20
  settings are warm starts only.
- Freeze proposal laws and selected controls before audit data. Preserve every
  failed attempt and its repair.

The executable is
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py`.
It loads and hashes the frozen M0 tensors, performs two GPU/XLA weighted
forward-KL screens on the training partition, selects a descriptive validation
arm, and evaluates the frozen selection on audit rows. It never launches HMC
and never uses a scalar or row-mapped target call.

## Gates

| Check | Role | Condition |
|---|---|---|
| device/memory/XLA/batch receipt | hard veto | policy verified before initialization |
| forward/inverse/Jacobian parity | promotion veto | exact finite checks pass |
| transformed-target two-mode canary | promotion criterion | no target/status or mode-label invalidity |
| loss/whitening/ESS | explanatory | may nominate repair only |
| HMC | continuation veto for this phase | never run here; separate plan required |

## Required artifacts

Training manifest, scope/tuning artifact, seed-wise curves and held-out
diagnostics, parity receipt, two-mode canary, device provenance, result note,
and refreshed Phase 5 adjudication statement. No posterior or HMC claim is
allowed from this screen.

## Executed receipts

- `phase4-attempt1` preserved the pre-import package-path harness failure.
- `phase4-attempt2` preserved the memory-growth ordering failure.
- `phase4-attempt3` preserved the one-sided audit metadata split failure.
- `phase4-attempt4` preserved the missing-two-component split failure.
- `phase4-attempt5` preserved the float32 probe dtype failure.
- `phase4-attempt6` passed the three-update GPU/XLA hard gates.
- `phase4-attempt7` passed the 20-update same-scope repair; whitening remained
  explanatory/tuning evidence only.
- `phase4-mutation-revalidation-attempt1` passed hard gates on the separate
  mutation candidate bank and did not replace the identity branch.

The result and repair note classify every failure and preserve all unique
directories. Phase 5 is now the active closeout phase; HMC remains out of
scope.
