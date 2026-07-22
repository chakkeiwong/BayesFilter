# Corrected Neural-Force HMC Reset Memo

Second supersession note, 2026-07-18: the native-tuning audit replaces the
resume path below.  Tuned performance claims are
`UNSUPPORTED_PENDING_NATIVE_RETUNING`; resume from
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

Supersession note, 2026-07-18: the matched-performance state below is
historical. The exact-gradient comparison repair completed PP-UKF, PP-SGQF,
and SIR-SGQF same-chart comparisons; STR-UKF remains partial because its exact
arm failed long-warm-up energy health twice. Resume from
`docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-reset-memo-2026-07-18.md`.

Program status: `COMPLETE`.

## Established State

- The primary corrected kernel is implemented in
  `bayesfilter/inference/neural_force_hmc.py`.
- Scalar residual-force training and frozen artifacts are implemented in
  `bayesfilter/inference/neural_force_training.py`.
- Campaign binding and tuning support are implemented in
  `bayesfilter/inference/neural_force_campaign.py`.
- Five Tier A posterior configurations passed one-seed corrected learned-force
  validity: LGSSM-KF, PP-UKF, PP-SGQF, SIR-SGQF, and STR-UKF.
- The zero-residual control also passed all five.
- Only LGSSM-KF has a complete matched descriptive performance-screen pass.
- The other four tested cells are
  `PERFORMANCE_NOT_DEMONSTRATED_MISSING_MATCHED_LEDGER`; no speed failure is
  inferred.
- Eight Tier B cells are `REQUALIFICATION_BLOCKED` because selected chart state
  is missing; they were not run on substitute charts.

Terminal result:
`docs/plans/bayesfilter-hnn-surrogate-hmc-terminal-result-2026-07-18.md`.

Machine-readable ledger:
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/final_cell_ledger.json`.

## Re-entry Rungs

1. For a matched performance extension on PP/SIR/structural cells, add and
   predeclare a same-chart true-gradient cost arm and complete amortization
   ledger. Do not infer it from existing ESS or runtime fields.
2. For Tier B, create a new fresh-chart campaign. Rebuild the exact historical
   target first, train a new chart under a target-specific protocol, and label
   it fresh evidence rather than requalification.
3. Preserve every future chart in a versioned repository artifact with target
   identity, architecture, all tensor values, transform conventions,
   log-Jacobian parity probes, and SHA-256 hashes.
4. Independent seeds or fixtures are appropriate if a stronger reliability
   statement is wanted. The current conclusion remains one-seed viability.

## Known Limits

- No statistically supported arm ranking exists.
- Residual learning was not shown to outperform the zero-residual force.
- Filter-defined posterior validity does not establish latent-model exactness.
- The structural raw-coordinate comparator remains source-geometry blocked.
- The historical Tier B NeuTra results remain context, not corrected-kernel
  evidence.

## Verification

Terminal verification is complete: `64` focused tests passed, JSON and Python
compile checks passed, `git diff --check` passed, the full 438-page monograph
built, all five claim-bearing result hashes replayed, and bounded read-only
Claude review returned `VERDICT: AGREE`. No mandatory phase remains.
