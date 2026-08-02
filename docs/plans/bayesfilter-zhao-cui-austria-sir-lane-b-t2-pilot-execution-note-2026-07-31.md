# Zhao-Cui Austria SIR Lane-B T2 Pilot Execution Note

Date: 2026-07-31

Status: `READY_AFTER_FINAL_CALIBRATION_PREPARATION`

This note governs only the six B4 GPU/XLA pilot arms in the frozen T2 plan.
It cannot read or generate the untouched role.

## Phase Audit

- Baseline: selected Lane-B T1 artifact plus the admitted B2 proposal and B3
  previous-marginal boundary. APF, UKF, source-replica, and generic retained-
  grid production routes are excluded.
- Objective: exact globally normalized empirical T2 cross-entropy. One update
  accumulates all 16 fixed-parameter 256-row gradients, clips once, and applies
  Adam once.
- Selection: validation normalized-log-density RMS among arms passing all hard
  gates. This is deterministic descriptive selection, not statistical ranking.
- Calibration: fixes the shift and checks an independent normalizer only. It
  does not select an arm.
- Vetoes: stale prepared inputs, role/seed/count mismatch, non-finite terms,
  XLA/GPU/memory-policy failure, calibration-validation disagreement, failure
  to improve on `0.95` times the constant-density RMS, peak allocation above
  6 GiB, artifact reload mismatch, or the ten-minute arm cap.
- Continuation: one rejected arm does not stop the ladder. No viable arm after
  all six is a repair trigger for target-specific frame/capacity tuning.
- Nonclaim: no value admission, untouched evidence, score, T20, HMC, production
  KR, posterior correctness, or Zhao-Cui superiority.

## Default Audit

`use_quantile_scale=True` is inherited from the admitted T1 runner and is a
warm-start hypothesis. The T2 plan's frozen quantile fraction `0.01`, expansion
factor `4`, and covariance jitter `1e-5` otherwise have no effect on the frame
when quantile scaling is disabled, so the enabled flag is the only coherent
interpretation of that frozen arm table. A failure remains evidence against
this pilot family, not evidence that the inherited flag is a T2 default.

All other arm values, counts, seeds, optimizer updates, role separation,
promotion criterion, and stop conditions are exactly those in the frozen T2
plan. Audit verdict: `PASS_FOR_EXECUTION` after the final calibration artifact
passes its prepared-data gates.
