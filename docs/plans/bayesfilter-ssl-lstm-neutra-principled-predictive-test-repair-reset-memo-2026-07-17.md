# SSL-LSTM NeuTra Principled Predictive-Test Repair Reset Memo

Date: 2026-07-17

Status: `REPAIR_IMPLEMENTED_CONFIRMATION_REMAINS_CLOSED`

## Durable State

- Chapter 28a now fully documents the SSL-LSTM model, scalar target, NeuTra
  history, predictive-law estimand, fixed-16 inconsistency, growing Bartlett HAC,
  and proper-score confidence-region decision.
- The historical fixed-batch APIs remain unchanged for receipt replay but must
  not support a general consistency claim when batch length is fixed.
- New prospective APIs live in
  `bayesfilter/inference/predictive_equivalence.py`:
  `growing_hac_bandwidth`, `chain_bartlett_long_run_covariance`,
  `proper_score_loss`, `quadratic_loss_confidence_bounds`, and
  `classify_proper_score_equivalence`.
- Raw zero-ridge HAC may be inference-admissible. Positive-ridge covariance is
  numerical diagnosis only and is fail-closed for inference.
- MMD is explanatory until its tolerance and growing-bandwidth uncertainty are
  separately calibrated.
- Focused repaired-statistics suite: `76 passed`.
- Lightweight equation/model suite: `7 passed`.
- Full book builds to `docs/main.pdf`; unrelated warnings remain elsewhere.
- The broader existing SSL-LSTM compiled suite was stopped at the 10-minute CPU
  cap after 39% with no observed failures; do not relabel it as passed.

## Scientific Boundary

No acceptable forecast-loss budget `K` has been selected. No direct
finite-sample simultaneous coverage/power study has been run for the repaired
decision. Therefore:

- Phase 9 remains closed;
- no G/H confirmation forecast may be opened;
- no HMC acquisition is justified;
- 4,096 and 8,192 remain historical feasibility signals, not sample-size
  decisions;
- no posterior correctness, predictive equivalence, material difference,
  method ranking, or default-readiness claim is supported.

## Next Action

Create one concise Tier-2 direct-calibration plan. Freeze `K`, horizon weights,
HAC multiplier, zero-ridge rule, sample ladder, fresh seeds, simultaneous
coverage target, false-decision bounds, and the role of MMD before execution.
Use controlled dependent forecast laws only. HMC and G/H confirmation require a
later separately authorized handoff after that design passes.
