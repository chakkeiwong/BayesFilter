# SSL-LSTM NeuTra Phase 8 Reset Memo

Date: 2026-07-17

Current state: `PHASE8_CONTROLLED_POWER_BLOCKER_PHASE9_CLOSED`

- G/H remain independently admitted peer posterior replications at 512
  retained draws per chain; neither is truth.
- Prefix draws `0..63` are permanently excluded pilot data. Confirmation suffix
  `64..511` remains unopened.
- Target pilot passed; receipt SHA-256 `5ae511c...`.
- Original 448-draw controlled nomination failed power; receipt SHA-256
  `ec112880...`.
- Bounded 1984-draw B/C/D power repair also failed; receipt SHA-256
  `56a34c4a...`.
- No MMD tolerance or repair arm is selected. Validation and Phase 9 are closed.
- Do not repeat either ladder, relax thresholds post outcome, open G/H
  confirmation forecasts, or acquire additional HMC draws without a new
  prospectively powered plan.
- The decisive repair-family blocker was true-equivalent persistent variance
  ratio `1.05`, not local-horizon material power.
- Machinery, GPU/XLA, covariance, MMD, lineage, and leakage gates passed; this
  is a statistical power blocker under the declared design.
- Full closeout:
  `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-result-2026-07-17.md`.
