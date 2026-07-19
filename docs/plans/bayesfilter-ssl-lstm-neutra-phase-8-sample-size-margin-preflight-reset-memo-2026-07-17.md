# SSL-LSTM NeuTra Phase 8 Sample-Size Preflight Reset Memo

Date: 2026-07-17

Current state: `FEASIBILITY_ENVELOPE_FOUND_PHASE9_AND_HMC_CLOSED`

- The 448- and 1984-draw formal designs remain failed historical baselines.
- No declared margin scenario is feasible at 1984 draws under the new
  operating-characteristic preflight.
- Asymptotic feasibility signals: arithmetic midpoint at 4096; both historical
  contracts at 8192. These are not selected designs.
- The arithmetic midpoint is mean `0.125`, log variance about `0.13597` (ratio
  about `1.1457`). It has no scientific loss/utility justification yet.
- No margin or MMD tolerance was selected. G/H confirmation remains unopened.
- No HMC acquisition is authorized. Estimated HMC-plus-forecast cost is about
  `1.4116` GPU-hours at 4096 and `2.8395` GPU-hours at 8192 for both charts.
- Next: justify the scientific margin, then direct finite-sample validation
  with fresh seeds. Do not acquire HMC first.
- Material receipt SHA-256:
  `ad13cede2f7ab23f18f956eb7eb39e729f1ed987e4175292cafd7ee59786d89d`.
- Full result:
  `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-preflight-result-2026-07-17.md`.
