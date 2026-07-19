# SSL-LSTM NeuTra Proper-Score Direct Calibration Reset Memo

Date: 2026-07-17

## Current State

The dual average/horizonwise proper-score implementation is complete and
tested. The direct controlled run completed at 4,096 and 8,192 draws per chain.
The current candidate failed decision power, not execution validity.

Immutable receipts:

- smoke SHA-256: `7554ac456684e02eb802f60320fb7fda927df5d0159bdbfee29402873159398b`;
- material SHA-256: `fc4781d98a69fbf1002c0f2b76955e023abde4471187c48a6df01f47e712ebf7`.

## What Must Not Be Rediscovered

- One equal-weight average threshold is impossible for the declared anchors:
  persistent negligible mean loss `0.00125` exceeds one-horizon material
  variance average loss, approximately `0.0012448`.
- The primary repair is one 20D joint region with average and all horizonwise
  loss extrema; threshold `K*=0.0068491` is frozen from loss-scale anchors.
- At 8,192 draws, persistent negligible mean equivalence was `5/256`; local
  variance material detection was `76/256`. Do not open G/H confirmation at
  this sample size.
- Zero invalid rows and zero false decisions occurred at both rungs. Do not
  misstate this as an implementation or numerical failure.
- The 256-replication `0.05/88` per-family coverage screen was underpowered: a
  true 95%-coverage procedure has only about 26% probability to certify the
  90% lower target. Do not use that screen alone to claim HAC undercoverage.
- Pooled descriptive coverage was `90.84%` at 4,096 and `92.54%` at 8,192;
  pooling does not establish every family.
- The first two smoke attempts wrote no receipts. The first found unsupported
  variant generation under XLA GPU; the second found compile-heavy unrolled
  loops. Dense batched generation and fixed-count XLA while-loops repaired
  both. The final immutable smoke passed.

## Next Boundary

No additional run is authorized by this memo. The next plan should preserve
the scientific threshold and HAC policy, choose a prospective larger draw
ladder, increase controlled replications to roughly 1,024 per family if the
same simultaneous screen is retained, use fresh seeds, and set a new resource
cap. HMC, retained archives, and G/H confirmation remain closed.

Primary result:
`docs/plans/bayesfilter-ssl-lstm-neutra-proper-score-direct-calibration-result-2026-07-17.md`.
