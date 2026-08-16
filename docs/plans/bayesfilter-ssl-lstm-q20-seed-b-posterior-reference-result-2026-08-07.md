# SSL-LSTM q=20 seed-B posterior/reference result (2026-08-07)

## Outcome

No posterior agreement verdict was issued. The independent deterministic
quadrature route was correctly vetoed by target-only evidence of two distinct
stationary modes. The declared plain-HMC repair was then correctly stopped by
the resource gate before tuning because the complete grid plus the minimum
sequential authority projected to `35,024.2 s`, above the authorized `20,000 s`
campaign cap.

This is an **inconclusive reference phase**, not a NeuTra rejection and not a
plain-HMC failure. Seed B remains a candidate that passed its own sequential
HMC screen; posterior correctness and agreement with an independent authority
remain untested.

## Evidence summary

| Phase | Status | Evidence |
|---|---|---|
| Quadrature preflight | Passed | CPU-only, XLA, current target signature, finite target at prior center and all independent MAP endpoints |
| Target-only MAP diagnostic | Reference veto | 9/9 endpoints stationary by score (`<=1e-5`); two basins: plus mode log density `-37.55317366`, minus mode `-37.60347349`; coordinate separation includes observation weight `+0.58942510` vs `-0.58769660` |
| Quadrature ladder | Not run | Single-center proposal invalid after the two-mode finding |
| Plain-HMC preflight | Passed | Current target signature, identity adapter, both mode starts finite, CPU-only XLA |
| Four-chain rate canary | Passed, descriptive only | Warm `54.10 s` for 2 results at `L=3`, `9.0172 s` per transition-leapfrog |
| Four-process one-chain rate canary | Passed, descriptive only | Critical path `1.9922 s` per transition-leapfrog; XLA, CPU-only, all four workers finite |
| Plain-HMC resource gate | Hard continuation veto | Implemented six-worker four-chain grid `145,853.9 s`; minimum four-chain sequential authority `81,155.2 s`; total `227,009.1 s` vs cap `20,000 s` |
| Plain-HMC tuning/sequential/comparison | Not launched | Launch would violate the predeclared cap and could not produce a complete authority |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not use quadrature as reference | Failed multimode single-proposal validity | Two distinct target-only stationary modes within 20 log units | Whether a multimode quadrature or mixture proposal can be made auditable | New reviewed multimode reference plan or plain-HMC budget increase | Quadrature cannot ever work; seed B is wrong |
| Do not launch plain-HMC campaign under current cap | Not evaluated; authority would be incomplete | Resource projection veto | Parallel worker scaling beyond four; exact tuning/grid variability | Obtain explicit larger compute budget, reduce target cost, or design a separately reviewed lower-cost authority | Plain HMC fails; NeuTra fails; posterior is invalid |
| Keep seed B as sequential-screen candidate | Its own screen passed | No hard veto in archived seed-B run; native divergence unavailable, not zero | No independent posterior authority; finite energy tails explanatory | Preserve candidate and seek an independently powered reference | Posterior correctness, superiority, model adequacy, default readiness |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Quadrature multimode veto and plain-HMC resource continuation veto supported. Seed-B sequential artifact had no hard vetoes; native divergence was not exposed. |
| Statistically supported ranking | None. No candidate/reference comparison was completed. |
| Descriptive-only differences | MAP log densities, mode coordinates, and CPU/XLA timing canaries. |
| Default readiness | Not established. |
| Next evidence needed | A valid multimode authority or a larger bounded plain-HMC campaign with full tuning and sequential R-hat/ESS screen, followed by the predeclared uncertainty-aware comparison. |

## Research-question guardian

| Question | Verdict |
|---|---|
| Did this invalidate the harness? | No. Focused tests, XLA/CPU preflights, target signatures, and rate canaries passed. |
| Did this invalidate the target or math? | No. It exposed target multimodality; all endpoint values/scores were finite and stationary. |
| Did the current candidate fail? | No. Seed-B’s own sequential screen remains passed. |
| Did the reference direction fail? | The single-center quadrature mechanism failed its multimode validity gate. Plain-HMC authority was not evaluated because it was under-budgeted. |
| What repair is triggered? | Increase bounded compute, reduce target runtime with a reviewed implementation change, or construct a reviewed multimode reference that explicitly sums both basins and checks cross-basin mass. |

## Run manifest

- Plan: `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plain-hmc-reference-plan-2026-08-07.md`
- Quadrature diagnostic artifacts: `docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json` and `r3/reference.json`
- Plain preflight: `docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/preflight/preflight.json`
- Four-chain rate: `docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/rate-final/rate.json`
- Four-process one-chain rate: `docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/parallel-chain-rate-final/parallel-chain-rate.json`
- Resource gate: `docs/plans/artifacts/ssl-lstm-q20-seed-b-plain-hmc-reference-2026-08-07/r1/resource-gate-final/resource-gate.json`
- Worker: `bayesfilter/testing/ssl_lstm_q20_plain_hmc_reference_worker.py`
- Supervisor: `docs/benchmarks/run_ssl_lstm_q20_seed_b_plain_hmc_reference_2026_08_07.py`
- Environment: conda `tfgpu`, TensorFlow/TFP, float64, CPU-only with `CUDA_VISIBLE_DEVICES=-1`, XLA enabled
- Shared worktree: dirty; unrelated concurrent changes preserved

## Post-run red team

The strongest alternative explanation is that a multimode-aware reference could
be much cheaper than the conservative plain-HMC projection. That is a valid next
hypothesis, but it was not tested here and cannot retroactively authorize a
single-center quadrature result. The result would be overturned by a reviewed
multimode authority whose truncation/mass and numerical error are independently
bounded, or by a larger owner-approved plain-HMC budget. The weakest evidence is
the four-process timing canary's single warm measurement per worker; it is enough
for a conservative resource stop, not a performance ranking.
