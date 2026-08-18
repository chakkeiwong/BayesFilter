# Strong-smooth weighted NeuTra result (2026-08-13)

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Candidate sampler evidence passed | Canonical sequential fixed-length HMC: retained max R-hat `1.00625 <= 1.01`; min bulk/tail ESS `776.7 / 697.0 >= 400`; all finite | No hard vetoes; native divergence is not exposed and therefore is not claimed to be zero | No normalized posterior/reference authority; one transport seed; high energy-proxy tails are explanatory only | Continue the regression to `nk_like_mild_smooth`; preserve this candidate for any later source-bound robustness arm | No posterior correctness, default promotion, ranking, or general NeuTra claim |

## Evidence contract

- Target: source-bound `nk_like_strong_smooth` formula from `dsge_hmc`, with frozen affine lift and source SHA-256 recorded in the run manifest.
- Transport: weighted forward-KL dense IAF, `(128,128)`, six stages, selected training update 8000 of 10,000; composed map `theta = mu + L @ IAF(z)`.
- Replay: 1,048,576 training rows, 65,536 disjoint selection rows, and 65,536 untouched audit rows, all CPU-generated with distinct stateless seeds.
- HMC: fixed-length TFP HMC only, `L` grid `(3,5,10,15,20,25)`, identity mass in transformed coordinates, XLA, float64, four chains, no NUTS.
- Sequential policy: `bayesfilter_neutra_sequential_hmc_v1`, warm-up and retained minimum 2,000/1,000 per chain, modern rank-normalized R-hat and bulk/tail ESS gates.

## Results

| Quantity | Result |
|---|---:|
| Training wall time | `912.3 s` |
| Selected training update | `8000` |
| Selection / untouched audit NLL | `14.7479 / 14.7468` |
| Audit importance ESS fraction / max weight | `0.36196 / 0.001258` |
| Gradient-clipped updates | `7737 / 10000` |
| Selected HMC `L` / epsilon | `3 / 0.4158491` |
| Warm-up / retained draws per chain | `2000 / 1000` |
| Retained max R-hat | `1.00625` |
| Retained min bulk / tail ESS | `776.7 / 697.0` |
| Sequential wall time | `25.3 s` |

The high clipping count is an optimization diagnostic and weakens any claim that the training curve alone demonstrates convergence. The actual downstream sequential HMC gates passed after the selected checkpoint was frozen and independently retuned.

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed: finite state/value/score/target status, movement, no declared hard vetoes |
| Statistically supported ranking | None; no arm-ranking uncertainty analysis was run |
| Descriptive-only differences | `L`, epsilon, acceptance, energy-proxy tails, training NLL, and clipping count |
| Default-readiness | Not assessed and not promoted |
| Next evidence needed | Repeat the same target-specific protocol on `nk_like_mild_smooth`; later compare predictive/output behavior where a normalized authority exists |

## Red team

The HMC pass may reflect a favorable identity-z mass and selected initial bank rather than a globally well-whitened target. The evidence would be overturned by a fresh initialization bank or independent target/reference check failing the same sequential gates. The weakest evidence is the lack of a normalized posterior authority and the large explanatory energy-proxy maximum; those remain explicit nonclaims.

Artifacts:

- `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/strong-smooth-serious-r1/`
- `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/strong-smooth-serious-r1-replay/`
- `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/strong-smooth-hmc-r2/` (small-arm rejection)
