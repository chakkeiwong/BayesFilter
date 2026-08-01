# Phase 2 Result: Frozen Score Bridge And Modern Tuning Gate

Date: 2026-07-13  
Status: `PASS_PHASE2_SCORE_AND_MODERN_TUNING_GATE`  
Plan: `docs/plans/bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-2026-07-13.md`

## Outcome

The two pre-runtime blockers found by the skeptical plan audit are closed.

1. Frozen dense-IAF artifacts now expose explicit scalar and batch score
   pullbacks plus explicit log-Jacobian scores through dense autoregressive,
   mixing, affine, and composed components.
2. The fixed-transport tuner now has an additive serious mode requiring exactly
   four chains, at least 1,000 retained verifier draws per chain, and
   `max(rank-normalized split R-hat, folded rank-normalized split R-hat)` in raw
   target coordinates before kernel handoff.

Legacy short acceptance-only tuner behavior remains available for its bounded
engineering uses. The serious campaign must opt into the modern mode.

## Evidence

| Check | Result |
| --- | --- |
| Explicit dense-IAF pullback versus autodiff | Passed at `1e-11` |
| Explicit logdet score versus autodiff | Passed at `1e-11` |
| Eager versus CPU/XLA explicit score | Passed at `1e-14` |
| Runtime tape/Jacobian source scan | No `GradientTape`, `batch_jacobian`, or `tape.` in frozen/tuner paths |
| Serious verifier below four chains | Rejected at config construction |
| Serious verifier below 1,000 draws/chain | Rejected at config construction |
| Real deterministic IID `[1000,4,2]` archive | Modern admission passed |
| Real scale-mismatch `[1000,4,2]` archive | Ordinary rank R-hat `<1.01`, folded R-hat `>1.01`, admission vetoed despite in-band acceptance |
| Focused score/tuner/convergence suite | `30 passed` |
| Broader frozen-score suite | `36 passed` |

## Repair Record

The first XLA parity assertion required bitwise equality and failed at maximum
absolute difference `2.22e-16`. The analytic score already matched the autodiff
oracle. The cross-backend assertion was corrected to a float64 tolerance of
`1e-14`; no score implementation change was required.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to trusted exact-target GPU/XLA canaries | Frozen transformed score is explicit/correct and serious tuner fails closed on folded R-hat or insufficient chains/draws | No Phase 2 veto fired | Exact 18D GPU training and CPU serious runtime not yet exercised | Build compact campaign driver; run trusted GPU probe, one-step canary, then short resume/freeze canary | Training quality, tuned LGSSM NeuTra convergence, posterior correctness, or superiority |

