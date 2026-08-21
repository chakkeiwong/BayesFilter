# NeuTra Banana Budget Repair And HMC Plan (2026-08-16)

## Research Intent Ledger

| Field | Predeclared statement |
|---|---|
| Main question | Does a longer target-specific cold reverse-KL budget repair the seed-sensitive banana coordinate-mean failure observed under the 3,000-update root-preserving transport? |
| Candidate mechanism | Same root-preserving `(32,32)` dense-IAF transport, float64/XLA, batch-native reverse-KL, and learning-rate schedule, with the joint budget increased from 3,000 to 6,000 updates. The original 3,000-update schedule boundaries at updates 1,800 and 2,550 remain fixed; extra updates use the final low learning rate. |
| Baseline | The previously viable 3,000-update root-preserving banana protocol, rerun on the same three fresh seeds for a paired diagnostic comparator. |
| Fresh seeds | Seeds `13, 14, 15`; these are disjoint from the terminal replication seeds `10, 11, 12` and are not selected after inspecting outcomes. |
| Primary repair gate | All three 6,000-update fresh runs pass untouched 131,072-draw exact-law coordinate-mean, coordinate-second-moment, and adjacent-cross-moment screens at the existing 99.9% intervals. |
| HMC gate | Only after the 6,000-update arm passes all three fresh seeds: tune `L=(3,5,10,15,20,25)` with `L=1` forbidden, then use the shared sequential HMC controller with the existing R-hat, ESS, finite-state, movement, energy, and exact-law retained-draw gates. |
| Hard vetoes | Nonfinite training/transport/audit values, invalid GPU/XLA/memory-growth provenance, failed exact-law repair gate, stale or reused artifact root, HMC convergence/ESS/finite/movement/energy failure, or failed retained exact-law screens. |
| Explanatory diagnostics | Reverse-KL loss, selected checkpoint, importance ESS fraction, ratio standard deviation, standardized coordinate discrepancies, runtime, acceptance, and leapfrog count. These do not promote a candidate. |
| Nonclaims | No universal training budget, no superiority ranking, no multimodal coverage, no SSL-LSTM transfer, and no production/default HMC readiness. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Root-preserving permutation and `(32,32)` width | Target-specific arm selected by the 2026-08-15 repair campaign | It is the only previously replicated banana arm and preserves the known triangular banana geometry | The architecture may still be insufficient | Exact-law moments and ratio diagnostics | Reviewed warm start, not universal default |
| `LR=5e-4` peak with the existing `1.0/0.1/0.01` schedule | Selected by the prior banana repair campaign | Keeps the repair isolated to budget rather than changing several controls at once | The learning rate may still be target-sensitive | Baseline and extended paired runs | Reviewed warm start |
| 6,000 updates | Repair hypothesis, derived as 2x the failed 3,000-update budget | Tests whether residual coordinate drift is a finite-budget effect while preserving objective, architecture, and the original LR phase boundaries | More updates may merely remain in the same basin or overfit the selection loss | Compare exact-law screens and selected/terminal losses | Hypothesis under test |
| Three fresh seeds | Existing replication minimum | Tests repeatability rather than a single favorable run | Still bounded evidence | Per-seed hard gate | Reviewed minimum |
| 131,072 audit draws and 99.9% intervals | Existing exact-law control protocol | Preserves the prior hard gate and avoids threshold drift | Multiple screens can produce occasional false vetoes; this is recorded, not relaxed post hoc | Per-screen standardized discrepancies and exact partition record | Reviewed inherited gate |

## Evidence Contract

| Item | Predeclared value |
|---|---|
| Comparator | Paired 3,000-update root-preserving banana arm on seeds `13,14,15`; no alternative architecture or LR is introduced. |
| Promotion criterion | The 6,000-update arm has `3/3` fresh exact-law passes. This establishes viability for this target-specific protocol only. |
| Promotion veto | Any failed exact-law screen, nonfinite state, or invalid provenance blocks HMC and candidate promotion. |
| Continuation veto | Broken target binding, corrupted artifacts, missing required diagnostics, or a runtime/infrastructure failure that prevents a valid comparison. A candidate failure alone is not a continuation veto. |
| HMC criterion | Sequential controller reaches warm-up and retained readiness under the shared thresholds and its retained draws pass exact-law screens. |
| Explanatory-only evidence | Loss, ESS fraction, ratio SD, runtime, acceptance, and `L` explain behavior but cannot override a hard veto. |
| Non-conclusions | Passing does not establish universal convergence, statistical superiority, multimodal coverage, or transfer to SSL-LSTM. |
| Artifact | Terminal root `docs/plans/artifacts/neutra-banana-repair-2026-08-16-r3/` with plan, manifest, progress, per-cell JSON, HMC archives, result, and SHA-256 hashes. |

## Skeptical Plan Audit

| Risk | Disposition |
|---|---|
| Budget change is confounded with architecture, permutation, or LR | Vetoed: baseline and repair use the identical target, transport, peak LR, fixed 3,000-update phase boundaries, batch, dtype, and XLA path; only update count changes. |
| Fresh seeds are selected after seeing failure | Vetoed: seeds `13,14,15` are fixed in this plan before execution and are disjoint from prior seeds. |
| Audit data leaks into training or selection | Vetoed: audit draws are generated after training from a separate stateless seed family and only nominate/veto; no checkpoint or LR is selected from them. |
| More updates are treated as proof of convergence | Vetoed: the exact-law gate remains primary; loss and ESS are explanatory only. |
| HMC is run after a failed proposal audit | Vetoed: HMC is conditional on `3/3` repair passes. |
| A short HMC run passes by acceptance alone | Vetoed: the shared sequential R-hat/ESS/finite/movement/energy and post-HMC exact-law gates remain mandatory. |
| Artifact overwrites historical evidence | Vetoed: the output root must be fresh and hashes exclude only the hash-list file itself. |

Audit verdict: the plan isolates one repair hypothesis, preserves the existing
exact-law and HMC gates, and has explicit stop conditions. It is fit for
execution under the bounded local GPU campaign budget.

## Pre-Rerun Review Of `r2`

The first implementation attempt completed its six training cells and HMC
run, but review found that it scaled the learning-rate phase boundaries with
the 6,000-update horizon. That changed both the budget and the schedule, so
`docs/plans/artifacts/neutra-banana-repair-2026-08-16-r2/` is retained only as
debugging/non-promotable evidence. Its proposal audits passed, but its HMC
retained exact-law screen failed; it is not used to support a scientific
conclusion. The runner was corrected to fix the original 3,000-update phase
boundaries before the terminal rerun.

## Execution

1. Run paired 3,000- and 6,000-update training cells for seeds `13,14,15`.
2. Write per-cell finite/provenance/training/audit artifacts and update progress after every cell.
3. If and only if all extended cells pass, tune and run sequential HMC from the final extended transport trained with seed `15`.
4. Write a result note with decision and inference-status tables, red-team analysis, and the next action.
