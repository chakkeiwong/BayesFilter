# Canonical LGSSM TF32 Horizon-Representative Balance Repair Plan

Date: 2026-07-18
Campaign ID: `canonical-lgssm-tf32-horizon-representative-balance-repair-20260718`
Status: `PLANNED_NOT_EXECUTED`

## Research Intent Ledger

| Field | Frozen statement |
| --- | --- |
| Main question | Can one fixed terminal-balance count pass the unchanged direct probability-marginal gates across representative T=10 active states in float32/TF32, while preserving the fused value/score and one-solve XLA topology? |
| Mechanism | Extend selection inputs to fixed T=10 prepared states and disjoint design/audit seeds; evaluate candidates `2,3,5,8` using only per-seed/time `TV_col` and `E_row`, then rerun same-count T=10 claim before considering T=50. |
| Baseline | Current pushed count-2 T=10 claim, which is finite and fast but fails `E_row` at one or more later states. |
| Promotion criterion | Smallest positive candidate passes every active seed/time on design and audit at `TV_col <= 1e-4`, `E_row <= 0.01`, with exact one shared state/sweep per active time step, no diagnostic work, replay equality, and valid charts/resets. |
| Veto | Any marginal/reset/chart failure, changed finite program, wrong count identity, source mismatch, non-XLA route, Python horizon unrolling, wrong chunk policy, OOM, cap breach, or incomplete per-seed/time artifact. |
| Forbidden claims | No claim that the selected count is universal, optimal, statistically superior, HMC-ready, or valid for T=50 until its own conditional gate passes. |

## Entry Conditions

- Pushed repair commit `1164c88` remains the source baseline.
- T=2 TF32 selection/audit and same-count float64 reference pass.
- T=10 16-seed claim is preserved as a failed comparator with exact artifact.
- No tolerance, chunk policy, Sinkhorn schedule, or GPU memory limit changes.

## Required Changes

1. Add per-seed/time `TV_col`, `E_row`, marginal-valid, reset-valid, and chart
   histories to the node schema. These are explanatory and veto diagnostics;
   aggregate maxima remain the promotion screen.
2. Create a marginal-only selector for T=10 using fixed seeds `81520..81527`
   for design and `81528..81535` for audit. Keep candidate counts exactly
   `2,3,5,8`, select the smallest positive design pass, and audit without
   retuning.
3. Bind the selected count, T=10 seed sets, source hashes, chunk policy,
   dtype, TF32, XLA, and plan identity into the selection artifact.
4. Run a fresh T=10 16-seed TF32 claim at the selected count. Do not run T=50
   unless this node passes.
5. If T=10 passes, run T=50 one-seed witness then its frozen 16-seed claim;
   otherwise write a blocker result and stop.

## Checks And Evidence Contract

- CPU-hidden focused tests for schema, selector, and mass/JVP parity.
- Trusted GPU preflight and 8192 MiB logical cap.
- XLA graph contains `StatelessWhile` and no Python horizon duplication.
- `K=N=1024`, exact one shared Sinkhorn/balance/transport state per active step,
  zero marginal tile and diagnostic solver work.
- Same-count float64 reference only for precision comparison, never selection.
- Fresh versioned artifacts; no overwrite.

## Budget And Stop Conditions

At most 8 GPU launches and 60 minutes total. Stop on exhausted budget,
selection failure with no declared candidate, any resource/identity artifact
failure, or T=10 claim veto. A failed candidate advances only to the next
predeclared candidate; it does not justify relaxing a gate or changing `N/K`.

## Handoff

Hand off to the T=50 conditional ladder only when the T=10 selector audit and
claim node both pass. Otherwise hand off a blocker result that classifies the
failure as marginal-state selection, numerical implementation, or resource,
and preserves the exact failing seed/time evidence.
