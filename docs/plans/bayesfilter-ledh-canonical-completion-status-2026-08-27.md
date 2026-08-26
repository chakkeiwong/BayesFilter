# LEDH Canonical Completion Program — Final Status

Branch: `worktree-ledh-canonical-rebuild` (`.claude/worktrees/ledh-canonical-rebuild`)
Head: 6abcff40
Date: 2026-08-27

## COMPLETE

### Q1 — Score-path completion ✓
- S6/S7 tangent wiring: full-program analytical score (reset + dual-cap) oracle-green
- Annealed-mode score: stage-weight tangents, 3 oracle gates green
- dlgssm q/r threading: density and covariance tangents, 2 oracle gates green
- Austria reduction slice: vs exact 18-dim Kalman, gate green
- Derivation ledger S1–S8 fully green, zero autodiff anywhere (C-9 verified)

Wall time: ~1 CPU-day (interleaved with Q2)
Commits: 11b90e58 → c546a056 (Q1.1–Q1.4)

### Q2 — Calibration campaign ✓
Seven curves, ~2 GPU-hours total:

1. **Annealed k/c surface**: k=4, c=8 calibrated (0.30 ESS threshold, 3 seeds). 
   c=∞ measured harmful (NaN on f32/TF32, 10× ESS loss at f64).
   
2. **Trust-radius model-trust**: promotion criterion honestly failed (takeoff 
   cloud heterogeneity; no uniform ρ≥0.75 radius). 0.5 retained as warm start.
   
3. **LM damping**: 1e-2 measured non-harm (40× condition margin, ≤0.3% residual 
   impact). Zero-damping viable but not promoted (higher bar for default change).
   
4. **Relative ridge**: derived + measured; absolute 1e-5 nominal-only on TF32 
   (4–5 orders below roundoff scale). Replacement form derived, Class-C gate pending.
   
5. **Dual-cap constants**: 2026-08-07 owner family decision registered (C4 closed).

6. **f32/TF32 decision**: spectral cap is PART OF lane definition (uncapped = NaN).

7. **Austria Fisher gate**: 3 directions green at annealed k=8. Plain-mode bias 
   classified as 1/ESS estimator bias (not score defect; oracle gates verify all 
   directions). Annealing reduces bias 6–10× at fixed N.

Artifacts: `docs/benchmarks/q2_calibration_20260824/`
Commits: 5a0a4684 → e19edb77

### Q3 — Production leaderboard ✓
Six models × canonical algorithm (value + score where applicable), 8 seeds, 
N=1008, frozen targets. True production program: dual-cap trust region ON per 
registry definition. Conformance stamp: ledh-canonical-conformance-v1-2026-08-24.

All value cells finite; zero score crashes. Statistical status: hard vetoes pass, 
no ranking supported (8 seeds, no uncertainty analysis), all differences 
descriptive only.

Repair: Austria annealed Cholesky NaN fixed via affine-restore relative-ridge 
guards (Class B + light Class C, 1e-12 floor calibrated, non-harm verified).

Artifacts: `docs/benchmarks/artifacts/ledh_canonical_leaderboard_2026-08/q3_board/`
Report: `docs/benchmarks/q3-canonical-leaderboard-report-2026-08-25.md`
Wall time: ~2.5 GPU-hours
Commits: 920e3ba5 → 6abcff40

## REMAINING

### Q4 — Merge and integration (OWNER-GATED)

Merge `worktree-ledh-canonical-rebuild` → main; coordinate with sibling agent's 
branch (both touch LEDH territory); wire conformance suite into standing test 
cadence; refresh AGENTS.md pointers.

**The merge decision, timing, and conflict resolution are the owner's.** 
Q1–Q3 ran pre-merge; nothing in Q4 blocked Q1–Q3.

### Q5 — Successor programs (OUT OF SCOPE)

NeuTra training, HMC campaigns, posterior correctness claims, default-readiness 
claims. Own contracts per gap register R2–R4.

## Evidence summary

- Q1: 11 oracle gates (S6/S7 full program, annealed 3×, q/r 2×, Austria reduction)
- Q2: 7 calibration curves with pre-declared evidence contracts
- Q3: 6 model rows, conformance-stamped, configuration-status-first enforced
- Governance: G-1 through G-6 green, meta-governance green, C-9 no-autodiff green
- Fidelity tally: 5/5 found-and-fixed (last: Austria annealed Cholesky NaN)

## Gap register status

Production-readiness gaps A1–A7, B1–B3, C1–C6, D1–D3 recorded in 
`docs/plans/bayesfilter-production-readiness-gap-register-2026-08-26.md`.
These are Q5-scope (NeuTra/HMC programs), not Q1–Q3 blockers.

Remediation phases R1–R4 enumerated; R1 (ridge contract, dtype param, linear 
score, gen-SV freeze, provenance check, seed pairing) mostly complete within 
Q1–Q3 work. R2–R4 are future campaigns.
