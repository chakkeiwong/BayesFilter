# SQMC Phase 3 Result Note: TUNED vs UNTUNED Comparison

**Date:** 2026-09-16  
**Campaign:** SQMC 4-route comparison (master program 2026-09-12)  
**Phase:** Phase 3 (TUNED vs UNTUNED paired comparison)  
**Model:** LGSSM T=20, N=1008, 3D diagonal, float64 GPU backend  
**Seeds:** 16 claim seeds (97701-97716), disjoint from 16 tuning seeds (50001-50016)  
**Oracle:** Exact Kalman filter (KSC transform for 3D diagonal LGSSM)  

---

## Executive Summary

Phase 3 completed successfully across all 4 routes. Statistical evidence supports tuning benefit for **1 route only** (iid_dual_cap). The remaining 3 routes show descriptively favorable or neutral L2 differences but **95% confidence intervals include zero**, rendering them statistically indistinguishable.

**Key finding:** Multi-objective Pareto tuning produced a statistically significant L2 reduction only for iid_dual_cap (-1.8% mean L2, 95% CI excludes zero). The other 3 routes passed hard constraints but showed no statistically supported improvement over warm-start controls.

---

## Comparison Results

### Summary Table

| Route                          | L2 Untuned | L2 Tuned | Δ L2    | 95% CI                    | Verdict               |
|--------------------------------|------------|----------|---------|---------------------------|-----------------------|
| iid_dual_cap                   | 1.4244     | 1.3992   | −0.0252 | [−0.0434, −0.0080]        | **favours_tuned**     |
| previous_inverse_cdf           | 1.4790     | 1.4717   | −0.0073 | [−0.0426, +0.0286]        | indistinguishable     |
| repaired_permutation           | 1.4912     | 1.4633   | −0.0279 | [−0.0587, +0.0035]        | indistinguishable     |
| repaired_permutation_ablation  | 1.5101     | 1.5184   | +0.0083 | [−0.0122, +0.0292]        | indistinguishable     |

### Statistical Evidence Detail

**iid_dual_cap:**
- Paired diff: −0.025229
- 95% CI: [−0.0434373, −0.0080302]
- Sign test: n=16, neg=12, pos=4, p=0.0768
- Verdict: **favours_tuned** (CI excludes zero)
- Cosine similarity also favours_tuned

**previous_inverse_cdf:**
- Paired diff: −0.007274
- 95% CI: [−0.0426159, +0.0285757]
- Sign test: n=16, neg=8, pos=8, p=1.0000
- Verdict: indistinguishable (CI includes zero)

**repaired_permutation:**
- Paired diff: −0.027917
- 95% CI: [−0.0586712, +0.0034696]
- Sign test: n=16, neg=12, pos=4, p=0.0768
- Verdict: indistinguishable (CI includes zero, barely)

**repaired_permutation_ablation:**
- Paired diff: +0.008320
- 95% CI: [−0.0122277, +0.0291892]
- Sign test: n=16, neg=7, pos=9, p=0.8036
- Verdict: indistinguishable (CI includes zero, descriptively neutral)

---

## Hard Constraint Status

All 4 routes passed the tuned-arm hard constraints on claim seeds:
- Cosine similarity ≥ 0.999: PASS (all routes ≥ 0.9995)
- Relative norm error ≤ 0.05: PASS (all routes ≤ 0.018)
- Fisher-scaled error ≤ 1.0: PASS (all routes ≤ 0.62)

**Interpretation:** Tuned controls preserved oracle fidelity. No route was vetoed for constraint violation.

---

## Observations

### Route Distinctness
L2 untuned baselines differ materially across routes (1.42 to 1.51), confirming that routes produce distinct score trajectories under identical controls.

### Tuning Benefit Magnitude
The statistically supported benefit (iid_dual_cap) is modest in absolute terms (−0.025 L2, −1.8% relative) but consistent across 12/16 paired seeds.

### Near-Miss Cases
- **repaired_permutation:** Large descriptive improvement (−0.028) but 95% CI barely includes zero (+0.0035 upper bound). With more seeds or tighter pairing, this route might cross the statistical threshold.
- **repaired_permutation_ablation:** Descriptively neutral (+0.008), CI firmly includes zero. No evidence of tuning benefit.

### Sign Test vs Bootstrap CI
Sign test p-values (e.g., p=0.0768 for iid_dual_cap and repaired_permutation) suggest directional evidence but do not alone establish statistical significance. The bootstrap 95% CI exclusion of zero is the primary criterion per the master program.

---

## Decision Table

| Criterion                     | Status                        |
|-------------------------------|-------------------------------|
| **Primary (L2 error)**        | 1/4 routes statistically improved |
| **Promotion veto**            | None (all hard constraints pass) |
| **Continuation veto**         | None                          |
| **Main uncertainty**          | Whether near-miss routes warrant extended replication |
| **Next justified action**     | Phase 4 interpretation and route recommendation |

---

## Non-Claims

**Not concluded from Phase 3:**
1. **Production readiness:** Oracle fidelity on LGSSM T=20 N=1008 at float64 does not establish HMC convergence benefit, production-scale performance, or TF32/float32 behavior.
2. **Route preference for other models/horizons:** These results apply only to LGSSM T=20 N=1008 with the exact tuning scope (dtype, backend, particle count, controls).
3. **Generalization beyond tuning scope:** Each model/horizon/route/dtype/backend combination requires its own tuning artifact per the LEDH Per-Scope Tuning Rule.
4. **Mechanism understanding:** Statistical evidence does not explain *why* iid_dual_cap benefits from tuning while other routes do not.
5. **Extended replication value:** Whether increasing claim seeds from 16 to 32+ would promote near-miss routes (e.g., repaired_permutation) to statistical significance is unknown.

---

## Artifacts

**Phase 3 result JSON (single route per file):**
- Only `docs/benchmarks/artifacts/sqmc-tuned-vs-untuned-lgssm-20260913/phase3-20260916/result.json` found via filesystem search (repaired_permutation)
- Full results extracted from task outputs:
  - `/tmp/.../tasks/bpfwr33iz.output` (iid_dual_cap, exit 0, wall time ~29 min)
  - `/tmp/.../tasks/b5c85fznr.output` (previous_inverse_cdf, exit 0, wall time ~34 min)
  - `/tmp/.../tasks/b5skgzylp.output` (repaired_permutation, exit 0, wall time ~34 min)
  - `/tmp/.../tasks/bxcu1ohbj.output` (repaired_permutation_ablation, exit 0, wall time ~30 min)

**Total Phase 3 wall time:** ~127 minutes (~2.1 hours, within 2-3 hour budget)

**Git commit:** Master program references 46b12fa9; Phase 3 execution used worktree commit 509871fd

---

## Next Step: Phase 4

Per master program lines 461-487, Phase 3 completion triggers **Phase 4: Route comparison analysis and interpretation**.

Phase 4 scope (from master program):
- Synthesize TUNED vs UNTUNED findings across all 4 routes
- Document tuning benefit magnitude and statistical support per route
- Route recommendation (if warranted) or declaration that no route dominates

**Phase 4 checkpoint:** After interpretation, present recommendation to user for final decision.

---

## Manifest

- **Plan file:** docs/plans/sqmc-master-program-2026-09-12.md
- **Result note:** docs/plans/sqmc-phase3-result-note-20260916.md
- **Git branch:** rqmc-sqmc-4route-comparison
- **Working directory:** /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
- **Environment:** tftwogpu conda env, CUDA_VISIBLE_DEVICES=1
- **Execution date:** 2026-09-16 11:17–12:19 (Phase 3 parallel execution)
