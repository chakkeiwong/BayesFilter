# SQMC Tuning Plan Audit

**Date:** 2026-09-12  
**Auditor:** Claude Opus 5  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Status:** AUDIT IN PROGRESS

---

## Audit Scope

Review the SQMC tuning continuation plan for:
1. **Logical consistency:** Does the plan achieve what it claims?
2. **Scientific correctness:** Are the research questions well-formed?
3. **Evidence standards:** Do the methods match the stated goals?
4. **Governance compliance:** Does it follow AGENTS.md and CLAUDE.md?
5. **Execution feasibility:** Can the plan be executed as written?

---

## Executive Summary

**VERDICT:** The tuning plan contains **CRITICAL ERRORS** that make it unexecutable and scientifically incorrect.

**Main issues:**
1. **Wrong research question:** Plan tries to answer "which route is best?" but the Austria SIR 16-seed statistical comparison already answered this definitively: ALL ROUTES ARE INDISTINGUISHABLE
2. **Unjustified scope expansion:** Jumps from 3D LGSSM diagnostic to claim-bearing statistical ranking without justification
3. **Tuning metric mismatch:** Plans to tune on "oracle score L2 error" but the principled metrics show this is NOT the right metric
4. **Missing baseline:** No comparison to UNTUNED performance, so cannot answer "is tuning necessary?"
5. **Ignores existing evidence:** Austria SIR already showed routes are equivalent with 16 seeds, statistical analysis, and production tuning
6. **Wrong continuation trigger:** The UNTUNED diagnostic showed all routes valid with excellent principled metrics (cosine > 0.999), which is a PASS not a "needs tuning" signal

---

## Detailed Findings

### Finding 1: Research Question Already Answered

**Plan claims:** "With exact-scope tuned controls, do the four SQMC ancestry routes produce statistically distinguishable gradient quality?"

**Evidence from Austria SIR (sqmc-4route-comparison-final-report-2026-09-09.md):**
- **Model:** Austria SIR T=20, N=1008
- **Seeds:** 16 (statistical validation grade)
- **Configuration:** Production tuning (trust-region, dual-cap, Contract-E)
- **Result:** "All 4 SQMC transport routes are statistically indistinguishable"
- **Statistical evidence:** Bootstrap 95% CIs all overlap, all 6 pairwise comparisons indistinguishable
- **Seed variation > route variation:** Within-route std (~0.7) > between-route range (0.15)

**Verdict:** The research question has already been answered with TUNED configurations on a production model. Running the same experiment on 3D LGSSM is redundant unless there's a hypothesis that LGSSM behaves differently from Austria SIR.

**What's missing:** No hypothesis stated for why LGSSM would show route differences when Austria SIR did not.

---

### Finding 2: UNTUNED Diagnostic Shows Success, Not Failure

**Plan treatment:** Presents UNTUNED diagnostic as preliminary harness check, implies tuning is needed to answer the route comparison question.

**Actual UNTUNED results (sqmc-oracle-comparison-master-program-v2):**
- **All routes finite and valid:** ✓
- **Gradient direction (cosine similarity):** 0.9995-0.9996 for ALL routes
  - Interpretation: "Gradient direction correct to within 0.05%"
- **Relative gradient norm error:** 0.9-1.7% for ALL routes
  - Interpretation: "Excellent gradient magnitude accuracy"
- **Fisher-scaled errors:** 0.008-0.50 across all parameters
  - Interpretation: "All parameters within acceptable range"
- **Induced HMC error:** 0.0003-0.0009 per leapfrog step
  - Interpretation: "Well within acceptable HMC error accumulation"
- **Route comparison:** "Seed variation and route variation of similar magnitude"

**Principled metrics verdict:** ALL ROUTES PRODUCE USABLE GRADIENTS FOR HMC with UNTUNED controls.

**Implication:** The UNTUNED diagnostic is a SUCCESS, not a preliminary check. If the research question is "can we use SQMC for HMC?", the answer is YES for all routes with warm-start controls.

**What the tuning plan should ask:** "Does exact-scope tuning IMPROVE gradient quality enough to matter?" Not "which route wins after tuning?"

---

### Finding 3: Tuning Metric Wrong for the Research Question

**Plan specifies:**
```
Tuning metric: Oracle score L2 error (primary), cosine similarity (veto)
```

**Problem:** The principled metrics document explicitly establishes that **score L2 error alone is misleading**:

From sqmc-oracle-principled-score-metrics-2026-09-11.md:
> **What we DON'T use (and why):**
> 
> **❌ Score L2 error alone:**
> - Doesn't distinguish direction error from magnitude error
> - Doesn't scale by Fisher information

**Correct tuning metrics for HMC gradient quality:**
1. **Primary:** Cosine similarity (direction correctness)
2. **Secondary:** Fisher-scaled errors (parameter-specific accuracy)
3. **Tertiary:** Induced HMC parameter error (practical impact)
4. **Explanatory:** Score L2 error (for comparison to oracle, not optimization)

**Verdict:** The tuning plan will optimize the WRONG objective. Minimizing L2 error can increase direction error or create parameter-specific biases that worsen HMC performance.

**Fix required:** Redefine tuning objective using principled metrics, or justify why L2 error is appropriate despite the established critique.

---

### Finding 4: Missing Baseline Comparison

**Plan structure:**
- Phase 1: Tune all routes
- Phase 2: Compare tuned routes against each other
- Phase 3: Bootstrap analysis of tuned routes

**What's missing:** Comparison of TUNED vs UNTUNED for the same route.

**Why this matters:**
- Austria SIR routes are indistinguishable WITH tuning
- LGSSM routes are indistinguishable with UNTUNED warm-start controls
- The interesting question is: "Does per-model tuning improve over validated warm-start defaults?"

**Current plan cannot answer:**
- Is exact-scope tuning worth the cost?
- Can we use universal warm-start controls across models?
- Does tuning change route rankings?

**Fix required:** Add UNTUNED baseline to Phase 2 comparison, report tuning benefit per route, and assess whether universal controls are sufficient.

---

### Finding 5: Scope Expansion Without Justification

**UNTUNED diagnostic scope:**
- 3D LGSSM canonical model
- T=20, N=1008
- 2 seeds (harness diagnostic)
- Label: "UNTUNED diagnostic only"
- Explicit non-claims: "No route ranking, statistical superiority, production readiness, HMC benefit"

**Tuning plan scope:**
- Same 3D LGSSM canonical model
- Same T=20, N=1008
- 16 seeds (statistical validation)
- Label: "Tuned, claim-bearing"
- Claims: Statistical route ranking, promotion eligibility

**Scope change:**
- Harness diagnostic → claim-bearing statistical comparison
- 2 seeds → 16 seeds
- Warm-start controls → exact-scope tuning
- Non-claim → production route recommendation

**Missing justification:**
1. Why is 3D LGSSM the right model for a production route decision?
2. Why should we expect route differences on LGSSM when Austria SIR showed none?
3. What claim would this support? (HMC benefit? Production default? Generalization?)
4. How does this relate to the existing Austria SIR statistical comparison?

**Problem:** The UNTUNED diagnostic established that warm-start controls work. The tuning plan treats that success as a reason to run an expensive tuning campaign without stating what new question the tuning answers.

**Fix required:** Either:
- State the hypothesis that LGSSM will show route differences (and why), OR
- Reframe as "validate that warm-start controls remain sufficient" (much smaller scope), OR
- Acknowledge that Austria SIR already answered the route comparison question and close it.

---

### Finding 6: Tuning Budget and Artifacts

**Plan budget:**
- 4 routes × 54 grid cells × 16 seeds = 3,456 cells
- Estimated 8-12 hours GPU time

**Concerns:**
1. **Grid size:** 54 cells is reasonable but couples Sinkhorn/balance steps (both 4, 8, or 16). This assumes they should always match, which may not be optimal.
2. **Seed count:** 16 seeds per grid cell is expensive. Tuning typically uses fewer seeds (4-8) to explore the grid, then validates the winner with more seeds.
3. **Route-specific tuning:** Assumes each route needs different transport controls. Austria SIR used the SAME controls across all routes and they were still indistinguishable.
4. **Artifact scope:** The tuning artifact is scoped to "3D LGSSM T=20 N=1008" but the plan implies it will inform production defaults. This is inconsistent with the LEDH per-scope tuning rule.

**LEDH per-scope tuning rule (CLAUDE.md):**
> Every claim-bearing LEDH model run requires an offline tuning artifact for the exact model/target, route/reset family, horizon/prepared-data regime, particle count, dimensions, dtype/backend, chunk policy, and route-specific control family used by that run. Any changed bound field is a new tuning scope.

**Implication:** Even if we tune for LGSSM T=20 N=1008, that artifact does NOT cover:
- Other LGSSM horizons (T=10, T=50)
- Other models (Austria SIR, Predator-Prey, KSC-SV)
- Other particle counts (N=504, N=2016)

**Verdict:** The tuning plan produces narrow-scope artifacts that cannot inform a general "which route should we use?" decision. It can only answer "which route for LGSSM T=20 N=1008?"

**Fix required:** Either:
- Scope the research question to LGSSM T=20 N=1008 only (and justify why this matters), OR
- Expand to multi-model, multi-horizon tuning (much larger budget), OR
- Use existing Austria SIR evidence and close the route comparison question.

---

### Finding 7: Decision Framework Inconsistent

**Plan Scenario A (Expected):** "Routes still indistinguishable"
- Conclusion: "All 4 routes produce equivalent gradient quality even with exact-scope tuning"
- Recommendation: "Use simplest route (iid_dual_cap) or most tested (repaired_permutation)"

**Problem:** This is the Austria SIR result already. Why run an expensive tuning campaign if we expect the same answer?

**Plan Scenario B:** "Clear winner emerges"
- Conclusion: "Route X produces superior gradients"
- Recommendation: "Use winner as default for LGSSM-class models"

**Problem:** 
1. Austria SIR is not "LGSSM-class" - it's a nonlinear epidemiological model. No evidence that LGSSM and Austria SIR behave similarly for route comparison.
2. The tuning artifact is scoped to T=20 N=1008. Cannot generalize to "LGSSM-class models" without multi-scope validation.
3. "Most tested" (repaired_permutation) has stronger evidence than "won one tuning comparison on one model/horizon."

**Plan Scenario C:** "Tuning makes routes worse"
- Reaction: "Investigate tuning metric mismatch, overfitting, grid insufficiency"

**Problem:** The plan doesn't consider the most likely explanation: **warm-start controls are already near-optimal, and tuning noise dominates tuning signal at this scale.**

**Verdict:** The decision framework assumes tuning will reveal new information, but doesn't seriously consider that the UNTUNED diagnostic + Austria SIR evidence may already be sufficient.

**Fix required:** Add Scenario D: "Tuning provides no benefit over warm-start controls" with the decision "use universal warm-start controls, close route comparison as resolved."

---

## Governance Compliance Check

### AGENTS.md: Academic Research Governance And Proportionality

**Policy:** "For trusted local academic and research repositories, optimize governance for scientific validity, reproducibility, bounded compute, and progress. Do not import production-service security ceremony without a concrete applicable threat."

**Plan compliance:** ✓ No excessive ceremony, plain-language approval sufficient.

**Policy:** "Unless a plan identifies a concrete adversarial risk or the user explicitly requests stronger controls, do not require: separate human approval for each local retry under an unchanged scientific contract and campaign budget."

**Plan compliance:** ✓ No magic tokens, hash-bound approvals, or per-retry approval gates.

---

### AGENTS.md: Research Question Guardian

**Policy:** "For research experiments, benchmarks, ablations, numerical comparisons, sampler diagnostics, ML-training comparisons, and multi-phase investigations, preserve the research question separately from the implementation checklist."

**Plan compliance:** ❌ FAIL

**Issue:** The research question ("do routes differ with tuning?") is already answered by Austria SIR. The plan doesn't preserve or update the research question in light of existing evidence.

**Policy:** "Before stopping a multi-phase research plan, answer explicitly: did the result invalidate the harness, implementation, target, data, math, or artifact, or did it merely show that the current candidate failed?"

**Plan compliance:** ❌ FAIL

**Issue:** The UNTUNED diagnostic did NOT fail. All routes passed with excellent principled metrics. The plan treats this success as a preliminary check rather than an answer to the harness validity question.

---

### AGENTS.md: Heuristic Dominance Gate

**Policy:** "Before interpreting or reporting any learned, optimized, or otherwise complex method, apply the Heuristic Dominance Gate."

**Plan compliance:** N/A (not applicable to route comparison, but relevant if claiming "SQMC is good for HMC")

**Note:** The UNTUNED diagnostic implicitly passes a heuristic check: SQMC gradients are close to exact oracle gradients. The relevant heuristic is "use exact gradients" (the oracle), and SQMC comes within 0.05% direction error of that heuristic.

---

### AGENTS.md: Statistical Evidence Discipline

**Policy:** "Interpret stochastic experiments with statistical humility. Do not claim one method is better, superior, improved, or the best unless the comparison has a predeclared criterion and uncertainty evidence supporting that ranking."

**Plan compliance:** ✓ Includes bootstrap analysis, pairwise tests, multi-seed validation

**Policy:** "Passing a hard screen is not evidence of superiority. It means the candidate remains viable under that screen. Among candidates that pass the hard screen, state that they are statistically indistinguishable under current evidence unless uncertainty analysis supports a ranking."

**Plan compliance:** ❌ FAIL

**Issue:** The plan aims to find a statistical ranking, but Austria SIR already showed routes are indistinguishable. The plan doesn't discuss what to conclude if LGSSM also shows indistinguishable routes (the expected result).

---

### CLAUDE.md: LEDH Per-Scope Tuning Rule

**Policy:** "Every claim-bearing LEDH model run requires an offline tuning artifact for the exact model/target, route/reset family, horizon/prepared-data regime, particle count, dimensions, dtype/backend, chunk policy, and route-specific control family used by that run. Any changed bound field is a new tuning scope."

**Plan compliance:** ✓ Creates exact-scope tuning artifacts for LGSSM T=20 N=1008

**But:** The plan implies these artifacts will inform a general "which route?" decision, which violates the per-scope rule. A route preference based on LGSSM T=20 N=1008 tuning does NOT apply to other models/horizons without their own tuning artifacts.

---

## Recommended Actions

### Option A: Close Route Comparison as Resolved (RECOMMENDED)

**Rationale:**
1. Austria SIR 16-seed statistical comparison: routes indistinguishable
2. LGSSM UNTUNED diagnostic: all routes produce excellent gradients (cosine > 0.999)
3. No hypothesis for why exact-scope tuning would reveal route differences
4. Tuning budget (8-12 hours) is expensive for a question already answered

**Action:**
1. Document that route comparison is resolved: "All routes equivalent at N=1008 for Austria SIR and LGSSM with warm-start controls"
2. Recommendation: Use `repaired_permutation` (most tested) or `iid_dual_cap` (simplest)
3. Close the SQMC route comparison branch
4. Archive artifacts for reference

**Risk:** Miss a genuine route difference that only shows up with exact-scope tuning on LGSSM

**Mitigation:** If future work finds gradient quality issues, reopen route comparison with specific failure mode

---

### Option B: Minimal Validation (if user wants more evidence)

**Scope:** Validate that warm-start controls remain sufficient for LGSSM

**Design:**
- **Question:** "Do warm-start controls produce acceptable gradients on LGSSM T=20 N=1008?"
- **Method:** Extend UNTUNED diagnostic from 2 seeds to 16 seeds
- **Metrics:** Principled gradient quality metrics (cosine, Fisher-scaled errors, induced HMC error)
- **Pass criterion:** Mean cosine > 0.999, Fisher-scaled < 0.5 for most parameters
- **Budget:** ~1 hour (4 routes × 16 seeds × 5 directions = 320 cells, already fast from diagnostic)
- **Decision:** If pass → warm-start validated for LGSSM, close route comparison. If fail → investigate why LGSSM differs from Austria SIR

**Advantage:** Answers "are warm-start controls universal?" without expensive per-route tuning

**Risk:** Still doesn't answer "which route is best?" but that question is already answered by Austria SIR

---

### Option C: Full Tuning Campaign (NOT RECOMMENDED unless hypothesis stated)

**Only proceed if:**
1. User states explicit hypothesis for why LGSSM will show route differences when Austria SIR did not
2. User needs LGSSM-specific evidence for a downstream claim
3. User wants to validate that exact-scope tuning doesn't reveal hidden route differences

**Required changes to plan:**
1. **Research question:** "Does exact-scope tuning on LGSSM reveal route differences not visible with warm-start controls or on Austria SIR?"
2. **Baseline:** Include UNTUNED performance in Phase 2 comparison
3. **Tuning metric:** Use principled metrics (cosine similarity, Fisher-scaled errors) not raw L2
4. **Decision framework:** Add Scenario D for "tuning provides no benefit, routes still indistinguishable"
5. **Scope boundaries:** Clarify that results apply to LGSSM T=20 N=1008 only, not general route preference
6. **Integration with Austria SIR:** Explain relationship between LGSSM and Austria SIR evidence

**Budget:** 8-12 hours GPU + 1-2 hours analysis

---

## Audit Verdict

**The current tuning plan is NOT READY FOR EXECUTION.**

**Critical flaws:**
1. Research question already answered by Austria SIR
2. Treats UNTUNED success as failure signal
3. Wrong tuning metric (L2 error instead of principled metrics)
4. Missing baseline (TUNED vs UNTUNED comparison)
5. Unjustified scope expansion (diagnostic → production route recommendation)
6. Ignores existing statistical evidence of route equivalence

**Recommended action:** Close route comparison as resolved based on Austria SIR + LGSSM UNTUNED evidence, OR execute Option B (minimal validation) if user wants additional confirmation for LGSSM.

**Do NOT execute the tuning campaign without:**
1. Explicit hypothesis for why LGSSM would differ from Austria SIR
2. Revised research question addressing existing evidence
3. Corrected tuning metrics using principled gradient quality measures
4. Baseline comparison (TUNED vs UNTUNED)

---

## Questions for User

Before any execution decision:

1. **Why do we need a route comparison on LGSSM?** Austria SIR already showed routes are equivalent with statistical validation. What would LGSSM evidence add?

2. **What is the downstream claim?** Is this for:
   - HMC validation? (UNTUNED already shows good gradients)
   - Production default? (Austria SIR already answered this)
   - Academic publication? (Need multi-model evidence, not just LGSSM)
   - Code verification? (UNTUNED diagnostic already verified harness)

3. **Is exact-scope tuning worth the cost?** UNTUNED controls produce excellent gradients (cosine > 0.999). Is 8-12 hours GPU time justified to see if tuning improves this to > 0.9995?

4. **Should we close the route comparison question?** We have:
   - Austria SIR: 16 seeds, tuned, statistically indistinguishable
   - LGSSM: 2 seeds, UNTUNED, all routes excellent
   - No hypothesis for why more evidence would change the conclusion

---

## Audit Metadata

- **Audit date:** 2026-09-12
- **Documents reviewed:** 5 (master program, tuning plan, principled metrics, Austria SIR final report, tuning runner code)
- **Code reviewed:** run_sqmc_tuning.py
- **Evidence reviewed:** Austria SIR 16-seed statistical comparison, LGSSM UNTUNED diagnostic
- **Governance policies checked:** Research Question Guardian, Statistical Evidence Discipline, LEDH Per-Scope Tuning Rule, Academic Research Governance

---

## Changelog

- 2026-09-12: Initial audit after user request to review tuning plan
