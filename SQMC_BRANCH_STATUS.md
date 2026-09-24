# SQMC Branch Status and Next Steps

**Date:** 2026-09-12  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Worktree:** `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`  
**Status:** RESOLVED - Ready for disposition decision

---

## Current State

### What Was Completed

✅ **Austria SIR 16-seed statistical comparison** (Sep 9, 2026)
- 4 routes × 16 seeds = 64 cells
- Bootstrap analysis, pairwise tests
- Result: All routes statistically indistinguishable
- File: `docs/plans/sqmc-4route-comparison-final-report-2026-09-09.md`

✅ **LGSSM oracle diagnostic with principled metrics** (Sep 11, 2026)
- 3D canonical LGSSM vs exact Kalman oracle
- 4 routes × 2 seeds, gradient quality analysis
- Result: All routes achieve cosine > 0.999 with warm-start controls
- File: `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md`

✅ **Principled score quality metrics defined** (Sep 11, 2026)
- Cosine similarity (gradient direction)
- Fisher-scaled errors (parameter-specific)
- Induced HMC parameter error
- File: `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md`

✅ **Tuning plan audit** (Sep 12, 2026)
- Found critical errors in proposed tuning campaign
- Identified that research question already answered
- File: `docs/plans/sqmc-tuning-plan-audit-2026-09-12.md`

✅ **Resolution and closure** (Sep 12, 2026)
- Documented route comparison resolution
- Recommendation: use repaired_permutation or iid_dual_cap
- File: `docs/plans/sqmc-route-comparison-resolution-2026-09-12.md`

### Latest Commit

```
c6a1185d SQMC route comparison: Resolution and closure
```

### Untracked Files

These files were created during the tuning plan development but are NOT committed:
- `docs/benchmarks/run_sqmc_tuning.py` - Tuning runner (not needed, plan not executed)
- `docs/plans/sqmc-oracle-tuned-continuation-plan-2026-09-12.md` - Unexecuted tuning plan (superseded by audit)

---

## Key Findings Summary

### Main Result

**All 4 SQMC ancestry routes are equivalent at N=1008:**
- IID Gaussian (identity ancestry)
- Randomized Halton + Hilbert inverse-CDF
- Randomized Halton + Hilbert one-to-one permutation
- Randomized Halton + Hilbert one-to-one permutation (conservative ablation)

### Evidence

1. **Statistical:** Austria SIR 16-seed comparison, bootstrap CIs, pairwise tests → indistinguishable
2. **Gradient quality:** LGSSM oracle diagnostic → all routes cosine > 0.999 with warm-start
3. **Variance decomposition:** Seed variation > route variation (std ~0.7 vs range 0.15)

### Recommendation

**Use `repaired_permutation` (most tested) or `iid_dual_cap` (simplest)**

Route choice is a workflow preference, not a performance optimization.

---

## Branch Disposition Options

### Option 1: Merge to Main (RECOMMENDED)

**What gets merged:**
- Austria SIR statistical comparison results
- LGSSM oracle diagnostic with principled metrics
- Principled score quality metrics standard
- Tuning plan audit (prevents future unnecessary work)
- Resolution document

**Advantages:**
- Makes evidence available in main branch
- Preserves route comparison resolution for future reference
- Documents principled metrics standard for oracle comparisons
- Prevents future agents from repeating this work

**Process:**
```bash
git checkout main
git merge --no-ff rqmc-sqmc-4route-comparison
# Resolve any conflicts if present
git push origin main
```

**Cleanup after merge:**
```bash
# Optional: delete branch after merge
git branch -d rqmc-sqmc-4route-comparison
git worktree remove /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
```

---

### Option 2: Archive Branch Without Merge

**When to use:**
- If main branch should stay focused on current work
- If Austria SIR and LGSSM oracle results are reference-only
- If route comparison is considered exploratory

**Advantages:**
- Keeps main branch history clean
- Evidence preserved on branch for future reference
- Can still retrieve if needed

**Process:**
```bash
# Tag the final state
git tag sqmc-route-comparison-resolved-20260912 rqmc-sqmc-4route-comparison

# Push tag for preservation
git push origin sqmc-route-comparison-resolved-20260912

# Optional: delete local branch (tag preserves it)
git worktree remove /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
git branch -d rqmc-sqmc-4route-comparison
```

**Retrieval later:**
```bash
# If needed, check out the tag
git checkout sqmc-route-comparison-resolved-20260912
```

---

### Option 3: Extract Key Artifacts, Then Archive

**What to extract:**
- Final reports (Austria SIR, LGSSM oracle)
- Principled metrics standard
- Resolution document

**Process:**
```bash
# On main branch
git checkout main

# Cherry-pick or copy key documents
cp /path/to/worktree/docs/plans/sqmc-4route-comparison-final-report-2026-09-09.md docs/plans/
cp /path/to/worktree/docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md docs/plans/
cp /path/to/worktree/docs/plans/sqmc-route-comparison-resolution-2026-09-12.md docs/plans/

# Commit extracted artifacts
git add docs/plans/
git commit -m "Add SQMC route comparison resolution (extracted from branch)"

# Then archive branch as in Option 2
```

**Advantages:**
- Main gets the conclusions without full branch history
- Clean extraction of key findings
- Branch preserved separately for detailed evidence

---

## Recommendation

**Use Option 1 (Merge to Main)** because:

1. **High-value evidence:** Austria SIR statistical comparison and LGSSM oracle diagnostic are rigorous, reusable evidence
2. **Prevents duplicate work:** Future agents need to know route comparison is resolved
3. **Standard establishment:** Principled metrics document establishes standard for future oracle comparisons
4. **Audit value:** The tuning plan audit documents why expensive tuning was unnecessary
5. **Clean branch:** Only 2 untracked files, no messy experimental debris

The merge brings valuable validated evidence into main without cluttering history.

---

## Files Summary

### Committed and Ready to Merge

**Key results:**
- `docs/plans/sqmc-4route-comparison-final-report-2026-09-09.md` - Austria SIR 16-seed comparison ⭐
- `docs/plans/sqmc-oracle-comparison-master-program-v2-2026-09-09.md` - LGSSM oracle diagnostic ⭐
- `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md` - Standard metrics ⭐
- `docs/plans/sqmc-route-comparison-resolution-2026-09-12.md` - Resolution ⭐
- `docs/plans/sqmc-tuning-plan-audit-2026-09-12.md` - Audit findings ⭐

**Supporting documentation:**
- `docs/plans/sqmc-4route-comparison-execution-plan-2026-09-09.md`
- `docs/plans/sqmc-oracle-decision-summary-2026-09-09.md`
- `docs/plans/sqmc-oracle-recovery-plan-20260909.md`
- `docs/plans/sqmc-oracle-executive-status-2026-09-09.md`
- `docs/plans/sqmc-oracle-production-verification-2026-09-09.md`

**Code:**
- `docs/benchmarks/run_sqmc_oracle_characterization.py` - LGSSM oracle runner
- `docs/benchmarks/smoke_test_oracles_sqmc_comparison.py` - Oracle verification
- `docs/benchmarks/analyze_sqmc_4route_comparison.py` - Statistical analysis
- `sqmc_principled_metrics.py` - Metrics calculator

**Artifacts:**
- `docs/benchmarks/artifacts/sqmc-oracle-characterization-canonical-20260909/` - Full LGSSM diagnostic results

### Untracked (Not Needed)

- `docs/benchmarks/run_sqmc_tuning.py` - Tuning runner for unexecuted plan
- `docs/plans/sqmc-oracle-tuned-continuation-plan-2026-09-12.md` - Superseded by audit

**Recommendation:** Delete these files before merge (they document an unexecuted, flawed plan)

---

## Next Action Required

**User decision needed:** Which branch disposition option?

1. **Merge to main** (recommended)
2. **Archive without merge** (if main should stay focused)
3. **Extract key artifacts, then archive** (if you want selective merge)

Once you decide, I can execute the merge/archive process.

---

## Post-Disposition

After branch disposition:

1. **Update any active plans** that reference "SQMC route comparison in progress"
2. **Document in main branch README** (if applicable) that route comparison is resolved
3. **Close any related issues/tickets** for SQMC route investigation
4. **Communicate to team:** Route choice is repaired_permutation or iid_dual_cap, both equivalent

---

## Current Working Directory

You are in the worktree:
```
/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
```

To return to main repo:
```bash
cd /home/chakwong/BayesFilter
```

To switch to main branch in main repo:
```bash
cd /home/chakwong/BayesFilter
git checkout main
```

---

## Summary for User

✅ **SQMC route comparison is RESOLVED**

**Finding:** All 4 routes statistically equivalent at N=1008

**Evidence:** 
- Austria SIR: 16 seeds, statistical validation
- LGSSM: Oracle diagnostic, principled metrics

**Recommendation:** Use repaired_permutation or iid_dual_cap

**Next:** Decide branch disposition (merge, archive, or extract)
