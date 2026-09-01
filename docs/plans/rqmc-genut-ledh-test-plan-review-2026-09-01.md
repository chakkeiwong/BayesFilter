# RQMC GenUT LEDH Test Plan — Skeptical Review

**Reviewer:** Claude Code (Opus 5)  
**Review Date:** 2026-09-01  
**Plan Date:** 2026-09-01  
**Review Mode:** Skeptical audit before execution

## Review Questions

### 1. Wrong Baseline Risk
**Question:** Is the MC baseline actually using `tf.random.normal` with properly seeded generators, or does it use the old consecutive-seed bug?

**Finding:** The plan names `tf.random.normal` but does not specify the seed hashing protocol. The memory file `tf-consecutive-from-seed-is-one-stream.md` documents that consecutive `from_seed(s)` and `from_seed(s+1)` share a Philox stream, so replication loops over consecutive seeds are pseudo-replications.

**Required fix:** The claim-bearing runner must use the `replication_generator(seed)` pattern from the memory file:
```python
def replication_generator(seed: int) -> tf.random.Generator:
    material = np.random.SeedSequence(int(seed)).generate_state(2, dtype=np.uint64)
    return tf.random.Generator.from_seed(int(material[0]))
```

**Verdict:** Plan is **incomplete**. Add explicit seed-hashing requirement to runner specification before execution.

---

### 2. Tuning Scope Match
**Question:** Do the rescued GenUT tuning artifacts match the dual-cap route, or were they tuned for a different route?

**Finding:** The plan states "The original GenUT tuning artifacts used N=1008 but predate the dual-cap mechanism." This means the rescued artifacts are **warm-start only** and cannot satisfy the per-scope tuning requirement.

**Required action:** Fresh tuning for all 3 models under the dual-cap route is mandatory. The plan correctly budgets "5 tuning runs per model" but does not state what happens if warm-starting from old artifacts produces a non-converged tuning result.

**Verdict:** Plan is **acceptable** but add: "If warm-start tuning does not converge after 5 attempts, run fresh cold-start tuning with expanded grid."

---

### 3. Proxy Metric Promotion Risk
**Question:** Is terminal log-likelihood the actual LEDH score-lane estimand, or is it a proxy?

**Finding:** The plan states "The LEDH score lane computes the incremental log-weight sum; we compare the terminal value across arms." This is correct per the production program: the score lane is the likelihood estimand under the standard particle filter identity.

**Cross-check required:** Verify that the production program's score lane actually computes `sum_t log(mean_n w_n^t)` and not a surrogate. The estimand gate `test_production_score_lane_value_is_likelihood_estimand` should confirm this.

**Verdict:** Plan is **acceptable** contingent on estimand gate passing. Add pre-execution checklist item: "[ ] Estimand gate passing for dual-cap route."

---

### 4. Statistical Power
**Question:** Are 3 seeds sufficient to detect a meaningful difference, or is the plan underpowered?

**Finding:** The plan uses 3 seeds and bootstrap 95% CI for statistical ranking. With 3 seeds, the bootstrap resamples from a very small empirical distribution. This is likely **underpowered** for small effect sizes.

**Calculation:** For a paired comparison (RQMC vs MC), 3 seeds gives 3 differences. A bootstrap 95% CI from 3 points has high variance. If the true effect size is <0.5 standard deviations, we likely cannot reject zero difference.

**Alternative:** The plan could use a sign test (requires median difference to be consistent across seeds) or increase to 5-7 seeds per arm. However, 3 seeds × 5 arms × 3 models = 45 runs is already substantial.

**Verdict:** Plan is **acceptable with caveat**. The 3-seed design can detect large effects (>1 SD) but will call small effects "statistically indistinguishable." This is conservative: we avoid promoting RQMC unless the benefit is unambiguous. Document this explicitly: "The 3-seed design favors the MC baseline; RQMC promotion requires a large, unambiguous improvement."

---

### 5. Heuristic Dominance Gate
**Question:** Does the plan satisfy the heuristic dominance gate from `CLAUDE.md`?

**Finding:** The plan compares 4 RQMC methods against MC baseline. MC initialization with `tf.random.normal` is already the simplest practitioner heuristic. The plan correctly avoids comparing only complex methods against each other.

**Salient situations:** The plan tests across 3 models (SV with Student-t obs, seasonal flow, 5D linear-Gaussian) which cover different regimes. However, it does not explicitly test conditional performance in high-volatility vs low-volatility regimes within Austria SV, or boundary vs interior states.

**Required clarification:** The plan should state whether per-regime conditional comparison is in scope. If not, the result note must say "Unconditional average only; regime-specific performance not evaluated."

**Verdict:** Plan is **acceptable** for unconditional comparison. Add to "What Will Not Be Concluded": "Not claiming regime-specific performance; results are unconditional averages across the full time series."

---

### 6. Failure Mode Coverage
**Question:** What happens if the RQMC initialization produces a cloud that looks good at t=0 but causes dual-cap saturation at t=50?

**Finding:** The plan includes dual-cap saturation >10% as a hard veto. This is correct: if the RQMC cloud geometry interacts badly with the dual-cap mechanism, the arm is disqualified.

**Additional diagnostic needed:** Per-time-step dual-cap activation rate (already in explanatory diagnostics) should be traced to see if saturation emerges late in the horizon. If an RQMC arm shows rising saturation trend but stays <10%, this is an explanatory signal that the cloud geometry is degrading.

**Verdict:** Plan is **acceptable**. The hard veto catches catastrophic saturation; the trace diagnostic catches degradation trends. No change required.

---

### 7. Infrastructure Realism
**Question:** Do the required runners actually exist, or is the plan assuming they will be written?

**Finding:** The plan lists 3 runners:
1. `tune_ledh_dual_cap_austria_sv_n1008.py` (and variants for KSC, J0)
2. `run_ledh_rqmc_initialization.py` (unified runner for all arms)
3. `assemble_rqmc_results.py` (result aggregator)

**Check required:** Search for these files or similar runners in the rescued content.

Let me check:
- The rescued test files from GenUT worktree included `test_ledh_pfpf_genut_initialization_designs.py` which imports `run_ledh_pfpf_genut_initialization_design_comparison.py`.
- This is a comparison runner, not a tuning runner.

**Verdict:** Plan is **blocked** until runners exist. The pre-execution checklist correctly includes "[ ] Tuning runner exists for each model" and "[ ] Claim-bearing runner exists" but these are not yet satisfied. Before executing the plan, we must either:
1. Locate existing runners in the rescued content, or
2. Write the 3 required runners.

This is a **continuation veto** for execution, not a flaw in the scientific design. The plan is sound but not yet executable.

---

### 8. Configuration-Status-First Enforcement
**Question:** Does the result assembler satisfy the configuration-status-first reporting rule from `CLAUDE.md`?

**Finding:** The plan states "Configuration-status-first reporting enforced in result assembler" as a checklist item. The rule requires that any artifact showing benchmark numbers must state:
1. Which program each cell ran (production or variant)
2. Per-scope tuning status (artifact path or UNTUNED)
3. What must not be concluded from the table

**Required verification:** The result assembler must emit a table with columns:
- `model`, `arm`, `seed`, `program` (should be `LEDH_PRODUCTION_PROGRAM_V1` for all rows)
- `tuning_artifact` (path to the per-scope tuning artifact)
- `terminal_log_likelihood`, `ess`, `runtime`, ...
- A header banner stating: "All arms use LEDH_PRODUCTION_PROGRAM_V1 with dual-cap enabled. Tuning artifacts listed per row. Do not compare rows with different tuning artifacts."

**Verdict:** Plan is **acceptable** but the result assembler does not exist yet (see finding #7). When writing the assembler, enforce this structure.

---

### 9. Budget and Stopping Rules
**Question:** What is the total compute budget, and what happens if we exhaust it mid-campaign?

**Finding:** The plan states:
- Tuning: 5 runs × 3 models = 15 tuning runs
- Claim-bearing: 3 seeds × 5 arms × 3 models = 45 runs
- Total: 60 runs

Each run is N=1008 particles for T=50 to T=945 time steps on GPU. Estimated runtime per run: ~5-30 minutes depending on model and horizon.

**Rough budget:** 60 runs × 15 minutes average = 15 hours wall time (assuming serial execution on 1 GPU).

**Stopping rule:** The plan includes a continuation veto for "Tuning fails to converge for >1 model after 5 attempts." This is correct. However, the plan does not state what happens if a claim-bearing run crashes due to GPU OOM or serialization corruption.

**Required clarification:** The plan should state: "Infrastructure failures consume campaign budget. After 2 repair attempts on the same run configuration, mark that cell as 'infrastructure failure' and continue to the next run. If >20% of runs hit infrastructure failure, stop and diagnose."

**Verdict:** Plan is **acceptable** but add the infrastructure-failure stopping rule to the continuation veto section.

---

### 10. Post-Mortem: How Could This Mislead Us?
**Question:** If all runs complete successfully and RQMC shows superior terminal log-likelihood, what alternative explanation could invalidate the conclusion?

**Scenario 1:** The RQMC cloud at t=0 happens to land closer to the true latent state by chance, giving it a head-start. This is not a flaw—it's the point of RQMC—but the conclusion should say "RQMC initialization improves particle efficiency" rather than "RQMC improves the filter."

**Scenario 2:** The dual-cap mechanism interacts with RQMC geometry in a way that reduces cap activation, making the route more stable. This is a valid benefit, not a confound. The explanatory diagnostics (dual-cap activation rate) will show if this is happening.

**Scenario 3:** The tuning was inadvertently easier for RQMC arms (more stable optimization landscape), so they ended up with better-tuned parameters. This would be a **tuning confound**. To rule this out, the tuning artifacts must record the tuning objective value and convergence diagnostics. If RQMC arms show materially different tuning objective values, this is evidence of a tuning difference rather than an initialization difference.

**Required mitigation:** Add to result note: "Compare tuning objective values across arms. If RQMC arms have materially lower tuning loss, this is evidence that RQMC interacts with tuning, not only initialization."

**Verdict:** Plan is **acceptable** but add tuning objective comparison to post-hoc analysis.

---

## Summary Verdict

**Scientific design:** Sound. The plan correctly identifies the mechanism under test (initialization only), uses the right baseline (MC), includes hard vetoes, and avoids proxy-metric promotion.

**Execution readiness:** **BLOCKED**. The required runners do not exist yet. Before executing:
1. Write or locate the 3 required runners (tuning, claim-bearing, assembler)
2. Add seed-hashing requirement to runner spec
3. Add estimand gate check to pre-execution checklist
4. Add infrastructure-failure stopping rule to continuation veto
5. Add tuning objective comparison to post-hoc analysis
6. Add regime-specific performance to "What Will Not Be Concluded"

**Statistical power:** Acceptable for detecting large effects (>1 SD). The 3-seed design is conservative: it favors the MC baseline and requires unambiguous improvement for RQMC promotion. Document this explicitly.

**Overall recommendation:** **REVISE AND IMPLEMENT RUNNERS BEFORE EXECUTION**. The plan's scientific logic is correct, but it is not yet executable. The next step is to write the 3 required runners, then re-audit the plan against the actual runner implementations.

---

**Review completed:** 2026-09-01  
**Reviewer signature:** Claude Code (Opus 5, 1M context)
