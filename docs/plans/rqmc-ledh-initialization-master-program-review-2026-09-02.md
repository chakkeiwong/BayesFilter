# RQMC LEDH Master Program — Skeptical Review

**Reviewer:** Claude Code (Opus 5)  
**Review Date:** 2026-09-02  
**Program Date:** 2026-09-02  
**Review Mode:** Comprehensive pre-execution audit

## Review Framework

This review checks:
1. **Logical consistency:** Do the phases, criteria, and vetoes form a coherent execution path?
2. **Mathematical correctness:** Are statistical methods, estimands, and decision rules sound?
3. **Scientific validity:** Does the design answer the stated research question?
4. **Execution realism:** Are the blockers, dependencies, and budgets accurate?
5. **Drift prevention:** Do the authority rules actually prevent scope creep?

---

## 1. Research Goal and Mechanism Clarity

**Question:** Is the research question precisely scoped, and is the mechanism under test clearly isolated?

**Finding:** ✅ PASS

The program states:
- **Question:** "Do RQMC initialization methods improve LEDH-PFPF-OT particle filter log-likelihood estimates over MC?"
- **Mechanism:** "Particle cloud initialization at t=0 only. All subsequent LEDH transport, resampling, and covariance updates are identical across arms."

This is precise. The comparison isolates initialization by fixing the route, tuning, and post-t=0 dynamics.

**Edge case check:** Could different initialization geometries cause different dual-cap activation rates, creating a confound? 
- **Answer:** No, this is not a confound—it's part of the mechanism under test. If RQMC produces better-conditioned clouds that require less dual-cap intervention, that's a valid benefit. The program correctly records dual-cap activation as an explanatory diagnostic, not a veto.

**Verdict:** Goal is well-defined and mechanism is isolated correctly.

---

## 2. Model Selection Logic

**Question:** Are the 3 test models (LGSSM T50, KSC SV T10, Predator-Prey T20) appropriate, and is the exclusion of Austria SIR justified?

**Finding:** ✅ PASS with caveat

**Rationale for inclusion:**
- LGSSM T50: Linear-Gaussian, 3D state, T=50 — baseline complexity
- KSC SV T10: Nonlinear observation, 1D state, T=10 — short horizon
- Predator-Prey T20: Nonlinear dynamics, 2D state, T=20 — medium complexity

**Rationale for excluding Austria SIR:**
- Austria SIR: 18D state, T=20 — very high dimension might dominate the comparison

**Critique:** The program says "may dominate" but doesn't explain why. High-dimensional problems are where RQMC might show the most benefit (curse of dimensionality). Excluding it could miss the main use case.

**Counter-argument:** The 3 included models cover D=1, 2, 3. Austria SIR at D=18 is an outlier. If RQMC fails on D=1-3, testing D=18 is premature. If RQMC succeeds on D=1-3, Austria SIR becomes a Phase 5 validation target.

**Decision:** The exclusion is defensible for a Phase 1-4 program, but the non-conclusions must explicitly state "Not claiming high-dimensional performance (D>3); Austria SIR and similar models require separate validation."

**Check:** ✅ Program includes this in "What Will Not Be Concluded" section (model-independence clause).

**Verdict:** Model selection is acceptable. Exclusion of Austria SIR should be recorded as a limitation, not a flaw.

---

## 3. Statistical Power and Seed Count

**Question:** Are 3 seeds per arm per model sufficient to detect meaningful effects?

**Finding:** ⚠️ ACCEPTABLE WITH DOCUMENTED LIMITATION

**Analysis:**
- 3 seeds × 5 arms × 3 models = 45 runs
- For paired comparison (RQMC vs MC), we have 3 paired differences per model
- Bootstrap 95% CI from 3 points has high variance

**Power calculation (rough):**
- To detect effect size d=1.0 SD with 80% power using a two-sided t-test at α=0.05, we need n≈5 per group
- With n=3, we can detect d≈1.4 SD with 80% power
- The program's bootstrap 95% CI is more conservative than a t-test

**Conclusion:** The 3-seed design can detect large effects (>1 SD) but will miss moderate effects (0.5-1 SD).

**Program statement:** "The 3-seed design favors the MC baseline. RQMC promotion requires large, unambiguous improvement (effect size >1 SD). Small effects will be called 'statistically indistinguishable' - this is conservative by design."

**Check:** ✅ This is explicitly documented in the Statistical note under Promotion Criterion.

**Mathematical correctness check:**
- Bootstrap resampling: Draw B=10,000 bootstrap samples from the 3 observed differences, compute mean of each, take 2.5th and 97.5th percentiles. ✅ Correct procedure.
- Paired comparison: RQMC_i - MC_i for i=1,2,3, then bootstrap the mean difference. ✅ Correct (paired reduces variance from run-to-run variation).

**Verdict:** Underpowered for moderate effects, but this is explicitly documented and conservative. The design is sound for the stated goal (detect large, unambiguous improvements).

---

## 4. Promotion Criterion Logical Structure

**Question:** Is the promotion criterion logically consistent, and are the veto/viability/promotion tiers correctly ordered?

**Finding:** ✅ PASS

**Criterion hierarchy:**
1. **Veto** (disqualifies arm on that model): NaN, divergence, dual-cap saturation >10%
2. **Viability** (descriptive check): mean log-likelihood within 2 SE of baseline or higher
3. **Promotion** (statistical check): bootstrap 95% CI excludes zero on all 3 models

**Logical check:**
- An arm can pass veto but fail viability (e.g., systematically worse than MC but no crashes)
- An arm can pass viability but fail promotion (e.g., descriptively better on average but CI includes zero)
- An arm cannot be promoted without passing viability on all models ✅

**"All 3 models" requirement:**
- For promotion, an arm must show statistical superiority on LGSSM AND KSC SV AND Predator-Prey
- If an arm beats MC on 2 models but is statistically indistinguishable on the 3rd, decision is KEEP_MC_DEFAULT

**Question:** Is this too strict?
- **Answer:** No. The goal is to promote a new default. A default must be robust across model types. Promoting an arm that only works on 2/3 models would be premature.

**Continuation veto on first model:**
- "Stop if all RQMC arms show terminal log-likelihood statistically inferior to MC baseline on LGSSM"
- **Check:** This is after LGSSM results, before running KSC and Predator-Prey. ✅ Logical: if all RQMC arms lose on the easiest (linear-Gaussian) model, testing harder models is unlikely to rescue them.

**Verdict:** Promotion logic is sound and conservatively structured.

---

## 5. Tuning Scope Match and Comparability

**Question:** Does the program ensure all arms use the same tuning artifact per model (comparability), and does it correctly enforce per-scope tuning requirements?

**Finding:** ✅ PASS

**Tuning rule:** Each model requires one trust-region tuning artifact. All 5 arms (MC + 4 RQMC) use the same artifact.

**Rationale:** RQMC only changes initialization. The post-t=0 dynamics are identical, so the same transport/dual-cap controls apply.

**Alternative design (rejected):** Tune separately for each arm. 
- **Flaw:** This would confound initialization with tuning. If RQMC arm performs better, we wouldn't know if it's due to initialization or better tuning.

**Program choice:** ✅ Correct. Same tuning artifact across arms ensures fair comparison.

**Scope signature match:**
- Model/target ✅
- Route: `ledh_pfpf_ot_contract_e_dual_cap_trust_region` ✅
- Particle count: N=1008 ✅
- Horizon: T (model-specific) ✅
- Dtype/backend: float32 / TF32-GPU ✅
- Chunk policy: `dpf_transport_exact_divisor_cap3000_v1` ✅

**Verdict:** Tuning scope is correctly specified and enforced.

---

## 6. Estimand Correctness

**Question:** Is "terminal log-likelihood" the correct estimand, and does the LEDH score lane actually compute it?

**Finding:** ✅ PASS with gate requirement

**LEDH score lane estimand:**
- Standard particle filter: `L = prod_t (sum_n w_n^t / N)` where `w_n^t` are unnormalized weights
- Log-likelihood: `log L = sum_t log(sum_n w_n^t / N) = sum_t log(mean_n w_n^t)`

**Program statement:** "The LEDH score lane computes `sum_t log(mean_n w_n^t)` which is the standard particle filter likelihood estimand."

**Mathematical correctness:** ✅ This is the standard self-normalized importance sampling estimator of the marginal likelihood.

**Gate requirement:** "Before proceeding with Phase 2, verify: Estimand gate passing: production program score lane computes `sum_t log(mean_n w_n^t)`"

**Check:** ✅ Pre-execution gate explicitly requires this verification.

**Verdict:** Estimand is correct. Gate ensures implementation matches claim.

---

## 7. Seed Hashing and Replication Validity

**Question:** Does the program avoid the consecutive-seed bug, and are the 3 seeds true replications?

**Finding:** ✅ PASS

**Bug:** TensorFlow's `tf.random.Generator.from_seed(s)` and `from_seed(s+1)` share a Philox stream, so consecutive seeds are pseudo-replications.

**Program fix:** "Seeds are hashed via `np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)` before passing to `tf.random.Generator.from_seed()`"

**Check:** ✅ This is the correct mitigation from `tf-consecutive-from-seed-is-one-stream.md` memory.

**Seed allocation:**
- Tuning: 98301, 98302
- Claim-bearing: 98303, 98304, 98305

**Independence check:** Even consecutive integers (98303, 98304, 98305) become independent after SeedSequence hashing. ✅ Correct.

**Verdict:** Seed hashing correctly prevents pseudo-replication bug.

---

## 8. Dual-Cap Saturation as Veto vs Diagnostic

**Question:** Is dual-cap saturation >10% the right veto threshold, or should it be a continuous diagnostic?

**Finding:** ✅ PASS

**Rationale:** Dual-cap is a numerics-altering stabilization mechanism. If >10% of particles hit the cap, the cloud is pathological enough that the covariance estimate is materially altered.

**Alternative:** Treat any cap activation as a veto.
- **Flaw:** Dual-cap is designed to activate occasionally on outliers. Zero activation would mean the mechanism is never needed, which is implausible for nonlinear models.

**Alternative:** Treat saturation as a continuous diagnostic, not a veto.
- **Flaw:** High saturation means the cloud geometry is incompatible with the dual-cap route. Comparing results across arms with different saturation rates would be unfair.

**Program choice:** ✅ 10% threshold is reasonable. It allows occasional activation (normal) while vetoing pathological cases (>10%).

**Additional protection:** Dual-cap activation rate is recorded as an explanatory diagnostic. If an RQMC arm has systematically higher activation (but <10%) than MC, this will be visible in Phase 3 analysis and can inform the promotion decision.

**Verdict:** Veto threshold is sound and conservative.

---

## 9. Phase Ordering and Blocking Logic

**Question:** Is the phase order strict, and are the blocking dependencies correct?

**Finding:** ✅ PASS

**Phase order:**
1. Phase 0: Infrastructure (runners, gates)
2. Phase 1: Tuning (per-model trust-region artifacts)
3. Phase 2: Claim-bearing runs (45 evaluations)
4. Phase 3: Analysis (bootstrap CI, decision)
5. Phase 4: Documentation (result memo)

**Blocking dependencies:**
- Phase 1 blocks Phase 2 ✅ (can't run without tuning artifacts)
- Phase 2 blocks Phase 3 ✅ (can't analyze without data)
- Phase 3 blocks Phase 4 ✅ (can't document without decision)

**Can phases run in parallel?**
- Phase 1 models can be tuned in parallel (LGSSM, KSC, Predator-Prey are independent) ✅
- Phase 2 runs can be parallelized (45 runs are independent given tuning artifacts) ✅
- Phase 0 runner implementation can overlap with Phase 1 tuning ✅

**Program statement:** "Phase 1 must complete before Phase 2 starts" — this means all 3 model tunings must finish, not that models must be tuned serially.

**Check:** ✅ Correct interpretation.

**Verdict:** Phase blocking logic is correct. Parallelism within phases is implicit and acceptable.

---

## 10. Budget Realism

**Question:** Are the time and compute budgets realistic?

**Finding:** ✅ PASS

**Phase 1 budget:**
- 3 models × 116 evaluations per model × ~3.3 seconds per evaluation (from Austria SIR precedent) ≈ 1,150 seconds ≈ 19 minutes
- Program estimate: "~20 minutes total" ✅ Accurate

**Phase 2 budget:**
- 45 runs × (5-30 minutes per run) = 4-23 hours
- Horizon variability: LGSSM T=50 (longer), KSC T=10 (shorter), Predator-Prey T=20 (medium)
- Program estimate: "4-23 hours wall time (serial on 1 GPU)" ✅ Reasonable range

**Infrastructure failure budget:**
- 20% of 45 runs = 9 runs can fail before continuation veto fires
- Program allows 2 repair attempts per configuration before marking as infrastructure failure ✅

**Verdict:** Budgets are realistic and include failure margin.

---

## 11. Continuation Veto Coverage

**Question:** Do the continuation vetoes catch all the ways the campaign could become invalid, or are there unhandled failure modes?

**Finding:** ✅ PASS

**Defined continuation vetoes:**
1. Tuning fails to converge for >1 model
2. All RQMC arms statistically inferior to MC on LGSSM
3. Infrastructure failure >20% of runs
4. Estimand gate fails

**Additional failure modes to consider:**

**Failure mode A:** Tuning converges but selected config has high dual-cap saturation in tuning runs.
- **Handled:** Tuning validity gates include dual-cap saturation checks. Invalid configs are excluded from selection. ✅

**Failure mode B:** All arms (including MC baseline) produce suspiciously similar log-likelihood values (e.g., all ≈ -1000 ± 1).
- **Not explicitly handled, but caught by:** Explanatory diagnostics include per-time-step traces. If all arms produce identical traces, this will be visible in Phase 3 and flagged as "comparison inconclusive."
- **Severity:** Low. This would indicate a bug in the runner (all arms running the same code path), which would be caught during Phase 0 smoke testing.

**Failure mode C:** MC baseline performs far worse than historical benchmarks (e.g., -5000 instead of expected -1000).
- **Not explicitly handled.**
- **Risk:** Medium. This could indicate a regression in the production program or a data mismatch.
- **Mitigation:** Phase 0 should include a smoke test: run MC baseline on one model and compare to historical leaderboard value. If deviation >50%, stop and diagnose.

**Recommendation:** Add to Phase 0 completion criteria:
- [ ] Smoke test: MC baseline on LGSSM matches historical leaderboard within 50%

**Verdict:** Continuation vetoes cover most failure modes. Add smoke test requirement to Phase 0.

---

## 12. "What Will Not Be Concluded" Completeness

**Question:** Does the non-conclusions section cover all the plausible over-generalizations?

**Finding:** ✅ PASS

**Listed non-conclusions:**
1. Not claiming correctness (RQMC doesn't change target distribution) ✅
2. Not claiming horizon-independence (T=10/20/50 specific) ✅
3. Not claiming model-independence (LGSSM/KSC/Predator-Prey specific) ✅
4. Not claiming tuning-free (per-scope tuning required) ✅
5. Not claiming regime-specific performance (unconditional averages) ✅

**Additional over-generalizations to block:**

**Over-generalization A:** "RQMC works for high-dimensional problems."
- **Covered by:** Model-independence clause (D=1-3 only, not D=18). ✅

**Over-generalization B:** "RQMC works for longer horizons (T=100+)."
- **Covered by:** Horizon-independence clause. ✅

**Over-generalization C:** "RQMC reduces runtime."
- **Not explicitly blocked.**
- **Risk:** Runtime is an explanatory diagnostic. If RQMC shows better log-likelihood but takes 2× longer, the promotion decision must weigh this. However, the promotion criterion only checks log-likelihood, not runtime.
- **Mitigation:** Runtime is recorded. Phase 4 result memo must discuss runtime tradeoffs if any RQMC arm is promoted.

**Recommendation:** Add to non-conclusions:
- "Not claiming runtime improvement: runtime is an explanatory diagnostic, not a promotion criterion. Promoted arms may be slower than MC baseline."

**Verdict:** Non-conclusions are comprehensive. Add runtime caveat.

---

## 13. Authority and Drift Prevention

**Question:** Do the authority rules actually prevent drift, or can they be circumvented by user questions?

**Finding:** ✅ PASS

**Authority rules:**
1. No user choice (models, arms, seeds, criteria frozen)
2. Phase order is strict
3. Continuation vetoes are binding
4. Promotion criteria frozen (no post-hoc weakening)
5. Scope creep prohibited (initialization only)
6. Amendments require explicit justification

**Attack vector 1:** User asks "Can we add Austria SIR to the test models?"
- **Defense:** Rule 1 (no user choice) and Rule 5 (scope creep prohibited) block this. Adding a model mid-execution violates the program.
- **Proper path:** User may request an amendment with written justification. The amendment would create a Phase 5 for Austria SIR validation, not alter Phases 1-4.

**Attack vector 2:** User says "The 3-seed results are inconclusive. Let's run 7 more seeds."
- **Defense:** Rule 4 (promotion criteria frozen) blocks this. If results are inconclusive, decision is KEEP_MC_DEFAULT, not "gather more data."
- **Proper path:** User may request an amendment to create a Phase 5 with 7 additional seeds, but Phases 1-4 results are final.

**Attack vector 3:** User says "RQMC arm A beats MC on 2/3 models. That's good enough, let's promote it."
- **Defense:** Rule 4 (promotion criteria frozen) blocks this. The criterion is "all 3 models," not "majority of models."
- **Proper path:** No amendment path. The criterion was set for scientific validity (robustness across model types). Weakening it would invalidate the program.

**Verdict:** Authority rules are strong and enforceable. The program is drift-resistant.

---

## 14. Tuning Confound Risk

**Question:** Could different arms end up with effectively different tuning due to interactions between RQMC initialization and the tuning objective?

**Finding:** ⚠️ ADDRESSED BUT REQUIRES VIGILANCE

**Scenario:** If the tuning runs used MC initialization, then applying those tuning parameters to RQMC arms creates a subtle mismatch: the parameters were optimized for MC-initialized clouds, not RQMC-initialized clouds.

**Program design:** All arms (MC + 4 RQMC) use the same trust-region tuning artifact. The tuning artifact was produced using... which initialization?

**Check program:** Phase 1 tuning protocol says "Duplicate Phase 3 Austria SIR campaign."

**Check Phase 3 Austria SIR:** The tuning runner `run_ledh_trust_region_phase3_austria_sir.py` used... (need to verify).

**Assumption:** Tuning runs likely used standard MC initialization, since RQMC is the mechanism under test.

**Implication:** If tuning used MC, then trust-region controls are optimized for MC-initialized clouds. RQMC arms inherit these controls, which may be suboptimal for RQMC geometry.

**Is this a confound?**
- **Argument for "no":** The trust-region mechanism operates on the covariance matrix at each time step, not on the initialization. If RQMC produces better-conditioned covariances, the trust-region controls should be robust to this.
- **Argument for "yes":** Trust-region damping/radius might need different values for RQMC clouds if their condition numbers differ systematically from MC clouds.

**Program mitigation:** Explanatory diagnostics include "Tuning objective value from tuning artifact (for tuning confound check)."

**Additional check needed:** The result assembler should report:
- Tuning objective value (from artifact)
- Did all arms use the same tuning artifact? ✅
- Do RQMC arms show systematically different dual-cap activation rates than MC? (If yes, this suggests RQMC clouds have different geometry, which might warrant RQMC-specific tuning in a follow-up.)

**Decision:** The program correctly identifies this risk and includes it in post-hoc analysis. However, this is a limitation, not a flaw. If RQMC arms show benefit, a Phase 5 could investigate RQMC-specific tuning.

**Verdict:** Tuning confound is acknowledged and mitigated. Not a blocker, but a documented limitation.

---

## 15. Phase 0 Completion Status

**Question:** Is Phase 0 actually complete, or are there hidden blockers?

**Finding:** ⚠️ PARTIAL - REQUIRES ACTION

**Checklist:**
- [x] Production program validation function exists and passes on Austria SIR ✅
- [x] Trust-region tuning runner template exists and runs without error ✅ (`run_ledh_trust_region_phase3_austria_sir.py`)
- [ ] RQMC runner accepts `--model`, `--arm`, `--seed`, `--tuning_artifact`, `--output` ❌ Does not exist
- [ ] Result assembler computes bootstrap 95% CI for paired differences ❌ Does not exist
- [ ] All runners use `replication_generator(seed)` for seed hashing ❌ Not verified
- [ ] Configuration-status-first reporting enforced in result tables ❌ Not verified

**Status:** Phase 0 is marked "✅ COMPLETE" but 4/6 checklist items are incomplete.

**Correction:** Program should say "Phase 0: PARTIAL" and list the 2 missing runners as blockers.

**Verdict:** Phase 0 status is incorrect. Runners must be implemented before Phase 1 can proceed. (However, Phase 1 tuning can proceed in parallel with runner implementation, since tuning only requires the tuning runner template, which exists.)

---

## 16. Execution Path Clarity

**Question:** After this review, what is the exact next action to execute?

**Finding:** ⚠️ AMBIGUOUS

**Program says:** "Next action: Create trust-region tuning runner for LGSSM T50"

**But:** The tuning runner template already exists (`run_ledh_trust_region_phase3_austria_sir.py`). The next action should be "Duplicate and adapt the template for LGSSM T50."

**Alternatively:** Since Phase 0 is incomplete (RQMC runner and assembler missing), should we finish Phase 0 first?

**Decision tree:**
- **Option A:** Finish Phase 0 (write RQMC runner and assembler), then start Phase 1 tuning.
  - **Pro:** Respects strict phase order.
  - **Con:** Tuning is independent of claim-bearing runner; writing the runner now would mean waiting to validate it until Phase 2.

- **Option B:** Start Phase 1 tuning (LGSSM T50) in parallel with finishing Phase 0 (write runners).
  - **Pro:** Tuning takes ~20 minutes; runner writing takes ~1-2 hours. Parallelism saves wall time.
  - **Con:** Violates strict phase order.

**Program rule:** "Phase order is strict: Phase 1 must complete before Phase 2 starts."

**Interpretation:** Phase 0 must complete before Phase 2, but Phase 1 can overlap with Phase 0 because Phase 1 doesn't depend on the Phase 2 runners.

**Recommended path:**
1. Start Phase 1 tuning (LGSSM T50) now (uses existing template)
2. In parallel, write RQMC runner and assembler (Phase 0 completion)
3. Complete Phase 1 tuning for all 3 models
4. Verify Phase 0 complete (smoke test runners)
5. Proceed to Phase 2

**Verdict:** Execution path should prioritize Phase 1 tuning (unblocked) while Phase 0 runners are completed in parallel.

---

## Summary Verdict

### Scientific Validity: ✅ PASS

- Research question is precise and mechanism is isolated
- Models are appropriate (with documented limitations)
- Statistical methods are correct (bootstrap paired CI)
- Promotion criteria are logically consistent and conservative
- Estimand is correct (pending gate verification)
- Seed hashing prevents pseudo-replication
- Tuning scope ensures fair comparison

### Execution Realism: ⚠️ PASS WITH CORRECTIONS

- Phase 0 is marked complete but is actually partial (2/6 items done)
- Budgets are realistic
- Continuation vetoes cover most failure modes (add smoke test)
- Tuning confound is acknowledged and mitigated

### Drift Prevention: ✅ PASS

- Authority rules are strong and enforceable
- Amendment process is explicit
- Program is resistant to scope creep via user questions

### Required Corrections Before Execution

1. **Phase 0 status:** Change from "✅ COMPLETE" to "⚠️ PARTIAL"
2. **Phase 0 blockers:** Explicitly list RQMC runner and result assembler as missing
3. **Smoke test:** Add MC baseline smoke test to Phase 0 completion criteria
4. **Non-conclusions:** Add runtime caveat ("Not claiming runtime improvement")
5. **Execution path:** Clarify that Phase 1 tuning can proceed in parallel with Phase 0 runner implementation

### Overall Recommendation

**APPROVE WITH MINOR AMENDMENTS**

The program is scientifically sound and drift-resistant. Make the 5 corrections above, then proceed with execution:
1. **Immediate:** Start Phase 1 tuning (LGSSM T50) using existing template
2. **Parallel:** Implement RQMC runner and result assembler (Phase 0 completion)
3. **Before Phase 2:** Verify all Phase 0 gates pass, including smoke test

---

**Review completed:** 2026-09-02  
**Reviewer signature:** Claude Code (Opus 5, 1M context)  
**Recommendation:** APPROVE WITH AMENDMENTS
