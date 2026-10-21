# Review verdict: Testing surrogate HMC across BayesFilter models

Reviewer: Claude Opus 5 (1M context)  
Review date: 2026-09-18  
Review target: [surrogate-hmc-all-models-master-program-2026-09-18.md](surrogate-hmc-all-models-master-program-2026-09-18.md)  
Source baseline: `8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`

## Summary

The master program defines a scientifically valid test of surrogate HMC driven by an imperfect full-DCTR force. It correctly separates three distinct objects—model posterior, finite-DCTR posterior, and proposal field—and applies the involution argument appropriately for a deterministic position-only force. The four comparisons (B/R, C/D, D/R, C/R) logically partition the research question and permit independent progress when dependencies fail. Evidence rules are statistically sound, coverage is honestly incomplete, and the program explicitly acknowledges missing implementations and unknown all-model feasibility.

Two material findings warrant revision before execution: the program must explicitly verify that force and endpoint evaluations use the same frozen DCTR realization within each trajectory (omega, c_v identity), and the mixed-precision LG1 scope requires clearer terminology to prevent misclassification. Two clarifications strengthen the design without changing it.

## Findings

### F1: Force and endpoint must use identical frozen DCTR realizations (MEDIUM)

**Location:** Section 3 (lines 99-105, 131-136), section 9 (consumer checks), section 10.1 (lines 656-662)

**Issue:** The program states that U_D uses "frozen random inputs omega, fixed value controls c_v" and allows force and endpoint to use "independently frozen numerical controls" when comparing forces. For the actual C arm inference, both the force g_D and the endpoint U_D must use the **same** frozen particle bank and controls within each trajectory to satisfy the declared MH invariance. The current text doesn't explicitly require or test this identity.

If force and endpoint use different omega or c_v at the same position z, the acceptance ratio compares U_D(z, omega_1) with U_D(z_new, omega_2), which is not detailed balance for either fixed realization. The program correctly freezes the "current endpoint cached" (section 3.1, line 169), but doesn't state that the force must query the same realization.

**Why it matters:** Different realizations at force versus endpoint calls would introduce subtle bias in the stationary distribution, invalidating the sampling claim even when both individually use valid finite approximations.

**Recommendation:** Add to section 9's executable checks (extend item 1 or 2) an explicit requirement:

> The C arm's force and endpoint evaluations at the same position within a trajectory use identical frozen randomness omega and controls c_v, verifiable by recording the DCTR input seed/hash or particle realization identifier per evaluation and asserting equality for paired force/endpoint calls at the same z.

Also clarify in section 3 (after line 136) that when comparing alternative force controls (arm E), the baseline endpoint frozen realization remains unchanged, but within each arm's sampling the force and endpoint must share their frozen inputs.

**Severity:** Medium. The implementation team might reasonably assume realization identity from "frozen endpoint" language, but explicit verification prevents a class of sampler bugs that are difficult to diagnose from posterior summaries alone.

---

### F2: Mixed-precision LG1 scope requires explicit classification (MEDIUM)

**Location:** Section 10.1 (lines 656-662), section 5 coverage table, section 13 result requirements

**Issue:** The LG1 pilot uses a mixed-precision design: "FP64 HMC state, momentum and Kalman endpoint arithmetic" as a "numerical-reference exception," with the DCTR field/value executing in "GPU FP32/TF32/XLA." The force is defined as `cast64(g_D_FP32(cast32(z)))` and the endpoint promotes its FP32 scalar. The program correctly says "do not label it an exact FP64 gradient," but doesn't state what terminology to use instead.

This scope is neither a pure FP64 reference (the DCTR components remain FP32/TF32) nor a pure FP32 sampler (the integrator and momentum are FP64). Without a distinct label, it risks being misclassified as "the FP64 baseline" when presenting results, or incorrectly compared with a later all-FP32 scope.

**Why it matters:** Clear scope identity prevents two errors: (1) overstating LG1's numerical qualification by calling it FP64 when the tested force/endpoint are FP32-derived, and (2) claiming a later FP32 sampler is validated by this pilot when section 10.1 explicitly says "A later all-FP32 sampler is a distinct scope; it is not validated by casting its force inside an FP64 integrator."

**Recommendation:**

1. In section 10.1, after line 662, add explicit terminology:

   > This scope is classified as **FP64-integrator/FP32-DCTR** inference: the HMC kernel, momentum sampling, and exact Kalman reference operate in FP64, while the full-DCTR force and endpoint are computed in GPU FP32/TF32/XLA and promoted at the consumer boundary. It is a numerical-reference scope for the mechanism test, not a production-precision configuration.

2. In the M01 coverage row (section 5), update the reference description to: "Exact Kalman likelihood in FP64; LG1 pilot uses FP64 integrator with FP32 DCTR; production FP32 sampler is a separate scope."

3. In section 13, require that `decision.json` and the result note include a `precision_classification` field with values such as `FP64_INTEGRATOR_FP32_DCTR`, `FP64_FULL`, `FP32_FULL`, to make scope identity machine-readable.

**Severity:** Medium. The program's text is careful to distinguish this from a pure-FP64 gradient, but without a positive classification term, result notes and cross-scope comparisons risk ambiguity.

---

### F3: Noise-scale prior justification when source scales are arbitrary (LOW / CLARIFICATION)

**Location:** Section 5 (lines 286-296, 341-347), section 5.1 (scope card completion)

**Context:** The proposed noise-scale inference rule uses `Normal(0, log(2)^2)` priors for lambda_Q, lambda_R, "justified by testing inference near the existing source scales without dropping normalizers." The text warns "Existing source scales may themselves be arbitrary; report sensitivity as a later distinct scope if it affects the interpretation."

**Observation:** If during P0 closure a fixture's source scales are discovered to be arbitrary (e.g., chosen for numerical convenience in a filtering test, not from data), the stated justification for the Normal(0, log(2)²) prior becomes circular. Proceeding with that prior risks either an unjustified inference or blocking the scope unnecessarily when a model-structure-based prior could be derived.

**Recommendation:** Clarify in section 5.1 (around line 338) that if P0 discovers a source scale is arbitrary:

> The noise-scale prior must be re-justified from identifiability analysis, prior predictive behavior relative to the problem's meaningful parameter range, or data-driven scale evidence, before launching that scope. If no justified prior can be constructed, mark the scope `SCOPE_UNRESOLVED` with reason `ARBITRARY_SOURCE_SCALE_NO_ALTERNATIVE_PRIOR`. A sensitivity analysis over reasonable priors is an acceptable resolution if it fits the stage budget.

This keeps the inference hypothesis honest without creating an automatic block.

**Severity:** Low. The program's existing language ("justified by testing") and the SCOPE_UNRESOLVED mechanism already permit this path; making it explicit reduces ambiguity during P0 execution.

---

### F4: Budget feasibility and measured forecasts (CLARIFICATION, not defect)

**Location:** Section 12 (lines 841-900), section 9.1 (missing commands)

**Context:** Section 12 allocates 63 GPU-hours and 15 CPU-hours across six stages, with 3 hours for P2 LG1. It explicitly warns: "These are allocation ceilings, **not a claim that full all-model validation fits**. Existing timings do not establish such feasibility." The forecasting rule is stated: "1.5 * (compilation + preparation + tuning + reference + planned inference + analysis) from measured components."

**Observation:** No measured timing for any component is provided in the plan. The runner commands don't exist yet (section 9.1: "P0/P1 must turn these specifications into actual CLI options and tested commands"). This is transparent, but leaves open the risk that even LG1 exceeds its 3-hour allocation, making later stages infeasible.

**Clarification:** This is acceptable for a pre-implementation plan document. The program correctly requires measured forecasts before launch (section 12, final paragraph: "launch readiness requires... a measured affordable forecast") and mandates dry-run printing of resolved commands. Section 13 requires honest incomplete-coverage reporting when budget runs out.

**No revision needed**, but note for execution: P0/P1 must produce measured LG1 component timings and verify the 3-hour P2 forecast before any P2 launch. If the forecast exceeds allocation, either the scope must be reduced (fewer arms/looks) or the honest outcome is `INCONCLUSIVE_BUDGET` for the full LG1 task, with that budget shortfall explicitly reported per section 13.

**Severity:** Clarification only. The program's structure already handles this correctly.

---

## Systematic audit against the six review questions

### 1. Research objective and probability targets (sections 1–4)

**Checked:**
- Three densities (π_M, π_D, g_D) are precisely separated in section 3, with π_M using the model likelihood L_M, π_D using the finite DCTR likelihood ell_DCTR with frozen (omega, c_v), and g_D potentially differing from both grad U_M and grad U_D.
- Chart Jacobian log|det J_b(z)| appears consistently in all posterior expressions.
- Involution argument (section 3.1) correctly establishes that a deterministic position-only field defines a valid MH kernel through the symmetric leapfrog step and momentum reversal, without requiring g = grad U.
- Per-chain independent acceptance U_i(z) + K_i(p) is explicit; no chain reduction in the energy.
- Qualification "empirical agreement does not make the two densities equal" (section 3, line 110) is appropriate.
- Section 3, line 109: "Use pi_R = pi_M only for an exact likelihood identity" correctly prevents a numerically qualified approximation from being called the model posterior.

**Finding:** F1 (frozen realization identity) applies here. Subject to that verification, the probability-target separation is correct.

**Verdict on question 1:** Passes with F1 revision. The program correctly tests an imperfect force on declared endpoints, with assumptions explicit.

---

### 2. Four comparisons and dependencies (sections 4 and 10)

**Checked:**
- Four arms (A, B, C, D) and four comparisons (B/R, C/D, D/R, C/R) are defined in section 4, table on lines 225-232 and lines 261-268.
- B versus R isolates force mechanism; C versus D tests sampling of pi_D; D versus R tests finite-likelihood bias; C versus R tests practical inference. Interpretations are logically correct.
- Section 4 explicitly states: "A failed conventional-HMC comparator triggers investigation; it is a global block only if evidence implicates the shared target or kernel." This prevents an optional exact-gradient failure from blocking the imperfect-force method.
- Section 4: "Do not make success of an optional exact-gradient computation a hidden requirement for an imperfect-force method."
- Section 4, lines 254-257: "Schedule B against the model reference first when it is ready. Qualification of D and completion of C must not delay that independent mechanism test." Dependency ordering is appropriate.
- Section 10 repair table (lines 757-768) maps each failure type to a bounded next action without global vetoes where not justified.

**No material defect found.** The four comparisons correctly partition the question, and optional comparator failures don't unnecessarily block required tests.

**Verdict on question 2:** Passes. Dependencies permit independent progress as designed.

---

### 3. Failure semantics and actual implementation (sections 3.2, 6.3, 9)

**Checked:**

*Failure semantics (section 3.2):*
- Distinguishes true zero-density, endpoint failure, force failure, and proposal overflow (lines 172-175).
- Deterministic position-only finite extension (e.g., zero force) is allowed if "reverse replay and the executed status branches satisfy the stated map" (lines 176-180).
- Requires either a reversibility proof or empirical demonstration that an unqualified branch doesn't fire (lines 181-189).
- Restricted posterior with excluded mass: correct total-variation and bounded-function error formulas (lines 193-208), with the important caveat that unbounded moments need explicit excluded-tail bounds. Appropriate rigor.
- Particle-marginal reference requirements (lines 210-218): correct statement that a frozen biased approximation or log-likelihood averaging is not a model-posterior reference.

*Tuning authorities (section 6.3):*
- DCTR controls require a repository-issued artifact with "imperfect proposal-field experiment" classification (lines 418-428).
- HMC tuning uses the public interface with explicit `DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS` (lines 438-463).
- Frozen after tuning; equal tuning access per arm; no cross-dataset reuse.

*Implementation contract (section 9):*
- Six executable checks listed (lines 667-698), including:
  1. Coordinate perturbation exercises real parameter dependence.
  2. Call chain reaches declared routes with correct coordinates, signs, shapes, dtype; row independence and batch permutation tests.
  3. Per-chain acceptance and proposal reversibility with counterexamples.
  4. Typed binding with explicit proposal-field semantics; artifact scope matching.
  5. Warm-up exclusion, fixed kernel, deterministic seeds, two-look accounting.
  6. Failure injection distinguishes rejection types; shared bugs stop dependents.

**Findings:** F1 (realization identity) applies to check 2. Subject to that addition, the checks are appropriate.

**Not checked in this review:** Actual implementation of the consumer, which section 9.1 explicitly says doesn't exist yet. The call-chain rule from the repository policy is correctly invoked (section 9, line 629: verify the consumer reaches the canonical route, not just that a general function exists).

**Verdict on question 3:** Passes with F1 revision. Failure semantics are rigorous; consumer tests are specified; implementation is an acknowledged prerequisite.

---

### 4. Coverage and references (sections 5–6)

**Checked:**

*Coverage (section 5):*
- 14 model families (M01–M14) plus 3 control families (C01–C03) in the table (lines 298-317).
- Each row lists concrete required scopes and identifies missing prerequisites.
- M01 row honestly says: "Full-DCTR inference wiring/tuning is missing. Restricted LG1 cannot complete all linear scopes."
- Noise-scale inference rule is explicitly defined (lines 286-296) with appropriate caveats for degenerate laws and existing proper priors taking precedence.
- Section 5.1 closure protocol (lines 319-353): requires equations or source anchors, all material scope-card fields, and `SCOPE_UNRESOLVED` for missing fields. Appropriate rigor.
- SIR-UKF owner exclusion preserved (M06 row and section 5.1).
- Closure deliverable: `model-register.json` with every discovery item, mapped scope/disposition, and reasons (lines 350-353).

*References (section 6.2):*
- Low-dimensional: exact or refined integration with at least three refinements and independent implementation (lines 393-404). Numerical error allowance ≤ one-tenth of posterior margin.
- High-dimensional: own mixing/MCSE requirements; UKF/SGQF/Laplace cannot certify the model posterior "merely because their output is finite" (lines 405-410).
- Reference validation cost is charged to the scope (lines 410-411).
- Fixed thresholds before opening surrogate results; independent construction/assessment or propagated uncertainty (lines 413-416).

**Finding:** F3 (noise-scale prior justification) provides a clarification for section 5.1 when source scales are arbitrary. F2 (mixed-precision classification) affects the M01 reference row description.

**Not checked:** Actual existence of references, which the coverage table honestly labels as missing or diagnostic-only for many families.

**Verdict on question 4:** Passes with F2/F3 clarifications. Coverage is honestly incomplete; closure protocol is rigorous; reference qualification rules are appropriate.

---

### 5. Evidence and defaults (sections 6–8, 11)

**Checked:**

*Posterior criteria (section 7):*
- Margins: 0.10 for standardized moments, 0.02/0.03 for probabilities (lines 476-484). Stated as "provisional scientific tolerances" to be translated per scope before results are inspected.
- Simultaneous intervals with familywise alpha=0.05 across m functions * K=2 looks (lines 503-509). Formula includes MCSE for both candidate and reference plus numerical bias term. Statistically appropriate for equivalence testing.
- Pass criterion: "**every** interval lies inside [-delta_j, delta_j]" (lines 520-521). Overlap with boundary is inconclusive. Correct.
- Section 7 explicitly warns against interpreting C/R agreement at two separate delta margins as establishing C/R at delta (lines 521-526).

*Sampling schedule (section 7.1):*
- Four chains, 250-transition chunks, 1000 ≤ n ≤ 5000 per chain, two looks at n and 2n reserved in advance (lines 532-541).
- Overdispersed initialization covering modes; not reference-only (lines 532-535).
- R-hat ≤ 1.01, ESS ≥ 400, qualified MCSE required (lines 542-546). Conservative.
- Confirmation requires both chain groups and each DCTR bank to pass (lines 551-557). Appropriate.

*Diagnostic roles (section 7.2):*
- Table on lines 560-571 correctly classifies diagnostics as continuation vetoes, promotion vetoes, repair triggers, or explanatory.
- "Force error, filtering moment error, finite energy-error size, acceptance: Explanatory/repair diagnostics. No fixed accuracy or acceptance threshold substitutes for posterior inference." Appropriate positioning.

*Cheap adversaries (section 8):*
- Three-to-five simple alternatives per model situation (lines 574-588), with rationales.
- Conditional evaluation by regime, tails, modes, initialization (lines 590-607).
- "Do not use heuristic results as a tuning objective" (line 608). Correct.
- Conditional heuristic loss is a promotion veto for practical usefulness, but "preserve any independently established sampling-accuracy result" (line 569).

*Defaults audit (section 11):*
- Table on lines 810-824 lists each choice with provenance, failure mode, earliest check, and promotion status.
- Several items explicitly say "baseline only if justified; otherwise unresolved" (e.g., line 817: damping/protection).
- N=1024 is "initial production-direction hypothesis... N not a tuned universal default" (line 816).
- Margins are "Proposed research criteria; domain review may tighten" (line 820).

**No material defect found.** Evidence rules are statistically sound; defaults are audited with honest status; heuristics correctly positioned as vetoes, not tuning targets.

**Verdict on question 5:** Passes. Statistical inference design is rigorous; defaults are examined; no silent promotions detected.

---

### 6. Executable next step, budget and outcomes (sections 9–13)

**Checked:**

*LG1 task (section 10.1):*
- Exact equations: x_t = phi * x_{t-1} + epsilon_t, epsilon ~ N(0,1); y_t = x_t + eta_t, eta ~ N(0,0.5); T=10; x_0 ~ N(0, 1/(1-phi²)) stationary (lines 776-781).
- Infer only phi ~ Uniform(-0.95, 0.95); data-generating phi=0.8.
- Chart: phi = -0.95 + 1.90*sigmoid(z) with full log Jacobian (lines 781-783).
- Endpoint and score include phi-dependent stationary initial law (lines 783-785).
- Attempt 4, fresh partitions; attempt 5 available for one repair (lines 787-791).
- N=1024, K=1024; controls to be nominated (lines 791-793).
- Arms A, B with their own tuning; physical mean, centered second moment, three CDF thresholds, one-step predictive event Pr(y_{T+1}>0 | y) (lines 794-802).
- Predictive probability integrates model's Kalman law over parameter posterior (lines 800-802). Correct interpretation.
- One dataset/bank/group for pilot; confirmation needs section 7 replication (lines 804-808).

**Finding:** F2 (mixed-precision classification) applies to section 10.1. The task is concrete subject to that clarification.

*Runner and commands (section 9.1):*
- Lines 707-722 explicitly say: "Proposed implementation paths, **not currently existing commands**."
- P0/P1 must implement the CLI, options, scope cards, dry-run, and interface (lines 716-722).
- No tests or numerical jobs were run to write this plan (lines 732-735). Honest.

*Budget (section 12):*
- Recovered spend: 0.43 GPU-hr, 0.90 CPU-hr (lines 841-844).
- Remaining envelope: 63.57 GPU-hr, 15.10 CPU-hr (line 844-845).
- Proposes 63 GPU-hr, 15 CPU-hr additional, within that remainder (lines 846-850).
- Allocation table (lines 852-862) sums to those caps.
- Explicit warning: "These are allocation ceilings, **not a claim that full all-model validation fits**. Existing timings do not establish such feasibility" (lines 863-865).
- Forecasting rule: 1.5 * measured components before launch (lines 864-868).
- LG1: "this revised plan explicitly proposes **two additional inference attempts**" (lines 879-882). Prior three attempts acknowledged as used.
- Launch readiness: complete card, qualified consumer, reference, tuning, command, forecast (lines 893-900). Appropriate gates.

**Finding:** F4 (budget feasibility) is a clarification, not a defect. The program correctly requires measured forecasts before launch and honest incomplete reporting.

*Required results (section 13):*
- Run manifest fields listed (lines 903-909).
- `decision.json` with separate fields for the four comparisons, validity, references, mixing, heuristics, efficiency, readiness, budget, next action (lines 910-919).
- Evidence states include `INCONCLUSIVE_BUDGET`, `SCOPE_UNRESOLVED`, `REFERENCE_BLOCKED`, `NOT_RUN` (lines 914-918). Honest vocabulary.
- Two mandatory tables (lines 922-933): decision table and inference-status table.
- Post-run red-team paragraph required (lines 934-938).
- Final cross-model table reports planned, attempted, passing, failed, unresolved separately; "do not average model verdicts" (lines 939-940).
- First deliverable: LG1 posterior comparison. Final deliverable: full per-model assessment. "Neither a numerical-containment report nor a force accuracy leaderboard completes this goal" (lines 942-947). Correct endpoint.

**Verdict on question 6:** Passes with F2 clarification and F4 note. LG1 task is concrete; budget accounting is honest about limits and missing feasibility evidence; result requirements are rigorous.

---

## Unchecked items (as instructed)

The following were not checked in this bounded review, as they require source inspection or implementation verification beyond the master-program document:

- Actual existence and faithfulness of the canonical DCTR route, Contract E-Chol, UKF covariance lifecycle, analytical recursive scores, and transport chunk policy in the repository code.
- Whether existing fixtures already implement the phi-dependent stationary LG1 prior and chart correctly.
- Actual runner, scope-card schema, tuning interface, and CLI commands (explicitly stated as missing in section 9.1).
- Quality of existing references for M02-M14 families (coverage table labels many as missing or diagnostic-only).
- Actual measured timing for any scope or component (none provided in this plan document).
- Whether the tuning contract `HMC_TUNING_INTERFACE_CAPABILITIES` and public `tune_hmc_kernel` support the required proposal-field semantics.
- Source faithfulness of any proposed Zhao-Cui references (section 9, line 625: requires paper and author-code anchors under repository policy).

These are appropriate implementation and source-verification tasks for P0/P1 execution, not defects in the master program's design.

---

## Alternative explanations and weakest evidence

**Strongest alternative explanation:** The program could "pass" by sampling a restricted posterior (silently excluding DCTR-invalid regions), using the same approximate likelihood as both candidate and reference (C versus D with D=C), or validating only easy LG1 while leaving difficult models unresolved. The program explicitly tests these:

- Restricted support must bound excluded mass and tail moments (section 3.2, lines 193-208).
- D must be "independently implemented at the sampler level" (section 4, line 231) to prevent D=C.
- M01 row says "Restricted LG1 cannot complete all linear scopes" (section 5, line 301).
- Section 13 requires reporting unresolved scopes explicitly (lines 939-947).

**Weakest evidence:** The allocations in section 12 have no supporting measurements. If P0/P1 discover that LG1 compilation, exact Kalman quadrature, A/B tuning, and four-chain sampling require 8 GPU-hours instead of 3, most of the plan becomes infeasible. The program acknowledges this (section 12, lines 863-868) and requires honest `INCONCLUSIVE_BUDGET` reporting (section 13, line 916). This is acceptable for a pre-implementation plan, but makes execution success uncertain.

**Result that would overturn the favorable assessment:** If during implementation the "imperfect proposal-field experiment" classification (section 6.3, lines 422-428) cannot be implemented without either (a) bypassing the canonical route admission rules, or (b) requiring exact-score agreement after all, then the program's premise collapses. The text explicitly prohibits both outcomes, but verification requires inspecting the actual tuning contract and artifact-issuer code.

---

## VERDICT: REVISE

**Reason:** Two material findings (F1, F2) warrant modest additions before execution. The program's scientific design is sound: it correctly tests surrogate HMC with an imperfect force, separates mechanism from model inference, applies valid statistical inference rules, and is honest about missing implementations and unknown feasibility. The findings strengthen rigor without changing the research structure.

**Required revisions:**

1. **F1 (force/endpoint realization identity):** Add explicit verification to section 9 that C arm force and endpoint evaluations at the same position use identical frozen omega and c_v, and clarify in section 3 that force-control comparisons hold baseline endpoint realization fixed.

2. **F2 (mixed-precision scope classification):** Add terminology "FP64-integrator/FP32-DCTR" to section 10.1, update M01 coverage row, and require `precision_classification` field in results.

**Recommended clarifications (F3, F4):** Strengthen section 5.1 handling of arbitrary source scales and note that P0/P1 must produce measured forecasts before P2.

**Advancement:** After these revisions, the master program defines a valid research design. Execution readiness still requires P0/P1 implementation of runners, scope cards, consumer wiring, and measured forecasts, as the program itself explicitly acknowledges. Agreement with this design does not certify unimplemented code, reference quality, posterior accuracy, or all-model feasibility within the stated budget.

---

## Comparison with local audit (optional)

The [local audit](surrogate-hmc-all-models-master-review-2026-09-18.md) was available but not loaded for this independent first reading, per the handoff instruction. A post-verdict comparison may identify overlapping or complementary findings; routine repairs after this material review do not require an indefinite review chain per the handoff guidance.
