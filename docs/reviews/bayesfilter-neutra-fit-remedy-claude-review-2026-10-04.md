# Claude Review: NeuTra Fit Remedy Proposal
**Date:** October 4, 2026  
**Reviewer:** Claude Opus 5.5  
**Review Status:** READ-ONLY BOUNDED REVIEW  
**Scope:** Diagnosis verification, proposal critique, revised remedy recommendation

This review addresses the user's request to fully analyze the current NeuTra fitting failures, thoroughly review the proposed remedy, and propose a stronger or more economical remedy where evidence supports it. I have challenged both the diagnosis and the plan as requested.

## Executive Summary

**The diagnosis is correct but incomplete.** The root-cause analysis accurately identifies the unconditional RKL handover as a proximate cause of coverage regression, and correctly demonstrates that forward-only fits already have excess tail/bridge mass. However, the diagnosis underweights capacity/expressivity questions and treats the October 2 joint-loss failures too lightly.

**The proposed remedy (P1-P6) is methodologically sound but operationally underspecified and possibly overscoped.** Its phased structure, comparator ladder, and evidence contracts are exemplary. Its critical gaps are: (1) no measured pricing for the complete proposal against remaining budget; (2) P1's author-reproduction ladder may be unnecessarily expensive while P3's IAF calibration may be under-resourced; (3) P4's representation escalation threshold is vague; (4) the proposal does not reconcile why joint loss already failed in October 2.

**A revised remedy prioritizing IAF calibration with guarded joint continuation, followed by conditional representation escalation, is more economical and equally discriminating.** The literature strongly supports joint objectives as the coverage-preserving mechanism; dismissing October 2's negative results without first repairing the known forward-fitting defects and controller bugs is premature.

## Causal Diagnosis Assessment

### What the evidence establishes (CHECKED DEFECTS)

1. **Unconditional RKL finish causes coverage regression in the width-64 three-component case.**  
   - Responsibility mass for component 2 drops from 0.21572 to 0.02240 (target 0.249591).
   - Paired change interval [-0.19753, -0.18913] is resolved.
   - The KL decomposition math is correct: weight penalty 0.00319→0.20984, weighted shape 1.69401→0.41727.
   - This is a *correct RKL optimization* sacrificing an already-underrepresented mode to reduce shape cost where q visits.
   - **Classification:** Checked defect in the training controller; supported causal mechanism.

2. **Forward endpoint maps already have excessive tail/bridge mass before RKL.**  
   - Width-64 maps put 8.39% (two-mode) and 21.62% (three-mode) probability outside all component ellipses (r²≤8).
   - True upper bound from chi-square tail: 1.83%.
   - Analytic responsibility-weighted second-moment checks show distorted component shapes (e.g., 14.34 vs. 1.0).
   - Heldout forward KL at width-64 endpoints: 0.1072 (two-mode), 0.4298 (three-mode) — not converged.
   - **Classification:** Checked defect in forward fitting; the RKL regression compounds an existing shape error.

3. **Numerical implementation is locally correct.**  
   - Inverse/forward roundtrip errors ~1e-14.
   - Log-determinant vs. full Jacobian ~1e-14.
   - Finite-difference directional derivatives ~1e-7 to 1e-9 (reasonable FD truncation).
   - Target score vs. analytic mixture score ~1e-15.
   - Attached RKL gradient vs. direct differentiation: exact agreement.
   - Zero clipped updates; zero sampled conditional-cap slopes below 0.1.
   - **Classification:** Local numerical checks passed; no gross implementation error found in density/gradient/inverse mechanics at the checked points.

4. **Training stopped at resource limits, not at stationarity.**  
   - All 48 October 2 fits reached budget caps.
   - October 4 width-64 heldout forward KL improving 0.3011→0.1072 (two-mode), 0.5892→0.4298 (three-mode) across checkpoints.
   - Training vs. heldout gap is descriptive, not a test (training set selected the map).
   - **Classification:** Checked defect in stopping logic; under-optimized fits are a plausible partial cause.

### What remains unresolved (SUPPORTED MECHANISMS vs. UNTESTED HYPOTHESES)

5. **Width/capacity vs. optimization effort are confounded.**  
   - Width 32→64 changed together with 4,096→8,192 updates at the same learning rate.
   - No matched-effort capacity test; no learning-rate calibration.
   - Initialization: one seed per target, author variance scale 0.02 vs. earlier 0.2 (labeled comparator only).
   - Three IAF stages with conditional affine transformations may be insufficient for the required deformation.
   - **Classification:** Supported mechanism (representational limitation) vs. untested hypothesis (under-optimization). The diagnosis does not settle this; P3 is designed to discriminate.

6. **Joint loss already failed in October 2 under bounded protocol.**  
   - 8 cases × 6 arms (parent, continue, forward, reverse, joint, joint-continuation) = 48 completed fits.
   - Zero maps passed fresh posterior confirmation.
   - All stopped at budget limits, not stationarity.
   - The proposal dismisses this as "already tried" but does not explain *why* it failed then.
   - **Classification:** Untested hypothesis that October 2's joint failure was due to inadequate budget/optimization rather than a fundamental joint-objective problem. P2+P3 must address this gap.

7. **Separated Gaussian mixtures demand sharp deformation.**  
   - 1D symmetric mixture p(x)=[N(-a,σ²)+N(a,σ²)]/2: exact quantile map T satisfies p(T(z))T'(z)=φ(z).
   - At z=0 (valley): T'(0) = σ exp(a²/2σ²). For a=5, σ=1: T'(0) ≈ 268,337.
   - Conditional cap on scale does not impose global Lipschitz bound (shift networks remain free).
   - **Classification:** Supported mechanism (mode separation creates steep gradients). However, this is a 1D example; 2D mixtures with positive density everywhere have different geometry. Not a disconnected-support impossibility.

8. **Small average KL does not control score residuals.**  
   - Counterexample: p(x)=φ(x)[1+ε sin(kx)], normalized. D(p||φ)=O(ε²) independent of k.
   - Score residual: ε k cos(kx)/[1+ε sin(kx)] grows with k.
   - Width-64 two-mode final map: median residual 0.351, p95=40.72; 83.6% of squared residual from 6.46% tail mass.
   - **Classification:** Supported mechanism (whitening is stronger than average likelihood fit). Checked via counterexample and empirical residuals.

### Strongest alternative explanation

**Better optimization within the same three-stage IAF could repair forward shape errors, and guarded joint continuation could then preserve coverage while improving geometry.**

The current evidence does not exclude this. It would require:
- Calibrated learning rate, batch size, and continuation schedule (P3).
- Matched capacity experiments at fixed optimization effort.
- Multiple fit seeds to assess initialization sensitivity.
- Explicit forward checkpointing with heldout-based selection before any RKL.
- Joint continuation with coverage regression checks and forward-likelihood anchoring (λ≥0).

This explanation is *not* refuted by October 2's joint failures, because those fits also stopped at budget limits and shared the same under-optimized forward baseline. The proposal must reconcile this.

### Weakest part of the diagnosis

**Attribution of forward fitting failure among optimizer, initialization, capacity, and finite teacher data is unresolved.** The diagnosis correctly identifies that forward fits are inadequate but does not discriminate the causes. Width and updates changed together; learning rate was not searched; one initialization per target; 4,096-sample fixed pool may be overfitting or insufficient. P3 is designed to address this, but the diagnosis presents "unfinished optimization" and "inadequate family" as equally plausible without ranking them.

---

## Findings Against the Proposal

Ordered by scientific impact, with concrete corrections.

### FINDING 1: No priced complete comparison against remaining budget (HIGH IMPACT)

**Location:** Entire proposal, especially P1-P6 and "Numerical choices" section.

**Issue:** The proposal states (line 261): "This remainder has not been shown sufficient for the full proposal." It requests measured pricing before launch but does not provide that pricing in the proposal itself. The terminal r3 state records 3,895.5 GPU-process seconds and 8,660.1 CPU-core seconds remaining. P1 alone (Gabrié reproduction + port + IAF adaptation) could consume substantial budget, yet no breakdown is given.

**Impact:** Without pricing, the proposal cannot be approved as a complete plan. It is a *design* awaiting a priced execution specification.

**Correction:**  
Before final approval, provide:
- Measured per-update cost for width 32, 64, and candidate RealNVP at typical batch sizes (64, 256, 400).
- Estimated setup/compile time per job.
- Total update budget for each P1 arm (Gabrié original, port, IAF substitution) × targets × seeds.
- Total update budget for P3's discriminating ladder (data refresh, LR/batch search, width search, joint branches) × 3 seeds × 2 dev targets.
- Reserve for failures/retries based on observed campaign failure rate (~50% in historical runs).
- Explicit decision: if complete P1-P6 exceeds remainder, which phases are mandatory vs. contingent?

**Revised priority:** Price P3 (IAF calibration) first as the highest-priority discriminating experiment. Price P1 (author reproduction) as a *conditional* investment if P3 suggests a port defect rather than under-optimization.

### FINDING 2: P1's author reproduction may be unnecessarily expensive for the causal question (MEDIUM-HIGH IMPACT)

**Location:** P1, lines 85-115.

**Issue:** P1 proposes reproducing Gabrié's Appendix G.1 Gaussian example (RealNVP, 6 coupling pairs, width 100, 1,500 updates) as the first baseline, followed by a TF port, then IAF substitution. The rationale is to "separate port defects from our later substitutions."

This is methodologically correct but possibly inefficient for the immediate causal question. The current failure is an *IAF* student failure on *exact teacher samples*. The failed maps are already TensorFlow, already checked numerically, and the training procedure is already batch-native. The relevant causal questions are:

1. Is the IAF family expressive enough (capacity)?
2. Is the optimization schedule adequate (learning rate, continuation, stopping)?
3. Does the controller's automatic RKL handover cause coverage regression?

Gabrié's successful *sampler* used RealNVP with concurrent adaptive sampling, not a fixed exact-sample IAF fit. Reproducing that full adaptive controller answers whether *Gabrié's method* works, but it does not directly discriminate our IAF calibration questions.

**Correction:**  
Reorder P1 and P3:

- **Start with P3 (IAF calibration):** Fixed exact-sample forward training with matched capacity/effort experiments, learning-rate search, and explicit checkpoint selection. This directly addresses under-optimization vs. capacity.
- **Escalate to P1 (author reproduction) only if:** P3 forward fits remain inadequate *and* local numerical checks or suspicious training dynamics suggest a port/implementation defect beyond mere under-tuning.

**Justification:** The substantial FAB parity evidence (1,804 FP64 checks passed), local IAF numerical checks, and the fact that forward KL *was improving* at budget exhaustion all suggest under-optimization more strongly than a port defect. Start with the hypothesis most consistent with current evidence.

**Conditional P1 design:** If P1 is executed:
- Gabrié Appendix G.1 separated-mode example is the right source baseline (directly addresses modes).
- TF port is necessary for framework parity.
- IAF substitution on the *same exact-iid control* (not the adaptive sampler) isolates the architecture.
- Budget: Gabrié's 1,500 updates × width 100 × 6 coupling pairs is ~O(10⁴) updates. Multiply by 3 seeds × 2 comparators (original JAX, TF port) × 2 targets ≈ 10⁵ update-equivalents. At ~0.1s/update (width-100 RealNVP guess), this is ~10⁴ seconds ≈ 3 GPU-hours. Affordable if P3 motivates it; expensive as a mandatory first step.

### FINDING 3: October 2 joint-loss failures are dismissed without explanation (HIGH IMPACT)

**Location:** Proposal lines 28-29, 49-54; root-cause memo lines 344-346.

**Issue:** The proposal states: "The October 2 campaign already tried forward, reverse, joint and joint continuation... Do not repeat those configurations while calling the combined objective new or guaranteed to work."

This is factually correct but causally incomplete. October 2 tried joint loss and it failed — but *why* did it fail? The proposal does not explain. Without that explanation, we cannot know whether:
- Joint loss is fundamentally inadequate for these targets (invalidates P2's joint continuation strategy), OR
- October 2's joint loss failed for the same reasons forward-only failed: under-optimization, poor initialization, inadequate budget.

**The math strongly supports joint loss as the coverage-preserving mechanism.** Forward-only D(p||q) places finite cost on missed mass; reverse-only D(q||p) does not. The proposal's own KL decomposition (lines 186-247 of root-cause memo) shows that correct RKL optimization can sacrifice modes when their shape cost is high. Joint objectives with λ>0 retain forward pressure. Noé, Gabrié, and the literature all use this.

**Correction:**  
P2 must include an explicit post-mortem of October 2's joint failures:
- Were forward endpoints already adequate before joint continuation?
- Did joint continuation improve, stagnate, or regress?
- What were the λ values, continuation budgets, and optimizer states?
- What diagnostics (loss trajectory, gradient norms, coverage checks) were recorded?

Then P2/P3 joint branches must differ from October 2 in *identified* ways:
- Start from a better-optimized forward baseline (P3's calibrated forward training).
- Use explicit coverage regression checks (responsibility mass, tail mass) at checkpoints.
- Test multiple λ schedules, not one arbitrary value.
- Use heldout-based checkpoint selection, not automatic handover.

If October 2's joint arms had *improving* forward endpoints that then regressed during joint continuation, that would support the proposal's joint-with-anchoring strategy. If October 2's forward endpoints were already poor (like r3's), then joint continuation cannot repair them — forward calibration (P3) is the prerequisite.

**Action:** Read October 2 result artifacts to determine whether forward baselines were adequate. If not accessible now, P3 must document its forward endpoints before joint branches to avoid repeating undiagnosed failures.

### FINDING 4: P3's IAF calibration may be under-resourced relative to its causal load (MEDIUM IMPACT)

**Location:** P3, lines 145-174.

**Issue:** P3 is the *primary* discriminating experiment for capacity vs. optimization. Its proposed ladder:
1. Fixed-bank vs. fresh-iid forward training (isolates finite-sample overfitting).
2. Learning rate and batch size search (isolates optimization schedule).
3. Width comparison at matched effort (isolates capacity).
4. Forward vs. joint continuation (isolates coverage-preservation mechanism).

This is excellent design. However, the proposal specifies "three independent fit seeds as an exploratory replication minimum" without justification. Three seeds can expose gross seed sensitivity but provide weak statistical power. More critically, P3 is expected to resolve:
- Optimal learning rate (needs ≥3 values × 3 seeds = 9 fits minimum).
- Optimal batch size (≥2 values if not grid-searched).
- Capacity at widths 32, 64, and possibly 96 or 128.
- Forward vs. joint performance.
- Two development targets.

Minimal coverage: 2 data-refresh arms × 3 LR × 2 batch × 3 widths × 2 objectives × 3 seeds × 2 targets = 432 trained maps at multiple checkpoints each.

**Correction:**  
P3 needs a *sequential* calibration strategy, not a full grid:

**Phase 3a (Forward foundation, ~50 maps):**
- Two targets, width 64, fresh-iid data, 3 seeds.
- Learning rate search: {3e-4, 1e-3, 3e-3} at batch 64. Pick best by heldout progress.
- Batch size: {64, 256} at best LR. Pick best by heldout progress/stability.
- Checkpoints: every 1,024 updates to 16,384, with heldout assessment.
- Stopping: plateau detection (loss improvement <1e-3 over 2,048 updates, or 16,384 cap).
- Cost: 3 LR × 3 seeds × 2 targets × ~8 checkpoints = ~144 checkpoint assessments. At ~10s/checkpoint, ~1,400s ≈ 0.4 GPU-hours.

**Phase 3b (Capacity test, ~30 maps):**
- Best LR/batch from 3a, fresh-iid data, 3 seeds, two targets.
- Widths: {32, 64, 96} at matched update budget (e.g., 12,288).
- Compare final heldout KL, tail mass, and residuals.
- Cost: 3 widths × 3 seeds × 2 targets = 18 fits × checkpoints ≈ 1 GPU-hour.

**Phase 3c (Joint guarded continuation, ~20 maps):**
- From best forward endpoints (width 64 or 96), test joint continuation.
- Lambda schedule: {1.0, 0.5, 0.1} (start equal forward/reverse, then favor reverse).
- Coverage check every 256 updates: responsibility mass, tail mass ≤ 2× forward endpoint.
- Fresh Adam optimizer per branch (document reset).
- Stop on coverage regression or 4,096 joint updates.
- Cost: 3 λ × forward-selected seeds × 2 targets ≈ 12-18 fits × checkpoints ≈ 0.5 GPU-hours.

**Total P3 cost estimate:** ~3 GPU-hours + CPU multicore exact-sample generation. Affordable within remainder if P1 is deferred.

### FINDING 5: P4's representation escalation threshold is vague (MEDIUM IMPACT)

**Location:** P4, lines 176-200.

**Issue:** P4 says "If the canonical map remains inadequate after the discriminating optimization checks, compare the successful author RealNVP and the already implemented... conditional DSF/NAF."

What does "inadequate" mean quantitatively? The proposal does not define the threshold. Without a threshold, P4's trigger is ambiguous.

**Correction:**  
Define P4 escalation threshold explicitly:

**Trigger P4 (representation change) if P3's best IAF achieves ALL of:**
- Forward heldout KL < 0.05 (near-entropy match),
- Tail mass outside r²≤8 ellipses < 3% (modest margin over 1.83% bound),
- Max responsibility error < 0.05 (tighter than 0.15 screen),

**AND** still fails fresh posterior confirmation with calibrated HMC.

**Do NOT trigger P4 if:**
- Forward heldout KL > 0.15 (density fit still inadequate; continue P3 optimization).
- Tail/responsibility checks fail (shape error; continue P3 or investigate initialization/capacity within IAF first).

**P4 conditional design if triggered:**
- Test NAF (already implemented) before RealNVP (requires new TF port or JAX subprocess).
- NAF has flexible scalar transformations; RealNVP has coupling freedom but affine coordinates.
- Price: NAF hidden layers, mixture components, and update budget must be calibrated (another ladder).
- Budget: 10-20 fits × checkpoints ≈ 1-2 GPU-hours.

### FINDING 6: P5's native-method assessment is sensible but must preserve causal separation (LOW-MEDIUM IMPACT)

**Location:** P5, lines 201-222.

**Issue:** P5 correctly separates native sampling accuracy from downstream IAF quality. The table of required work is excellent. However, the proposal does not emphasize strongly enough that a failed common student (P3 IAF not passing) **blocks interpretation of P5 teacher results**.

The corrected r2/r3 campaign's exact-teacher banks *passed* their screens but the IAF student failed. This means:
- We cannot conclude FAB/Gabrié/AIS/SMC/AFT/CRAFT failed until the IAF can learn from exact samples.
- Native method experiments are valuable in parallel if budget allows, but they are *diagnostics of the sampler*, not tests of the end-to-end NeuTra pipeline.

**Correction:**  
P5 execution order:
1. **P3 must produce at least one passing IAF on exact samples before P5 native cells execute.** Otherwise P5 results are uninterpretable for the IAF training question.
2. Native method experiments can run in parallel with P3 if budget allows, but label them as "sampler validation, IAF applicability pending."
3. If P3 passes and P5 native methods fail, *then* attribute failure to the teacher, not the student.
4. If P3 passes on exact samples but fails on a native method's output, investigate the teacher's coverage (missing modes, poor bridge sampling) as a separate diagnosis.

### FINDING 7: Geometric penalty (in P4) needs clearer scope and cost-benefit analysis (LOW IMPACT)

**Location:** P4, lines 193-199; also referenced in literature remedies memo.

**Issue:** The proposal mentions "the existing derived geometric objective as a local extension" if density fit is adequate but geometry remains harmful. It asks to "check its total inverse-map parameter derivative, activation/knot regularity, derivative cost and numerical equivalence."

This is sensible caution but lacks specifics. What geometric objective? The monograph derives score residual L2 penalties and discusses rare-region information requirements. Is this proposing:
- A score-residual penalty added to forward KL?
- A stratified sampling scheme with reweighting?
- A separate geometric loss phase?

**Correction:**  
Clarify geometric penalty scope:

**If P3+P4 produce low forward KL but poor whitening:**
- **Diagnostic first:** Measure score residuals on a 10,000-point Gaussian grid. Report median, q95, q99, and mass-weighted residual.
- **Candidate mechanism:** Weighted score-residual penalty: L_geom = E_q[||∇log p_z(z) + z||²]. This is the squared Gaussian score residual.
- **Total derivative required:** Yes, this is second-order in the target (Hessian appears via chain rule) OR can be estimated via finite differences of score. Check cost.
- **Integration:** Add as λ_geom * L_geom to forward objective, calibrate λ_geom by validation.
- **Budget:** 5-10 fits to calibrate λ_geom ≈ 0.5 GPU-hours.
- **Stop condition:** If geometric penalty requires Hessian and target does not provide it, or if finite-difference cost exceeds 2× forward-only cost, document as too expensive and defer.

Do not implement this until P3/P4 density fit succeeds but whitening fails. It is a conditional repair.

### FINDING 8: P6's final generalization and posterior assessment are sound (NO CHANGES NEEDED)

**Location:** P6, lines 224-251.

**Assessment:** P6's design is exemplary:
- Freezes procedure before final targets (2103/2104, 1103/1104 reserved).
- Uses `tune_fixed_transport_hmc_kernel` with identity latent mass (correct interface).
- Requires fresh fixed-map posterior confirmation, not just density diagnostics.
- Separates rare-event precision questions from ordinary HMC diagnostics.

**No correction needed.** P6 is the correct terminal validation gate.

---

## Revised Phase Sequence and Numerical Choices

### Recommended ordering

**Phase 0 (Infrastructure, unchanged):** Reconcile evidence, executable comparisons, verify controls. ~0.1 GPU-hours.

**Phase 3 (IAF Calibration, PROMOTED TO FIRST):**  
Calibrate the canonical IAF with exact samples before testing alternative architectures or native methods.

- **3a. Forward baseline:** LR search {3e-4, 1e-3, 3e-3}, batch {64, 256}, fresh-iid vs. fixed-bank, author initialization 0.02. Checkpoints to 16,384 updates with plateau detection. Widths 32, 64 at matched settings. Three seeds, two dev targets.
  - **Cost:** ~2 GPU-hours.
  - **Success criterion:** Heldout forward KL < 0.15, tail mass < 5%, responsibility error < 0.15 on at least one width/LR/batch combination.
  - **Failure mode:** If all configurations fail these loose bounds, consider P4 (representation) or P1 (port audit).

- **3b. Capacity:** Width ladder {32, 64, 96} at best LR/batch from 3a, matched update budget 12,288. Fresh-iid data, three seeds, two targets.
  - **Cost:** ~1 GPU-hour.
  - **Success criterion:** Width 64 or 96 achieves forward KL < 0.10, tail < 3%, responsibility < 0.10.

- **3c. Joint guarded continuation:** From best forward checkpoints, test joint training with λ={1.0, 0.5, 0.1}. Fresh Adam optimizer per branch. Coverage regression checks (responsibility, tail mass) every 256 updates. Stop on regression or 4,096 joint updates.
  - **Cost:** ~0.5 GPU-hours.
  - **Success criterion:** Joint continuation maintains or improves coverage (responsibility error ≤ forward baseline) while reducing RKL and improving score residuals.
  - **Failure mode:** If joint continuation regresses coverage for all λ, investigate: (a) forward baseline still inadequate, (b) λ schedule wrong, (c) optimizer reset vs. warm-start, (d) need explicit forward-likelihood anchoring term.

**Phase 2 (Controller repair, MERGED INTO PHASE 3):**  
P2's controller improvements are incorporated into Phase 3:
- Explicit forward checkpointing (no automatic RKL handover).
- Heldout-based selection before any continuation.
- Matched optimizer state across branches (document reset vs. warm-start).
- Plateau detection instead of fixed budget.

**Phase 1 (Author reproduction, CONDITIONAL):**  
Execute P1 only if Phase 3 suggests a port defect:
- **Trigger:** Phase 3 training dynamics show suspicious behavior (e.g., gradient noise inconsistent with batch size, loss trajectories suggesting implementation bug, or numerical checks failing during training).
- **Do NOT trigger if:** Forward KL is simply not converging due to insufficient updates, capacity, or poor LR — those motivate extending Phase 3 or escalating to Phase 4.
- **Design:** Gabrié Appendix G.1 two-mode example, exact reproduction, TF port, IAF substitution on exact-iid control. JAX subprocess with isolated dependencies.
- **Cost:** ~3 GPU-hours if triggered.

**Phase 4 (Representation, CONDITIONAL):**  
Execute P4 only if Phase 3's best IAF achieves forward KL < 0.05, tail < 3%, responsibility < 0.05 but still fails posterior confirmation.
- **First candidate:** NAF (already implemented), calibrate mixture components and hidden layers.
- **Second candidate:** RealNVP (requires TF port or JAX subprocess).
- **Third candidate:** Extended-depth IAF (6 stages instead of 3) as explicit departure.
- **Cost:** ~2 GPU-hours per architecture tested.

**Phase 5 (Native methods, IN PARALLEL WITH PHASE 3 IF BUDGET ALLOWS):**  
Can run concurrently with Phase 3 since native methods do not depend on IAF success. However, *interpretation* of native→IAF pipeline requires Phase 3 to pass first.
- **FAB:** Resolve α=2 integrability, AIS accuracy, compare full training run. ~1 GPU-hour.
- **Gabrié:** Adaptive controller reproduction. ~1 GPU-hour.
- **AIS/SMC:** Annealing overlap, mutation, resampling, ancestry. ~0.5 GPU-hours.
- **AFT/CRAFT:** Full multi-stage training. ~1 GPU-hour each.

**Phase 6 (Generalization and posterior confirmation, TERMINAL):**  
Freeze procedure from Phase 3 (or 4 if escalated), apply to final targets 1103/1104, 2103/2104. Fixed-map HMC with calibrated kernel. Fresh posterior assessment.
- **Cost:** ~1 GPU-hour.

### Total estimated cost

- **Mandatory path (P0+P3+P6):** ~4 GPU-hours.
- **Conditional escalations (P1+P4):** +3-5 GPU-hours if triggered.
- **Native methods (P5):** ~4 GPU-hours, can run parallel.
- **Total maximum:** ~13 GPU-hours ≈ 46,800 GPU-process seconds.

**Remaining budget:** 3,895.5 GPU-process seconds ≈ 1.08 GPU-hours.

**CRITICAL GAP:** The mandatory path alone (~4 GPU-hours) exceeds the remaining budget by ~4×. The proposal cannot execute as written without additional allocation.

### Recommendations given budget constraint

**Option A (Minimal diagnostic, fits in budget):**  
Execute Phase 3a only (forward baseline calibration):
- LR search {1e-3, 3e-3}, batch 64, width 64 only, fresh-iid, three seeds, two dev targets.
- Checkpoints to 8,192 updates (half the original r3 allocation).
- Cost: ~1 GPU-hour.
- **Outcome:** Determines whether better LR/stopping improves forward fits within canonical IAF. If successful, request allocation for Phase 3b/3c. If unsuccessful, motivates Phase 4 or P1.

**Option B (Request additional allocation):**  
Request 10 GPU-hours (36,000 GPU-process seconds) for complete Phase 0+3+6, with contingency for one conditional escalation (P1 or P4).
- **Justification:** Root-cause diagnosis and proposal development consumed negligible compute. Failed r3 forward maps are under-calibrated, not exhaustively tested. Literature strongly supports joint objectives. One discriminating IAF calibration study is justified before abandoning the canonical architecture.

**Recommendation:** Pursue Option B. Request 10 additional GPU-hours with the argument that r3's failures were due to under-optimization and unconditional RKL handover, both of which are repairable within the canonical IAF.

---

## Mathematical Derivations and Source Anchors

### Joint objective coverage preservation (LITERATURE SUPPORT)

**Claim:** A positive forward-KL term in the joint objective prevents complete mode loss during reverse-KL optimization.

**Source:** Noé et al. (2019), Methods equation (9); Gabrié et al. Section III.B, Algorithm 1.

**Derivation for disjoint regions:**  
Partition space into regions A_k with p(A_k)=w_k and q(A_k)=a_k. The joint objective is:

J_λ(q) = D_KL(q||p) + λ D_KL(p||q)

For regional weights only (holding conditional shapes fixed):

J_λ = Σ_k [a_k log(a_k/w_k) + λ w_k log(w_k/a_k)]

Taking ∂J_λ/∂a_k = 0 under constraint Σa_k=1 (Lagrange multiplier μ):

log(a_k/w_k) + 1 - λ w_k/a_k + μ = 0

At λ=0 (pure RKL), this gives a_k ∝ w_k√(1+μw_k), which can suppress small w_k.

At λ>0, rearranging: a_k log(a_k/w_k) = λ w_k/a_k - 1 - μ.

For large λ (forward-dominated), a_k → w_k recovers the target weights.

**Key point:** As a_k→0 for fixed w_k>0, the forward term λ w_k log(w_k/a_k) → +∞, creating a divergent penalty. This prevents complete mode collapse in the population objective. Finite optimization, model capacity, and local minima can still cause failures, but the *objective itself* resists mode loss when λ>0.

**Conclusion:** The math supports joint objectives. October 2's failures need investigation, not dismissal.

### Small KL does not control score residuals (COUNTEREXAMPLE)

**Claim:** Small average forward KL does not bound score residuals.

**Source:** Root-cause memo lines 298-324; original construction.

**1D counterexample:**  
Let p(x) = φ(x), q_ε,k(x) = φ(x) [1 + ε sin(kx)] / Z.

By symmetry, E_φ[sin(kx)] = 0, so Z = E_φ[1 + ε sin(kx)] = 1 + ε E_φ[sin(kx)] = 1 + O(ε²).

More precisely, cosh(t) ≤ exp(t²/2) for small t, so Z ≤ E_φ[exp(ε sin(kx))] ≤ exp(ε²/2).

Thus D_KL(φ || q) = log Z ≤ ε²/2, independent of k.

But the score residual is:

∇ log q - ∇ log φ = ∇ log[1 + ε sin(kx)] = ε k cos(kx) / [1 + ε sin(kx)]

For |ε| < 1, the denominator is bounded away from zero, so the residual magnitude is O(εk), which grows with k.

**Conclusion:** Small average KL is necessary but not sufficient for good whitening. The proposal's emphasis on score residual diagnostics (1,000-point probes) is correct.

### Separated mixture valley derivative (STEEP DEFORMATION REQUIRED)

**Claim:** The quantile map for a separated symmetric Gaussian mixture has an exponentially large derivative at the valley.

**Source:** Root-cause memo lines 278-295; standard change-of-variables.

**Setup:** p(x) = [N(-a,σ²) + N(a,σ²)]/2, symmetric 1D mixture. Let T be the increasing quantile map satisfying F_p(T(z)) = Φ(z).

**Derivation:**  
Differentiating F_p(T(z)) = Φ(z):

p(T(z)) T'(z) = φ(z)

At z=0 (median), T(0)=0 by symmetry. The mixture density at the valley is:

p(0) = [φ(a/σ) + φ(-a/σ)]/(2σ) = φ(a/σ)/σ = (1/σ√(2π)) exp(-a²/(2σ²))

Thus T'(0) = φ(0)/p(0) = σ exp(a²/(2σ²)).

For a=5, σ=1: T'(0) = exp(12.5) ≈ 268,337.

**Limitations:** This is a 1D example. 2D mixtures with positive density everywhere have different geometry. The valley-crossing path can follow a curved route with bounded local derivatives if the map has sufficient nonlinearity. However, affine autoregressive stages with conditional caps may struggle to represent such sharp but smooth transitions.

**Conclusion:** Mode separation demands sharp deformation. Capacity questions are legitimate. However, this is not a disconnected-support impossibility — the mixtures have positive density everywhere, and coupling flows (RealNVP) can represent them with sufficient depth.

---

## Revised Proposal Summary

### Core changes from original proposal

1. **Reorder P1 and P3:** Start with IAF calibration (P3), escalate to author reproduction (P1) only if port defect suspected.
2. **Make P1 conditional:** Trigger only on suspicious training dynamics, not as mandatory first step.
3. **Sequence P3 as 3a→3b→3c:** Forward baseline search, capacity ladder, guarded joint continuation. Each phase informs the next.
4. **Define P4 escalation threshold quantitatively:** Forward KL < 0.05, tail < 3%, responsibility < 0.05, but posterior still fails.
5. **Price the complete proposal:** Mandatory path ~4 GPU-hours, exceeds remaining budget 4×. Request 10 additional GPU-hours.
6. **Reconcile October 2 joint failures:** Explain why they failed before recommending joint continuation again. Answer: likely same under-optimization affecting all arms; forward baseline was inadequate.
7. **Preserve P5/P6 as designed:** Native methods in parallel (with causal separation), terminal generalization and posterior assessment.

### Comparison matrix for revised proposal

| Phase | Original proposal | Revised proposal | Rationale for change |
|---|---|---|---|
| P0 | Reconcile evidence, comparisons | Unchanged | Correct as-is |
| P1 | First: reproduce Gabrié/Noé | Conditional: only if P3 suggests port defect | Efficiency: under-optimization more likely than port defect given FAB parity evidence |
| P2 | Repair controller | Merged into P3 | Controller fixes (no automatic RKL, heldout selection) are implementation details of P3 |
| P3 | Calibrate IAF after P1/P2 | First: 3a→3b→3c sequential calibration | Priority: IAF calibration is the primary causal question |
| P4 | Conditional representation | Conditional with quantitative threshold | Clarity: define "inadequate" threshold |
| P5 | Native methods | Parallel with P3, interpretation requires P3 success | Efficiency: can run concurrently; causal separation preserved |
| P6 | Generalization, posterior | Unchanged | Correct as-is |

### Smallest experiment that could overturn the diagnosis

**Hypothesis to overturn:** "Better optimization within three-stage IAF can produce adequate forward fits, and guarded joint continuation can preserve coverage while improving whitening."

**Discriminating experiment:**
- Phase 3a forward baseline at calibrated LR, fresh-iid data, width 64, three seeds, two dev targets, checkpoints to 16,384 updates with plateau detection.
- **If forward heldout KL < 0.10, tail mass < 3%, responsibility error < 0.10 at any checkpoint:** Hypothesis supported; IAF capacity is adequate, proceed to joint continuation (3c).
- **If all configurations fail these bounds even at plateau:** Hypothesis weakened; escalate to P4 (representation) or investigate initialization/batch effects.

**Cost:** ~1 GPU-hour.

**Current budget allows:** No. Requires allocation request.

---

## What Was Inspected and What Remains Unreviewed

### Inspected in this review

**Primary sources:**
- Handoff memo (complete)
- Proposed remedy plan (complete)
- Root-cause analysis memo (complete)
- Literature remedies memo (complete)
- October 2 controlled repair results (summary table, 100 lines)
- Upstream training evidence audit (150 lines)

**Code inspection:**
- `bayesfilter/testing/neutra_scientific_campaign.py` lines 180-362: training blocks, evaluation, freeze/assess, teacher generation, forward-to-RKL handover at line 352.
- `bayesfilter/inference/neutra_weighted_training.py` lines 490-569: forward-KL gradient implementation, minibatch weighting, loss normalization.
- `bayesfilter/inference/neutra_transport_core.py` lines 360-445: RKL gradient, path vs. standard estimator, stage construction, author initialization.
- `bayesfilter/inference/neutra_transport.py`: referenced but not read in detail (configuration surface).

**Documentation:**
- `docs/reference/neutra-implementation.md` (100 lines): canonical architecture definition, source correspondence, author IAF masks/initialization.
- `docs/chapters/ch26f_neutra_controlled_training.tex` (100 lines): exact transport control, Gaussian HMC baseline, leapfrog resonances.

**Evidence artifacts (from memo references):**
- `docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r3/state.json`: remaining budget.
- `docs/plans/artifacts/neutra-scientific-2026-10-04/root-cause-r2/result.json`: numerical checks, KL decomposition, density probes.
- `.localresources/` literature sources (via memo citations): Noé, Gabrié, FAB, NAF/splines, AFT/CRAFT, EMUS.

### NOT inspected (out of bounded scope or time)

**Code:**
- Complete `neutra_transport.py` implementation details (constructors, checkpoint loaders, frozen-map interface).
- `hmc_candidate_set_*` modules (HMC execution, tuning, retained sampling) — these are consumers, not causes of the forward-fit failure.
- FAB equivalence test source in detail (manifest and results read, but not the 1,804 individual checks).
- Native method implementations (Gabrié/AIS/SMC/AFT/CRAFT) — these did not execute in r3, so not directly relevant to current failure.

**Artifacts:**
- Individual r3 forward/final checkpoint files (only summary diagnostics read).
- October 2 detailed loss trajectories, gradient norms, per-arm diagnostics beyond the summary table.
- Historical FAB equivalence artifacts (`artifacts/neutra-fab-equivalence-2026-09-25/`) beyond manifest.

**Documents:**
- Full monograph chapters on NeuTra training, rare regions, HMC diagnostics (only excerpts read).
- Complete September 24 q20 study results (canonical architecture designation context, not directly relevant to mixture failures).
- Upstream source code in `.localresources/` (only cited line ranges and functions, not full modules).

**Justification for scope:**
The handoff memo explicitly requested "a review exactly from this path" and "inspect the precise cited sources when this memo asks for them." It also stated "do not review unrelated repository work." I prioritized:
1. The failure mechanism (r3 execution source, root-cause findings).
2. The proposed remedy phases and their justification.
3. Literature support for joint objectives and representation alternatives.
4. Implementation authority (canonical IAF definition, training interfaces).

I did not exhaustively audit all historical artifacts or the full inference stack because the current failure is localized to forward-fit inadequacy and the RKL handover, both of which are well-documented in the inspected materials.

### What would strengthen this review

**If budget/time allowed:**
- Read October 2 detailed per-arm results to determine whether forward baselines were adequate before joint continuation failed. This would inform whether joint loss itself is problematic or whether October 2's joint arms inherited bad forward endpoints.
- Inspect r3 forward learning histories (loss, gradient norms, per-checkpoint heldout diagnostics) to confirm that optimization was steadily improving vs. plateaued/oscillating.
- Compare r3's 4,096-sample fixed pool with fresh-iid exact-sample generation to test finite-data overfitting hypothesis directly.

**Not essential because:**
- The diagnosis of "under-optimized forward fits + unconditional RKL handover" is well-supported by existing evidence.
- The proposed remedy (P3 calibration) directly tests the remaining hypotheses (capacity, LR, continuation).

---

## VERDICT: REVISE

**Primary reasons:**

1. **Missing pricing against remaining budget.** The proposal cannot execute as written. Mandatory path (~4 GPU-hours) exceeds remainder (~1 GPU-hour) by 4×. Requesting additional allocation is reasonable, but the proposal must state this explicitly and provide measured estimates.

2. **Inefficient phase ordering.** P1 (author reproduction) is expensive (~3 GPU-hours) and may not discriminate the immediate causal question (under-optimization vs. capacity vs. port defect). Start with P3 (IAF calibration), escalate to P1 only if training dynamics suggest implementation error.

3. **October 2 joint failures dismissed without explanation.** The proposal recommends joint continuation (P3c) after joint loss already failed in October 2. It must explain *why* October 2 failed and how P3 differs. Answer: October 2's forward baselines were likely inadequate (same under-optimization); P3 calibrates forward training first.

4. **P4 escalation threshold undefined.** "Inadequate" needs quantitative definition to avoid ambiguous escalation.

**Substantive agreement:**

The diagnosis correctly identifies the proximate causes (unconditional RKL handover, under-optimized forward fits) and the proposed phased structure (P0-P6) is methodologically excellent. The evidence contracts, comparator ladder, and skeptical review are exemplary. The proposal's caution about false architectural impossibility claims and its insistence on exact-sample controls are correct.

**Recommended revisions:**

1. **Reorder phases:** P0 → P3 (IAF calibration) → P1 (conditional author reproduction) → P4 (conditional representation) → P5 (parallel native methods) → P6 (terminal validation).

2. **Price the complete proposal** with measured per-update costs, setup time, failure reserves. If complete execution exceeds remainder, either:
   - Request additional allocation (recommend 10 GPU-hours for P0+P3+P6 + one conditional escalation), OR
   - Execute minimal diagnostic (P3a forward baseline only, ~1 GPU-hour) to determine whether additional allocation is justified.

3. **Explain October 2 joint failures** before recommending joint continuation. Most likely explanation: forward baselines were inadequate (same under-optimization affecting all arms), so joint continuation could not repair them. P3's calibrated forward training addresses this prerequisite.

4. **Define P4 threshold:** Escalate to representation change only if forward KL < 0.05, tail < 3%, responsibility < 0.05, but posterior still fails.

5. **Sequence P3 as 3a→3b→3c:** Forward baseline (LR/batch search), capacity (width ladder), guarded joint continuation. Each phase informs the next.

**What this revision improves over the original proposal:**

- **Efficiency:** Starts with the highest-priority discriminating experiment (IAF calibration) instead of expensive author reproduction that may not discriminate the immediate question.
- **Budget clarity:** Forces explicit pricing and allocation request instead of assuming budget sufficiency.
- **Causal coherence:** Reconciles October 2's joint failures instead of dismissing them, strengthening the rationale for P3c's guarded joint approach.
- **Decisiveness:** Defines quantitative thresholds for escalation (P1, P4) instead of vague "inadequacy."

**What the revision preserves:**

- The phased structure and evidence contracts (excellent).
- The comparator ladder (naive/classical/plain/enhanced).
- The skeptical review discipline and stop-condition framework.
- The terminal generalization and posterior assessment (P6).
- The causal separation between density fit, sampling accuracy, and whitening.

**Why REVISE rather than AGREE:**

The original proposal is a strong *design* but not a complete *execution specification*. It requires measured pricing, explicit allocation request (or scope reduction), and phase reordering to become executable. These are not minor editorial changes — they change the cost, schedule, and risk profile. However, the core scientific approach is sound, and the revisions strengthen rather than replace it.

---

## Additional Technical Notes

### On the r3 training implementation

**Verified correct:**
- Minibatch sampling from log-weighted pool (line 191) with uniform within-batch weights: this correctly estimates the empirical forward cross-entropy ∑ w̄ᵢ log q(xᵢ).
- Inverse-map parameter gradient included in forward training (line 498-507 `_train_step_impl`).
- RKL attached-score gradient (line 386-388) matches the standard estimator formula.
- Adam optimizer state preserved across rung boundaries within one training phase (line 214 offset tracking).

**Unconditional RKL handover (line 352-354):**
- Creates fresh Adam optimizer (resets momentum).
- Runs 256 RKL updates unconditionally, ignoring forward endpoint pass/fail.
- No coverage regression check.
- This is the confirmed defect; P2's correction is to make this an explicit branch with checkpoint selection and coverage guards.

**Width/update confounding (line 330-331, `neutra_scientific_design.py`):**
- Width 32 gets 4,096 updates; width 64 gets 8,192 updates.
- Both labeled "same LR" but different architectural cost per update.
- Rungs at {512, 1024, 2048, 4096, 8192} are reporting checkpoints, not continuation decisions.
- P3's correction is to test widths at matched update budgets (e.g., 12,288 for all widths).

### On the canonical IAF architecture

**Source correspondence verified:**
- `hoffman_block_masks_v1` matches author exclusive-first, inclusive-later mask pattern (TFP `masked_autoregressive_default_template`).
- ELU activation, coordinate reversal between stages, three-stage composition all match author code.
- Conditional scale `b + c*tanh(h/c)` with free bias outside cap matches author `neutra-utils.py:739` option.
- Author initialization variance scale 0.02 (fan-in, truncated-normal corrected) matches `hoffman_variance_scaling`.

**What differs from generic "NeuTra" or "IAF":**
- Generic constructor `NeuTraTransportConfig(kind="iaf")` defaults to **legacy degree masks**, not author masks. Must use `hoffman_author_iaf` profile or specify explicitly.
- Author paper used dimension-wide hidden layers; q20 study used width (16,16) for dimension 4 — this is a local calibration choice, not source-mandated.
- Path gradients (Vaitl/Roeder mechanism) are optional; standard estimator is the r3 choice.

**Implication:** The "canonical IAF" architectural designation is well-defined and source-grounded. Training recipe calibration (LR, batch, updates, stopping) remains an open question, which P3 addresses.

### On forward KL geometry

**Why forward KL alone is insufficient for whitening:**

The forward KL is D_KL(p||q) = ∫ p(x) log[p(x)/q(x)] dx = E_p[log p - log q].

At a global optimum where q=p, the score residual ∇log p_z(z) + z = 0 everywhere (Gaussian latent). However, at a local optimum with small but nonzero D_KL(p||q):

1. **Coverage guarantee:** Forward KL diverges if q misses any region where p>0 (mode dropping is penalized).
2. **No score guarantee:** The counterexample p=φ, q=φ[1+ε sin(kx)]/Z shows D_KL(φ||q)=O(ε²) but score error O(εk).

**Why this matters for HMC:**

Hamiltonian dynamics integrate ẋ = ∇_p H, ṗ = -∇_x H with H(x,p) = U(x) + K(p). In latent coordinates, U(z) = -log p_z(z) and K(v) = v²/2 (unit mass).

The leapfrog force is -∇U(z) = ∇log p_z(z). If p_z is standard Gaussian, this is exactly -z (linear restoring force). Score residuals create nonlinear force errors, which accumulate over L leapfrog steps. Large residuals in low-probability regions still affect trajectories that pass through them during momentum-driven excursions.

**Conclusion:** Forward KL is necessary (covers modes) but not sufficient (does not guarantee whitening). The 1,000-point Gaussian probe checking score residuals is the right diagnostic.

### On joint objectives and mode preservation

**Why λ>0 prevents mode collapse (population level):**

Partition space into disjoint regions A_k. For simplicity, hold conditional shapes fixed and vary only regional probabilities a_k=q(A_k).

J_λ = D_KL(q||p) + λ D_KL(p||q)  
    = ∑_k [a_k log(a_k/w_k) + λ w_k log(w_k/a_k)]

As a_k → 0 for fixed w_k > 0:
- RKL term a_k log(a_k/w_k) → 0 (no penalty for losing the mode).
- Forward term λ w_k log(w_k/a_k) → +∞ (divergent penalty).

This creates a barrier against complete mode loss. However:
- Finite model capacity may prevent reaching low J_λ anywhere.
- Local optimization may get stuck in bad local minima.
- Insufficient forward weight λ may allow partial mode suppression.

**Why October 2 joint training could still fail:**

Even with λ>0, joint training can fail if:
1. Forward baseline is already poor (modes present but distorted).
2. Optimization budget insufficient to improve from that poor baseline.
3. λ mis-calibrated (too small → mode suppression still possible; too large → RKL term ineffective).
4. Optimizer reset vs. warm-start mishandled.

P3c's guarded joint continuation addresses these by: (a) starting from a better-calibrated forward baseline, (b) testing multiple λ values, (c) explicit coverage checks, (d) documenting optimizer reset.

---

## Closing Assessment

This proposal represents careful, disciplined scientific thinking. Its phased structure, comparator design, evidence contracts, and skeptical self-review are exemplary. The diagnosis of the r3 failures is accurate: unconditional RKL handover caused coverage regression on an already under-optimized forward fit.

The main gaps are operational, not conceptual:
1. Missing pricing creates execution uncertainty.
2. Phase ordering prioritizes completeness over efficiency.
3. October 2 joint failures need reconciliation, not dismissal.
4. Escalation thresholds need quantitative definition.

These are all repairable. The revised proposal addresses them while preserving the core scientific approach. With measured pricing, phase reordering (P3 first), and explicit reconciliation of October 2, this becomes an executable, resource-bounded research plan.

The proposal's caution about not overinterpreting failures, not claiming impossibility without matched capacity tests, and separating density/sampling/whitening is exactly right. Its insistence on exact-sample controls and independent confirmation is correct. The literature support for joint objectives is strong, and the proposed guarded continuation approach is sound.

**Final recommendation:** Revise per the detailed findings above, price the complete proposal (or scope a minimal diagnostic), request allocation if needed, then execute. The scientific approach is sound; it needs operational refinement to become executable within known constraints.

---

**VERDICT: REVISE**

The proposal's scientific approach is sound and its diagnostic rigor is exemplary. Revisions are needed for: (1) explicit pricing and allocation request, (2) phase reordering for efficiency, (3) reconciliation of October 2 joint failures, (4) quantitative escalation thresholds. These changes strengthen an already strong design and make it executable within resource constraints.

