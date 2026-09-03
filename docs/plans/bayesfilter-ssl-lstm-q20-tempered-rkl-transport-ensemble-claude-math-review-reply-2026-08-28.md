# Stage 1: Mathematical Review Response

**Date:** 2026-08-28  
**Reviewer:** Claude Code (Opus 5)  
**Target:** `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`

## Review scope and method

This is a read-only bounded review of the mathematical propositions, proofs, and claim boundaries in the specified LaTeX document. I inspected:

- All propositions, theorems, corollaries, lemmas, and counterexamples (lines 94–583)
- The mixture reverse-KL identity and its gradient interpretation (Proposition 3.2, lines 230–256)
- The separated-region optimum for alpha (Proposition 5.1, lines 266–292)
- The beta=0 diversity note (lines 607–614)
- The proper bridge construction (Section 7, lines 358–404)
- Fixed versus state-dependent chart mixtures (Propositions 8.2, 8.3; Counterexample 8.1, lines 455–481)
- Replica-exchange detailed balance (Propositions 9.1, 9.2; Theorem 9.1, lines 483–534)
- The exact cold marginal claim (Theorem 9.1, lines 520–534)
- The distinction between invariance and discovery (Propositions 9.2, 6.2, lines 536–554, 330–356)

## Findings

### CLARITY ONLY

**Finding C1 (line 103):** The expression "$\log Z$" appears as an additive constant in equation (2). While mathematically correct, the proof would be clearer if it explicitly stated that $Z$ is the normalizing constant from line 82 and that this term is constant with respect to the transport parameters, hence drops out of gradients. The current proof is correct but could be more pedagogically explicit about why this term is "trainable term" versus constant.

**Status:** The mathematics is correct; this is a presentation suggestion only.

---

**Finding C2 (lines 360–374):** The proper tempering section defines the geometric bridge (eq. 16) and then specializes to likelihood tempering (eq. 17). The text correctly notes that "an unnormalized uniform endpoint on $\mathbb{R}^d$ is not an admissible $\beta=0$ reference" (line 373). However, the document does not explicitly state the integrability requirement: $g_0$ must be a normalized probability density with $\int g_0(\theta)\,d\theta = 1$ and finite $Z_\beta$ for all $\beta \in [0,1]$ requires additional moment/tail conditions when $\widetilde\pi$ has heavy tails or $g_0$ and $\widetilde\pi$ have mismatched support. 

**Status:** The stated mathematics is correct under the assumption "assume $0 < Z_\beta < \infty$ throughout" (line 366). A fully formal treatment would make the sufficient conditions for finite $Z_\beta$ explicit, but the current statement is adequate for the working-note genre and the assumption is clearly flagged.

---

**Finding C3 (lines 471–477, Counterexample 8.1):** The state-dependent chart-selection counterexample is correct and important. The example uses a two-point discrete state space for clarity. A reader might wonder whether the failure mode extends to continuous state spaces with smooth chart-selection rules. The mathematics as stated is correct; a remark that smooth state-dependent weighting also breaks detailed balance without additional correction (e.g., Gibbs, augmented state, or separate Metropolis) would strengthen the pedagogical message, but this is not a mathematical defect.

**Status:** Correct as stated; additional elaboration would be helpful but not required for correctness.

### Assessment of specific scrutiny points

**Mixture reverse-KL identity (Prop 3.2, lines 230–249):**  
The proof is correct. The split $q_\alpha = \sum_i \alpha_i q_i$, change of variables $\theta = T_i(z)$ per component, and the identity $q_i\{T_i(z)\}|\det DT_i(z)| = \rho(z)$ from the pushforward are all valid. The claim that "no target-distributed particle is required" is supported: the expectation is over the base Gaussian $\rho$, and each $T_i(Z_0)$ is evaluated at the unnormalized target $\widetilde\pi_\beta$. **CORRECT.**

**Separated-region optimum for alpha (Prop 5.1, lines 266–292):**  
The decomposition (eq. 18) correctly uses the partition assumption and the definitions $q_\alpha|_{A_i} = \alpha_i q_i$ and $\pi|_{A_i} = p_i \pi_i$. The Lagrange-multiplier first-order condition and normalization yield equation (19). The exponential bias term $e^{-\delta_i}$ arises from $\delta_i = \mathrm{KL}(q_i \| \pi_i)$, so $\alpha_i^* \propto p_i e^{-\delta_i}$ correctly shows that the variational weights confound regional mass and approximation error unless all $\delta_i = 0$ (Corollary 5.1). **CORRECT.**

**Beta=0 diversity (lines 607–614):**  
The algorithm description states: "Full optimization at $\beta=0$ makes every component distribution target the same reference law and can erase distributional diversity" (lines 607–609). This is correct under the setup: at $\beta=0$, $\widetilde\pi_\beta = g_0$ (from eq. 16), so minimizing $\mathrm{KL}(q_i \| g_0)$ independently for each $i$ drives every $q_i \to g_0$. The prescription to use "predeclared fresh restarts or branching at one or more positive temperatures" (line 612) is a design response, not a mathematical claim, and is appropriately scoped. **CORRECT and appropriately qualified.**

**Proper bridge (Prop 7.1, lines 376–398):**  
Equation (16) defines the geometric bridge, and equation (20) gives the energy interpolation $U_\beta = (1-\beta)U_0 + \beta U_1$ up to a constant. The proof takes negative logarithms of the geometric-mean form and the constant cancels in differences (eq. 21). The statement "An unnormalized uniform endpoint on $\mathbb{R}^d$ is not an admissible $\beta=0$ reference" (lines 373–374) is correct: $g_0$ must integrate to a finite positive value. The bridge is proper in the sense that $\pi_\beta$ is a well-defined probability measure for all $\beta \in [0,1]$ under the stated integrability assumption. **CORRECT.**

**Fixed versus state-dependent chart mixtures (Prop 8.2, 8.3; Counterexample 8.1, lines 455–481):**  
Proposition 8.2 (lines 455–463): The proof correctly invokes linearity of the preservation equation $\pi_\beta K_\beta = \pi_\beta$ and the fact that each $K_{\beta i}$ preserves $\pi_\beta$ (Prop 8.1). The requirement that $\gamma_i$ are "fixed and state-independent" is explicit (line 457). **CORRECT.**

Counterexample 8.1 (lines 471–477): The discrete example is valid and demonstrates that a state-dependent mixture can fail to preserve the target. The text correctly notes this is "excluded from the initial proposal" (line 480) and requires separate justification. **CORRECT.**

**Replica-exchange detailed balance (Prop 9.1, Theorem 9.1, lines 483–534):**  
Proposition 9.1 (lines 493–509): The swap acceptance ratio (eq. 24) is the standard adjacent-replica-exchange Metropolis ratio. The proof correctly observes that the product form $\Pi = \prod_\ell \pi_{\beta_\ell}(\theta_\ell)$ causes all unaffected factors and all normalizing constants to cancel, leaving exactly the ratio in (24). The symmetric proposal gives detailed balance. **CORRECT.**

Theorem 9.1 (lines 520–534): The proof invokes: (i) within-temperature kernels preserve the product target because each factor kernel preserves its marginal (Prop 8.2); (ii) swaps preserve the product target (Prop 9.1); (iii) compositions and fixed mixtures of kernels with a common invariant preserve that invariant. All three steps are standard and correct. The conclusion that the $\beta_L = 1$ marginal is $\pi$ follows from the product structure. **CORRECT.**

**Exact cold marginal versus discovery (Prop 9.2, lines 536–554):**  
The proposition states that invariance does not imply mixing or discovery. The identity-kernel example and the closed-region argument (lines 544–548) are both correct. The text explicitly separates "exact finite values, local acceptance, or adjacent swaps" from "global mixing" evidence (lines 550–554), and correctly identifies round trips, hot-level forgetting, cold diagnostics, and initialization forgetting as "separate evidence requirements" (lines 551–554). **CORRECT and strongly stated.**

**Finite-query non-identification (Prop 6.2, lines 330–356):**  
The bump construction (eq. 15) is valid: a smooth bump $h$ supported in a ball $B$ disjoint from the finite query set $S$ can be added with arbitrary coefficient $c > 0$, and all derivatives vanish on $S$. This proves that no finite set of function and derivative evaluations can certify exhaustive discovery. The statement "No finite generic computation can certify exhaustive global discovery without additional structure" (lines 353–355) is correct and appropriately cautious. **CORRECT.**

---

## Assessment of claim boundaries and qualifications

**Lines 56–77 (Section 1):** The status and correction are clearly stated. The document explicitly retracts the *scientific recommendation* (line 62) for adaptive replay as the primary repair, while not retracting the conditional-convergence mathematics. The replacement is described as a two-layer system: fresh-Gaussian reverse-KL training (lines 65–67) and frozen-chart exact HMC (lines 68–69). The claim "The final cold marginal is exact under the assumptions proved below" (lines 70–71) is supported by Theorem 9.1. The limitation "its finite run need not have mixed" (lines 70–72) is explicit. **APPROPRIATE.**

**Lines 74–78:** The note explicitly disclaims favorable scaling: "It contains no claim that the proposed method avoids the curse of dimensionality or that a finite number of charts discovers every important mode" (lines 76–78). This is consistent with Propositions 6.1, 6.2, and 9.2. **APPROPRIATE.**

**Lines 300–303 (following Corollary 5.1):** The text states: "The $\alpha_i$ are mixture-density parameters. They are not posterior mode-mass authority unless the regional approximation conditions are established" (lines 301–303). This correctly distinguishes the variational optimum from a posterior inference and is consistent with the bias term $e^{-\delta_i}$ in equation (19). **APPROPRIATE.**

**Lines 405–409 (Prop 7.2):** The statement "Tempering is therefore an optimization and discovery mechanism during training, not evidence that the optimizer found distinct modes" (lines 418–419) correctly interprets Proposition 7.2: the cold objective $\mathrm{KL}(q_\alpha \| \pi)$ is the same regardless of the warm path. **APPROPRIATE.**

**Lines 536–554 (Prop 9.2):** The limitations on invariance versus discovery, mixing, and finite-run diagnostics are explicit and correct. **APPROPRIATE.**

**Lines 670–676 (Section 11):** The "nonconclusions" paragraph explicitly lists what the theorems do *not* establish: exhaustive discovery, geometric ergodicity, favorable scaling, bounded component/temperature counts, statistical superiority, or correctness of the finite SSL–LSTM implementation. This is a strong and appropriate boundary. **APPROPRIATE.**

---

## Verdict Summary

I found **no mathematical errors** in the propositions, proofs, or counterexamples. The three clarity-only findings (C1, C2, C3) are presentation suggestions that would improve pedagogical completeness but do not affect correctness. 

The claim boundaries are stated strongly and consistently throughout:
- Invariance is distinguished from discovery (Prop 9.2, lines 536–554).
- Variational weights are distinguished from posterior mode masses (Corollary 5.1, lines 294–299; discussion lines 300–303).
- The proper bridge is required and improper endpoints are rejected (lines 373–374).
- Finite-query non-identification is proved (Prop 6.2).
- Exactness of the cold marginal under the stated kernel assumptions is proved (Theorem 9.1), but finite mixing is explicitly not claimed (lines 70–72, 536–554).
- Beta=0 lineage collapse is correctly identified and a design response is prescribed (lines 607–614).
- No favorable scaling or exhaustive discovery claim is made (lines 76–78, 670–676).

The mathematics is **sound under its stated assumptions**, and the limitations are **stated strongly enough** for a research working note.

---

## VERDICT: AGREE

The mathematical propositions and proofs are correct under their stated assumptions. The claim boundaries are appropriately strong. The Stage 2 implementation-plan review may proceed.

---

## Optional strengthening suggestions (not required for AGREE verdict)

1. **C1 resolution:** After equation (2), add one sentence: "The term $\log Z$ is constant with respect to the transport parameters and vanishes under differentiation."

2. **C2 resolution:** After line 366, optionally add: "Sufficient conditions for finite $Z_\beta$ include: $g_0$ and $\widetilde\pi$ have common support, or $g_0$ dominates $\widetilde\pi$ on its support, or both have exponentially decaying tails."

3. **C3 resolution:** After Counterexample 8.1 (line 477), optionally add a remark: "Smooth state-dependent chart weighting $\gamma_i(\theta)$ also breaks detailed balance without an additional Metropolis correction or augmented-state construction."

These are **optional enhancements** only. The current document is mathematically correct and ready for Stage 2.
