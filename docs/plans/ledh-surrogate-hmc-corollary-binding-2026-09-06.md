# LEDH Surrogate-Force HMC — Corollary 5.2 Verification and Binding

**Date:** 2026-09-06  
**Gap G1 resolution:** Read the theorem, verify premises, map onto construction.

---

## 1. The theorem statement (variance note lines 953–976, 980–995)

**Corollary 5.2 (Surrogate-force HMC targets the exact executed scalar):**

> Consider the algorithm: (i) refresh $p \sim \mathcal{N}(0, M)$; (ii) propose
> $S(\theta, p)$ with $S = \mathcal{F} \circ \Psi^L$ built from *any*
> deterministic momentum-independent force $F$; (iii) accept with probability
> $\min\{1, e^{H(\theta,p) - H(S(\theta,p))}\}$ where $H$ uses the exact
> executed potential $U = -\widehat{L}^N$.
>
> Every step leaves $\pi(\theta, p) \propto e^{-H(\theta,p)}$ invariant, hence
> the $\theta$-marginal $\propto e^{-U(\theta)}$ is preserved, **for any quality
> of $F$**. The choice of $F$ affects only the acceptance probability and mixing.

**Proof sketch:** Step (i) is a Gibbs update from the exact conditional under
$\pi$. Steps (ii)–(iii) form an MH kernel with an involutive measure-preserving
proposal supplied by the two leapfrog propositions (volume-preservation and
reversibility). A composition of $\pi$-invariant kernels is $\pi$-invariant.
Since $F$ enters only through the proposal path, not through $\pi$ or $H$, its
quality cannot affect invariance.

---

## 2. The premises

**From Proposition 1 (leapfrog volume-preservation, line 874):**

> Each substep of the kick-drift-kick leapfrog, and hence $\Psi$ and any iterate
> $\Psi^L$, preserves Lebesgue measure on $\mathbb{R}^P \times \mathbb{R}^P$,
> **for every measurable $F$**.

**From Proposition 2 (leapfrog reversibility, line 891):**

> For every $F$ and every $L \geq 1$, $S = \mathcal{F} \circ \Psi^L$ is an
> involution: $(\mathcal{F} \circ \Psi^L) \circ (\mathcal{F} \circ \Psi^L) =
> \text{id}$.

**From Remark 5.3 (requirements, lines 980–995):**

> $F$ must be a **deterministic function of $\theta$ alone**: all Monte Carlo
> seeds inside $F$ must be frozen, and $F$ may not depend on momentum or on the
> trajectory's history.
>
> Under those conditions Corollary 5.2 licenses using, as leapfrog force:
> (i) a **damped or clipped variant of the recursive score** (e.g. computed with
> larger $\lambda, \delta$ than the value program), (ii) the Fisher-identity
> estimator with its own frozen seeds, or (iii) a deterministic teacher or
> sigma-point score — while the acceptance step keeps the **exact same-scalar
> energy**, so correctness never depends on score variance again.

---

## 3. What $\lambda$ and $\delta$ are (Definition 1, line 87)

From the scope definition:

> Contract-E ridge $\lambda > 0$, correction strength $\rho \geq 0$, damping
> $\delta > 0$, correction counts, pairwise controls.

The Cholesky reset uses ridged covariances:
$$L_E L_E^\top = \widetilde{P} + \lambda I, \quad L_w L_w^\top = P_w + \lambda I$$

So **$\lambda$ is the reset ridge parameter** (ridges the covariances before
Cholesky factorization). $\delta$ is "damping" but its specific role is not
defined in the extracted section — it appears in the program signature but the
note does not give its mechanics.

---

## 4. What "exact executed potential" means (Definition 1, line 108)

$$\widehat{L}^N(\theta) = \sum_{t=1}^T \widehat{\ell}_t(\theta)$$

where $\widehat{\ell}_t = \log(\frac{1}{N} \sum_{i=1}^N e^{\ell_{t,i}})$ is the
log-mean-exp marginal likelihood increment at time $t$. This is the finite-$N$,
finite-seed, fixed-branch value — **all noise frozen**.

The note explicitly states (line 132):

> $\widehat{L}^N$ is a biased approximation of $\log p_\theta(y_{1:T})$ at
> finite $N$.

---

## 5. The fixed-branch assumption (Assumption 2, lines 113–120)

> Throughout, $\theta$ lies in an open set on which: every Cholesky argument in
> the program is strictly positive definite; every Sinkhorn row mass exceeds its
> floor; the cost-scale maximum branch is locally constant; and **no validity
> gate changes state**. On this set every stage map is continuously
> differentiable in all of its arguments.

**Proposition 6 (parameter-dependent reset triggers break continuity, line
1166):**

> If the program applies the reset at time $t$ **if and only if**
> $\text{ESS}_t(\theta) < \tau N$ (a $\theta$-dependent condition), and some
> $\theta^*$ satisfies $\text{ESS}_{t^*}(\theta^*) = \tau N$ with
> $\nabla_\theta \text{ESS}_{t^*}(\theta^*) \neq 0$, then
> $\theta \mapsto \widehat{L}^N(\theta)$ is **discontinuous** at $\theta^*$, and
> no same-scalar derivative claim holds across the crossing.

The note's conclusion: **$\theta$-independent reset schedules are the safe
policy** for the repository's fixed-branch score contract.

---

## 6. Binding to the surrogate-force HMC construction

### 6.1. Does Corollary 5.2 apply to the LEDH construction?

**Yes, if the premises hold:**

| Premise | Status | Notes |
|---|---|---|
| $F$ deterministic in $\theta$ | ✅ if seeds frozen | Remark 5.3 explicit |
| $F$ momentum-independent | ✅ | LEDH score depends only on $\theta$ |
| No trajectory-history dependence | ⚠️ **verify** | No cached cloud, no call-count branch |
| $H$ uses exact executed $U = -\widehat{L}^N$ | ✅ | Value path frozen-seed |
| Value and force share frozen noise | ⚠️ **verify** | Current risk: A3 |
| Fixed branch (no $\theta$-dependent gates) | ⚠️ **verify** | Dual-cap, trust-region are $\theta$-dependent |

### 6.2. What "biased force" means formally

From Remark 5.3: force computed with **larger $\lambda, \delta$** than the value
program. So:

- **Value:** $U(\theta) = -\widehat{L}^N(\theta; \lambda_{\text{value}}, \delta_{\text{value}}, \text{frozen seeds})$
- **Force:** $F(\theta) = -\nabla_\theta \widehat{L}^N(\theta; \lambda_{\text{force}}, \delta_{\text{force}}, \text{same frozen seeds})$

with $\lambda_{\text{force}} > \lambda_{\text{value}}$ or
$\delta_{\text{force}} > \delta_{\text{value}}$.

This is **not** $\nabla U + \epsilon$. It is $\nabla \widetilde{U}$ where
$\widetilde{U}$ is a **different functional** (different reset, different
transport, different cloud). The gradient is exact *for that functional*, but
the functional differs from the acceptance one.

**Pre-mortem A3's claim is correct:** if $\lambda$ enters the transport, same
noise through two different resets produces two different clouds. The cost is 2×
per step (run filter twice), and "damping" is a discrete choice between
different transports, not a smooth perturbation.

### 6.3. What the corollary *does not* say

The corollary says nothing about:
- Whether the construction is **efficient** (acceptance rate, ESS)
- Whether it **mixes well**
- Whether the biased force is a **good approximation** to the exact force
- What $\lambda_{\text{force}}$ to choose

Remark 5.3 is explicit: "No claim is made about acceptance rates or mixing;
those are empirical questions for a planned experiment."

---

## 7. Corrections to the pre-mortem and gap audit

### 7.1. G1 (Corollary 5.2 never read) — **RESOLVED**

The theorem exists, is correctly cited, and says what the program claims. The
premises are listed and understandable. G1 is **closed as verified**, but three
sub-premises remain to check (seeds frozen everywhere, no history dependence,
fixed branch).

### 7.2. Pre-mortem claim "force is a perturbed gradient" — **WRONG**

I wrote (pre-mortem A3):

> "the force is not a perturbed gradient of the value — it is the gradient of a
> different functional evaluated on a different cloud."

**Correction:** This is exactly what Remark 5.3 describes as admissible. The
corollary applies to *any* deterministic $F$, including the gradient of a
different reset configuration. My framing as a defect was wrong. It is a
**cost and interpretability issue** (2× cost per step, discrete transport
choice), not a theoretical blocker.

### 7.3. Gap audit claim "θ-hashed seeding makes U discontinuous" — **CORRECT**

G20 / pre-mortem B4 is validated by Proposition 6 (line 1166): a
$\theta$-dependent branch (and `hash(θ)` is maximally $\theta$-dependent) makes
$\widehat{L}^N(\theta)$ discontinuous at every branch boundary. The note's
verdict: "$\theta$-independent reset schedules are the safe policy."

**Resolution:** one trajectory-level seed, frozen for the entire HMC chain. The
seed may be a function of the **initial** $\theta_0$ (part of the frozen
context), but not of the **current** $\theta$ during leapfrog.

### 7.4. Gap audit claim "single ω doesn't generalise" — **CORRECT**

Pre-mortem A2 / G13 is consistent with the note. The corollary targets
$\pi_N^\omega(\theta) \propto \exp(-U_N^\omega(\theta))$ **for one draw ω**.
Evidence at one ω is one sample from a distribution over pseudo-posteriors.

---

## 8. Open verification tasks (the three ⚠️ items)

### V1. Are all Monte Carlo seeds frozen? `[premise check]`

**Where seeds enter:**
- Initial-noise matrix (Definition 1, line 85)
- Innovation rows $e_{t,i}$ (line 85)
- Sinkhorn noise (entropic OT)
- Residual design $\Xi$ (line 87, but this is deterministic per Contract-E)

**Required:** every seed must be a deterministic function of one trajectory-level
seed $\omega$, and $\omega$ must be **frozen for the entire HMC chain** (not
per-step, not per-trajectory within a chain).

**Check:** instrument every TF random op in the LEDH value and force paths,
assert they all resolve to slices of one master seed.

### V2. Is the force history-independent? `[premise check]`

**Forbidden:**
- Cached particle cloud from a prior evaluation
- Call-count conditional
- Step-index conditional
- Accumulated float state (running mean, EMA)

**Check:** run involution test (forward L steps, flip $p$, backward L steps,
assert return to start). If the force depends on anything but $\theta$, this
fails.

### V3. Is the branch fixed? `[premise check]`

**Risk:** dual-cap and trust-region are control-flow gates. If they depend on
$\theta$ (e.g. "apply dual-cap if $\|\nabla_\theta \widehat{L}^N\| > \tau$"),
Proposition 6 applies and $U$ is discontinuous.

**Check:** audit every `if` statement in the reset and correction paths. A gate
is safe if it depends only on:
- Fixed hyperparameters (N, d, T, λ)
- Validity checks that hold on an open set (PD, floor)
- Time index $t$

A gate is **unsafe** if it depends on:
- Any empirical quantity that varies with $\theta$ (weights, ESS, residuals,
  condition numbers)
- Gradient norm
- Previous iterations' outcomes

---

## 9. Updated gap status

| Gap | Status after reading Corollary 5.2 |
|---|---|
| G1 | ✅ Resolved: theorem exists, correctly cited, premises listed |
| A3 | ⚠️ Reframed: not a blocker, but a cost/interpretability issue |
| B4 | ✅ Validated: θ-hashed seeding breaks continuity (Prop 6) |
| A2 | ✅ Validated: single ω is one pseudo-posterior sample |
| V1 | ⚠️ Open: verify all seeds frozen |
| V2 | ⚠️ Open: verify force history-independent |
| V3 | ⚠️ Open: verify branch fixed (dual-cap, trust-region) |

The foundational theorem is sound. Three runtime premises remain to verify
before the construction can be called Corollary-5.2-compliant.

---

**END OF COROLLARY 5.2 BINDING**
