# RQMC LEDH V2 Design: Mathematical Formalization and Audit

**Date:** 2026-09-06  
**Status:** MATHEMATICAL AUDIT — formal verification required before implementation  
**Related:** rqmc-ledh-init-v2-design-2026-09-06.md  

## Purpose

Formalize the V2 campaign design as a mathematical proposition, state all assumptions explicitly, and verify that the design eliminates the V1 confound while preserving scientific validity.

---

## Executive Summary

**Verdict**: V2 design eliminates the V1 process-noise confound and is mathematically sound **if GenUT is dropped**. Three GenUT-specific confounds remain unfixable:

1. **Process-noise confound (V1)** — ✅ **FIXED** by `SeedSequence.spawn(2)`
2. **Moment pre-alignment confound (GenUT)** — ❌ **UNFIXABLE** without changing filter
3. **Index-wise injection confound (GenUT)** — ⚠️ **FIXABLE** with permutation, but still leaves #2
4. **Low diversity (GenUT)** — ⚠️ **PARTIALLY FIXABLE** with symmetric design (only 3-7 distinct particles)

**Recommendation**: Drop `genut_guided`, add Latin Hypercube Sampling. This produces a clean 5-arm comparison (mc, sobol_owen, halton_owen, halton_reverse, lhs) with no confounds.

**Statistical framework**: Bootstrap estimand valid for n=10 seeds. Sensitivity analysis protocol specified for small-n bootstrap variance.

---

## Background: V1 Confound

The V1 runner used one `tf.random.Generator` for both initial and process noise:

```python
rng = replication_generator(seed)
initial_noise = _generate_initial_noise(arm, state_dim, seed, rng)  # may advance rng
process_noise = _generate_process_noise(horizon, state_dim, seed, rng)
```

**Confound**: For arm $a$ and seed $s$:
- MC: $\text{rng}$ advances by $N \cdot d$ elements → process noise at offset $N \cdot d$
- RQMC: $\text{rng}$ not advanced → process noise at offset $0$

So $\mathcal{W}^{\text{MC}}_{s} \neq \mathcal{W}^{\text{RQMC}}_{s}$ (different process-noise trajectories), violating the "identical dynamics" requirement.

---

## V2 Fix: Independent Seeding via SeedSequence

### Notation

- $s \in \mathbb{N}$: replication seed (e.g., 98401)
- $d \in \mathbb{N}$: state dimension
- $T \in \mathbb{N}$: time horizon
- $N \in \mathbb{N}$: particle count (1008)
- $a \in \mathcal{A}$: arm index (mc, sobol\_owen, halton\_owen, halton\_reverse, genut\_symmetric)

### Random Generators

Let $\text{SeedSequence}(s)$ be the NumPy SeedSequence with entropy $s$, and let $\text{spawn}(n)$ derive $n$ independent child sequences.

Define:
$$
\begin{align}
\text{ss} &= \text{SeedSequence}(s) \\
[\text{ss}_{\text{init}}, \text{ss}_{\text{proc}}] &= \text{ss.spawn}(2) \\
s_{\text{init}} &= \text{ss}_{\text{init}}\text{.generate\_state}(1)[0] \\
s_{\text{proc}} &= \text{ss}_{\text{proc}}\text{.generate\_state}(1)[0]
\end{align}
$$

Then:
$$
\begin{align}
G_{\text{init}}(s) &= \text{tf.random.Generator.from\_seed}(s_{\text{init}}) \\
G_{\text{proc}}(s) &= \text{tf.random.Generator.from\_seed}(s_{\text{proc}})
\end{align}
$$

### Initial Noise Generation

For arm $a$ and seed $s$, let:
$$
\mathcal{Z}^{(a)}_0(s) = \begin{cases}
G_{\text{init}}(s)\text{.normal}([N, d]) & \text{if } a = \text{mc} \\
\Phi^{-1}(\text{Sobol}_{\text{Owen}}(d, N, s)) & \text{if } a = \text{sobol\_owen} \\
\Phi^{-1}(\text{Halton}_{\text{Owen}}(d, N, s)) & \text{if } a = \text{halton\_owen} \\
\Phi^{-1}(\text{Halton}_{\text{Reverse}}(d, N, s)) & \text{if } a = \text{halton\_reverse} \\
\text{GenUT}_{\text{symmetric}}(d, N, s) & \text{if } a = \text{genut\_symmetric}
\end{cases}
$$

where $\Phi^{-1}$ is the standard normal quantile function.

**Key property**: $\mathcal{Z}^{(a)}_0(s)$ may or may not use $G_{\text{init}}(s)$, but it **never uses** $G_{\text{proc}}(s)$.

### Process Noise Generation

For all arms and all seeds:
$$
\mathcal{W}(s) = G_{\text{proc}}(s)\text{.normal}([T, N, d])
$$

**Critically**: $G_{\text{proc}}(s)$ is initialized from $s_{\text{proc}}$ and **never called before** this line, regardless of arm.

---

## Proposition 1: Process-Noise Independence

**Statement**: For all arms $a, a' \in \mathcal{A}$ and all seeds $s \in \mathbb{N}$:
$$
\mathcal{W}(s) \text{ is deterministic given } s \text{ and independent of } a
$$

That is:
$$
\mathcal{W}^{(a)}(s) = \mathcal{W}^{(a')}(s) = \mathcal{W}(s) \quad \forall a, a'
$$

**Proof**:

1. By construction, $s_{\text{proc}} = \text{SeedSequence}(s)\text{.spawn}(2)[1]\text{.generate\_state}(1)[0]$ is deterministic given $s$ and does not depend on $a$.

2. $G_{\text{proc}}(s) = \text{tf.random.Generator.from\_seed}(s_{\text{proc}})$ is deterministic given $s_{\text{proc}}$.

3. The only call to $G_{\text{proc}}(s)$ is:
   $$
   \mathcal{W}(s) = G_{\text{proc}}(s)\text{.normal}([T, N, d])
   $$
   which occurs **after** $\mathcal{Z}^{(a)}_0(s)$ is generated, and $\mathcal{Z}^{(a)}_0(s)$ never modifies $G_{\text{proc}}(s)$.

4. Therefore, $G_{\text{proc}}(s)$ is in the **same internal state** when called for all arms $a$, given seed $s$.

5. Since TensorFlow's `normal` is deterministic given generator state, $\mathcal{W}(s)$ is independent of $a$.

**Verification protocol**: For each seed $s$, compute:
$$
h_a(s) = \text{SHA256}(\mathcal{W}^{(a)}(s))
$$

The campaign is valid only if:
$$
|\\{h_a(s) : a \in \mathcal{A}\\}| = 1 \quad \forall s
$$

(i.e., all arms produce the same hash for process noise at each seed).

---

## Proposition 2: Mechanism Isolation

**Statement**: Under V2, the only difference between arm $a$ and arm $a'$ for seed $s$ is the initial particle cloud $\mathcal{Z}^{(a)}_0(s)$ vs $\mathcal{Z}^{(a')}_0(s)$. All subsequent dynamics are identical.

**Formally**: Let $\mathcal{L}^{(a)}(s)$ denote the terminal log-likelihood for arm $a$ and seed $s$. Then:
$$
\mathcal{L}^{(a)}(s) = f(\mathcal{Z}^{(a)}_0(s), \mathcal{W}(s), \theta, y_{1:T}, D)
$$

where:
- $\theta$: model parameters (fixed)
- $y_{1:T}$: observations (fixed)
- $D$: reset design (fixed, arm-independent)
- $\mathcal{W}(s)$: process noise (Proposition 1: arm-independent)
- $f$: LEDH evaluator (deterministic given inputs)

**Corollary**: The causal effect of initialization is:
$$
\Delta^{a \to a'}(s) = \mathcal{L}^{(a')}(s) - \mathcal{L}^{(a)}(s) = f(\mathcal{Z}^{(a')}_0(s), \cdots) - f(\mathcal{Z}^{(a)}_0(s), \cdots)
$$

where "$\cdots$" denotes the common arguments $(\mathcal{W}(s), \theta, y_{1:T}, D)$.

**Proof**: By Proposition 1, $\mathcal{W}(s)$ is arm-independent. The evaluator $f$ is a pure function (TensorFlow graph, deterministic). $\theta$, $y_{1:T}$, and $D$ are fixed inputs. Therefore, the only degree of freedom is $\mathcal{Z}^{(a)}_0(s)$.

---

## Proposition 3: Statistical Estimand

**Statement**: The bootstrap 95% CI for $\mathbb{E}_s[\Delta^{\text{mc} \to a}(s)]$ (where $s \sim \text{Uniform}(\{98401, \ldots, 98410\})$) is an unbiased estimator of the **average causal effect of initialization** for arm $a$ vs MC, under the assumption that the 10 seeds are exchangeable.

**Formally**: Let:
$$
\bar{\Delta}^{\text{mc} \to a} = \frac{1}{n} \sum_{i=1}^{n} \Delta^{\text{mc} \to a}(s_i)
$$

where $n = 10$ and $s_i \in \{98401, \ldots, 98410\}$.

Bootstrap resampling gives:
$$
\bar{\Delta}^{\text{mc} \to a}_{\text{boot},b} = \frac{1}{n} \sum_{i=1}^{n} \Delta^{\text{mc} \to a}(s_{\sigma_b(i)})
$$

where $\sigma_b$ is a random permutation with replacement.

The 95% CI is:
$$
[\bar{\Delta}^{\text{mc} \to a}_{2.5\%}, \bar{\Delta}^{\text{mc} \to a}_{97.5\%}]
$$

where percentiles are taken over $B = 10{,}000$ bootstrap samples.

**Interpretation**:
- If $\bar{\Delta}^{\text{mc} \to a}_{2.5\%} > 0$: arm $a$ is **statistically superior** to MC (higher terminal log-likelihood)
- If $\bar{\Delta}^{\text{mc} \to a}_{97.5\%} < 0$: arm $a$ is **statistically inferior** to MC
- Otherwise: **statistically indistinguishable** under current evidence

**What is NOT being estimated**:
- Effect of RQMC **process noise** (would require $\mathcal{W}^{(a)}(s) \neq \mathcal{W}^{\text{mc}}(s)$)
- Effect of RQMC **resampling sequences** (not tested; resampling uses the same RNG for all arms)
- Hardware-specific effects (one GPU, one dtype)
- Generalization to $N \neq 1008$

---

## Assumption Audit

### A1. SeedSequence Spawn Independence

**Assumption**: `numpy.random.SeedSequence(s).spawn(2)` produces two cryptographically independent child sequences.

**Source**: NumPy documentation states SeedSequence uses SipHash-2-4 for avalanche properties.

**Verification**: Standard NumPy test suite.

**Risk**: Low (well-tested library behavior).

### A2. TensorFlow Generator Determinism

**Assumption**: `tf.random.Generator.from_seed(s).normal(shape)` is deterministic given seed $s$ and shape.

**Source**: TensorFlow documentation guarantees determinism for stateful generators.

**Verification**: Regression test in Phase 0.

**Risk**: Low (core TensorFlow guarantee).

### A3. Evaluator Purity

**Assumption**: The LEDH evaluator $f$ is a pure function: same inputs → same output.

**Source**: LEDH uses `@tf.function` with stable `input_signature` and no Python callbacks.

**Verification**: Existing LEDH tests verify determinism.

**Risk**: Low (already tested in production LEDH).

### A4. Seed Exchangeability

**Assumption**: Seeds 98401-98410 are exchangeable (no systematic ordering effects).

**Source**: Seeds are arbitrary integers chosen to avoid overlap with V1 (98301-98303).

**Verification**: Sensitivity analysis (drop highest/lowest seed, check if ranking changes).

**Risk**: Medium (n=10 is small; outliers could dominate).

**Mitigation**: Report per-seed table; if one seed drives the verdict, flag it.

### A5. Bootstrap Validity

**Assumption**: Bootstrap resampling gives valid CIs for $\mathbb{E}_s[\Delta]$ when $n=10$.

**Source**: Bootstrap is valid for smooth statistics when $n \geq 10$ (Efron & Tibshirani 1993).

**Risk**: Medium (n=10 is at the lower bound).

**Mitigation**: Report both bootstrap CI and sign test (exact, distribution-free).

### A6. No Confound Leakage

**Assumption**: The `_generate_initial_noise` implementations for RQMC arms (Sobol, Halton, GenUT) do **not** call `G_proc(s)` or modify any state that `G_proc(s)` depends on.

**Source**: Code inspection (Sobol/Halton use scipy/tfp; GenUT is deterministic).

**Verification**: Phase 0 confound regression (hash check).

**Risk**: Low (straightforward to verify).

---

## Critical Defect: GenUT Identity to Reset Design

### The Problem

Inspection reveals:
$$
\mathcal{Z}^{(\text{genut\_symmetric})}_0(s) = D
$$

where $D = \text{target}[\text{'design'}]$ is the **reset design** used by the LEDH filter at every resampling step.

**Implication**: The `genut_symmetric` arm is **not testing initialization quality**. It is testing whether using the reset design as the initial cloud affects performance compared to MC initialization.

This is a **different research question**:
- V2 (as written): "Does RQMC initialization improve particle coverage at $t=0$?"
- GenUT (actually): "Does initializing with the filter's own reset design (vs MC) affect terminal likelihood?"

### Why This Matters

1. **Not RQMC**: GenUT is deterministic (no quasi-randomness).
2. **Not independent of dynamics**: If the reset design is well-tuned for the model, initializing with it may give an advantage unrelated to spatial coverage.
3. **Unfair comparison**: GenUT gets to "know" the filter's internal structure; MC/Sobol/Halton do not.

### Resolution Options

**Option A — Drop GenUT entirely**:
- Replace with Latin Hypercube Sampling (LHS), which is a genuine RQMC method
- Pros: clean RQMC-vs-MC comparison
- Cons: loses the "reset design as initialization" question

**Option B — Keep GenUT but reframe**:
- Rename arm to `reset_design_init`
- Update evidence contract to state this arm tests a **different mechanism**
- Do not include it in the "RQMC superior" count
- Pros: preserves the question "does reset-aware initialization help?"
- Cons: muddies the primary research question

**Option C — Fix GenUT degeneracy properly**:
- Use **full symmetric design** (not just positive quadrant replication)
- For $d=3$: $2d+1 = 7$ distinct particles, but the zero-weight origin gives only 6
- This is still low diversity but represents a valid cubature method
- Pros: tests whether low-order cubature (not RQMC) helps initialization
- Cons: still deterministic, not quasi-random

**Recommendation**: **Option A** (drop GenUT, add LHS). The V2 design aims to test RQMC, and GenUT confuses that question.

---

## Revised Arm Definitions (with Option A)

1. **mc**: $\mathcal{Z}^{(\text{mc})}_0(s) = G_{\text{init}}(s)\text{.normal}([N, d])$
2. **sobol\_owen**: $\mathcal{Z}^{(\text{sobol})}_0(s) = \Phi^{-1}(\text{Sobol}(d, N; \text{scramble=Owen, seed=}s))$
3. **halton\_owen**: $\mathcal{Z}^{(\text{halton\_o})}_0(s) = \Phi^{-1}(\text{Halton}(d, N; \text{scramble=Owen, seed=}s))$
4. **halton\_reverse**: $\mathcal{Z}^{(\text{halton\_r})}_0(s) = \Phi^{-1}(\text{Halton}(d, N; \text{scramble=Reverse, seed=}s))$
5. **lhs**: $\mathcal{Z}^{(\text{lhs})}_0(s) = \Phi^{-1}(\text{LatinHypercube}(d, N; \text{seed=}s))$

All arms use $\mathcal{W}(s) = G_{\text{proc}}(s)\text{.normal}([T, N, d])$ (Proposition 1).

---

## Formalization for MathDevMCP

The above propositions and assumptions constitute the **mathematical claim** of the V2 design:

**Claim**: V2 isolates the causal effect of initialization on terminal log-likelihood, free of process-noise confound.

**Proof obligations**:
1. Proposition 1 (process-noise independence) ✓ proved above
2. Proposition 2 (mechanism isolation) ✓ follows from Prop 1 + evaluator purity
3. Proposition 3 (statistical estimand) ✓ standard bootstrap theory

**Assumptions requiring verification**:
- A1-A3: Low-risk library guarantees (defer to standard tests)
- A4: Exchangeability (sensitivity analysis in Phase 3)
- A5: Bootstrap validity (n=10 is marginal; augment with sign test)
- A6: No confound leakage (Phase 0 hash regression)

**Open questions for MathDevMCP audit**:
1. Does the SeedSequence derivation guarantee $s_{\text{init}} \perp\!\!\!\perp s_{\text{proc}} \mid s$?
2. Is the bootstrap CI valid for $n=10$ when the underlying distribution may be heavy-tailed?
3. Does the GenUT arm (if kept) invalidate the "RQMC vs MC" framing?

---

## Next Steps

1. **MathDevMCP audit**: Verify Propositions 1-3 and assumptions A1-A6
2. **Owner decision**: Option A (drop GenUT + add LHS) vs Option B (reframe) vs Option C (accept low diversity)?
3. **Phase 0 implementation**: If audit passes, implement runner with confound checks
4. **Phase 0 validation**: Hash regression, distinctness check, smoke test

Campaign execution waits for audit completion and owner approval.
