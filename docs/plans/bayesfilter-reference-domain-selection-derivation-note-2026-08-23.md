# Reference-Domain Selection — Derivation Note — 2026-08-23

Status: `DERIVATION_NOTE` (supersedes the "Option A vs Option B" framing of
`bayesfilter-p1b-attempt04-result-2026-08-22.md` §Decision: the construction
question is answered by derivation where derivation reaches, and the residual
empirical questions are named. Owner input is required only for the
program-level budget call, not for the mathematics.)

Diagnostic script: `docs/benchmarks/check_reference_domain_selection_20260823.py`
(NumPy closed-form/quadrature checks; diagnostic code under the backend rule).
Paper: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024`
(Prop. 9 bounded-space remark, txt:1372; §5.2 linear preconditioning, txt:1581;
weighted/unbounded theory delegated to Cui & Dolgov 2022, Cui et al. 2023 —
fetch into `.localresources` before C2 implementation).

## 0. Setting and candidates

Step target (unnormalized), joint over current/previous blocks, x ∈ R^{2n}:

    q_t(x_c, x_p) = f(x_c | x_p, θ) · g(y_t | x_c, θ) · π̂_{t-1}(x_p).

The declared represented object is NOT the bare square. Per the Aug-6
derivation note and the paper's own construction (txt:556-569: the
defensive term ensures the target is absolutely continuous w.r.t. the
approximation, i.e. bounded target/approximation ratio; txt:712 eq. (16):
per-block product λ(x_t)λ(θ)λ(x_{t-1}); txt:1428-1445: the error analysis
carries τ_t explicitly):

    q̄_t = e^{-c_t} ( h_t² + τ_t Z_{h,t} λ_t ),    q_t ≈ q̄_t  w.r.t. μ_t,

where c_t is the frozen row-constant log-shift, λ_t is a probability
density w.r.t. μ_t, and the engine uses the relative-τ convention
(defensive mass proportional to the fitted mass Z_{h,t} = ⟨h_t, h_t⟩_μ;
`squared_tt_engine_adapted_tf.py:307-315,371`). The increment is

    Ẑ_t = ∫ q̄_t dμ_t = e^{-c_t} (1 + τ_t) Z_{h,t},

exact (Gram chain plus defensive mass), and retention carries the
defensive marginal in closed form (Aug-6 note, a_t formula). The
defensive term is a safety component of the declared program — it floors
the retained density so that zeros of the fitted polynomial cannot become
-inf log-targets or unbounded ratios at later rows — and is therefore
scored as its own objective (O7). The first draft of this note
abbreviated it away; corrected 2026-08-23 (owner catch). All candidates keep the per-step frozen affine
(triangular) preconditioner T_t built from deterministic moment hints — the
validated Aug-20/21 machinery. The candidates differ ONLY in (μ_t, basis,
domain):

- **C1 (incumbent)**: μ_t = uniform on the κσ moment box B_t ⊂ R^{2n};
  Legendre basis; boxes chained by retention (containment constraint).
- **C2 (Gaussian reference)**: μ = N(0, I_{2n}) on R^{2n} in whitened
  coordinates u = T_t(x); normalized probabilists' Hermite basis; no boxes.
- **C3 (compactified, author-code route)**: μ_t = uniform on [-1,1]^{2n};
  coordinates z = A(T_t(x)) with A the per-axis `AlgebraicMapping`
  bijection u ↦ u/√(s²+u²) (P86 contract, `bases.py:116`); Legendre/Lagrangep
  basis on the box; no truncation (A is a diffeomorphism R → (-1,1)).

## 1. Properties: mathematical definitions

Hard requirements (binary):

- **R1 (exact normalizer + analytic adjoint).** The map cores ↦ Ẑ_t must be
  a finite composition of closed-form operations: mass matrices
  B_ij = ∫ b_i b_j dμ must be exactly computable constants, and the manual
  adjoint of the declared finite program must exist in closed form.
- **R2 (well-defined recursion).** supp(row law at step t) ⊆
  dom(retained object of step t−1). If violated, the recursion forces either
  truncation of the retained object or evaluation outside its domain.

Objectives (quantitative):

- **O1 (structural truncation mass).**
  ε_trunc(n) := q_t(R^{2n} \ dom_t) / q_t(R^{2n}) — step-target mass outside
  the representable domain, at zero fit error. Required: ε_trunc ≡ 0, or at
  minimum non-increasing in n.
- **O2 (zero-fit-error bias floor).**
  b(n) := lim_{fit error → 0} d(retained_t, true filter marginal_t).
  Required: b ≡ 0 (the recursion must converge to the truth as
  approximation error vanishes).
- **O3 (per-axis approximation efficiency).** ℓ(ε) := smallest per-axis basis
  cardinality with relative L²(μ) error ≤ ε for the program's representative
  target family (whitened near-Gaussian after preconditioning; LGSSM = exactly
  Gaussian is the calibration member).
- **O4 (TT rank demand).** Rank r(ε) on the same family. (Derivation
  separates candidates only through O2's shape distortion; otherwise
  empirical.)
- **O5 (fit conditioning / kernel cost).** Boundedness of basis values on the
  effective row support (design-matrix conditioning) and the ℓ-driven cost
  channels: design matrix width ℓr², mass-matrix work ℓ², XLA graph size.
- **O6 (θ-derivative compatibility).** The exact score of the declared
  program (maps frozen) must exist with closed-form adjoint nodes.
- **O7 (defensive floor support and tail order).** supp(λ_t) must cover
  the recursion's evaluation set (a floor with compact support is
  inoperative exactly where the risk lives), and the tail order of λ_t's
  physical-space pullback bounds the target-to-floor ratio at
  re-expression points.

Engineering re-derivation cost is recorded separately (§4) — it is a budget
input, not a mathematical objective, and must not be allowed to masquerade
as one. That conflation is how C1 survived three repairs.

## 2. Derivations per candidate

### R1 — all three PASS

- C1: certified (Aug-8 block certificates; adjoint vs forward-JVP 1e-12).
- C2: normalized Hermite mass matrix is the identity:
  ∫ Hē_j(u) Hē_k(u) dN(0,1)(u) = δ_jk with Hē_k := He_k/√(k!), by standard
  orthogonality of probabilists' Hermite polynomials under the Gaussian
  weight. Numerical check: max |B − I| = 1.33e-15, degrees ≤ 12 (script §C).
  Gram chains, retention marginals, and the defensive term (λ := 1 w.r.t. η,
  i.e. the reference density itself) carry over with B = I.
- C3: the reference measure is still uniform on the box — compactification is
  absorbed into target-side Jacobian factors √(J(z)) at row evaluation. Mass
  matrices, Gram chains, and the adjoint algebra are **unchanged** from C1.

Consequence: R1 never forced the bounded box. The analytic-derivative
requirement does not discriminate among the candidates.

### R2 — C1 FAILS at n ≥ 4; C2, C3 PASS

C1: the retained polynomial is undefined outside its own box, so each step's
previous-block box must nest inside the prior box. The containment fixed
point pins the effective previous-block half-width at κ* ≈ 2.0–2.2σ
independent of requested κ (measured twice, κ-grid; design note §11 — this is
an empirical input to the derivation, marked as such). C2/C3: dom = R^n
(full support; A is a bijection), so R2 holds vacuously — containment is not
a constraint that exists.

### O1 — ε_trunc: C1 dimension-growing; C2, C3 identically zero

C1, whitened previous block, ideal centered box at the capped κ*:

    ε_trunc(n) ≥ 1 − (2Φ(κ*) − 1)^n

(product of per-axis standard-Gaussian box masses; anisotropic containment
shrink only worsens it). Computed (script §A): at κ* = 2.2:
n=2 → 5.5%, n=4 → 10.7%, n=8 → 20.2%; at κ* = 2.0: 8.9% / 17.0% / 31.1%.
The attempt04 measured truncation ratios at n=4 (26–65%) sit above the ideal
bound, consistent with the measured shrink factors 0.4–0.7. The geometric
growth in n is the derivation-level statement of "dimension-lethal".
C2/C3: dom = R^n ⇒ ε_trunc ≡ 0 identically.

### O2 — bias floor: C1 nonzero and irreducible; C2, C3 O(τ), vanishing under the paper's τ coupling

C1: at zero fit error the retained object is exactly the box-conditioned
marginal, which differs from the true marginal by the ε_trunc(n) mass and its
shape. Conditioning does not commute with the recursion (the composition of
per-step conditionings is not the conditioning of the composition), so the
deficiency compounds. Removing it requires κ → ∞, which R2's containment
fixed point forbids. Hence b(n) > 0 and increasing in n: **no rank or degree
repairs it** — this is the theorem-shaped content behind the attempt04 flag.
C2/C3: the defensive term itself injects a mixture bias — at zero fit
error the retained μ-density is (q_μ + τ Z_h λ)/((1+τ) Z_h), so

    TV(retained, true) = (τ/(2(1+τ))) ∫ |λ − q_μ/Z| dμ ≤ τ/(1+τ),

**dimension-free** (the mixture weight does not grow with n), and
controlled two ways: under the paper's own coupling
τ_t ≤ ‖φ_t − √q_t‖²_{L²} (Lemma 1 / Prop. 11 hypothesis, txt:570-575 and
1445-1450; the engine's relative-τ convention rescales the absolute
constant by Z_{h,t}) the bias is second-order in the fit error, so b → 0
in the fit→0 limit and the limit statement survives; under the program's
current fixed-τ policy (τ = 1e-6, P1B plan) it is a constant ≤ 1e-6 with
worst-case T·τ ≈ 1.2e-4 over T = 120, well under the 2.5e-3 bar. The
fixed-τ-vs-paper-coupled choice is hereby flagged as an unexamined
default for the C2 derivation note's default audit. C1 carries the same
O(τ) term PLUS the truncation part, which is the binding defect:
irreducible, dimension-growing, κ-uncorrectable. (First draft wrote
b ≡ 0 unconditionally for C2/C3 — corrected 2026-08-23, second owner
catch.)

### O3 — per-axis cardinality: C2 ≪ C1 < C3 on the near-Gaussian family

C2: if the whitened step target is exactly Gaussian, the fitted object is
g = √(q∘T⁻¹ · |det L| / η) ≡ const — **degree 0**. The preconditioner absorbs
the Gaussian part by construction; deviations from Gaussianity set the
degree (this is precisely the paper's Fig. 2 rank-collapse mechanism, and the
weighted approximation theory is Cui & Dolgov 2022's subject — delegated, to
be anchored when the PDFs land in `.localresources`).

C1: the fitted object √(truncated Gaussian) is entire on the box ⇒ geometric
Legendre decay. Measured: ℓ(1e-3) = 7, ℓ(1e-4) = 9 (script §D). Note this
efficiency is *purchased by the truncation that O1/O2 eliminate* — the box
makes the target easy by cutting exactly the mass that invalidates the
recursion.

C3: the fitted object is √(F1) with F1(z) = η(x(z)) · s(1−z²)^{−3/2},
x(z) = sz/√(1−z²). As z → ±1, the exponent −x²/2 = −s²z²/(2(1−z²)) → −∞
faster than the Jacobian blows up, so F1 → 0 but with an essential
singularity at the endpoints in the analytic continuation ⇒ subgeometric
Legendre convergence. Measured (script §D): ℓ(1e-3) = 33 at map scale s=1
(63 at s=0.5, 21 at s=2), vs 7 for C1's comparator. Independent
corroboration: the authors' own route pays exactly this — `Lagrangep(4,8)`
is ~33 dofs/axis, and the paper's experiments run ℓ = 33.

### O4 — rank: C1 invalidated; C2 vs C3 not separable by derivation

C1's box-conditioned retained shape (first-order at n≥4, measured) feeds a
distorted target into the next fit — a recursion-instability channel C2/C3
lack. Beyond that, rank demand on non-Gaussian targets is an empirical
question for both C2 and C3; no ranking is claimed.

### O5 — conditioning and cost

- C1: Legendre values bounded on [-1,1]; mild. (Moot given R2/O2.)
- C2: normalized Hermite grows in the tail: |Hē_k(z)| ≤ ~3.4 at z=3 and
  ~25.7 at z=4 for k ≤ 16 (script §B). Rows drawn from η essentially never
  exceed |z| = 4, and the L²(η) fit norm damps tails — bounded, recorded
  constant; not disqualifying.
- C3: basis values bounded, but O3's ℓ ≈ 33–63 inflates every ℓ-driven cost:
  design width ℓr², mass work ℓ², and XLA graph size — the same compile
  channel that produced attempt04's rank-10 9-axis timeouts, now driven by
  degree instead of rank.

### O6 — all three PASS

Maps are frozen per step in all candidates (deterministic hints; the paper's
particle estimate of (μ_t, Σ_t) in §5.2 is replaced by frozen hints exactly
as in the current program — this is the one place the analytic-derivative
requirement genuinely forces deviation from the paper, and it is
candidate-independent). Mass matrices are θ-independent constants in all
three. The adjoint machinery differentiates target values at rows — same
nodes, different basis evaluations.

### O7 — defensive floor: the paper's domination hypothesis; C1 fails it intrinsically; λ is free in C2/C3

The paper defines λ generically — "some reference tensor-product
probability density λ(x) such that sup_x π(x)/λ(x) < ∞" (txt:553-556,
eq. (13)) — and its guard theorem is exactly the ratio bound

    sup_x p(x)/p̂(x) < (ẑ/(τ z)) · sup_x π(x)/λ(x),

i.e. the floor performs its declared function (absolute continuity,
CLT-valid weights, bounded log-ratios) **iff λ dominates the target's
tails**. Lemma 1 / Proposition 11 price the floor's L² cost at a factor
√2 under τ_t ≤ ‖φ_t − √q_t‖². Consequences:

- **C1: intrinsic FAIL.** Any λ supported on the box gives
  sup q/λ = ∞ for an unbounded-support target: the paper's own
  hypothesis is violated by construction, so the guard cannot perform its
  declared function in exactly the region R2/O1 indict. Third independent
  elimination ground.
- **C2 and C3: λ's tail order is a free design choice, so O7 does not
  discriminate between them.** Both admit tensor-product λ of any tail
  order with the closed-form algebra intact: ∫ λ dμ = 1, per-axis product
  marginals in retention, θ-independence in the adjoint. λ := reference
  density (Gaussian for C2; compactified-uniform, i.e. t₂-like, for C3)
  is merely the default; a heavier-tailed product λ (Student-t family)
  restores domination for any declared target tail class in either
  candidate (log-space evaluation of λ = t/η keeps C2 float-safe). The
  first amendment credited C3 a strict O7 advantage; that was an artifact
  of freezing λ := reference — withdrawn.

The floor splits the tail requirements into two channels, which the first
draft conflated:

1. **Increment accuracy — NOT relaxed by the floor.** The floor's mass is
   λ-shaped, not target-shaped; Lemma 1's √2 factor shows it only adds L²
   error. Un-fitted tail mass still biases Ẑ_t one-for-one. The O3
   measured ℓ(ε) numbers stand unchanged as increment-accuracy
   statements; what governs missed tail mass is preconditioning quality
   (whether the whitened target's mass sits where rows sit) —
   candidate-independent.
2. **Recursion stability — what the floor regularizes.** With domination,
   log q̄ is floored at log(τ Z_h λ) and the represented-to-true ratio is
   bounded by the display above, so fit zeros and missed tails produce
   bounded, tail-mass-weighted errors at later rows instead of -inf
   log-targets or unbounded weights. This is the "regularizes the
   unboundedness of the fit" effect, and it is available to C2 and C3
   equally through the λ choice.

## 3. Evaluation table

| Property | C1 box (incumbent) | C2 Gaussian/Hermite | C3 compactified |
|---|---|---|---|
| R1 exact normalizer + adjoint | PASS (certified) | PASS (B = I, checked 1.3e-15) | PASS (mass matrices unchanged) |
| R2 recursion well-defined | **FAIL n≥4** (containment cap κ*≈2, measured) | PASS | PASS |
| O1 ε_trunc | ≥ 11–17% at n=4, geometric in n | 0 identically | 0 identically |
| O2 bias floor (fit→0) | > 0, irreducible | 0 | 0 |
| O3 ℓ(1e-3), whitened Gaussian | 7 (bought by truncation) | ~1 (g ≡ const; deviations set ℓ) | 21–63 (scale-dep.; authors pay 33) |
| O4 rank | invalidated via O2 | empirical | empirical |
| O5 conditioning/cost | mild (moot) | tail factor ≤ ~26 (recorded) | ℓ² and compile-channel inflation |
| O6 θ-adjoint | PASS | PASS | PASS |
| O7 defensive floor (λ domination, eq. (13)) | **FAIL** intrinsic (sup q/λ = ∞ off-box) | pass (λ tail order free) | pass (λ tail order free) |

## 4. Engineering re-derivation surface (budget input, not an objective)

- C1: zero. (Eliminated on R2/O1/O2 regardless.)
- C2: new bases module (normalized Hermite + Gauss–Hermite/η-Sobol row law),
  B = I simplifies Gram chains, defensive term = η, retention re-derivation
  under η, adjoint re-certification (U-MAP-MOM-1-style unit ladder). Largest
  surface, but every mass matrix is the identity — the algebra *shrinks*.
- C3: smallest surface (Jacobian factors at rows only; June P86 contract,
  Lagrangep mass integrals, and a value-path fit smoke already exist) — but
  O3's ℓ-inflation is a permanent per-axis tax paid at every step of every
  run, vs C2's one-time derivation cost.

## 5. Verdict licensed by the derivations

1. **C1 is mathematically eliminated at n ≥ 4** — R2 fails and O1/O2 give a
   dimension-growing irreducible bias floor. This is not a preference and is
   not revisitable by tuning (κ-invariance is measured; the ε_trunc bound is
   an identity given κ*).
2. **C2 dominates C3 on the derived objectives** (O3 decisively on the
   calibration family, O5's cost channel; ties elsewhere). The first
   amendment briefly credited C3 a heavy-tail O7 advantage; the second
   round withdraws it: per the paper's own eq. (13) hypothesis, λ's tail
   order is a free closed-form design choice in both C2 and C3, so O7
   discriminates only against C1. Heavy-tail risk decomposes into
   increment accuracy (governed by preconditioning quality;
   candidate-independent) and recursion stability (governed by λ
   domination; free in both). The mathematics selects **C2**. C3's
   re-derivation-cost edge remains a budget input, not an objective.
3. **What remains genuinely empirical** (named, not decided): rank demand
   r(ε) under C2 on non-Gaussian targets (SV); the Hermite tail-conditioning
   constant in practice at n ≥ 8; XLA behavior of Hermite kernels.
4. **What remains genuinely owner-level**: whether the n ≥ 4 program line
   gets the C2 derivation+implementation budget at all (a direction/budget
   call — the mathematics ranks the candidates, it does not command the
   spend).

## 6. Scientific consequence for the ladder design

Under C2 with exact hints, the LGSSM step target is Gaussian and g ≡ const:
r*(n) on LGSSM becomes small **by construction** and measures hint quality,
not TT capacity. The feasibility question r*(n) must therefore move to the
non-Gaussian arm (SV or nonlinear fixtures) or to deliberately degraded
hints. attempt05's plan must redeclare its question accordingly — carrying
attempt04's question forward unchanged would produce a pass-for-wrong-reason
artifact.

## 7. Non-claims and red team

Not claimed: any C2 rank statement; any posterior/HMC readiness; that the 1-D
ℓ(ε) quadrature numbers transfer to n-D targets beyond the structural
conclusion (they are exact quadrature computations, not MC estimates, but
they are computed on the easiest family member). Strongest alternative
reading: C3's ℓ-inflation might be acceptable at moderate n if XLA compile
scales better with ℓ than with r — the recorded compile blowups were
rank-driven; a degree-driven probe would discriminate cheaply if C2's
derivation stalls. Weakest evidence: the Cui & Dolgov weighted approximation
theory is cited-but-not-yet-inspected; fetching and reading it is a
prerequisite for the C2 derivation note, per the literature discipline.

Amendment 2026-08-23 (owner catch): the first draft scored the candidates
against a program with the defensive term abbreviated away. The term is a
declared safety component (paper txt:556-569 and eq. (16); Aug-6 derivation
note q̄_t = e^{-c_t}(φ_t² + τ_t λ_t); engine
`squared_tt_engine_adapted_tf.py:307-315,371`) and is now carried in §0,
O7, the table, and the verdict. Effect of the correction: no conclusion
reverses for the declared family; the C1 elimination is strengthened (its
floor shares the box's support, so the guard is inoperative off-box); the
"dominates on every derived objective" sentence was too strong and is
amended — O7 favors C3 in the heavy-tail regime. Also now recorded as a
non-claim: C2's O3 efficiency statement does not extend to
heavier-than-Gaussian whitened tails, where the Hermite rate degrades and
the paper's own answer is the §5.3 nonlinear layer.

Second amendment 2026-08-23 (owner challenge: "the safety term
regularizes the unboundedness of the fit — does the tail analysis
change?"): yes, in two places, and the corrections strengthen the verdict
rather than weakening it. (i) O2's "b ≡ 0" for C2/C3 was unconditional
and wrong as written: the floor injects a τ/(1+τ) mixture bias —
dimension-free, second-order under the paper's τ_t ≤ ε_t² coupling
(Prop. 11), ≤ 1e-6 under the program's fixed-τ policy; corrected in
place, and the fixed-vs-coupled τ policy is flagged as an unexamined
default for the C2 derivation note's audit. (ii) O7 was restructured
around the paper's own domination hypothesis sup q/λ < ∞ (eq. (13)):
C1 violates it intrinsically for unbounded-support targets — a third
independent elimination ground — while for C2/C3 the λ tail order is a
free closed-form design choice, so the first amendment's "C3 strictly
stronger on O7" is withdrawn. The tail analysis is now split into
increment accuracy (the floor does not relax it; the O3 measured numbers
stand) and recursion stability (the floor controls it via domination;
candidate-independent). The preceding paragraph's heavy-tail non-claim is
re-scoped accordingly: the Hermite rate concern belongs to the bulk
increment-accuracy channel when heavy tails carry real mass — a
preconditioning-quality question — while the stability risk is handled by
the λ choice in either candidate. O1/O3/O5 derivations and all measured
numbers are unchanged by this amendment.
