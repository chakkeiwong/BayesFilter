# C2 Gaussian-Reference Program — Derivation Note — 2026-08-24

Status: `DRAFT_FOR_REVIEW` (material review required before engine work;
this review is also the human certification step the MathDevMCP ch38
audits defer to). Implements owner decisions D1–D3 of
`bayesfilter-squared-tt-program-reset-memo-2026-08-24.md`. Selection
basis: monograph ch38 §40.8–40.9 (commit `347c5fd5`) and
`bayesfilter-reference-domain-selection-derivation-note-2026-08-23.md`.

Source anchors (local copies, text-extracted, technical sections read):

- ZC24 = `zhao-cui-tensor-train-sequential-learning-jmlr-2024`:
  §5.2 linear preconditioning; eq. (13)/(16) defensive form; Lemma 1,
  Prop. 11 (τ coupling), Cor. 12 (error accumulation).
- CD22 = `cui-dolgov-2022-deep-composition-sirt` (FoCM 22:1863–1922):
  §2.1 weighted-L² frame (the whole SIRT theory is generic in a product
  weight λ); §3 squared construction; **Prop. 2** marginalization
  recursion via per-coordinate mass Cholesky M_k = L_k L_kᵀ + core QR;
  §3.2.3 unbounded-domain options — including the authors' own warning
  that domain truncation "may lead to a biased estimator", and that the
  Hermite caveat is CDF-inversion cost/monotonicity **for sampling**;
  §3.2.4 diffeomorphic mapping; Thm 1 (Hellinger bound), Thms 3–4 (DIRT
  composition).
- CDZ23 = `cui-dolgov-zahm-2023-self-reinforced-approx` (arXiv
  2303.02554): §3.3 Hermite on R^d with two caveats — (i) CDF inversion
  at tails (sampling-only), (ii) the tail condition (2.11):
  sup target/weight bounded, i.e. exactly the O7 domination condition —
  with the mapped-basis construction (3.10)–(3.11) as the heavy-tail
  alternative; §4 self-reinforced (composed-KR) preconditioning,
  Prop. 4.1 convergence under a bridging condition.

Caveat disposition (read before adopting, per the literature rule): both
papers' Hermite objections concern operations this program does not
perform. We never invert a fitted CDF (no transport sampling; rows come
from Φ⁻¹∘Sobol against the *reference*, increments and retention are
Gram-chain contractions), so caveat (i) is out of scope. Caveat (ii) is
the domination condition already carried as O7 with the λ escalation
rule (D3). The truncation route both papers warn about is C1, already
eliminated.

## 1. Program objects under C2

Coordinates. Per-step frozen triangular preconditioner from
deterministic hints (M2/M3/M1-DETACHED contracts unchanged):
u = T_t(x) = L_t^{-1}(x − m_t) on R^{2n}; reference measure
μ = N(0, I_{2n}) with density η.

Basis. Normalized probabilists' Hermite per axis,
Hē_k = He_k/√(k!), k = 0..ℓ−1. Mass matrix B = I exactly (ch38
Prop. on Hermite mass identity; CD22/CDZ23 orthonormal-frame analogue).
Three-term recurrence He_{k+1} = u·He_k − k·He_{k−1}; normalized form
Hē_{k+1} = (u·Hē_k − √k·Hē_{k−1})/√(k+1). Derivative recurrence:
He_k' = k·He_{k−1} ⟹ **Hē_k' = √k·Hē_{k−1}** — one shift-and-scale,
same kernel shape as the Legendre engine's derivative path.

Fit target (μ-density at rows):

    F_t(u) = q_t(T_t^{-1}(u)) · |det L_t| / η(u),

with q_t = (transition × observation × retained) as in the bounded
program. Rows: u_i = Φ^{-1}(Sobol) (owen-scrambled, reusing
`EngineConfig.row_design="sobol"`); the row law IS the reference, so
the fit is least-squares in L²(μ) with unweighted rows.

Represented object and increment (unchanged in form):
q̄_t = h_t² + τ_t Z_{h,t} λ_t as μ-density;
Ẑ_t = e^{−c_t}(1+τ_t) Z_{h,t}; with B = I, Z_{h,t} is the plain
Frobenius contraction of the core chain — every mass-matrix
multiplication in the bounded engine's Gram chain drops out.

Retention. CD22 Prop. 2 with M_k = I: step 1 (Cholesky of the mass
matrix) collapses to C_k = B_k; the recursion is core-unfold + thin QR
only. The retained object is stored as (cores, coordinate map
(m_{c,t}, L_{c,t}), c_t, τ_t) exactly as today, but its domain is all
of R^n: re-expression at step t+1, u_old = L_{c,t}^{-1}(x_p − m_{c,t}),
is defined for every row. **No containment constraint exists; nothing
chains; the shrink machinery and the truncation-mass correction are
deleted, not ported.**

Conversion terms (replaces the box program's row-constant ±n log 2 and
log-det constants). The retained μ-density lives in its own u_old
coordinates; entering step t+1's fit target in u_new coordinates:

    log F_{t+1}(u_new) = log g + log f
      + log p_ret(u_old) + log η(u_old) − log|det L_{c,t}|
      + log|det L_{t+1}| − log η(u_new),

where u_old is affine in u_new (composition of frozen affine maps).
Two structural differences from the bounded program, both exact:
(a) the reference-density ratio η(u_old)/η(u_new) is **row-dependent**
(exp of a quadratic in u_new), not a row constant — it joins the target
evaluation like any kernel factor; (b) it is θ-free under frozen maps,
so it adds no adjoint nodes. Consistency check that the review must
verify: in the exactly-Gaussian case this quadratic is precisely what
cancels against f, g, and the constant retained factor to make
F_{t+1} ≡ const (ch38 degree-collapse proposition) — a sign error here
breaks the oracle gate, which is the point of the gate.

Defensive term (D3). Default λ ≡ 1 as μ-density (physical floor
τZ_h·η: Gaussian tails). Coupling τ_t = min(τ_max, ε̂_t²) with ε̂_t the
fit's weighted residual on the fitted rows. Honesty note: in-sample ε̂
typically underestimates the true L²(μ) error (Defect 2 lesson), but
the asymmetry is safe in the direction that matters — the Lemma-1/Prop.
11 bias bound needs τ ≤ ε_true², and ε̂² ≤ ε_true² preserves it; an
underestimate only weakens the floor's stabilization margin, never the
bias order. Escalation: if a declared target family fails the
domination check sup q/λ < ∞ in whitened coordinates, replace λ by a
Student-t product per axis (log-space evaluation; closed-form product
marginals; θ-free) — CDZ23 (2.11) is the same condition, and their
mapped-weight (3.11) is the fallback shape.

Score path. Node inventory relative to the certified bounded adjoint:
mass-matrix contraction nodes drop (B = I); basis and basis-derivative
evaluations swap recurrences (above); the η-ratio conversion is a
θ-free target factor; maps remain frozen (M2/M3/M1-DETACHED as
declared). Expected: the adjoint graph is isomorphic with fewer nodes.
The I-P2-4-style forward-JVP-vs-adjoint fixture (1e-12) must be re-run
under the new basis before any score claim.

## 2. Validation ladder (before any C2 claim)

1. U-HERM-1: mass identity and derivative recurrence against 60-node
   Gauss–Hermite quadrature (already checked to 1.3e-15 at the
   selection stage; promote to a unit test).
2. U-RET-1: retention parity — CD22 Prop.-2-with-identity-mass QR
   recursion vs dense quadrature marginal on a 2-axis fixture (the
   U-MAP-MOM-1 analogue).
3. Conversion-term closure: single-step re-expression identity on a
   Gaussian fixture (the quadratic-cancellation check above), value and
   score.
4. **LGSSM oracle gate** (D2): under exact hints the whitened step
   target is Gaussian and F ≡ const, so the full T = 120 filter must
   reproduce the Kalman log-likelihood to float64 accumulation error
   (gate 1e-8 total, expected far better) at ℓ = 1, r = 1 — and,
   run again at ℓ = 13, r = 6, must not degrade (fit-of-constant
   conditioning check). This gate is a *correctness oracle*, not rank
   evidence: ch38 §40.9 records why r*(LGSSM) is trivial by
   construction under C2.
5. Hermite conditioning diagnostic at n ∈ {4, 8}: design-matrix
   condition number vs the Legendre baseline at matched ℓ, rows from
   Φ⁻¹(Sobol) (tail factor ≤ ~26 at |z| = 4, k ≤ 16, measured).
6. XLA kernel parity: eager-vs-jitted value on the oracle fixture
   (the Legendre engine's 1e-12 gate pattern), fresh-process compile
   battery per the LLVM lesson.

## 3. The SV arm question (declaration only; attempt05 plan will carry
the full evidence contract)

Question: r*(n) — smallest TT rank meeting 2.5e-3 nats/step against a
resolved reference — on the stochastic-volatility family under C2, in
the paper's own log-volatility coordinates (ZC24 transforms SV to
unbounded coordinates for exactly this reason, txt:2156).
Domination pre-check (gate before any run): in whitened log-vol
coordinates the SV step target's tails are Gaussian-dominated — the
transition/prior factor is Gaussian; the observation factor
N(y; 0, e^x) decays like e^{−x/2} as x → +∞ and double-exponentially as
x → −∞, so the product is dominated by the Gaussian prior tails; the
formal statement and its verification belong to the attempt05 plan, and
failing it triggers the λ escalation of D3, not a program change.
Reference ladder: exact-moment oracle unavailable (non-Gaussian), so
the comparator is the resolved-quadrature / long-particle reference per
ch38's Defect-2 ledger rules. Not concluded here: any rank prediction.

## 4. What the review must check (material review scope)

1. The conversion-term derivation of §1 (the single place a sign can
   hide); its Gaussian-cancellation consistency with the ch38
   degree-collapse proposition.
2. The τ-coupling estimator's direction-of-safety argument.
3. The retention-under-identity-mass claim against CD22 Prop. 2 as
   read (steps (27)–(29)): that dropping the Cholesky step is exactly
   M_k = I and nothing else changes.
4. The SV domination sketch (§3) at the level of "gate is well-posed",
   not at the level of the attempt05 contract.
5. That no bounded-program machinery survives by accident (shrink,
   containment assertions, truncation correction, box constants
   ±n log 2) — deletion list to be enumerated at implementation.

Non-claims: no rank statement for any n; no HMC/posterior claim; no
statistical comparison of C2 vs C3 runtimes; the oracle gate, once
green, certifies implementation correctness only.
