# C2 Gaussian-Reference Derivation Note — Independent Material Review — 2026-08-24

Target: `bayesfilter-c2-gaussian-reference-derivation-note-2026-08-24.md`
(status `DRAFT_FOR_REVIEW`). Review scope: the note's own §4, verified by
re-derivation, not narrative. Verdict classifications per the repo's
plain-language rule.

Sources actually inspected (text-extracted local copies, line anchors):

- ch38 §40.8–40.9: Defect 3 (tex:495–679), Reference-Domain Selection
  (tex:680–1128), including the Hermite mass-identity proof
  (tex:763–795), the degree-collapse proposition (tex:957–978), and the
  domination proposition (tex:911–933).
- CD22: squared construction and γ term (txt:690–734); Prop. 2 with
  steps (27)–(29) (txt:847–903); full Appendix B proof (txt:3982–4441);
  §3.2.3 (txt:1130–1146); §3.2.4 (txt:1147–1187).
- ZC24: eq. (13), the ratio display, and Lemma 1 (txt:548–590); their
  Prop.-2 restatement, eq. (14) and either-end marginalization
  (txt:592–620, 1086); Prop. 11 and Cor. 12 (txt:1428–1459).
- CDZ23: defensive form (2.10) and tail condition (2.11) (txt:270–300);
  §3.3 with both Hermite caveats and mapped basis (3.10)–(3.11)
  (txt:1039–1146).
- Selection derivation note 2026-08-23 (whole); adapted-coordinate-maps
  design note 2026-08-20 §1 conversion conventions and §§9–10
  shrink/truncation machinery.

## Item 1 — Conversion-term display and Gaussian cancellation: CORRECT

### Re-derivation of the display

Convention (fixed by the program and by the design note's §1 audited
form): a "retained μ-density in its own coordinates" p_ret on
u_old ∈ R^n means the physical retained density is recovered by the
change-of-variables for densities under the affine map
x_p = m_{c,t} + L_{c,t} u_old:

    π̂_t(x_p) = p_ret(u_old) · η_n(u_old) / |det L_{c,t}|,
    u_old = L_{c,t}^{-1}(x_p − m_{c,t}),

where η_n is the n-dimensional standard normal density. (Equivalently
p_ret(u_old) = π̂_t(x_p) |det L_{c,t}| / η_n(u_old), the same convention
the note's §1 fit-target line declares for F_t.)

The step-(t+1) fit target is, by the note's own definition,

    F_{t+1}(u_new) = q_{t+1}(T_{t+1}^{-1}(u_new)) · |det L_{t+1}| / η_{2n}(u_new),
    q_{t+1}(x_c, x_p) = f(x_c|x_p) · g(y_{t+1}|x_c) · π̂_t(x_p).

Substituting the retained convention into q_{t+1}:

    log F_{t+1}(u_new)
      = log f + log g
        + [ log p_ret(u_old) + log η_n(u_old) − log|det L_{c,t}| ]
        + log|det L_{t+1}| − log η_{2n}(u_new).

This is exactly the note's display, term for term: +log η(u_old),
−log|det L_{c,t}|, +log|det L_{t+1}|, −log η(u_new). Every sign and
every |det L| placement checks. u_old is affine in u_new (previous-block
subvector of m_{t+1} + L_{t+1}u_new composed with the frozen retained
map) — claim (a) row-dependent quadratic ✓; the whole conversion factor
contains only frozen map constants — claim (b) θ-free ✓ (under the
declared M1-DETACHED retained detachment; a θ-carrying retained object
would be a different program).

Cross-check against the convention replaced: the bounded design note's
audited previous-block conversion "+ log|det L_p| − log|det L_{map,t−1}|"
is the same two-leg chain (retained μ_old → physical → μ_new) with both
reference densities uniform, so the 2^{±n} and box-volume factors cancel
to row constants; under C2 the two legs' Gaussian densities do not
cancel row-wise and survive as the η(u_old), η(u_new) terms. The note's
"replaces the box program's row-constant ±n log 2 and log-det constants"
is an accurate description of that difference.

### Gaussian-cancellation consistency: holds, no sign error

Exactly-Gaussian step with exact-moment maps. At step t the retained
object collapses (ch38 degree-collapse proposition, whose proof I
checked: N(m + Lu; m, Σ)|det L| = η(u) because LᵀΣ^{-1}L = I and
|det L| = |det Σ|^{1/2}): p_ret ≡ Z_t constant. The defensive term does
not disturb this: λ ≡ 1 as μ-density marginalizes to 1 under the product
reference, so zero-fit-error retention gives (1 + τ_t)Z_t, still
constant. Then in the display:

    log η_n(u_old) − log|det L_{c,t}| turns p_ret's conversion legs into
    log N(x_p; m_{c,t}, Σ_{c,t})  (retained Gaussian reconstructed),

    log f + log g + log N(x_p; ·) = log p(y_{t+1}|y_{1:t})
                                    + log N(x; m_{t+1}, Σ_{t+1})
    (LGSSM joint exponent is quadratic; exact one-step Bayes algebra),

    log N(x; m_{t+1}, Σ_{t+1}) + log|det L_{t+1}| − log η_{2n}(u_new) = 0
    (degree-collapse identity again, exact maps),

leaving log F_{t+1} = log[(1+τ_t)Z_t] + log p(y_{t+1}|y_{1:t}),
constant in u_new. I verified the same chain explicitly in the scalar
LGSSM (f = N(x_c; a x_p, σ_f²), g = N(y; c x_c, σ_g²)); every constant
lands in the marginal likelihood, which is exactly what the T = 120
oracle gate accumulates against the Kalman recursion. The row-dependent
η-ratio quadratic is precisely the previous-block Gaussian part, as the
note asserts.

Minor notation (no error): η is used at two dimensions in one display
(η(u_old) is η_n, η(u_new) is η_{2n}); worth one clause. See also F6 on
the e^{−c_t} placement.

## Item 2 — τ-coupling direction of safety: core claim CORRECT as
conditioned; the τ → 0 endpoint is a material gap (F2)

Lemma 1 (ZC24 txt:578–586) hypothesis: τ ≤ ‖φ − √π‖²_{L²}. Derivation
of the bound it buys (reconstructed, since the direction matters): for
a, b, c ≥ 0, (√(a+b) − √c)² = a + b + c − 2√((a+b)c) ≤ (√a − √c)² + b
because √((a+b)c) ≥ √(ac). With a = h², b = τ Z_h λ, c = q̄-target,
integrating over μ:

    ‖√(h² + τZ_hλ) − √q‖²_{L²(μ)} ≤ ε_true² + τ Z_h,   ε_true = ‖h − √q‖.

(In the paper's absolute normalization, ≤ ε_true² + τ.) So:

- Guaranteed under ε̂ ≤ ε_true: τ = min(τ_max, ε̂²) ≤ ε_true² preserves
  the Lemma-1 hypothesis; the L² error keeps the √2 factor
  (≤ √2 ε_true) and the ch38 mixture bias TV(p̂, p) ≤ τ/(1+τ) ≤ ε_true²
  stays second order. The note's direction-of-safety sentence is
  correct for the bias order: smaller τ can only shrink the defensive
  contribution. Prop. 11 (txt:1434–1449) is a direct restatement and
  adds nothing that changes this.
- NOT guaranteed, and understated by "weakens the stabilization
  margin": there is no lower bound on τ. The stabilization guarantee is
  the ratio display sup p/p̂ < (ẑ/(τz)) sup q/λ — it scales as 1/τ and
  is vacuous at τ = 0. ε̂ = 0 is reachable: exactly representable
  targets (the oracle gate at ℓ = 1 — harmless there since h is a
  positive constant), interpolating fits, and precisely the Defect-2
  scenario the note itself cites (in-sample residual near zero while
  the true L²(μ) error is large). In that last case the coupling turns
  the floor OFF in exactly the situation the floor exists for: h can
  have zeros where q > 0 with no absolute-continuity guarantee — the
  −inf log-target / unbounded-ratio mode returns. Under the repo's
  Safety Guardrail Reversed Burden rule, a reachable zero-protection
  setting requires a principled justification or a clamp; the note
  records neither a τ_min nor a justification, and τ_max has no stated
  value or provenance. This is finding F2 (blocking).
- Also not guaranteed: ε̂ ≤ ε_true itself. In-sample optimism is
  systematic (the fit minimizes the empirical residual), not a
  pointwise inequality; a realization can overestimate. The overestimate
  direction is bias-harmless — the bound above gives
  ‖√q̄ − √q‖² ≤ ε_true² + ε̂², second order in the measured quantity —
  but then the paper's bound holds in ε̂, not ε_true; one sentence
  should say so (F4).
- Normalization gap (F3): Lemma 1's τ is absolute
  (π̂ = φ² + τλ); the engine's convention is relative
  (q̄ = h² + τ Z_h λ, i.e. τ_abs = τ_rel Z_h). The hypothesis
  τ_abs ≤ ‖h − √q‖²_abs therefore requires ε̂ to be the
  Z_h-normalized (relative) residual: τ_rel = ε̂_rel² with
  ε̂_rel² ≈ ‖h − √q‖²/Z_h. The note says only "the fit's weighted
  residual on the fitted rows" — under-specified; the selection note
  carried the rescaling remark, this note dropped it.

## Item 3 — Retention under M_k = I vs CD22 Prop. 2: CORRECT

Call-chain check of the recursion as printed (txt:847–891) and proved
(txt:3982–4441): the mass matrix M_k enters at exactly one site. In the
proof, the marginalization over x_k produces the Gram matrix
𝕄_k[α,β] = Σ_ℓ ∫ P_k^{(α,ℓ)}P_k^{(β,ℓ)} λ_k dx_k, which reduces via
(83) to B_k-contractions against M_k[i,j] — the only appearance of the
basis integrals. The Cholesky L_k L_kᵀ = M_k then defines
C_k = B_k ×_2 L_k in (27); the thin QR (28) and the A_{k−1}×R_k
contraction (29) involve no basis integral at all. With the normalized
Hermite basis, M_k = I exactly at every degree (ch38 mass-identity
proposition; I checked the generating-function proof:
∫G(z,s)G(z,t)φ(z)dz = e^{st} via the normal MGF, coefficient matching
gives ∫He_j He_k dμ = δ_{jk} k!), the Cholesky factor of I is uniquely
I, hence C_k = B_k and step 1 degenerates to the identity — the
recursion is core-unfold + thin QR + (29), exactly as the note claims.

What Prop. 2 assumes, checked against C2: (i) a product weight
λ(x) = ∏ λ_k(x_k) — μ = ∏ N(0,1) ✓; (ii) per-axis SPD mass matrix
(txt:4189) — orthonormality gives I ✓; (iii) basis square-integrability
under λ_k — polynomials under the Gaussian weight ✓; (iv) the
marginalization weight equals the fit weight axis by axis — both are
the standard normal ✓. The proof is domain-generic (all integrals over
X_k with weight λ_k); §3.2.3 explicitly contemplates Hermite on
(−∞, ∞), and the Hermite caveat stated there attaches to CDF
computation/inversion for the transport, not to the marginalization
recursion. The defensive bookkeeping γ∏_{i>k} λ_i(X_i) in (23)–(24) is
weight-normalization-invariant (each factor is 1 for a probability
weight), and ZC24's restatement (14) plus txt:1086 confirms either-end
marginalization at O(ℓr³), so retaining the current block is within the
source recursion. Nothing else in the recursion changes. The note's
per-axis recurrences also check: Hē_{k+1} = (u Hē_k − √k Hē_{k−1})/√(k+1)
and Hē_k' = √k Hē_{k−1} both follow from He_{k+1} = uHe_k − kHe_{k−1},
He_k' = kHe_{k−1} by dividing through by √((k+1)!) and √(k!).

## Item 4 — SV domination sketch: gate well-posed; the sketch is WRONG
relative to the stated gate (F1)

Well-posedness: the gate is sup_u F_{t+1}(u)/λ_μ(u) < ∞ in whitened
coordinates with the declared λ (default λ_μ ≡ 1, i.e. the reference
density) — a well-defined condition once coordinates, hints, and λ are
fixed. Well-posed ✓.

Asymptotics check: N(y; 0, e^x) = (2π)^{−1/2} exp(−x/2 − y²e^{−x}/2).
As x → +∞ this tends to (2π)^{−1/2}e^{−x/2}: the e^{−x/2} claim is
correct. As x → −∞ it decays double-exponentially for y ≠ 0; at y = 0
it GROWS like e^{|x|/2} (density of a collapsing Gaussian at its
center). The stated asymptotic needs a y ≠ 0 qualifier (F5); the
product remains Gaussian-dominated at y = 0 via the transition factor,
so only the characterization, not the conclusion, fails there.

The material flaw: "the product is dominated by the Gaussian prior
tails" is true and irrelevant to the gate. The gate compares the target
to the whitened reference η, not to a prior-variance Gaussian, and
those two have different tail variances. Derivation along the current
block, x_p fixed, x_c → +∞:

    log q_{t+1} = −(x_c − a)²/(2σ_f²) − x_c/2 + O(1)
                = −(x_c − (a − σ_f²/2))²/(2σ_f²) + O(1),

a Gaussian tail with exactly the transition variance σ_f² (the
observation factor is asymptotically log-linear, so it shifts the mean
and cannot narrow the variance; the retained factor is O(1) in x_c).
The reference along the same ray:

    log η(T_{t+1}(x)) = −(x_c − b)²/(2s²) + O(1),
    s² = 1/[Σ_t^{−1}]_{cc} = hint conditional variance of x_c.

Hence

    log F_{t+1} = (1/s² − 1/σ_f²) · x_c²/2 + O(x_c),

and sup F = ∞ whenever s² < σ_f². Exact-moment hints give s² < σ_f²
generically: ∂²/∂x_c² log g = −y²e^{−x_c}/2 ≤ 0, so the observation
factor is log-concave in x_c and Brascamp–Lieb gives
Var_q(x_c|x_p) ≤ σ_f², strict where y is informative; the Gaussian
hint's conditional variance is the best-linear residual
E_q[Var_q(x_c|x_p)] plus a second-order nonlinear-mean term. The
mechanism is structural: the SV observation narrows the bulk but its
log-density flattens in the +x_c tail, so the target's tail variance
(σ_f²) exceeds its bulk conditional variance (what the hint whitens
by), and the reference under-covers the tail. Conclusion: with the
default λ ≡ 1 the domination pre-check should be EXPECTED TO FAIL for
SV under exact or near-exact hints — the note's sketch predicts the
opposite. Wrong relative to the stated gate.

Consequences and repairs (both closed-form, neither in the note as the
expected path): (i) the D3 Student-t escalation dominates any Gaussian
tail — under this finding it is the expected default for the SV arm,
not a contingency; (ii) alternatively a hint-variance floor at the
transition variance along x_c (with predict-variance hints, s² = σ_f²
and log F = −x_c/2 + O(1), bounded) restores domination at a
degree-collapse cost that would need its own audit. Named as open for
the attempt05 formal statement (not settled here, either way): the
retained floor term τZ·1 (w.r.t. μ_old) injects an old-reference
Gaussian factor into q_{t+1}; conditional screening
(y_{t+1} ⊥ x_p | x_c) suggests the previous-block direction may be
benign, but the formal domination statement must cover it. This finding
does not touch the C2 selection itself — O7 remains satisfiable by the
λ choice, exactly as ch38's domination proposition says — but the note
must not hand attempt05 an inverted expectation.

## Item 5 — Surviving machinery and unexamined defaults

Checked the C2 note against the bounded design note's machinery
inventory (shrink/containment §§9–10, truncation-mass correction
V_phys, center/slack logic, ±n log 2 constants, containment
assertions): all are explicitly deleted or replaced in the C2 note; I
found no silent survival in the note's text. The deferral of the
concrete deletion list to implementation is acceptable and declared
(§4.5). Declared carryovers (M2/M3/M1-DETACHED contracts, sobol row
design, U-MAP-MOM-1/I-P2-4-style fixtures) are appropriate and marked.

Unexamined defaults found (all in the D3 paragraph): τ_max with no
value or provenance; the absence of a τ_min (the reachable-zero
protection choice — Class C burden, part of F2); the definition and
normalization of ε̂ (F3). The selection note's flagged audit item
(fixed τ vs paper coupling) is answered by D3's coupling — modulo F2.

## Caveat disposition: CORRECT against the papers as read

- CD22 §3.2.3 (txt:1130–1146): truncation "may lead to a biased
  estimator" — quoted faithfully. The Hermite caveat there is the cost
  of CDF evaluation (error/erfc/erfi functions approximated
  numerically) and monotonicity/uniqueness of the inverse-CDF solution
  at the tails — operations of the sampling transport. The program
  inverts only the exact reference CDF (Φ^{-1}∘Sobol), never a fitted
  CDF; retention is Gram-chain contraction. Out of scope as claimed.
- CDZ23 §3.3 (txt:1039–1146): caveat (i) is "inverting the distribution
  function ... numerically challenging towards the tails" —
  sampling-only ✓; caveat (ii) is the tail condition (2.11)
  sup f_X/λ < ∞ (txt:282–289) — exactly the O7 domination condition ✓,
  with the mapped basis (3.10)–(3.11) as the heavy-tail alternative ✓.
  Note that finding F1 is this caveat materializing for SV with λ = η:
  the note carries the condition correctly and mispredicts its outcome.

## Findings by severity

- **F1 (major, blocking).** §3 domination sketch: wrong relative to the
  stated gate. Domination by a prior-variance Gaussian does not imply
  sup F < ∞ against the whitened reference; under exact-moment hints
  the ratio diverges along +x_c (derivation above), so the λ ≡ 1
  pre-check should be expected to fail for SV and the Student-t
  escalation is the expected outcome. Repair: re-scope the sketch's
  conclusion and record the expected-escalation (or hint-inflation)
  path before attempt05 inherits it.
- **F2 (major, blocking).** §1 D3 coupling τ = min(τ_max, ε̂²) has no
  lower clamp: ε̂ = 0 (reachable, and precisely in the cited Defect-2
  in-sample-optimism scenario) turns the defensive floor off entirely —
  absolute continuity and the ratio guard vanish, not "weaken". Under
  the repo's Class-C zero-protection rule this needs a τ_min with
  provenance or a recorded justification; τ_max also needs a value and
  provenance. The direction-of-safety claim for the bias order stands;
  the safety story for the guard's declared function does not.
- **F3 (minor).** ε̂'s normalization is unstated; under the engine's
  relative-τ convention the Lemma-1 hypothesis transfers only if ε̂ is
  the Z_h-normalized residual (τ_abs = τ_rel Z_h ≤ ‖h − √q‖²_abs).
- **F4 (minor).** ε̂ ≤ ε_true is systematic-typical, not guaranteed;
  state that in the overestimate event the bound holds with ε̂ in place
  of ε_true (bias still second order in the measured quantity).
- **F5 (minor).** The double-exponential x → −∞ claim fails at y = 0
  (growth e^{|x|/2}); needs the y ≠ 0 qualifier. Conclusion unaffected.
- **F6 (nit).** η used at dimensions n and 2n in one display without
  remark; §1's represented-object line drops e^{−c_t} from q̄_t while
  Ẑ_t retains it, so Ẑ_t = ∫q̄_t dμ fails as displayed (the selection
  note's convention q̄_t = e^{−c_t}(h² + τZ_hλ) is the consistent one).

Verified correct alongside: item 1's display and cancellation, item 3's
retention claim, the Hermite mass identity and both recurrences, the
caveat disposition, and the note's non-claims discipline (no rank, no
HMC, no C2-vs-C3 runtime claims).

VERDICT: DISAGREE — blocking findings F1 (SV domination sketch wrong
relative to the stated gate) and F2 (τ-coupling reaches τ = 0 with no
clamp or justification). Both are repairable by small edits to §3 and
the D3 paragraph; items 1 and 3, the load-bearing derivations, stand.

## Re-verification of repairs (2026-08-24, commit 37d94faa)

Checked the full diff 90cd98c5 → 37d94faa against each finding.

- **F1 discharged.** §3 now states the gate against the whitened
  reference, carries the tail-variance derivation correctly (σ_f² tail
  vs s² ≤ σ_f² whitening, strictness via log-concavity/Brascamp–Lieb;
  the log F display matches this review's), declares the λ ≡ 1
  pre-check expected-to-fail for SV, promotes the Student-t escalation
  to the expected SV configuration, marks the hint-variance-floor
  alternative as requiring its own audit, and assigns the
  retained-floor coverage question to attempt05.
- **F2 discharged as a safety design.** τ_t = clamp(ε̂_t², 1e-6, 1e-4)
  removes the reachable-zero endpoint; the Class-C justification is in
  the acceptable form (inherited measured no-harm evidence plus an
  explicit bound — the cited 46× n=2 margin figure is not independently
  re-checked here, but the sub-τ_min mixture bound τ/(1+τ) ≤ 1e-6 is
  exact algebra independent of it, and declining the Lemma-1 √2 claim
  in that regime is correct: the hypothesis τ ≤ ε_true² can fail there
  while the mixture bound cannot). τ_max = bar/25 now has value and
  provenance.
- **F3, F4, F5, F6 discharged.** Relative-residual definition with the
  τ_abs = τ_rel·Z_h transfer ✓; overestimate direction bounded via
  ‖√q̄ − √q‖² ≤ ε_true² + τ_abs with no ε_true guarantee claimed ✓;
  y ≠ 0 qualifier with the y = 0 growth behavior and transition-factor
  control ✓; e^{−c_t} restored so Ẑ_t = ∫q̄_t dμ holds as displayed,
  η dimension clause added ✓.

**New finding F7 (moderate, blocking until reconciled), introduced by
the F2 repair.** The clamp makes ladder item 4 arithmetically
unsatisfiable as written. At the oracle (exact fit, ℓ = 1) ε̂² ≈ 0, so
the clamp binds: τ_t = τ_min = 1e-6 at every step. The increment is
Ẑ_t = e^{−c_t}(1 + τ_t)Z_{h,t}, and the program's documented
likelihood convention includes the defensive mass (the selection note's
O2 budgets T·τ ≈ 1.2e-4 as a real bias of the reported likelihood).
Hence the T = 120 accumulated log-likelihood carries
Σ_t log(1 + τ_min) = 120·log(1 + 10⁻⁶) ≈ 1.2×10⁻⁴, which exceeds the
declared 1e-8 gate by four orders — a single step's 10⁻⁶ already does.
The pre-repair coupling gave τ = 0 at the oracle, so this
inconsistency did not exist before. One-sentence repair, in order of
preference: (i) declare the oracle comparison as
Σ_t (log Ẑ_t − log(1 + τ_t)) — exact closed-form algebra, tests the
h-chain at machine precision while exercising the production clamp
path; or (ii) declare the gate's τ configuration a named
smoke/reference exception (weaker: the gate then never exercises the
clamp). Until §2.4 says which, the note's declared correctness oracle
cannot pass under the note's declared default policy — wrong relative
to the stated target, internal to the note.

VERDICT: DISAGREE (after repairs, 2026-08-24) — F1–F6 discharged; F7
alone blocks, and a one-sentence reconciliation of ladder item 4 with
the τ_min clamp discharges it.
