# Adapted Coordinate Maps (Linear Bridging) — Design Note — 2026-08-20

Status: `DRAFT_FOR_REVIEW` (program-contract change: per-step coordinate
maps; touches both engines and the retention pipeline. Engine work does
not start until this note's contract is stable; the step-level evidence
already exists in `bayesfilter-row-design-v2-proposal-rows-note-2026-08-19.md`
Section 3b.)

Paper anchors (read from `.localresources/papers/zhao-cui-...jmlr-2024`):
Section 5.1 Eqs. (30)-(33) (bridging density rho_t, KR map T_t, TT fits
the ratio pushforward q_{#,t}; composed pullback with |grad(S∘T)|);
Section 5.2 (Gaussian bridging => affine lower-triangular T_t from
estimated moments); Fig. 2 (rank collapse under preconditioning).
This note implements the LINEAR (5.2) instance only; the nonlinear
tempered instance (5.3, DIRT-style layering) is explicitly deferred.

## 1. What changes in the declared program

Today both engines hard-code the identity-scaled map
x = hw * z on every axis at every step (single global box). v1 of this
contract replaces that with PER-STEP, PER-BLOCK frozen affine maps:

    step t transition fit:  x_c = m_c,t + L_c,t z_c   (current block)
                            x_p = m_p,t + L_p,t z_p   (previous block)

with (m, L) computed from ALREADY-AVAILABLE state before the step's
rows are drawn (V1-compliant: no adaptation inside the step; the maps
join seeds as frozen step inputs and enter the scope identity).

Block-diagonal L only (v1): the retention split z_c | branch | z_p must
remain an axis split, so cross-block correlation stays in TT rank, as
in the probe. Full 2n-triangular maps (paper's L_t on the joint vector)
would rotate the split and are out of v1 scope.

The fit target's conversion terms generalize (A14 machinery), stated in
the REFERENCE-TYPED convention the engine actually uses (the previous
block's retained density is a reference-measure density in its OWN
coordinates, so it does NOT get a physical-density conversion; the
current engine's conversion is n(log hw + log 2), current block only):

    current block (physical kernels -> mu_new):  + log|det L_c| + n log 2
    previous block (retained mu_old -> mu_new):  + log|det L_p| - log|det L_map,t-1|

where L_map,t-1 is the retained object's own stored map matrix (for the
present identity-scaled program, log|det| = n log hw — recovering the
probe's validated constant log|det L_c| + log|det L_p| + n log 2
- n log hw). Both terms are row-constant for affine maps and enter
log f_ref before the smooth shift; the end-to-end account is validated
by the probe (Section 3b arm, -0.30 vs Kalman). The
RetainedQuadraticForm for step t stores coordinate_map_t =
(m_c,t, L_c,t) — the type already carries a coordinate_map field and
dual evaluators, so retention needs no type change, only non-identity
map values.

AUDIT NOTE (2026-08-20 pre-implementation review): the original
draft of this section wrote "+ 2n log 2, replacing 2n(log hw+log 2)",
which silently switched the previous block to the physical evaluator
and misstated the baseline conversion; corrected above to the
reference-typed form matching the engine and the validated probe.

## 2. The one nontrivial coupling: previous-block re-expression

The retained object from step t-1 is a density in ITS OWN z-coordinates
(map_t-1). At step t the previous-block rows are drawn in the NEW
z_p-coordinates. The prefix TT must therefore be evaluated at

    z_old = map_{t-1}^{-1}(x_p) = L_{c,t-1}^{-1} (x_p - m_{c,t-1}).

Two regimes:
- z_old inside [-1,1]^n: exact evaluation (affine-in-affine; no
  approximation introduced).
- z_old OUTSIDE the old box: the retained polynomial has no declared
  meaning there. v1 contract: the new previous-block map must be chosen
  INSIDE the old box image (shrinkage constraint below), and a
  fail-closed assert rejects steps whose row image leaves the old box
  by more than a declared epsilon (1e-12 slack for roundoff). This
  REMOVES the probe's clip impurity by construction instead of hiding it.

Shrinkage constraint: choose (m_p,t, L_p,t) as the moment box of the
PREVIOUS step's retained law intersected with the previous box: i.e.
kappa-sigma box clipped into map_{t-1}([-1,1]^n). For LGSSM-family
fixtures at kappa=3 this is non-binding after t=1; for heavy-tailed
retained laws it binds and the intersection is the declared behavior.

## 3. Moment sources (scope-identity field `map_moment_source`)

- M1 `retained_exact` (v1 default): mean/cov of the retained law
  computed EXACTLY from the RetainedQuadraticForm (quadratic-form
  moments are Gram-chain integrals of the same kind as Z_h; one
  derivation lemma + pairing test required). Current-block moments from
  a declared one-step predictive inflation: m_c = A-image handled
  model-free as m_p propagated through NOTHING — v1 uses the SAME
  moments for the current block inflated by a declared factor
  (kappa_c >= kappa_p), because the engine is model-free (no access to
  A). This is the weakest point of v1; the probe used exact filtered
  moments, so M1's degradation must be measured before adoption
  (probe arm M1 below).
- M2 `adapter_hint`: optional adapter callback returning per-step
  (m, L) hints (e.g. from a cheap EKF/UKF companion filter — the
  paper's own 5.2 uses a particle estimate). Fail-closed validation:
  hints must be finite, L PD, and produce in-old-box rows.
- M3 `fixed`: the current global box (backward compatibility; exactly
  reproduces today's program — the n=2 regression guard).

## 4. Score-path consequences (P2 adjoint engine)

The maps are FROZEN per step, so they carry NO theta-derivative terms
when moments come from M3 or M2-with-frozen-hints. For M1, the moments
depend on theta THROUGH the retained object of step t-1 — the exact
score then needs d(m,L)/dtheta terms (moment tangents of the retained
law: same Gram-chain tangent machinery as dot_Zh; derivable, but new
adjoint nodes). v1 DECISION: claim-bearing score runs use M2/M3 or
M1-DETACHED (maps treated as frozen constants; the resulting score is
exact for the DECLARED program "filter with maps as given", which is
the same epistemic status as frozen seeds). Full M1 differentiation is
deferred to its own derivation note (UB-1 style) if HMC campaigns need
retained-adaptive maps. This choice is recorded plainly: with
M1-DETACHED the score is exact for a program in which the maps are
data, not functions of theta — same contract as v0's frozen rows.

## 5. Validation ladder (before any n>=4 claim)

1. Unit: map round-trip, Jacobian conversion (int 1 dmu = 1 under
   composed maps), retained-moment lemma pairing test (M1).
2. Step probe arm M1: rerun the 3b probe with M1 moments instead of
   Kalman moments — quantifies the model-free degradation.
3. n=2 regression: M3 must reproduce the current engine bit-identically
   (same maps => same program); M1/kappa=3 must pass the existing n=2
   smoke gate (2e-2) and ladder tolerance at r=6.
4. n=4 smoke: single seed, r=6, Sobol 8192 under M1 — success bar for
   the DIRECTION at engine level: per-step |err| <= 0.1 (probe residual
   -0.30 was WITH clip impurity; the engine version removes it).
5. Only then: ladder attempt04 plan amendment (n in {2,4}, declared
   budgets, three seeds) for a real r*(4).

## 6. Non-claims

No posterior/HMC claim; no claim the affine instance suffices for
non-Gaussian-family targets (the paper's 5.3 tempering exists precisely
because it often does not); no n=8 claim (coverage decay may need the
nonlinear instance there); M1's model-free current-block inflation is a
hypothesis until arm M1 measures it.

## 7. Probe arm M1 RESULTS (2026-08-20) — contract decision

Rung 1 (moment lemma) GREEN: `retained_moments_tf.py` +
U-MAP-MOM-1 (chain vs dense quadrature, 1e-10, 2/2 passed).

Rung 2 (`/tmp/rowdesign_v2_probe.log`, t=1 step, n=4):

| arm | moments | err | ESS | clip |
|---|---|---|---|---|
| affine-k3 | exact Kalman | -0.298 | 174 | 3.4% |
| m1-k3-infl1.5 | retained + inflate 1.5 | -0.876 | 31 | 2.7% |
| m1-k3-infl2.0 | retained + inflate 2.0 | -2.051 | 6 | 2.7% |

READING: M1's PREVIOUS-block moments are sound (clip 2.7%), but the
model-free CURRENT-block heuristic (same center, inflated) is the
binding defect: the true current-block target center is shifted by the
dynamics and pulled toward y_t, which "same center + inflation" cannot
express — inflation only dilutes (ESS 31 -> 6, err back to uniform
levels at 2.0). M1-as-drafted is 3x worse than exact moments though
still 2x better than uniform.

DECISION (measured, per Section 3): M2 `adapter_hint` is PROMOTED to
the v1 default for the CURRENT block (a cheap companion-filter
predictive moment — the paper's own 5.2 route); M1 `retained_exact`
remains the v1 default for the PREVIOUS block (measured sound, and the
retained object is the authoritative source there). The all-M1 route
is retained as a fallback labeled with this measured caveat. The
moment-source scope field becomes per-block:
`map_moment_source = {prev: retained_exact | fixed, curr: adapter_hint | fixed}`.

Next: engine implementation per Sections 1-2 with this per-block
contract; validation ladder rungs 3-5 unchanged.

## 8. Pre-implementation skeptical audit (2026-08-20)

Checked before code:
- CONVERSION TERM DEFECT FOUND AND FIXED (Section 1 audit note): the
  draft's "+2n log 2" formula silently applied a physical-density
  conversion to the previous block, whose retained density is
  reference-typed in its own coordinates; corrected to the per-block
  form that reproduces the probe's validated constant.
- Normalizer semantics: Z_h and the relative tau_abs = tau * Z_h_prev
  are defined w.r.t. each step's OWN reference measure; the maps change
  the measure via the explicit conversion terms only — no silent
  renormalization. The t=0 step keeps the global box (its target is
  not concentrated; localization evidence).
- v1 scope guard: adapted mode is defined for scattered rows only;
  quadrature_order + adapted is rejected fail-closed (diagnostic
  tensor-grid rows would need their own box logic — out of scope).
- Rung-3 bit-identity: map_mode="fixed" must keep the EXISTING code
  path untouched (a generalized path with identity maps would change
  float op order); adapted mode is a separate branch.
- M2 hint contract: optional adapter field
  `predictive_moment_hint(y_t) -> (m_c, L_c)`; fail-closed checks
  (finite, PD L); ladder fixtures supply it from a companion Kalman
  filter in the benchmark script — model knowledge lives in the
  BENCHMARK (which owns the model anyway), not the engine.
- Score path untouched in this change (value engine first; the adjoint
  engine port follows the same-scalar rule V5 in a follow-up with its
  own parity gate — v1 claim scope is VALUE evidence only).
AUDIT VERDICT: proceed.

## 9. Rungs 3-4 results, kappa grid, and v2 TRIANGULAR escalation (2026-08-20, owner-approved)

Rung 3 (n=2): 4.2-4.5e-3/step vs the 2.5e-3 bar — formal miss with
BENIGN telemetry (shrink 1.000 everywhere, z_old_max <= 0.99, per-step
errors at the fixed engine's own noise scale on this fixture seed).
Classified: needs a seed-matched fixed-vs-adapted comparison before
being called noise; NOT a validated regression, NOT a pass.

Rung 4 (n=4) round 1 (kc=kp=3): 0.73/step, shrink 0.74-0.83 (containment
truncation). Round 2 (kc=4, kp=3): 0.82/step (coverage dilution, the
probe's kappa=4 signature). Kappa grid (single seed):

| kc / kp | per-step | min shrink |
|---|---|---|
| 3.0 / 2.5 | 0.715 | 0.844 |
| 3.5 / 2.5 | 0.702 | 0.858 |
| 3.5 / 3.0 | 0.808 | 0.720 |
| 3.0 / 3.0 | 0.729 (round 1) | 0.737 |

VERDICT: no kappa seam; the block-diagonal affine instance floors at
~0.7/step at n=4 (still ~3x better than fixed maps; ~7x short of the
bar). The binding structure: the step target's cross-block correlation
(x_c ~ A x_p ridge) cannot be rotated away per-block, so the previous-
block box must cover the FULL x_p marginal while the target visits only
the slice conditioned on x_c — per-slice coverage collapses again.

### v2: joint lower-triangular map (paper 5.2's actual form)

Map, ordered CURRENT block first:

    x_c = m_c + L_cc z_c
    x_p = m_p + L_pc z_c + L_pp z_p        (L joint lower-triangular)

Derivation points (amendment-level; full write-up follows in the
implementation docstring):
1. RETENTION SPLIT SURVIVES: mu stays a product over z-axes; suffix
   marginalization over (branch, z_p) is the same exact Gram
   contraction; the retained object's own coordinate map is
   (m_c, L_cc) — affine in z_c alone. This was the reason triangular
   was deferred in Section 1; the current-block-first ordering
   dissolves it.
2. CONVERSION: block-triangular Jacobian |det L| = |det L_cc||det L_pp|;
   for fixed z_c the x_p slice is affine in z_p with constant L_pp, so
   the previous-block conversion keeps the Section 1 audited form with
   L_pp in place of L_p; current-block term unchanged with L_cc.
3. CONTAINMENT: z_old = L_old^{-1}(x_p(z_c, z_p) - m_old) now depends
   on both blocks; the closed-form shrink generalizes with transfer
   T = L_old^{-1} [L_pc | L_pp] (n x 2n row-sums) and the same
   center/slack logic; shrink scales (L_pc, L_pp) jointly (the
   conditional structure is preserved under joint scaling).
4. MOMENT SOURCE (M2-joint): the hint returns the JOINT (2n) mean and
   covariance of (x_t, x_t-1) — the paper's own 5.2 object (they
   estimate it with a particle step; the benchmark's companion Kalman
   supplies the exact filtered joint with lag-one cross-covariance).
   Cholesky of the joint covariance in (c, p) order yields
   (L_cc, L_pc, L_pp) directly. Retained moments remain a telemetry
   cross-check on the p-marginal.
5. WHAT THIS BUYS: for the LGSSM fixture the conditional
   x_p | x_c has covariance = Sigma_pp - Sigma_pc Sigma_cc^{-1} Sigma_cp,
   much smaller than Sigma_pp; the z_p box now tracks the slice, which
   is exactly the collapsed direction identified above.

Scope-identity: map_form in {blockdiag_affine, triangular_affine};
kappa fields as before. Non-claims: single-seed evidence chain
unchanged; no claim triangular suffices at n=8 (5.3 nonlinear remains
the next rung of the paper's hierarchy).

## 10. v2 triangular results (2026-08-21): fits SOLVED, containment truncation now the binding defect

Rung 3 (n=2): per-step 1.10e-2 — worse than block-diagonal's 4.2e-3 —
but with fit rms COLLAPSED 40x (1.9e-4..6.9e-4 vs 1.4e-2..3.7e-2): the
triangular coordinates essentially solve the representation problem at
n=2. Decisive telemetry: the 32768-row and 8192-row cells give the
IDENTICAL total gap (8.763e-2 to four digits) — the residual error is
ROW-INDEPENDENT, i.e. a deterministic bias, not sampling/fit error.
Shrink 0.84..0.96 (the L_pc row-sums now trigger containment even at
n=2, where block-diagonal had shrink 1.0).

Rung 4 (n=4): 0.61/step (vs 0.70 block-diagonal floor); step 1 reaches
-0.12 (near the bar) but steps 2+ sit at -0.5..-0.96 with shrink
0.65..0.90 and fit rms still 0.16..0.21.

MECHANISM (now isolated): the containment shrink truncates the
integration box below the target's support; the mass outside the
shrunken kappa-sigma polytope is lost DETERMINISTICALLY from every
increment (hence row-independence). The chained-box constraint is the
driver: each step's previous-block box must fit inside the LAST step's
current-block box, so shrink compounds pressure backward.

WHY THIS IS GOOD NEWS: under triangular coordinates, box width and
target coverage are DECOUPLED (the conditional is tracked by L_pc; a
wider box costs only polynomial resolution, which the 40x fit-rms
collapse shows is now cheap). The obvious next arm — impossible under
block-diagonal, natural now — is WIDER kappas: kappa_c large enough
that next-step containment never binds, kappa_p wide enough to cover
the tails. Predicted signature if correct: shrink -> 1.0, the
deterministic bias -> Gaussian tail mass of the un-shrunk box
(~1e-3-scale at kappa 4), fits stay resolved.

Next arm: triangular kappa sweep (kc, kp) in {(5,4), (6,4), (6,5)} at
n=4 plus one n=2 cell — if shrink releases and the row-independent bias
collapses, rung 3 AND rung 4 both close with tuned-wide kappas.

## 11. Wide-kappa arm REFUTED: the chained-box fixed point (2026-08-21)

| kc / kp (triangular, n=4) | per-step | min shrink | EFFECTIVE kp (shrink*kp) |
|---|---|---|---|
| 3.0 / 3.0 | 0.61 | ~0.72 | 2.2 |
| 5.0 / 4.0 | 0.87 | 0.500 | 2.0 |
| 6.0 / 4.0 | 1.23 | 0.466 | 1.9 |
| 6.0 / 5.0 | 1.38 | 0.406 | 2.0 |

Section 10's prediction is REFUTED, and the telemetry shows a FIXED
POINT: the effective previous-block width shrink*kappa_p is pinned at
~2.0-2.2 sigma REGARDLESS of the requested kappas, while wider kappa_c
only dilutes fit resolution (errors rise monotonically with kc).

Mechanism: the containment cap is relative — the p-box must fit inside
the OLD current-box, and the L1 row-sum bound over the 2n-column
transfer L_old^{-1}[L_pc | L_pp] costs an O(sqrt(2n))-conservative
factor plus the genuine corner spread; since L_old itself scales with
kappa_c, requested widths cancel and the cap is invariant. A ~2-sigma
effective box truncates real tail mass every step (the row-independent
deterministic bias of Section 10).

CANDIDATE REPAIRS (design fork):
(a) Row-exact containment: shrink to the ACTUAL frozen rows' z_old
    max, not the worst-case corner bound (rows are frozen and known at
    map-construction time — still deterministic and fail-closed).
    Expected gain modest (Sobol rows do approach corners).
(b) Truncation-mass correction: the retained object is exactly
    integrable; compute the retained mass NOT covered by the new box
    image and correct the increment. Stays in the bounded-Legendre
    program; new derivation needed for the correction term's place in
    the declared program.
(c) Gaussian-reference program variant: the SOURCE construction
    (Section 5 reference eta on R^m) never chains boxes at all — the
    bounded-box chaining is OUR program's artifact, not the paper's.
    Faithful but a major program change (bases, mass matrices, Gram
    chains are all bounded-Legendre today).

## 12. Truncation-mass correction (owner-selected fork; derivation v1)

The uncorrected increment computes the predictive mass restricted to
the new box image: M_in = exp(shift) Zc_new / Zc_prev-normalization.
The deterministic bias of Sections 10-11 is exactly the mass of the
step integrand outside that image. That missed mass is a SCALAR (it
enters only the likelihood increment, never the retained object), so
it needs no TT fit — plain frozen quadrature suffices, and its Monte
Carlo error is benign because the correction is small relative to M_in.

Estimator (v1, frozen Sobol, seed-deterministic):
- sample z ~ uniform([-1,1]^2n); map x_c = m_c + L_cc z_1 (new c-box),
  x_p = map_old(z_2) (FULL old p-box);
- keep only points OUTSIDE the new box image: z_p_new =
  L_pp^{-1}(x_p - m_p - L_pc z_1) with |z_p_new|_inf > 1;
- integrand: p_ret_phys(x_p) * exp(log_trans + log_obs) using the
  retained object's own physical evaluator (normalization consistent
  with the increment's telescoping);
- M_out = (V_phys / N_shell) * sum(kept integrand),
  V_phys = 2^n |det L_cc| * 2^n |det L_old|;
- corrected increment = logaddexp(increment_uncorrected, log M_out).

Declared residuals (v1 non-claims): x_c outside the new c-box is NOT
captured (bounded by observation-likelihood tail mass at kappa_c
sigma); the RETAINED law remains box-conditioned (the correction fixes
likelihood mass only — posterior-shape truncation is second-order in
the tail mass and stays a recorded approximation). Both effects shrink
as kappas widen, which the correction now makes affordable.

## 13. Truncation correction results (2026-08-21): n=2 SOLVED; n=4 residual isolated to fit quality

Rung 3 (n=2): PASS at 5.4e-5/step (bar 2.5e-3; 46x margin) at the
ladder budget, 8.6e-5 at the cheap 8192-row budget. The correction
paid back the Section 10 deterministic bias exactly; combined with the
triangular map's fit-rms collapse, the adapted engine at n=2 is ~40x
more accurate than the fixed engine's best at 4x fewer rows. The
n=2 program is comprehensively solved.

Rung 4 (n=4): 0.45/step (from 0.61). Step 1 now +0.06 (INSIDE the
bar); steps 2+ sit at -0.44..-0.83 with fit rms 0.16..0.22. The
correction removed ~0.15/step (its share of the bias); the remaining
error correlates with fit rms, NOT with shrink/coverage telemetry:
step 1 (fit rms 0.19, but old box = wide global box so the retained
law entering it is exact) is fine, while steps 2+ inherit RETAINED
OBJECTS fitted at rms ~0.2 — the truncation correction integrates the
retained DENSITY, which is itself off.

ISOLATION: at n=4/r=6/deg12 the TRANSITION-STEP fit in triangular
coordinates still has rms ~0.2 (vs ~4e-4 at n=2): the 9-axis branch
target is under-resolved in RANK or DEGREE — but now in coordinates
where those are the real and only lever. This is exactly the question
the P1B ladder exists to answer (r*(4) under a valid design), no
longer masked by coverage/truncation defects.

STATUS: rung 4's 0.1 bar is NOT met at r=6/deg12, and the evidence
says it is a resolution bar, not a design defect. The design-note
ladder therefore hands over to the ladder campaign: attempt04 with
map_form=triangular_affine + truncation correction, rank ladder
r in {6, 8, 10} (+ degree arm if rank saturates), n in {2, 4},
three seeds — the program's first valid r*(4) measurement.
