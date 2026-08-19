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

The fit target's conversion term generalizes (A14 machinery):

    log f_ref = log p_ret,t-1(x_p) + log p(x_c|x_p) + log p(y_t|x_c)
                + log|det L_c| + log|det L_p| + 2n log 2

(replacing 2n(log hw + log 2)); the RetainedQuadraticForm for step t
stores coordinate_map_t = (m_c,t, L_c,t) — the type already carries a
coordinate_map field and dual evaluators, so retention needs no type
change, only non-identity map values.

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
