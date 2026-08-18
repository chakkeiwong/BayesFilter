# Master Program Plan: Generic Zhao-Cui-Family Squared-TT Filtering with Analytical Score (Revision 4)

Date: 2026-08-15 (revision 4 on 2026-08-17: program v0.2/v0.3 semantics
and the rank-conditioning tuning gate folded in from the execution-log
Addenda; revision 3: owner decisions on tau policy and structural scope
incorporated; revision 2 was the post-Codex-audit correction)
Status: `REVISED_AWAITING_FOCUSED_REAUDIT`
Audit chain:
- request: `bayesfilter-zhao-cui-generic-program-codex-audit-request-2026-08-15.md`
- audit: `bayesfilter-zhao-cui-generic-program-codex-audit-reply-2026-08-15.md`
  (verdict `REVISE_BLOCKED_BEFORE_P1_IMPLEMENTATION`)
- author reply + erratum:
  `bayesfilter-zhao-cui-generic-program-fable-reply-to-codex-2026-08-15.md`
- unblock artifacts: UB-1 derivation note + UB-2 source ledger (landed)
- re-audit handoff:
  `bayesfilter-zhao-cui-generic-program-codex-reaudit-handoff-2026-08-15.md`

Owner decisions (2026-08-15, recorded as binding):
- D1: `tau` is a per-scope TUNED parameter, never a silent default. Setting
  tau=0 without a scope-specific starvation diagnostic is disallowed for
  claim-bearing runs. Rationale: in high dimension a fit can be good in
  aggregate yet near-zero in regions the true filter visits; tau is the
  declared insurance against that failure mode and must be tuned per model.
- D2: the structurally deterministic (endogenous/exogenous split) case is
  IN-PROGRAM, resolved by the exact Dirac-substitution formulation
  (Section 3.6): filtering integrates over declared stochastic variables
  only, and the endogenous state is computed from the deterministic law as
  an exact change of variables. This case is universal in the target model
  class (DSGE) and the program is not complete without it.

Block scope (per Codex follow-up): bounded P1A may begin only after the
corrected derivation note and the source-classification ledger land; the
LGSSM ladder (P1B) and the full score engine (P2+) remain blocked until the
focused mathematical test artifacts exist.

## 0. Mission (unchanged)

Build one generalizable squared-TT filtering algorithm of Zhao-Cui family
provenance that:
1. accepts any repository state-space model through a declared adapter
   interface (variable n, m, p);
2. evaluates the filtering log-likelihood with polynomial-in-n cost and no
   dense `q^n` object anywhere in the runtime path;
3. produces the exact analytical gradient of the declared finite program for
   all p parameters, suitable for HMC and MLE;
4. is admitted per model/scope through tuning procedure v1.1 (Section 8);
5. joins the leaderboard as one named algorithm alongside SVD-UKF, LEDH,
   SGQF, GenUT, mixture-Kalman, which remain independent comparator
   algorithms.

Naming rule (audit source-faithfulness finding): "Zhao-Cui" is family
provenance only. The frozen weighted-ridge-ALS runtime and the analytical
score route are repository constructions and are classified as such in the
route ledger (Section 10). No claim that the runtime is the author
algorithm.

Scale target: n ~ 60-100, m ~ 80-100, T ~ 120, p ~ 300 (NAWM-class). The P6
rehearsal runs n=100, m=100, T=120, p=300 and is a resource/consistency row
only (Section 9).

## 1. Corrected understandings ledger (binding)

A1-A9 from revision 1 remain in force (goal is the generic algorithm; five
per-model backends are legitimate comparator algorithms; generic != tuning-
free; fitted-frozen-branch != adaptive; batched tangents; sigma=1 contract
narrowness; dense retention located at filtering.py:4048; single-path gaps
descriptive). Revision 2 adds the audit corrections:

| # | Correction (source) |
|---|---|
| A10 (F1) | An exact end-block marginal of a squared TT is a **quadratic form / sum of squares** `H_left(x) E_right H_left(x)^T`, not one scalar squared TT. Anchors: Zhao-Cui Prop. 2 / Eq. (14); `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-85` |
| A11 (F2) | Freezing the ALS schedule freezes the discrete branch, **not** the operator: design matrices contain left/right environments built from other cores, which move with theta after the first parameter-dependent update. The exact tangent is an **ordered total-derivative replay** of every core update with `dot_A`, `dot_W`, `dot_rho` terms. Donor: `zhao_cui_moment_teacher_als.py:403-475` (the only live-`dot_A` ordered replay in the repo — the scalar path's call at filtering.py:1253 passes all-zero dot cores and is NOT a donor; reply erratum E16) |
| A12 (F3) | The `<= 6x` gradient/value ratio is an empirical gate, not an analytic expectation. Forward-with-dot_A, chunked-forward, and adjoint modes are all open candidates until P2A measures them. Dominant uncounted work: `dot_A^T W A`-type contractions `O(p N c^2)` per update and tangent-environment construction; a materialized `[p,N,c]` stack is ~127 MiB at r=3 and ~900 MiB at r=8 |
| A13 (F4) | Likelihood increment must use the complete normalizer `log(Z_h + tau Z_0) + s_t`. Owner decision D1 (2026-08-15): tau is a per-scope TUNED parameter with a declared starvation-diagnostic selection step (Section 8); tau=0 is a legitimate tuning OUTCOME but never an untested default. Existing tau=0 admissions on low-dimensional scopes remain valid for their declared scopes (correct programs, passed gates); the policy error would be promoting tau=0 to high-dimensional scopes untested |
| A14 (F5) | The engine owns coordinate maps, Jacobians, reference weights, and the physical->reference measure conversion `log q_omega(z) = log q_x(R(z)) + log|det DR(z)| - log omega(z)` (pattern: filtering.py:2312-2322). Adapters return physical-coordinate log densities and batched JVPs only |
| A15 (F6) | No measure-zero tie claim. Contract: deterministic branch selection at max-shift ties + status telemetry; piecewise smoothness holds where the maximizer is unique |
| A16 (F7) | For structurally degenerate transitions, a TT over the declared stochastic coordinates alone does not define the next retained object. Two transition modes are required: `density_kernel` and `structural_substitution` (formerly the open "innovation_pushforward" item). Owner decision D2 + the Dirac-substitution derivation (Section 3.6) RESOLVE the design for the invertible-completion case: the retained object is an ordinary full-state density; the fit variable is `(x_t, m_{t-1})` (dimension n + n_stochastic, not 2n); the singular case (completion map not invertible in `k_{t-1}`, e.g. phi -> 0) is explicitly out of scope for v1 and fenced by an adapter invertibility precondition |
| A17 (arith) | Rank cost multipliers vs r=3 at N=512,b=12: r=8 -> 104.4x, r=16 -> 4677x (worse still once N must exceed c=b r^2=768 at r=8). No wall-clock claims anywhere until a measured GPU/XLA artifact exists |
| A18 (E16) | Code-verification discipline: verifying a function is called is not verifying it is active; activation checks must inspect actual argument values |

## 2. Asset inventory (corrected)

| Asset | Role |
|---|---|
| Multistate fixed-design TT path (filtering.py) | Trunk for the value recursion structure. Defects: dense-grid retention (4048), per-parameter score loop (1463), **zero-dot_A tangent** (1253) |
| `zhao_cui_moment_teacher_als.py:403-475` | **Sole donor** for the ordered total-derivative ALS replay (live dot_A, per-update value/JVP residual checks — inherit those checks) |
| Batched actual-SV TT route | Donor for batch-native/XLA layout; model welded inline; to be reproduced by engine+adapter then retired to comparator |
| Squared-TT core (`squared_tt.py`, `tt.py`, `derivatives.py`) | Sound: Gram normalizer verified by audit; primitives (`fixed_design_lsq_derivative` incl. dot_A/dot_N/dot_b, `tt_evaluation_derivative`, `squared_tt_log_normalizer_derivative`) FD-tested at `test_fixed_branch_derivatives.py:326-413` |
| `@TTSIRT/marginalise.m` (pinned author source) | Structural reference for the retained quadratic-form type |
| Other filters (SVD-UKF, UKF/CKF/GenUT, SGQF, LEDH, mixture-Kalman, lane modules) | Independent comparator algorithms / leaderboard rows / parity authorities (NOT independent same-target truth) |

## 3. Corrected mathematical program

### 3.1 Objects

Functional TT `h` with frozen ranks over the mapped box; squared density
`p(z) = (h(z)^2 + tau q_0(z)) / Z`, `Z = Z_h + tau Z_0`. Gram-chain
normalizer exact in `O(d b^2 r^4)` (audit-confirmed;
`squared_tt.py:164-175`).

**Retained object (corrected, A10).** After fitting the adjacent-pair TT at
step t and integrating out the previous-state block, the retained filtered
law is the quadratic-form type:

    RetainedQuadraticForm (working name SquaredTTMarginalFactor):
      - retained_prefix_cores          (TT block over x_t axes)
      - suffix_gram  E_right           (r x r PSD, from integrated block)
      - defensive_marginal q0_ret_ref  (reference-coordinate; tau per-scope
                                        tuned, owner decision D1)
      - normalizer   Zc_ref = Z_h + tau Z_0,ref (complete, A13; single
                                        stored scalar, reference measure)
      - coordinate map R + reference weight omega (A14; dual evaluators
                                        evaluate_reference_density /
                                        evaluate_physical_density per UB-1)
      - tangent state: dot_prefix_cores, dot_E_right (per parameter)

    p_ret_ref(z)  = (H_left(z) E_right H_left(z)^T + tau q0_ret_ref(z)) / Zc_ref
    p_ret_phys(x) = p_ret_ref(R^{-1}(x)) * omega(R^{-1}(x)) / J_R(R^{-1}(x))

evaluated directly as a quadratic form (no runtime Cholesky/QR gauge). The
next step's target assembly consumes this type natively; nothing densifies.
The rank ladder tracks fitted joint ranks AND retained boundary
rank/conditioning of E_right.

### 3.2 One filter step

**Program version note (revision 4).** The declared finite program is
v0.3, incorporating two in-execution repairs diagnosed FD-quality-first
(execution log Addenda 3-4). Both apply to BOTH engines (same-scalar, V5):

- **v0.2 relative defensive mass**: the engine-level defensive term is
  `tau_abs,t = tau * Z_h,t-1` (defensive FRACTION, shift-invariant), not
  absolute tau. The retained-object API (P1A) keeps its absolute-tau
  field; the ENGINE computes the relative form when assembling step t.
  Rationale: absolute tau made the value jump ~2e-5 under 1e-6
  theta-perturbations at shift-branch switches (C0 violation).
- **v0.3 smooth shift**: `s_t = logsumexp(log f_ref) - log N_rows`
  replaces the argmax max-shift. Rationale: argmax switches are generic
  along theta at fit resolution and each causes an O(fit-error) value
  jump, making FD invalid almost everywhere. With the smooth shift the
  program has NO shift-branch structure and needs no tie machinery
  (strengthens F6; the A15 tie/status contract is retired for these
  engines).

(a) Target on frozen designs, assembled by the ENGINE in reference measure
(A14):

    log f_ref = log p_ret,t-1 + log p_theta(x_t|x_t-1) + log p_theta(y_t|x_t)
                + log|det DR| - log omega
    g_t = exp((log f_ref - s_t)/2),
    s_t = logsumexp over grid of log f_ref - log N_rows   (v0.3 smooth
    shift; the argmax max-shift and its tie telemetry are retired)

(b) Frozen-schedule ALS over the 2n-core adjacent TT: per core,
`c = (A'WA + rho I)^{-1} A'W g_t` where A contains the moving environments
(A11). Discrete branch frozen; operator theta-dependent.

(c) Increment: `log Zhat_t = s_t + log(Z_h,t + tau_abs,t Z_0)` with the
v0.2 relative defensive mass (A13).

(d) Retention: build `RetainedQuadraticForm` for x_t by exact suffix
contraction (A10).

### 3.3 Exact score = ordered total-derivative replay (corrected, A11)

For each core update in the frozen order, with N = A'WA + rho I:

    N dot_c = dot_b - dot_N c
    dot_N   = dot_A' W A + A' W dot_A   (+ A' dot_W A + dot_rho I if active)
    dot_b   = dot_A' W g + A' W dot_g   (+ A' dot_W g)

where dot_A carries the tangent of the left/right environments accumulated
from all previously updated cores (ordered replay), and dot_g carries model
tangents plus the retained-object tangents (dot_prefix, dot_E_right) from
step t-1. Primal factorization of N shared across all tangent columns.
Chain closes with `dot Z_h` (bilinear Gram), `dot s_t` (branch-consistent),
quotient/log terms with complete Z, and the retained-type tangent for t+1.

Score mode (forward-with-dot_A vs chunked-forward vs adjoint) is decided by
the P2A measurement, not assumed (A12).

### 3.4 Differentiability status

The v0.3 program is smooth in theta away from declared floors (the
argmax branch structure is removed by the smooth shift; the branch
factor uses a PD Cholesky with a DECLARED relative Gram floor,
`branch_gram_floor = 1e-12`, the same factor in both engines per V5).
Its tangent is the ordered total derivative above. Derivative solves
(forward dot_c and adjoint lambda) must route through the SAME scaled
augmented factorization as the value solves (`scaled_normal_solve`;
conditioning repair, Addenda 4-5): raw normal-equation derivative solves
lose digits at ill-conditioned fits and are a diagnosed defect, not an
implementation choice. The analytic score is exact for the declared
finite program, not for the adaptive author algorithm (excluded, V1) and
not for the true likelihood (same-target gates measure that separately).
Existing Method A FD evidence covers those declared per-model routes
only — it does NOT validate the new retained type, the multi-sweep
total-derivative replay, the batched all-parameter engine, or NAWM-scale
cost (audit "existing evidence boundary" adopted). Verified P2 state
(Addendum 5): adjoint vs forward-JVP (I-P2-4, FD-independent) at
1e-12 for n in {1,2}; FD gate green at n in {1,2} in well-conditioned,
quadrature-resolved regimes.

## 3.5 Tau policy (owner decision D1; re-audit Finding 4 incorporated)

`p = (h^2 + tau q_0)/Z` with `Z = Z_h + tau Z_0` everywhere. The defensive
family `q_0` must be product-form so its retained marginals stay closed
form, and every candidate `q_0` must be NORMALIZED under the declared
measure (equivalently: tune a normalized defensive mixture mass, never an
unidentified `tau*q_0` scale — tau values are not comparable across
differently normalized families). Per scope, tau is selected by the tuning
procedure (Section 8, step T-tau); tau and q_0 enter the scope identity, so
any HMC/MLE run targets one fixed declared (tau, q_0) — still an
exact-gradient surrogate.

**Epistemic status (binding, per re-audit):** the T-tau step is a
fail-closed VIABILITY screen, not a bias or accuracy control. Where no
same-target reference exists, the selection is labeled
`viability_tuning_only` in the scope artifact: it is not evidence of low
approximation bias, posterior correctness, or superiority. Zhao-Cui
Lemma 1 relates the defensive constant to approximation error for the
SOURCE construction only; no transfer to the frozen-ALS program is claimed
without a separately checked transfer argument.

Trade-off recorded per scope: too-small tau risks region starvation
(log-collapse on tail observations); too-large tau injects per-step mixing
bias that accumulates over T (measured against the same-target reference
where affordable; otherwise bounded only descriptively via the Section 8
sensitivity table).

## 3.6 Structural substitution mode (owner decision D2; resolves F7 for the invertible case)

Setting (Ch18b): `x_t = (m_t, k_t)`; declared stochastic block
`m_t = T_m(m_{t-1}, eps_t; theta)` with proper density
`p_theta(m_t | m_{t-1})`; deterministic completion
`k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)`. The transition kernel is
degenerate:

    p(x_t | x_{t-1}) = p_theta(m_t | m_{t-1}) *
                       delta( k_t - T_k(k_{t-1}, m_{t-1}, m_t) ).

**Exact substitution (restricted subclass; re-audit Finding 5).** Adapter
precondition: for every (m_{t-1}, m_t, theta) in the declared support
region, the endogenous-block map `k_{t-1} -> T_k(k_{t-1}, m_{t-1}, m_t)`
is a GLOBAL single-valued diffeomorphism from the relevant k_{t-1} support
onto its image (not merely pointwise nonsingular), with inverse
`k_{t-1} = S(k_t; m_{t-1}, m_t; theta)` depending smoothly on parameters,
declared support image/boundary indicator, and no unhandled inverse
branches. The Jacobian factor is
`J = |det D_{k_t} S| = 1 / |det D_{k_{t-1}} T_k|`. Integrating the delta
against `k_{t-1}` then gives a PROPER density recursion over the full
state:

    p_t(m_t, k_t) ∝ p_theta(y_t | x_t) *
        ∫ p_{t-1}( m_{t-1}, S(k_t; m_{t-1}, m_t) )
          p_theta(m_t | m_{t-1}) * J  dm_{t-1}
    (off-image (m_t, k_t) points have explicit zero support).

Consequences (with re-audit qualifications):
- The retained object is an ordinary full-state density -> the
  RetainedQuadraticForm type applies unchanged. "No information loss" holds
  relative to the already-retained full-state approximation under the
  global inverse/support conditions; it is NOT a claim about the exact
  filtering law or about the error of the next TT fit.
- The TT fit variable is `(m_t, k_t, m_{t-1})`: raw dimension
  `n + n_stochastic`, not `2n`. This is a variable-count statement, NOT a
  complexity result: the k_t axes still need resolution, and composition
  with nonlinear S and J may increase basis-size or TT-rank demands enough
  to erase the nominal saving. Whether the reduction is material is
  MEASURED at P2S (dimension/rank telemetry), not assumed.
- `S` and `J` are smooth in theta on the declared support, adding tangent
  terms to the score chain. **Load-bearing score term (re-audit Finding
  6):** because the previous physical state now moves with theta, the
  retained-law derivative is a TOTAL derivative:
  `d/dtheta log p_ret(m_prev, S(theta); theta)
   = partial_theta log p_ret + grad_{k_prev} log p_ret . dot_S`,
  where the SPATIAL gradient of the retained density is an ENGINE
  obligation (a retained-evaluator spatial JVP, including the defensive
  component and, for reference-coordinate evaluation, propagation through
  `R^{-1}(S)` and the declared measure conversion). dot_S and dot_log_J
  alone do NOT complete the score. UB-3 must define and FD-test this
  spatial JVP.
- Toy check (Ch18b worked example A): `k_{t-1} = (k_t - gamma m_t^2)/phi`,
  `J = 1/|phi|` for phi != 0.

**Fenced out of v1 (V13), with the re-audit's taxonomy:** (i) finite
many-to-one completions (a branch SUM over inverse branches — still an
ordinary density, but not implemented in v1); (ii) rank-deficient
completions (e.g. phi -> 0), where the joint law is genuinely
manifold-supported. These are DIFFERENT cases and must not be conflated;
both are out of v1 scope, never silently regularized (Ch18b labeling
policy). Near-singular guard: a condition number alone CANNOT detect
Jacobian inflation (scalar toy: cond=1 for all phi != 0 while J = 1/|phi|
diverges); V13 therefore requires minimum-singular-value / inverse-norm
and log-J bounds plus J-weighted row/mass, nonfinite, floor, and support
telemetry.

**Scope honesty (re-audit Finding 5):** Ch18b's general pushforward
identity explicitly does NOT require invertibility of T_k (chapter
assumptions near lines 1616-1628); v1's substitution route is one useful
invertible-completion subclass, not closure of the general structural
case. The program's structural claim is limited accordingly.

**Companion derivation note required (UB-3)** before the structural score
column is claimed: the recursion above pushed through the UB-1 chain
(target assembly on `(m_t, k_t, m_{t-1})` rows; S/J tangents; the
retained-evaluator spatial JVP of Finding 6; retention split at the
`(m_t, k_t)` boundary; support/branch status), with the dense
`(x_{t-1}, eps_t)` quadrature as toy-scale arbiter and the Ch18b
validation-gate list as the admission checklist.

## 4. Design constraints / vetoes (revised)

V1 fixed-branch (no runtime adaptation) — unchanged.
V2 no dense `q^n` object in any runtime path (instrumented assertion I-P1-3).
V3 no runtime SVD/rank re-truncation — unchanged.
V4 all-p tangents in one pass sharing factorizations; per-parameter path
   re-execution disqualified. (The <=6x ratio is a P2A GATE, not a premise.)
V5 same-scalar property, FD-quality-first gates — unchanged.
V6 TF/TFP float64 reference semantics; NumPy diagnostics-only — unchanged.
V7 theta-batch native — unchanged.
V8 floor/status honesty — unchanged.
V9 (revised per D1) complete-normalizer and measure-conversion formulas
   everywhere; tau is scope-tuned via the declared starvation-diagnostic
   step — claim-bearing runs with an untuned tau (including untested tau=0)
   are disallowed.
V10 (new) retained objects are RetainedQuadraticForm; stamping a rank>1
   suffix Gram as a scalar square is a defect (U-MARG-TYPE-1 enforces).
V11 (new) GPU phases (including P1B ladders) configure and record TF memory
   growth before device initialization.
V12 (new) no wall-clock/hardware-time claims without a measured artifact on
   the actual device+XLA route.
V13 (revised per re-audit) structural adapters must declare GLOBAL
   single-valued invertibility of the endogenous-block completion map on
   the declared support (with support image/boundary indicator, smooth
   parameter dependence, no unhandled branches), plus minimum-singular-
   value / inverse-norm and log-J bounds — a condition number alone is
   insufficient (scalar counterexample: cond=1, J=1/|phi| divergent).
   Violation is a hard veto. Finite many-to-one and rank-deficient cases
   are distinct, both out of v1, never silently regularized.

## 5. Architecture (revised)

    [Adapter]  physical-coordinate log densities + batched theta-JVPs
               (density_kernel mode), or declared stochastic-block density
               `p_theta(m_t|m_{t-1})`, completion map T_k, its inverse S in
               k_{t-1}, Jacobian J, their JVPs, and an invertibility/
               conditioning declaration (structural_substitution mode,
               Section 3.6, V13); parameter chart; structural state
               partition metadata; support/scale hints; manifest.
    [Engine]   owns coordinate maps, Jacobians, reference weights, measure
               conversion (A14); frozen designs; ordered ALS with moving-
               environment tangent replay (A11); Gram normalizers (complete
               Z, scope-tuned tau per D1); RetainedQuadraticForm retention
               + tangents (A10); theta-batch axis; status; XLA-compilable.
    [Tuning]   procedure v1.1 (Section 8): calibration / validation /
               untouched-claim partitions; fail-closed scope artifacts;
               includes the T-tau defensive-mass selection step.

## 6. Phases (revised sequence with unblock conditions)

### Unblock artifacts (before any P1A implementation)
- UB-1 **Score derivation note**: LANDED 2026-08-15 —
  `bayesfilter-zhao-cui-generic-program-ub1-score-derivation-note-2026-08-15.md`
  (ordered total-derivative replay with moving environments; retained
  quadratic-form evaluator + suffix-Gram tangent; complete-Z chain; tie/
  status contract; batched-p organization with P2A cost caveat). Awaiting
  one focused review per the audit's re-audit scope.
- UB-2 **Source-classification route ledger**: LANDED 2026-08-15 —
  `bayesfilter-zhao-cui-generic-program-source-route-ledger-2026-08-15.md`.
- UB-3 **Structural substitution derivation note** (new, D2): LANDED
  2026-08-17 —
  `bayesfilter-zhao-cui-generic-program-ub3-structural-substitution-derivation-2026-08-17.md`
  (substitution recursion in branch-axis form; moving-point retained
  tangent incl. spatial gradient, R^{-1}(S) propagation, dot log J;
  binds U-STRUCT-SPATIAL/MOVING/J tests). Required before P2S
  implementation; NOT required for P1A/P1B/P2A/P2 on density_kernel
  models. Awaiting its focused review at the P2S boundary.

### P0 — Contract + skeleton (may proceed now, records revised semantics)
- Engine/adapter contract with: RetainedQuadraticForm type, two transition
  modes, measure-ownership line, ordered-tangent-replay semantics, scope
  identity fields (full audit list). No behavior change; suite green.
- Named open design item: pushforward retained law (A16), arbiter = dense
  `(x_{t-1}, eps_t)` integration at toy scale.

### P1A — Retained quadratic-form object at n<=3 (bounded; after UB-1/UB-2)
- Implement RetainedQuadraticForm + evaluator + manifest + tangents.
- Tests: U-MARG-TYPE-1, U-MARG-DERIV-1, U-MEASURE-1, U-TAU-1 (all must
  exist and pass before P1B).
- Gate: retained evaluator == brute-force integration (1e-10); tangents ==
  FD; both measure conventions.

### P1B — Value-only LGSSM ladder (blocked until P1A passes)

**Execution status 2026-08-16** (result note
`bayesfilter-squared-tt-engine-p0-p1a-p1b-smoke-result-2026-08-16.md`):
P1A PASSED (all gates). P1B smoke at n=1 REJECTED the naive
single-sqrt-refit target assembly at the declared tolerance, with a
diagnosed mechanism: at tau=0 the step target contains |h_prev| kinks
along the previous fit's zero set (fit-rms plateau invariant to
degree/rank); tau>0 smooths (20x gap improvement at tau=1e-3, sharp
over-mixing at 1e-2) — empirically vindicating D1 and the F4 mixing-bias
concern. Promotion veto on the candidate; NOT a continuation veto.

**Standing P1B amendment (required before the ladder):**
branch-decomposed target assembly. With E = L L' (PD Cholesky, smooth),
the step target factorizes exactly as
`f = sum_g (u_g(z_prev))^2 G + tau G`, `u_g = H_L L[:,g]`, and every
branch `u_g sqrt(G)` is SMOOTH (no kinks; G > 0 for Gaussian kernels) —
this is the author's own retained-multiple-functions structure
(`marginalise.m:35-37,63-65`, Prop. 2). Cost: boundary rank grows
additively per step, so a **declared branch-compression policy** is
required and touches V3: candidate designs are (a) frozen spectral-gap-
guarded eigen-truncation of the PSD Gram (needs a smoothness analysis and
a V3 amendment), or (b) fixed branch budget with Gram-weighted grouping.
The compression design note must land before implementation (same
discipline as UB-1), then the smoke gates (currently strict-xfail in
`tests/highdim/test_p1b_engine_v0_lgssm_smoke.py`) must flip to pass
before the ladder runs.

- n in {2,4,8,16,32,64}, T in {8,120}; exact Kalman truth; easy AND
  adversarial covariance/coordinate-order structures; predeclared tolerance
  (declared in the P1B experiment plan BEFORE execution); untouched
  validation paths; memory-growth recorded (V11); no-dense-grid assertion.
- Outcomes: r*(n) curve (fitted ranks and E_right conditioning), promotion
  criterion = same-target value error; fit residual is explanatory/repair;
  condition/nonfinite = hard veto; rank growth = feasibility diagnostic and
  continuation veto only at the predeclared resource bound.
- Pivot branch unchanged (structure exploitation) if the curve is bad.

### P2A — Total-gradient cost prototype (before full engine)
- Three-way measured comparison: forward+dot_A / chunked-forward / adjoint,
  at p in {3, 30, 300}: runtime, peak allocator bytes, same-scalar FD.
- Solver-reuse obligations (recheck Finding 2, binding): (a) scaled-primal
  vs normal-equation solution agreement on easy fixtures AND near
  column-scale floors / condition thresholds; (b) derivative consistency
  against the ACTUAL scaled primal solver (`_solve_scaled_augmented_ridge`
  -> `fitting.py:984-1010`), not only against the normal-equation
  primitive; (c) runtime and peak memory measured WITH and WITHOUT genuine
  factorization reuse — "shared factorization" is a goal to be demonstrated
  or abandoned here, not an assumption.
- Full-horizon stress (recheck Finding 5): in addition to the short
  mode-selection prototype, one full T=120 tangent-state run per candidate
  mode measuring memory/retracing/allocator behavior; short-prototype
  evidence alone is explicitly insufficient for full-horizon feasibility.
- Decision recorded; <=6x is the gate the chosen mode must meet at p=300.

### P2 — Batched score engine (blocked until UB-1, P1A, P2A)
- Implement the chosen mode over full horizon; per-update value/JVP residual
  checks inherited from the donor.
- Gates: I-P2-1 FD suite (LGSSM rungs, predator-prey, SIR d=18 T>0,
  structural T=20) at MULTIPLE parameter points including near-boundary,
  near-condition-threshold, near-tie points; I-P2-2 p-scaling; I-P2-3
  same-scalar.

### P2S — Structural substitution mode (new phase, D2; after UB-3 + P2)
- Implement the Section 3.6 recursion in the engine: target assembly on
  `(m_t, k_t, m_{t-1})` rows with S/J substitution; adapter global-
  invertibility declaration (V13); score chain extension with the COMPLETE
  moving-point total derivative (recheck Findings 5/6, binding):
  `partial_theta log p_ret + grad_{k_prev} log p_ret . dot_S` — i.e. the
  retained-evaluator SPATIAL JVP (including the defensive component and
  propagation through `R^{-1}(S)` under the declared measure convention),
  the coordinate-map inverse JVP, dot_S and dot_log_J adapter terms, and
  support/branch status; each term FD-tested per UB-3. dot_S/dot_log_J
  alone are NOT sufficient.
- Toy validation on the Ch18b worked-example model (rho, sigma, phi, gamma,
  R): value AND score vs dense `(x_{t-1}, eps_t)` quadrature reference at
  T in {5, 20}; Ch18b validation-gate list (metadata, constraint-support,
  linear-recovery, degenerate-transition tests) as admission checklist;
  U-STRUCT-PUSHFORWARD-1 retargeted to this mode.
- Singular fence check: a phi-near-0 configuration must trip the V13
  minimum-singular-value / log-J veto (condition number alone is proven
  insufficient), not produce a silent answer.
- Dimension telemetry: record fit-variable dimension (n + n_stochastic)
  vs the naive 2n AND the realized basis/rank demands, so the reduction
  claim is measured, not assumed.

### P3 — Batch-native/XLA port
- Eager-vs-XLA parity 1e-12 (value and score); memory growth; throughput
  recorded (no feasibility language).

### P4 — Adapter suite + reproduction gates
- LGSSM, actual SV, KSC SV, predator-prey, Austria SIR (density_kernel),
  structural model (Ch18b, structural_substitution mode via P2S),
  synthetic p=300 fixture.
- I-P4-1 SV near-bit reproduction of the admitted batched route; I-P4-2
  Kalman value+score; I-P4-3 scalar-path parity; I-P4-4 lane-module parity
  (labeled comparator parity, NOT independent truth); I-P4-5 Ch18b gate
  list; structural recursion gate (retained law == dense (x,eps) reference).

### P5 — Tuning procedure v1.1 execution + scope admission
- Calibration/validation/untouched-claim partitions, disjoint seeds/paths;
  fail-closed artifact consumption (U-SCOPE-FAILCLOSED-1); dominance
  comparison vs historical hand configs.

### P6 — Leaderboard + HMC campaign + NAWM-representative gate
- Full leaderboard per Section 9.
- HMC admission separate campaign (NeuTra governance).
- NAWM-scale synthetic row: resource/consistency ONLY.
- Any `NAWM_FEASIBLE`-class claim additionally requires the audit's
  ten-condition target-representative structural ladder (adopted verbatim:
  matched stochastic dimension and deterministic-completion pattern,
  representative coupling, full score-path parameter dependence, target
  horizon/data regime, multiple frozen coordinate orderings incl.
  adversarial, low-dim reductions with independent references, untouched
  validation across a declared parameter region, full telemetry, measured
  GPU/XLA runtime+memory, recorded extrapolation rule). Passing supports
  "feasible on a NAWM-representative synthetic contract" — never a claim
  about the actual NAWM target without running it.

## 7. Test program (normative; audit additions incorporated)

Unit (mathematical): U-GRAM-1/2/3, U-LSQ-1/2, U-EVAL-1, U-SQN-1,
U-MARG-TYPE-1, U-MARG-DERIV-1, U-ALS-REPLAY-1 (ordered two-sweep, nonzero
dot_A on later updates, every intermediate core vs FD), U-ALS-BATCH-1,
U-TAU-1, U-MEASURE-1, U-SHIFT-1/2 (persistent-tie construction), U-FLOOR-1,
U-JAC-1 (end-to-end through leaderboard value construction), U-DATA-1,
U-BATCH-1, U-ADAPT-1 (+ semantic adapter checks, not only graph identity),
U-ADAPTER-JVP-1, U-STRUCT-1, U-STRUCT-PUSHFORWARD-1, U-SCOPE-FAILCLOSED-1,
U-TUNE-1.

Integration: I-P0-1 suite-green; I-P1A gates; I-P1B ladder (+ I-P1-3
no-dense assertion); I-P2A cost prototype; I-P2-1/2/3; I-P3-1/2; I-P4-1..5
+ structural recursion gate; I-P5-1 dominance; I-P6-1 rehearsal
(resource-only claims); I-LB-1 leaderboard schema.

Statistical: S-1 (>=8 paired seeds for any leaderboard cross-family
interval; estimand predeclared; no family-wide "beats" language), S-2
two-step stability rule.

## 8. Tuning procedure v1.1

Scout -> resolution ladder -> rank/sweep selection -> **T-tau defensive-mass
selection (D1)** -> score admission -> scope binding, now with:
- **rank-conditioning gate (revision 4, from Addenda 4-5)**: rank
  selection must keep the retained suffix Gram numerically
  well-conditioned for score-bearing scopes — a rank above the retained
  law's effective rank produces a rank-degenerate Gram whose near-null
  Cholesky column rotates erratically with theta, injecting O(fit-error)
  value wiggles that invalidate FD and degrade the score path (diagnosed
  at n=2 rank 3: lambda3/lambda1 ~ 1e-13). The score engine's Gram
  conditioning veto (default 1e12) is the fail-closed backstop; tuning
  must select ranks that clear it with margin. Resolution selection must
  also keep the per-core ALS design full column rank (quadrature/rows
  per axis >= basis functions per axis; the qo=8-vs-deg-10 defect of
  Addendum 5 is the recorded counterexample);
- three disjoint partitions (calibration / validation / untouched claim),
  disjoint seeds and data paths; final claims only from the untouched
  partition after controls freeze;
- **T-tau step (mandatory for every scope; re-audit Finding 4 repairs
  incorporated)**: candidate q_0 families NORMALIZED under the declared
  measure (tune normalized defensive mass, not raw tau*q_0 scale);
  validation rows must include INDEPENDENTLY GENERATED stress rows spanning
  the declared HMC parameter region, observation tails, support boundaries,
  and long-horizon states (not only same-family rows as calibration);
  diagnostics are DIMENSIONLESS: p_ret/q_0 ratios, defensive fraction
  tau*q_0/(h^2 + tau*q_0), target-to-fit importance ratios or ESS where the
  row target is evaluable, weighted target mass, boundary/tail mass, floor
  activations; select the smallest tau (declared grid including 0) with no
  starvation diagnostic firing; after freezing (tau, q_0), evaluate the
  full predeclared sensitivity table on the UNTOUCHED claim partition as a
  veto/descriptive check only — a failure there triggers FRESH tuning
  partitions, never on-partition reselection; where no same-target
  reference exists the artifact is labeled `viability_tuning_only`;
  tau=0 is admissible only as the OUTCOME of this step; the artifact
  records the full diagnostic table for every grid point;
- fail-closed scope artifacts; scope identity binds model/target id,
  horizon, data regime, n/m/p, basis + coordinate maps, row design + seeds,
  rank vector, sweep schedule, ridge/stabilization policy, defensive
  density family + selected tau, dtype, backend, XLA mode, parameter
  chunking, structural integration-space metadata (mode + invertibility
  declaration for structural_substitution scopes);
- all tolerances declared in the experiment plan before execution.

## 9. Leaderboard specification (revised claim schema)

Rows: LGSSM (ladder), actual SV, KSC SV, predator-prey T in {20,40},
Austria SIR T in {20,40}, structural deterministic model (Ch18b) T in
{20,100}, NAWM-scale synthetic (resource row).

Columns per (model, algorithm): value; same-target gap where a reference
exists; score status + FD relerr; wall time (measured artifacts only);
scope artifact id; status flags; claim label; **reference_authority**.

Reference vocabulary (audit-revised):
- `EXACT_ORACLE` (exact Kalman on LGSSM, declared target only);
- `REFINED_NUMERICAL_REFERENCE` (dense quadrature WITH two-step refinement
  certificate);
- comparator/parity authorities (lane modules, other filters) — never
  same-target truth;
- Austria SIR internal consistency -> `DIAGNOSTIC_ONLY` or narrow
  `SURROGATE_USEFULNESS`;
- NAWM synthetic -> resource/consistency only.
`CERTIFIED_APPROXIMATION` must name authority, frozen tolerance, untouched
claim data, and the passed refinement/uncertainty gate. Cross-family
continuous differences remain descriptive unless the paired estimand and
uncertainty procedure were predeclared (S-1).

## 10. Source-classification route ledger (`EXACT_ANCHORS_RECORDED`; UB-2 is the sole binding ledger)

The SOLE binding source-faithfulness ledger is UB-2 revision 2:
`docs/plans/bayesfilter-zhao-cui-generic-program-source-route-ledger-2026-08-15.md`
(status `EXACT_ANCHORS_RECORDED`; includes the Lemma 1 transfer caveat, the
restricted structural-subclass scope, and the extended forbidden-claims
list). The synchronized summary below is informational; on any
discrepancy, UB-2 governs. Author-code paths are relative to
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/`.

| Operation | Classification | Exact anchor (paper; author code) |
|---|---|---|
| Squared-TT nonnegative density `(h^2 + tau q0)/Z` | `source_faithful` | Eq. (13), Lemma 1; `@TTSIRT/eval_potential_reference.m:21,33` |
| Exact `h^2` mass / Gram normalizer | `source_faithful` | Prop. 2 / Eq. (14), paper text lines 549-626; `@TTSIRT/marginalise.m:25-51` (squared-mass propagation, `fun_z` line 51) and `:85` (complete defensive mass) |
| End-block marginal quadratic form | `source_faithful` | Prop. 2 / Eq. (14); `@TTSIRT/marginalise.m:25-85` |
| Adaptive TT-cross/SVD construction (EXCLUDED from runtime, V1) | `source_faithful` (excluded) | Sec. 3; `@TTFun/cross.m:1-60`, `@TTFun/build_basis_svd.m:31` |
| Frozen weighted ridge ALS | `extension_or_invention` | no author counterpart (nearest: cross/AMEN basis builds, not copied) |
| Ordered total-derivative score replay | `extension_or_invention` | no fit-through score route found in the inspected pinned snapshot (`@TTFun/grad_reference.m:1-79` is evaluation-gradient only) |
| RetainedQuadraticForm type + dual measure evaluators | `extension_or_invention` (structure anchored to Prop. 2) | `@TTSIRT/marginalise.m:25-85` for structure |
| Structural substitution mode (restricted invertible subclass) | `extension_or_invention` (vs Zhao-Cui) | Ch18b (incl. general pushforward requiring NO invertibility, ~lines 1616-1628 — v1 is narrower) |

Binding caveats carried from UB-2: no Lemma 1 accuracy transfer to the
frozen-ALS program without a checked argument; the v1 structural route does
not close the general (non-invertible) structural case; `fixed_hmc_adaptation`
rows: none claimed in v1.

## 11. Pre-mortem (revised top risks)

1. Retained quadratic-form rank/conditioning growth (now measured directly
   in P1B via E_right telemetry).
2. Ordered-replay tangent cost: dot_A contractions may dominate; P2A decides
   the mode from measurements, adjoint fully open.
3. NAWM inference risk: gates can pass on easy proxies — closed by the
   ten-condition representative ladder (P6) and by refusing feasibility
   language elsewhere.
4. Structural substitution risks concentrate in the S/J substitution
   quality and the invertibility fence (V13): a near-singular completion
   map inflates J and can destabilize the fit. Mitigation: conditioning
   bound in the adapter declaration; dense toy arbiter at P2S; the truly
   singular case is fenced out of v1 by owner decision D2.
5. Tau selection can be gamed by a too-coarse grid or too-easy validation
   rows; mitigation: the T-tau artifact records the full diagnostic table
   for every grid point, and the untouched claim partition never
   participates in selection.

## 12. Definition of done (revised)

One engine + adapter suite passing P4 gates, including the structural
substitution mode (P2S) validated against the dense (x,eps) toy reference
and the Ch18b gate list; scope artifacts via v1.1 with untouched-claim
partitions and per-scope tuned tau (T-tau diagnostic tables recorded);
P2A-chosen score mode meeting its measured gate at p=300; r*(n) and E_right
conditioning curves published with honest verdicts; leaderboard rows with
the revised reference vocabulary; the NAWM-representative gate either
passed (enabling the representative-feasibility claim) or its failure
recorded with the extrapolation rule.
