# UB-1 Derivation Note: Ordered Total-Derivative Score for the Generic Squared-TT Filter

Date: 2026-08-15 (revision 3, 2026-08-16: measure-qualified closure repairs
per `codex-reply-to-fable-reaudit-response-2026-08-16.md`; content gate
passed per `codex-reply-to-fable-response2-2026-08-16.md`)
Status: `CONTENT_GATE_PASSED_IMPLEMENTATION_TESTS_PENDING` (derivation
artifact; the named tests must still be written and pass before P1B)
Notation: unqualified `Z_0` anywhere in this note is an alias for
`Z_0,ref`, the defensive normalizer under the declared reference measure;
`Zc` is an alias for `Zc_ref`.
Program plan (rev 2):
`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
Audit findings addressed: F1 (retained type), F2 (moving ALS environments),
F4 (complete normalizer), F5 (measure conversion), F6 (ties).

Donor implementations (verified live in this session):
- Ordered ALS value+JVP replay with moving-environment `dot_A`:
  `bayesfilter/highdim/zhao_cui_moment_teacher_als.py:351-513`
  (`fixed_als_value_jvp`; per-update value/JVP residual records).
- Per-update total-derivative solve (`dot_N`, `dot_b` complete):
  `bayesfilter/highdim/derivatives.py:521-603` (`fixed_design_lsq_derivative`).
- Environment tangent for the design matrix:
  `bayesfilter/highdim/derivatives.py:490-518` (`differentiate_design_matrix`).
- NOT a donor: `filtering.py:1253` (dot-core tuple is all zeros there; reply
  erratum E16).

Classification (route ledger rows 7-9): the analytical score below is an
`extension_or_invention` for the repository's declared finite program.
Zhao-Cui provides no HMC score route.

---

## 1. The declared finite program (value path)

Fixed offline (tuning scope artifact): product basis `{phi_k}` per axis with
mass matrices `M^(i)`; quadrature/fit rows `Z = {z_j}` with weights `W`;
ridge `rho`; ranks; sweep order `pi` and sweep count `S`; coordinate map
`x = R(z)` with reference weight `omega`; defensive `tau` (default 0) and
`q_0`; floors; deterministic seeds. None of these depend on `theta`.

Per step `t = 1..T`, with retained object `P_{t-1}` (Section 2):

**(V1) Target assembly (engine-owned measure conversion, F5; measure
contract fixed per re-audit Finding 1).** The retained object exposes TWO
non-interchangeable evaluators:

    evaluate_reference_density(z):  p_ret_ref(z)   [density wrt mu(dz) = omega(z) dz]
    evaluate_physical_density(x):   p_ret_phys(x)  [density wrt Lebesgue dx]

related by `p_ref(z) = p_phys(R(z)) J_R(z) / omega(z)` with
`J_R = |det DR|`. The defensive marginal q0_ret is owned by the SAME
convention as the evaluator that includes it, and its normalizer Z_0 is
declared under that convention. The DECLARED engine assembly uses the
reference-measure evaluator for the retained (previous-state) factor, so
only the current-block conversion appears:

    log f_ref(z_curr, z_prev) = log p_ret_ref(z_prev)                  [Sec. 2, reference]
                              + log p_phys(x_curr | x_prev)             [adapter, physical]
                              + log p_phys(y_t | x_curr)                [adapter, physical]
                              + log J_curr(z_curr) - log omega_curr(z_curr)
    with x_* = R(z_*).

The previous-state conversion is ABSORBED in p_ret_ref by construction and
must not be applied again (double-conversion is the Finding-1 defect; the
physical-evaluator variant with both-block conversion terms is the
documented alternative and the two must never be mixed). U-MEASURE-1 tests
a complete TWO-STEP recursion, including the defensive component, against
direct physical quadrature under both conventions.

**(V2) Shift and square-root target.**

    s_t(theta)  = max_j log f_t(z_j; theta)
    g_t(z_j)    = exp( (log f_t(z_j; theta) - s_t) / 2 )

**(V3) Ordered ALS.** Initialize cores `G^(0)` (frozen). For sweep
`s = 1..S`, core `i` in order `pi`: build the design

    A_i(theta) = [ L_i(z_j) (x) phi(z_j,i) (x) R_i(z_j) ]_j          (row j)

where `L_i`/`R_i` are the left/right environments contracted from the
*current* values of all other cores (theta-dependent after the first
parameter-dependent update — audit F2), and solve

    N_i c_i = b_i,   N_i = A_i' W A_i + rho I,   b_i = A_i' W g_t .   (*)

Update core `i` with `c_i`; continue in order. After `S` sweeps the fitted
square-root TT is `h_t`.

**(V4) Likelihood increment (complete normalizer, F4).**

    Z_h,t = <h_t, h_t>_omega            [Gram chain, exact]
    Zc_t  = Z_h,t + tau * Z_0
    log Zhat_t = s_t + log Zc_t
    log Lhat(theta) = sum_t log Zhat_t

**(V5) Retention (corrected type, F1; measure-qualified per recheck
Finding 1).** Split the fitted TT after the current-state block:
`h_t(z_curr, z_prev) = H_L(z_curr) H_R(z_prev)` with
`H_L(z) in R^{1 x r_c}` (row) and `H_R(z) in R^{r_c x 1}` (column), where
`r_c` is the boundary rank at the split. Exact marginalization over the
previous block under the reference measure:

    E_t = int H_R(u) H_R(u)' omega_prev(u) du   in R^{r_c x r_c},  PSD   (Gram chain)

The retained object is defined ONCE, as a REFERENCE-measure density, and
the physical evaluator is derived from it:

    p_ret_ref,t(z)  = ( H_L(z) E_t H_L(z)' + tau q0_ret_ref(z) ) / Zc_ref,t
    p_ret_phys,t(x) = p_ret_ref,t(R^{-1}(x)) * omega(R^{-1}(x)) / J_R(R^{-1}(x))

where `q0_ret_ref` is the defensive marginal REPRESENTED IN REFERENCE
COORDINATES (its physical counterpart is obtained by the same conversion)
and `Zc_ref,t = Z_h,t + tau Z_0,ref` is the single stored normalizer
scalar, declared under the reference measure. There is no separately
represented `Z_phys`: the physical evaluator reuses `Zc_ref,t` because the
conversion factor `omega/J_R` is exactly the density-of-measures
`d mu / d Leb`, so total mass is preserved (`int p_ret_ref d mu =
int p_ret_phys dx = 1`); U-MEASURE-1 asserts this numerically.

`RetainedQuadraticForm(prefix cores of H_L, E_t, tau, q0_ret_ref, Zc_ref,t,
coordinate map R, reference weight omega)`; evaluators
`evaluate_reference_density(z) -> p_ret_ref` and
`evaluate_physical_density(x) -> p_ret_phys` as in (V1). This is Zhao-Cui
Prop. 2 / Eq. (14) structure; it is generally NOT one scalar squared TT
(audit F1), and the engine never converts it to one. Step t+1's (V1)
consumes `log p_ret_ref` (the reference evaluator) directly. Boundary-rank
`r_c` and the conditioning of `E_t` are telemetry.

---

## 2. Retained-object evaluation and its tangent

All quantities in this section are REFERENCE-measure objects (suffix
`_ref`); the physical evaluator is the fixed conversion of (V5) and needs
no separate tangent derivation (the conversion factor is
theta-independent). Evaluation at a reference row `z`:

    v(z)         = H_L(z)                 (1 x r_c, TT contraction of prefix cores)
    q(z)         = v(z) E v(z)'            (scalar, >= 0)
    p_ret_ref(z) = (q(z) + tau q0_ret_ref(z)) / Zc_ref

Tangent state per parameter: `dot_prefix_cores` and `dot_E`. Then

    dot v(z) = tt_evaluation_derivative(prefix, dot_prefix)(z)         [derivatives.py:606]
    dot q(z) = 2 v(z) E dot v(z)' + v(z) dot E v(z)'
    dot log p_ret_ref(z) = (dot q + tau dot q0_ret_ref) / (q + tau q0_ret_ref)
                           - dot Zc_ref / Zc_ref

with `dot q0_ret_ref = 0` (fixed defensive family) and
`dot Zc_ref = dot Z_h` (tau, Z_0,ref fixed). All bilinear/quadratic in
known tangents; no new approximations. Note `E = E(theta)` and
`v = v(theta)` both move; both terms of `dot q` are mandatory.
`dot log p_ret_phys(x) = dot log p_ret_ref(R^{-1}(x))` since the conversion
factor carries no theta dependence.

---

## 3. The ordered total derivative (score path)

We differentiate `log Lhat` along a parameter direction `e_k`; all `dot`
quantities are directional. Everything below is exact for the declared
program on its smooth branch (Section 5).

### 3.1 Target and shift tangents

    dot log f_t(z_j) = dot log p_ret_ref,t-1(z_prev,j)  [Sec. 2, needs dot_prefix_{t-1}, dot_E_{t-1}]
                     + dot log p_theta(x_curr|x_prev)   [adapter JVP]
                     + dot log p_theta(y_t|x_curr)      [adapter JVP]
    (the measure terms log|det DR| - log omega are theta-independent: dot = 0)

    j* = argmax_j log f_t(z_j)        (deterministic tie rule, Sec. 5)
    dot s_t = dot log f_t(z_{j*})
    dot g_t(z_j) = 0.5 * g_t(z_j) * ( dot log f_t(z_j) - dot s_t )

### 3.2 Ordered ALS replay (audit F2 core)

State: `(cores, dot_cores)`, initialized `(G^(0), 0)` (frozen initial cores
have zero tangent). For each update `(s, i)` in the frozen order:

1. Environment tangents. With per-row core matrices
   `C_m(z_j) = sum_k G_m[.,k,.] phi_k(z_j,m)` and their tangents
   `dot C_m` (from current `dot_cores`), the left/right environments and
   their tangents satisfy the product-rule recursions

       L_i = C_1 ... C_{i-1},   dot L_i = sum_{m<i} C_1..dot C_m..C_{i-1}
       R_i = C_{i+1} ... C_d,   dot R_i analogous,

   accumulated in one forward and one backward pass
   (`_left_and_dot_environments` / `_right_and_dot_environments`,
   derivatives.py). Then

       dot A_i = [ dot L_i (x) phi (x) R_i + L_i (x) phi (x) dot R_i ]_j
                                                       [differentiate_design_matrix]

   This is nonzero from the first update whose environment contains an
   already-updated (theta-dependent) core — the term my original memo
   omitted.

2. Per-update total-derivative solve. Differentiating (*):

       dot N_i = dot A_i' W A_i + A_i' W dot A_i        (+ A' dot_W A + dot_rho I, inactive: W, rho frozen)
       dot b_i = dot A_i' W g_t + A_i' W dot g_t
       N_i dot c_i = dot b_i - dot N_i c_i

   [`fixed_design_lsq_derivative`.] **Solver-reuse status (corrected per
   re-audit Finding 2): the donor primal solves a column-scaled AUGMENTED
   least-squares system (`_solve_scaled_augmented_ridge` ->
   `fitting.py:984-1010`), while the derivative primitive independently
   forms the unscaled normal equations and calls `tf.linalg.solve`
   (`derivatives.py:550-582`). These agree on the exact ridge minimizer in
   exact arithmetic but are DIFFERENT solvers with different conditioning;
   "shared factorization" is a P2A design GOAL, not current donor
   evidence.** P2A obligations therefore include: scaled-primal vs
   normal-equation solution agreement (easy fixtures AND near column-scale
   floors / condition thresholds); derivative consistency against the
   actual scaled primal solver; runtime/peak-memory with and without
   genuine factorization reuse. Cost claims in Section 4 inherit this
   caveat.

3. State update: `cores[i] <- c_i`, `dot_cores[i] <- dot c_i`. Later
   updates therefore see both the new value and the new tangent — the
   "ordered replay" property. Per-update residual telemetry (value solve,
   JVP solve, weighted fit, weighted fit-JVP) is inherited from the donor
   and REQUIRED in the engine.

After `S` sweeps: `(h_t, dot h_t)` as (cores, dot_cores).

### 3.3 Normalizer tangent

`Z_h = <h, h>_omega` is a quadratic form in the cores, so

    dot Z_h = 2 <h, dot h>_omega

computed by the same Gram chain with, summed over axes, one core pair
replaced by (core, dot_core) — equivalently the bilinear chain
[`squared_tt_normalizer_derivative` / `squared_tt_log_normalizer_derivative`,
derivatives.py:647]. With the complete normalizer (F4):

    dot log Zhat_t = dot s_t + dot Z_h,t / (Z_h,t + tau Z_0)

and `score_k = sum_t dot log Zhat_t` for direction `e_k`.

### 3.4 Retention tangent (the genuinely new derivation)

Split `(h_t, dot h_t)` at the boundary. Prefix tangents are the
`dot_cores` of the current-block cores directly. For the suffix Gram, with
`H_R` multilinear in the suffix cores:

    E_t     = int H_R H_R' omega(du)
    dot E_t = int ( dot H_R H_R' + H_R dot H_R' ) omega(du)

computed by the same transfer-matrix chain as `E_t` with, summed over suffix
axes, one core pair replaced by (core, dot_core):

    T_m       = einsum(G_m, G_m, M^(m))          [existing pattern, squared_tt.py:169]
    dot T_m   = einsum(dot G_m, G_m, M^(m)) + einsum(G_m, dot G_m, M^(m))
    dot E_t   = sum_{m in suffix} T_first..dot T_m..T_last

`dot E_t` is symmetric by construction (sum of a matrix and its transpose
under the pairing); the implementation must assert symmetry to tolerance
rather than symmetrize silently. `(dot_prefix, dot E_t, dot Zc_t)` is the
tangent state consumed by step t+1's Section 3.1. This closes the recursion:
the score chain never leaves polynomial-size objects.

### 3.5 What is new vs the donor (honest scope)

The donor gives 3.2 verbatim for one fitted TT with fixed rows. New in this
program, requiring fresh implementation + tests:
(a) the retained-object evaluator/tangent (Sec. 2) feeding `dot log f`;
(b) the suffix-Gram tangent `dot E_t` (3.4);
(c) the complete-normalizer chain hookup (3.3 with tau);
(d) the batched-parameter organization (Sec. 4);
(e) the time recursion tying (a)-(d) across t.
None of these exist today; P2 is not a port.

---

## 4. Batched-parameter organization (all p directions in one replay)

Every tangent quantity above is linear in the direction. Batch with a
leading parameter axis `[p, ...]`:

- Adapter JVPs delivered as `[p, N]` stacks (U-ADAPTER-JVP-1 checks vs tape).
- Environment tangents: the forward/backward recursions of 3.2(1) run once
  per update with `dot C_m` carrying a `[p]` axis — einsum-batched.
- Per-update solve: factor `N_i` once; solve `p` right-hand sides
  `dot b_i^(k) - dot N_i^(k) c_i` as a multi-RHS batch.
- Gram tangents (3.3, 3.4): bilinear chains batched over the `[p]` axis.

Cost caveat (audit F3, binding): the `dot A_i' W A_i`-type contractions are
`O(p N c^2)` per update and a materialized `[p, N, c]` stack is ~127 MiB at
(p=300, N=512, c=108) and ~900 MiB at r=8. Whether batched-forward,
chunked-forward (frozen chunk policy over the p axis), or adjoint replay of
the scalar chain wins is DECIDED BY P2A MEASUREMENT; the `<= 6x` figure is a
gate, not a claim of this note. The adjoint candidate differentiates the
identical program (same smooth branch), reusing `N_i` factorizations
transposed; its derivation, if selected, extends this note before
implementation.

---

## 5. Smoothness domain and status contract (F6)

The program is smooth in `theta` on the open set where: (i) the argmax
`j*` is unique at every step; (ii) no normalizer/denominator floor
activates; (iii) every `N_i` passes the condition veto. On that set the
score above is the exact directional derivative of `log Lhat`.

- Ties (i), corrected per re-audit Finding 3: at a true tie the max-defined
  scalar has NO ordinary derivative (its directional derivative is the max
  over active branches, not the lowest-index branch derivative).
  Deterministic tie-breaking selects an implementation branch; it does not
  restore differentiability. Contract: a detected tie (two rows within the
  declared tie tolerance of the max) **invalidates the score-bearing
  evaluation for claim use** (hard status veto). The selected-branch
  derivative may be emitted as DIAGNOSTIC TELEMETRY only. (The alternative
  — redefining the finite target as the selected branch, a different
  declared scalar — is documented but not adopted.) U-SHIFT-2 asserts the
  claim veto at constructed persistent ties; tie-neighborhood FD tests must
  not treat a branch derivative as the derivative of max.
- Floors (ii) and condition vetoes (iii): fire status flags and invalidate
  the evaluation for claims (V8); the score solve additionally returns
  `DERIVATIVE_SOLVE_FAILURE` semantics from the donor.

---

## 6. Correctness obligations (tests this note binds)

| Identity derived above | Test |
|---|---|
| 3.2 full ordered replay, nonzero dot_A on later updates, every intermediate core vs FD | U-ALS-REPLAY-1 |
| batched replay == loop of single-direction replays | U-ALS-BATCH-1 |
| Sec. 2 evaluator == brute-force marginal integration; rank>1 E never stamped scalar | U-MARG-TYPE-1 |
| Sec. 2 / 3.4 tangents (dot_prefix, dot_E) vs FD | U-MARG-DERIV-1 |
| 3.3 complete-normalizer value+score identity at tau>0 | U-TAU-1 |
| V1/F5 measure conversion end-to-end both conventions | U-MEASURE-1 |
| 3.1 tie behavior under constructed persistent ties | U-SHIFT-2 |
| full-path score vs FD at multiple theta points (boundary/threshold/tie neighborhoods) | I-P2-1 |
| same-scalar property (score differentiates the exact emitted value) | I-P2-3 |

FD companions follow the FD-quality-first protocol (step ladders, checking
the FD baseline before blaming the analytic side).

## 7. Nonclaims

This note establishes the exact tangent of the declared finite program on
its smooth branch. It does not establish: cost of any mode (P2A), rank
sufficiency (P1B), approximation quality vs the true likelihood
(same-target gates), HMC readiness (P6 campaign), or the retained-law
design for innovation-pushforward transitions (open item A16 — this note
covers `density_kernel` mode only; the pushforward mode requires its own
retention derivation before any structural-row score claim).

---

## Addendum A (2026-08-17): Manual adjoint (reverse) sweep — the P2 score mode

Trigger: P2A measurement (result note
`bayesfilter-p2a-cost-prototype-result-2026-08-17.md`) disqualified
forward tangent replay at p=300 (ratio ~326x vs the <=6x gate; the
per-parameter dot_A construction is O(p N c^2) as audit F3 warned). Per
Section 4's provision, the adjoint derivation extends this note before
implementation. This remains a MANUAL derivation (Method A discipline):
every adjoint map below is the explicit transpose of a forward-linear map
already derived in Sections 2-3; no autodiff is invoked anywhere.

### A.1 Setup

The value program is a finite composition. Write its state after update
step u (a (sweep, core) pair in the frozen order, plus per-step assembly,
normalizer, and retention nodes) as s_u, with s_0 = frozen inputs and the
final node emitting the scalar log Lhat. Sections 2-3 derived, for every
node, the forward-linear tangent map ds_{u+1} = F_u[ds_u; dtheta_u],
where dtheta_u collects the direct theta-dependence entering at node u
(model-density JVP rows, dot s_t shift branch, dot L from dot E, etc.).

The gradient is assembled by one reverse sweep: initialize the adjoint of
the final scalar at 1 and propagate

    bar s_u = F_u^T [bar s_{u+1}],
    grad_theta += (direct-dependence maps at node u)^T [bar s_{u+1}],

i.e. each parameter receives contributions ONLY through the adapter
JVP-transposes (VJPs) at the nodes where theta enters directly. Cost:
one forward value pass (with checkpoints) + one reverse pass whose per-
node cost matches the forward node cost — O(1) x value in flops,
independent of p except for the final cheap VJP accumulations.

### A.2 Node-by-node adjoints (transposes of Sections 2-3)

1. **Core solve** (forward: N dot_c = dot_b - dot_N c with
   N = A'WA + rho I):
   given bar_c, solve the TRANSPOSED system lambda = N^{-T} bar_c
   (N symmetric -> same LU/Cholesky factors reused, one extra solve).
   Then pairing <bar_c, dot_c> = <lambda, dot_b - dot_N c> with
   dot_b = dot_A' W g + A' W dot_g and
   dot_N c = dot_A' W (A c) + A' W (dot_A c):

       <bar_c, dot_c> = <lambda, dot_A' W (g - A c)>
                        + <lambda, A' W dot_g>
                        - <lambda, A' W dot_A c>
       hence
       bar_g = W A lambda
       bar_A = W (g - A c) lambda'  -  (W A lambda) c'      (an N x cdim matrix)
   Both terms use quantities already present (residual g - A c, lambda,
   c); no p-dependence.
2. **Design assembly** (forward: dot_A from environment tangents,
   Sec. 3.2(1)): the adjoint distributes bar_A onto the core adjoints
   bar_G_m of every OTHER core through the transposed left/right
   environment recursions — one backward and one forward accumulation
   pass mirroring `_left_and_dot_environments` /
   `_right_and_dot_environments` with cotangents. Same einsum shapes
   transposed.
3. **Sqrt target** (forward: dot_g = 0.5 g (dot log f - dot s)):
       bar_logf += 0.5 g bar_g;   bar_s -= sum(0.5 g bar_g);
   with the argmax-gathered branch routing bar_s to row j* (Sec. 5 tie
   contract unchanged).
4. **Target assembly** (forward: dot log f = dot log p_ret_ref
   + adapter JVP terms): bar_logf splits into
   (a) retained-evaluator cotangent -> A.2.5;
   (b) adapter VJPs: the adapter contract gains
       `transition_log_density_vjp(x_c, x_p, cotangent) -> bar_theta`
       (and observation/initial analogues) — the transpose of the JVP
       obligation already in the contract; for TF-defined model densities
       these VJPs are closed-form or per-model manual, recorded per
       adapter as before.
5. **Retained evaluator** (forward: Sec. 2 dot log p_ret_ref):
   bar over (q + tau) rows distributes to
       bar_v = 2 (bar_row / (q+tau)) (v E)        [prefix evaluation adjoint]
       bar_E += (bar_row / (q+tau)) v' v          [Gram adjoint, symmetric]
       bar_Zc -= sum(bar_row) / Zc
   and bar_v propagates through the transposed prefix TT contraction to
   bar_prefix cores (transpose of `prefix_row_vectors_tangent`).
6. **Branch factor** (forward: dot L from dot E by Cholesky JVP): the
   adjoint is the Cholesky VJP (Phi-operator transpose), closed form,
   donor pattern `higher_moment_contract_e.py:55` transposed; P2
   smoothness guard (PD + conditioning veto) unchanged.
7. **Normalizer chain** (forward: dot Z_h = 2 <h, dot h>): adjoint
   bar_h += 2 bar_Z h through the transposed Gram chain (replace one
   core by cotangent in the same transfer-matrix pattern).
8. **Retention split** (forward: (dot_prefix, dot_E, dot_Zh) from
   dot cores): adjoints of the suffix/prefix Gram chains, same
   transposed-transfer pattern as 7; emits bar over the fitted cores of
   step t, which feed 1-2 of step t's reverse sweep, closing the
   time recursion in reverse order t = T..1.

### A.3 Checkpointing contract (the T=120 stress object)

Stored per update: design row-set id (frozen seed), LU/Cholesky factors
OR the recompute seed, solution c, residual g - A c, shift branch index,
and per-step (E, L, Zc, prefix cores). Recompute-from-seed is admissible
(deterministic program); the store-vs-recompute trade is measured by the
P2A full-horizon stress ON THIS MODE. All status/veto semantics (ties,
floors, condition, PD-Gram) bind identically to the forward chain.

### A.4 Obligations bound by this addendum

| Identity | Test |
|---|---|
| adjoint gradient == forward-replay gradient at p=3 (both manual) | I-P2-4 (new; FD-independent cross-check) |
| adjoint gradient vs FD at multiple theta points | I-P2-1 (unchanged) |
| per-node adjoint vs forward-tangent inner-product identity <bar_out, F[din]> == <F^T[bar_out], din> on random cotangents (each node class) | U-ADJ-NODE-1 (new) |
| solve-node adjoint vs FD through `_solve_scaled_augmented_ridge` | U-ADJ-SOLVE-1 (new) |
| <=6x gate at p=300 measured on the adjoint implementation | P2A gate (carried) |
