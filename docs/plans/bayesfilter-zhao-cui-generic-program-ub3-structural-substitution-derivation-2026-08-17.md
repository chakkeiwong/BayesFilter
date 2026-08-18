# UB-3 Derivation Note: Structural Substitution Mode Through the Score Chain

Date: 2026-08-17
Status: `DERIVATION_FOR_REVIEW` (unblock artifact for P2S only; the
density_kernel track P1B/P2A/P2 does not depend on this note)
Program plan: `bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
(rev 3, Section 3.6 and phase P2S).
Companions: UB-1 (density_kernel score chain, content gate passed);
branch-axis design note 2026-08-16 (the target assembly this note extends).
Authority for structural doctrine: Ch18b
(`docs/chapters/ch18b_structural_deterministic_dynamics.tex`), validation
gates section; owner decision D2.

## 0. Scope and preconditions

Model class (Ch18b structural split): `x_t = (m_t, k_t)`,
`dim m = n_s` (declared stochastic block), `dim k = n_d`
(deterministic completion), `n = n_s + n_d`:

    m_t = T_m(m_{t-1}, eps_t; theta),   p_theta(m_t | m_{t-1}) proper density
    k_t = T_k(k_{t-1}, m_{t-1}, m_t; theta)   (deterministic)
    y_t ~ p_theta(y_t | x_t)

V1 precondition (V13, plan): for every (m_{t-1}, m_t) in the declared
support box, the map `k_{t-1} -> T_k(k_{t-1}, m_{t-1}, m_t; theta)` is a
global diffeomorphism of the k-block with inverse
`k_{t-1} = S(k_t; m_{t-1}, m_t; theta)` and Jacobian

    J(k_t; m_{t-1}, m_t; theta) = |det d S / d k_t|   (n_d x n_d block det)

with a declared minimum-singular-value bound; violation is a hard veto.
The singular case (e.g. phi -> 0 in the Ch18b toy) is out of scope for v1.

## 1. The exact substitution recursion (value path)

Substituting the Dirac kernel and integrating against `k_{t-1}` (one
change of variables per (m_{t-1}, m_t) slice, legitimate by the V13
diffeomorphism):

    p_t(m_t, k_t) = (1/L_t) p_theta(y_t | x_t) *
        int p_{t-1}(m_{t-1}, S(k_t; m_{t-1}, m_t)) p_theta(m_t | m_{t-1})
            J(k_t; m_{t-1}, m_t) d m_{t-1}                          (1)

`L_t` = likelihood increment. The recursion state is the FULL-state
filtered density (no information loss: `k_{t-1}` is recovered exactly
inside the integrand, "computed, not noised" per Ch18b).

### 1.1 Engine realization (branch-axis form)

The engine fits, per step, one functional TT over the block

    (z_c, g, w) := (current-state axes for x_t = (m_t, k_t),
                    branch axis g in {0..B-1},
                    previous STOCHASTIC axes for m_{t-1})

— dimension `n + 1 + n_s` (vs `2n + 1` for density_kernel: the k_{t-1}
axes are GONE; that is the structural dimension reduction, now
`n + n_s` continuous axes).

Reference-measure target on frozen rows, with x_c = R_c(z_c),
m_p = R_s(w), and the previous retained object
`P_{t-1} = RetainedQuadraticForm` over the full previous state
(prefix basis over n axes):

    k_prev(row) = S(k_t; m_p, m_t; theta)             [adapter map]
    z_prev(row) = R^{-1}( m_p, k_prev )               [engine map]
    q_row       = H_L(z_prev) E H_L(z_prev)'          [prefix eval]
    G_row       = p_theta(m_t|m_p) p_theta(y_t|x_c) J(row)
                  * conv_c * conv_s                    [smooth kernel > 0]
    f_row       = (q_row + tau) G_row / Zc_{t-1}

where conv_c, conv_s are the current-block and m-block measure
conversions (log|det DR| - log omega terms, theta-independent). Branch
targets (branch design note): with any factor L L' = E,

    u_g(row) = H_L(z_prev(row)) L[:, g],   u_B = sqrt(tau)
    target(row, g) = u_g(row) * sqrt(G_row) * exp(-s_t / 2)

All rows are smooth in every continuous variable AND in theta (S, J, the
densities, and H_L are smooth; V13 keeps J bounded). Retention: split
after the z_c block exactly as in density_kernel mode; the retained
object is again a full-state RetainedQuadraticForm. Increment:
`log Lhat_t = s_t + log Zc_t - log Zc_{t-1}` (branch engine convention).

Note the composition subtlety the plan text calls out: `z_prev` depends
on the ROW through (k_t, m_p, m_t) via S, so the previous prefix TT is
evaluated at MOVING points — this is the "moving-point retained density"
of the re-audit; its theta-derivative therefore has a spatial term
(Section 2.2).

## 2. Score chain (P2S obligations, per plan Findings 5/6 text)

Per parameter direction, on top of the UB-1 chain the structural mode
adds exactly three new tangent contributions inside `dot log f_row`
(equivalently in dot target):

### 2.1 Kernel terms (adapter JVPs, standard)

    dot log G_row = dot log p_theta(m_t|m_p) + dot log p_theta(y_t|x_c)
                    + dot log J(row)

`dot log J` is the trace identity
`tr( (dS/dk_t)^{-1} d(dS/dk_t)/dtheta )` or supplied directly by the
adapter as a JVP (preferred; for the Ch18b toy, J = 1/|phi| gives
dot log J = -dot phi / phi).

### 2.2 Moving-point retained term (the load-bearing new term)

    d/dtheta [ log( q(z_prev(theta)) + tau ) ]
      = [ dot q_param + grad_z q . dot z_prev ] / (q + tau)

with:
- `dot q_param` = the UB-1 Section 2 parameter tangent
  (2 v E dot_v' + v dot_E v') evaluated at z_prev — carried by
  (dot_prefix_{t-1}, dot_E_{t-1});
- `grad_z q = 2 (H_L E) . dH_L/dz` — the retained-evaluator SPATIAL
  gradient, computed by the same TT contraction with one basis factor
  replaced by its derivative (`LegendreBasis1D.derivative` exists,
  bases.py:295); includes the defensive component trivially
  (grad tau = 0);
- `dot z_prev = dR^{-1}/dk_prev . dot S` where
  `dot S = dS/dtheta |_{row}` is an adapter JVP and `dR^{-1}` is the
  (affine) inverse coordinate map derivative — the `R^{-1}(S)`
  propagation named in the plan.

### 2.3 Previous-normalizer term

    - dot Zc_{t-1} / Zc_{t-1}    (carried scalar, UB-1)

### 2.4 What stays identical to UB-1

Ordered ALS replay with moving environments (dot_A terms), branch-factor
tangent `dot L` from `dot E` (Cholesky JVP; donor
`higher_moment_contract_e.py:55` / `ledh_contract_e_reset_tf`), Gram
normalizer tangents, retention tangents, complete-Z discipline, tie/floor
status semantics. The P2 smoothness guard (PD Gram + conditioning veto on
E before Cholesky) applies unchanged; V13 adds the S-map
minimum-singular-value veto.

## 3. Correctness obligations (tests this note binds)

| Identity | Test |
|---|---|
| Substitution recursion (1) == dense (x_{t-1}, eps_t) integration on the Ch18b toy, T in {1, 5} | U-STRUCT-PUSHFORWARD-1 (retargeted) |
| Engine value == (1) at toy scale, T in {5, 20} | I-P2S-1 (dense reference arm) |
| Spatial gradient grad_z q vs FD in z | U-STRUCT-SPATIAL-1 |
| Full moving-point tangent (2.2) vs FD in theta at fixed program | U-STRUCT-MOVING-1 |
| dot log J adapter JVP vs FD | U-STRUCT-J-1 |
| phi-near-0 trips the V13 veto (no silent answer) | I-P2S-2 |
| Ch18b gate list (metadata, constraint-support, linear recovery, degenerate transition) | I-P4-5 |

## 4. Nonclaims

Exactness claims are for the declared finite program on the V13-fenced
subclass only. No claim for non-invertible completions (shock-history
retention or labeled regularization remain future work per Ch18b policy).
No rank, cost, or accuracy claim — those are P2S's measured gates. The
dimension-reduction statement (n + n_s + 1 axes vs 2n + 1) is structural
bookkeeping; whether it converts into practical rank savings is measured
at P2S (plan telemetry obligation).
