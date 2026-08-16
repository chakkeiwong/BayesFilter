# Audit Request Memo: Generic Zhao-Cui Squared-TT Filtering with Batched Analytical Score

Date: 2026-08-15
From: Claude (BayesFilter session, generic Zhao-Cui program)
To: Codex (independent auditor)
Status: `AUDIT_REQUESTED`
Plan under audit:
`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`

---

## 1. What we are asking Codex to do

Audit the referenced program plan as a skeptical developer and applied
mathematician, and return a written verdict per Section 10. Specifically:

1. Check every mathematical claim in Sections 3-5 of this memo (derivations
   included below so the audit does not depend on trusting the author).
2. Check the complexity accounting in Section 6 against the target scale
   `n = 100` states, `m = 100` observations, `T = 120`, `p = 300`.
3. Check the prior-error ledger (Section 7) against the plan: verify each
   previously corrected misunderstanding is structurally prevented, not just
   mentioned.
4. Check the test program (Section 8) for completeness: every mathematical
   identity claimed must have a unit test; every phase gate an integration
   test; flag anything untested.
5. Check the leaderboard specification (Section 9) for wrong-target
   comparisons, missing same-target references, or claim-language violations.
6. Flag any place where the plan could pass all its own gates and still
   mislead us (pre-mortem review).

Repository governance context binds this audit: TF/TFP float64 backend rule,
fixed-variant policy (no runtime adaptation), Method A same-scalar manual
score rule, batch-native NeuTra rule, per-scope tuning rule, and the
Chapter 18b structural validation gates
(`docs/chapters/ch18b_structural_deterministic_dynamics.tex`, final section).

---

## 2. Why this program exists: the concrete target case

Our workhorse inference is HMC (and sometimes MLE). Both require the exact
gradient of the log-likelihood surrogate actually evaluated. The typical
serious application is a data-rich structural macro model (NAWM-class):

| Quantity | Symbol | Target value |
|---|---|---|
| State dimension | n | ~100 (60-100 range) |
| Observation dimension | m | ~100 (80-100 range) |
| Horizon | T | ~120 (quarterly, 30 years) |
| Parameters | p | ~300 |

Why no existing route suffices at this scale:

- **Exact Kalman** applies only to the linearized model. The scientific
  interest is the nonlinear filtering problem (higher-order pruned solutions,
  occasionally-binding constraints, SV blocks) where the Kalman likelihood is
  wrong relative to the declared model.
- **Particle filters** give a non-differentiable, noisy likelihood; no HMC.
- **Gaussian closures (UKF/SGQF/GenUT/SVD-UKF)** are fast and differentiable
  and remain leaderboard comparators, but they are single-Gaussian moment
  closures; the accuracy question for strongly nonlinear/non-Gaussian
  filtering laws is exactly what a density-based route should answer.
- **Dense grid/mixture references** are exact-target but cost `O(q^n)`:
  unusable beyond n ~ 3.
- **Existing repository Zhao-Cui routes** are per-model constructions; the
  only generic multistate TT path re-imports `O(q^n)` at its retention step
  (`_multistate_tt_grid_retained_from_density`, filtering.py:4048) and its
  score path re-runs the full recursion per parameter
  (filtering.py:1463ff) — disqualifying at p = 300.

The squared-TT route is the only family in the repository whose value AND
exact program gradient both scale polynomially in (n, m, p) — if and only if
the two defects above are repaired. That is the program.

---

## 3. Mathematical object and derivations (audit target A)

### 3.1 Functional TT and the squared density

Fix a product basis `{phi_k}_{k=1}^b` per axis on a mapped box
`Omega = prod_i [a_i, b_i]` (mapped Legendre in-repo). A functional TT with
frozen ranks `r_i` is

    h(z_1..z_d) = sum_{alpha} prod_{i=1}^d G_i[alpha_{i-1}, k_i, alpha_i] phi_{k_i}(z_i),

cores `G_i in R^{r_{i-1} x b x r_i}`, `r_0 = r_d = 1`. Storage `O(d b r^2)`.

The filtering density approximation is the **squared-TT density**

    p(z) = (h(z)^2 + tau q_0(z)) / Z,   Z = Z_h + tau Z_0,
    Z_h = int_Omega h(z)^2 omega(dz),

with `q_0` a fixed defensive density and `tau >= 0` (tau = 0 in the SVX
route). Nonnegativity is structural (a square), not enforced by clipping —
no non-smooth operation is introduced for positivity. This is the
Zhao-Cui (JMLR 2024) squared-Rosenblatt construction; pinned source audit
`third_party/audit/zhao_cui_tensor_ssm_p10/`.

### 3.2 Exact normalizer by Gram chain (the enabling identity)

Claim: `Z_h` is computable exactly (float64) in `O(d b^2 r^4)` flops.

Derivation. Substituting the TT expansion,

    Z_h = sum_{alpha, beta, k, l} prod_i G_i[a_{i-1} k_i a_i] G_i[b_{i-1} l_i b_i] M^(i)_{k_i l_i},
    M^(i)_{kl} = int phi_k(z_i) phi_l(z_i) omega_i(dz_i),

because the box measure factorizes and each axis integral hits exactly one
basis pair. Define per-axis transfer matrices

    T_i[(a b),(a' b')] = sum_{k,l} G_i[a k a'] G_i[b l b'] M^(i)_{kl}   (r^2 x r^2)

then `Z_h = (T_1 T_2 ... T_d)` collapsed against boundary vectors. Each `T_i`
costs `O(b^2 r^4)`; the chain is `d` products of `r^2 x r^2` matrices.
Implementation: `SquaredTTDensity.sqrt_square_normalizer` (squared_tt.py:164,
the einsum `alb,AmB,lm->aAbB`). With orthonormal bases `M = I`.

Audit check: verify the einsum indices implement exactly this contraction,
and that the mass matrices `M^(i)` correspond to the declared measure
convention (REFERENCE_MEASURE vs LEBESGUE) — a mismatch here would silently
rescale likelihood increments.

The same contraction with a subset of axes integrated out yields exact
marginals: marginalizing axis i replaces its pairing by `M^(i)` contraction,
leaving a squared-TT object on the remaining axes. This is the basis of the
P1 retention design.

### 3.3 One filter step (fixed-variant form)

Given retained `p_{t-1}` (squared-TT over `x_{t-1}`) and data `y_t`:

(a) **Target assembly** on frozen designs: the unnormalized joint over the
adjacent pair,

    f_t(x_t, x_{t-1}; theta) = p_{t-1}(x_{t-1}) p_theta(x_t | x_{t-1}) p_theta(y_t | x_t),

square-root target `g_t = exp((log f_t - s_t)/2)` with max-shift
`s_t = max_grid log f_t`.

(b) **Frozen-design ALS fit**: for each of the `2n` cores in a frozen sweep
order, a frozen number of sweeps `s`, solve

    c = (A' W A + rho I)^{-1} A' W g_t                        (*)

with A = basis design at frozen nodes, W = frozen weights, rho = frozen
ridge. Everything on the left side of (*) is theta-independent.

(c) **Likelihood increment**: `log Zhat_t = log Z_{h_t} + s_t` via 3.2;
`log Lhat(theta) = sum_t log Zhat_t`.

(d) **Retention**: `p_t = (h_t^2 + tau q_0)/Z_t` marginalized to `x_t`
(exact, by 3.2), kept in squared-TT form. [Today's multistate code instead
evaluates `p_t` on a dense `q^n` tensor grid — the defect P1 removes.]

### 3.4 Differentiability: fitted is not adaptive (audit target A-critical)

Claim: `theta -> log Lhat(theta)` is piecewise-smooth with closed-form exact
tangent, the non-smooth set being measure-zero argmax ties plus explicitly
flagged floor activations.

Argument, term by term:

1. Model log-densities are smooth in theta (adapter obligation).
2. `g_t` is smooth in `(log f_t, s_t)`; `s_t` is a max of finitely many
   smooth functions -> piecewise smooth, kink set = argmax ties
   (measure-zero; tangent uses the argmax-gathered branch).
3. (*) is a **linear** map from `g_t` to `c` with theta-independent,
   frozen, factorizable operator. Hence
   `dot c = (A'WA + rho I)^{-1} A'W dot g_t` — exact, one multi-RHS
   back-substitution per tangent. No pivoting, no rank decisions, no
   iteration-to-tolerance: the sweep count is frozen, so the composition of
   `s x 2n` such solves is a fixed smooth map.
4. TT evaluation and the Gram chain are polynomial (multilinear/quadratic)
   in the cores: `dot Z_h = 2 <h, dot h>_omega` by bilinearity, computed by
   the same chain with one core replaced by its tangent.
5. Quotients and logs are smooth away from floors; floor activations raise
   status flags and invalidate the evaluation (governance rule V8).

What breaks differentiability is precisely what the **adaptive** paper
algorithm does: TT-cross pivot selection, rank adaptation, tolerance-driven
sweep counts — discrete, data-dependent choices. The fixed-variant policy
moves all discreteness into the offline tuning procedure, so the runtime
object is a fixed smooth program. The analytical score is the exact gradient
of the declared finite program `log Lhat` — NOT of the idealized adaptive
algorithm and NOT of the true likelihood. HMC targeting `log Lhat` with its
exact gradient is MH-correct for the surrogate posterior; surrogate-vs-true
distance is a separately measured approximation question (same-target
gates), never assumed.

Existing empirical support: five Method A models pass same-program FD gates
at 1.4e-8 .. 2.0e-10 (multimodel campaign result 2026-08-04); per-primitive
FD tests in `tests/highdim/test_fixed_branch_derivatives.py`.

Audit check: confirm no code path in the proposed engine makes a runtime
data-dependent discrete choice (V1), and that the retention design (P1)
introduces neither SVD re-truncation (non-smooth at degenerate singular
values, V3) nor dense grids (V2).

### 3.5 Batched tangent propagation (audit target A)

Claim: all `p` parameter tangents can ride one recursion at
`<= ~6x` value cost, in the same pattern as the UKF/SGQF/LEDH analytic
scores (which propagate `(d mu_t / d theta_j, d P_t / d theta_j)` for all j
simultaneously against shared factorizations).

Reason: every tangent term in 3.4 is **linear in the tangent**:

- `dot g_t^(j) = 0.5 g_t (dot log f_t^(j) - dot s_t^(j))` — elementwise,
  batched over a leading p-axis;
- `dot c^(j)` — the operator `(A'WA + rho I)` is factorized ONCE per core
  solve; p tangents are p extra right-hand sides (multi-RHS solve,
  `O(N c + c^2)` each versus `O(N c^2 + c^3)` to factor);
- `dot Z_h^(j) = 2 <h, dot h^(j)>` — bilinear, batched;
- retained-density tangents — quotient rule, batched.

The current multistate score path violates this by re-running the entire
path per parameter index; the plan disqualifies that pattern (V4).
Cost model: value `C_v`; gradient `C_v + p * (multi-RHS + bilinear terms)`
where the per-parameter increment is a small fraction of the shared work.
The `<= ~6x` figure at p = 300 is an estimate to be verified by the P2
scaling harness; the audit should treat it as a gate, not a fact.

Model-side tangents `d/dtheta log p_theta` on N grid rows: supplied by the
adapter as batched JVPs; TF autodiff of *model densities inside the adapter*
is acceptable (the Method A manual-score rule binds the engine chain);
closed-form preferred where cheap; choice recorded per adapter.

---

## 4. Complexity accounting at the target scale (audit target B)

Assumptions for the concrete count: b = 12 basis functions/axis, rank r
frozen (table below), N = 512 fit rows/core, s = 2 sweeps, TT over the
adjacent pair -> 2n axes, float64.

### 4.1 General formulas

Memory:

    cores + retained:      O(n b r^2)
    frozen designs:        O(N b r^2) per axis pattern
    tangent stack:         O(n b r^2 p)          [all-parameter batch]
    NO term exponential in n, m, or p.

Per-step time:

    target evaluation:     O(N (c_trans(n) + c_obs(n,m) + n b r^2))
    core fits:             O(s n (N b^2 r^4 + b^3 r^6))
    Gram normalizer:       O(n b^2 r^4)
    TT retention:          O(n b^2 r^4)
    gradient increment:    O(p) multi-RHS/bilinear terms (Section 3.5)

`m` enters ONLY through `c_obs`: for dense Gaussian observation noise, one
`m x m` Cholesky per step (`O(m^3)`) plus `O(m^2)` per sample row. At
m = 100: ~1e6 flops for the factorization + ~5e6 per step for N = 512 rows
— negligible against the fits. Observation dimension is a non-problem for
this architecture; it never touches the tensor format.

### 4.2 Concrete numbers, n = 100, m = 100, T = 120, p = 300, r = 3

    cores (one step):        2*100*12*9      ~ 21.6e3 floats  ~ 0.17 MB
    tangent stack (p=300):                       ~ 52 MB
    fits per step:  2*200*(512*144*81 + 1728*729) ~ 2.9e9 flops
    value pass (T=120):                          ~ 3.5e11 flops
    gradient pass (batched, <= ~6x):             ~ 2e12 flops

On a modern fp64 GPU (or 32-core AVX-512 CPU at ~1e11 flops/s effective):
value ~ seconds; full 300-parameter gradient ~ tens of seconds CPU,
seconds GPU. HMC with ~1e3-1e4 gradient evaluations per effective sample is
then feasible in GPU-hours, not weeks. (These are order estimates; the P2
harness measures the real constants and is a gate.)

### 4.3 The honest nonlinearity: rank

All of 4.2 assumes frozen r ~ 3 suffices. Rank is where the curse of
dimensionality relocated; it is a property of the model's filtering-law
correlation structure, not of the algorithm. Cost scales as r^4 (fits) and
r^6 (solves):

    r = 3   -> baseline above
    r = 8   -> ~50x value cost; N must grow past 512 (c = b r^2 = 768 columns)
    r = 16  -> ~2000x; impractical at T = 120

Consequence: the P1 LGSSM n-ladder (n up to 64+, exact Kalman truth)
measuring the rank-sufficiency curve r*(n) is the program's go/no-go
instrument, run BEFORE the batched-score investment. A bad curve triggers
the declared pivot (structure-exploiting transitions, coordinate ordering)
rather than silent scope creep. Audit check: confirm the plan really orders
P1 before P2/P3 spend and that the pivot branch is concrete.

DSGE-specific rank remark for the auditor: pruned perturbation solutions
have block-structured transitions (first-order block driving higher-order
blocks) and structurally deterministic completions (Section 9.3). Both
plausibly cap the filtering law's effective rank; neither is proven here.
`not checked` — exactly what the ladder measures.

---

## 5. Architecture under audit

    [Adapter]  log p_theta(x_t|x_{t-1}), log p_theta(y_t|x_t), initial law,
               batched theta-JVPs of all three on given point sets;
               parameter chart; structural state partition (Section 9.3);
               support/scale hints; manifest.
    [Engine]   frozen designs; ALS fits + multi-RHS tangent solves;
               Gram-chain normalizers + bilinear tangents; squared-TT
               retention + retained tangents; theta-batch axis; status
               flags; XLA-compilable. Model-independent by construction.
    [Tuning]   procedure v1: scout -> resolution ladder -> rank/sweep
               selection -> score admission (FD-quality-first) -> frozen
               scope artifact. Per-scope admission; warm starts only across
               scopes.

Program-level vetoes V1-V8 (fixed-branch, no dense retention, no runtime
SVD truncation, batched tangents, same-scalar, TF/float64, batch-native,
floor honesty) are stated in the plan; the audit should treat any
achievable violation as a finding.

---

## 6. Phases (summary; full text in the plan)

    P0  contract + engine skeleton (no behavior change; suite stays green)
    P1  TT-native retention + LGSSM n-ladder r*(n)   <- go/no-go science
    P2  batched-tangent score through retention + p-scaling harness
    P3  batch-native/XLA port (1e-12 eager parity)
    P4  adapter suite + reproduction gates (near-bit vs admitted routes)
    P5  tuning procedure v1 execution + scope admission
    P6  leaderboard + HMC admission under NeuTra governance + NAWM-scale
        synthetic rehearsal (n=100, m=100, T=120, p=300)

---

## 7. Prior-error ledger (audit target C)

Each row: the error we actually made in this program's history, and the
structural prevention now in the plan. Codex: verify each prevention exists
and is testable, not aspirational.

| # | Error made | Prevention in plan |
|---|---|---|
| E1 | Goal drift: SV-specific work (sigma-generalizing the batched SV route) treated as the program | Mission statement fixes the generic algorithm as the deliverable; SV is one adapter; memory note recorded |
| E2 | Misread the five per-model score backends as a defect | They are five comparator algorithms; leaderboard design is intentional; only the ZC family is being unified |
| E3 | "Generic" misread as tuning-free | Tuning procedure v1 is a first-class deliverable (P5); per-scope artifacts mandatory; ad-hoc hand-tuning disqualified |
| E4 | "Squared-TT is fitted, hence adaptive, hence non-differentiable, hence no HMC" | Section 3.4 derivation: frozen-design LSQ is a smooth linear image; only the adaptive paper variant is non-differentiable and it is excluded by V1; FD evidence cited |
| E5 | Accepted per-parameter score recursion | V4 disqualifies it; P2 gate requires all-p batched tangents at <= ~6x value cost, verified at p = 300 |
| E6 | Called adjoint mode a "precondition" at p = 300 (wrong after E5) | Adjoint demoted to optional optimization; batched forward mode is the required pattern (matches UKF/SGQF/LEDH practice) |
| E7 | Suspected sigma=1 batch TT restriction was a bug | Recorded as contract narrowness; its real defect (inline-welded model) is exactly what the adapter boundary removes |
| E8 | Believed "exact-transformed ZC TT fails on SIR because of dimensionality" without locating the mechanism | Mechanism located: dense-grid retention (filtering.py:4048). P1 removes it; SIR T>0 score is an explicit P2 gate |
| E9 | Interpreted single-path gaps as findings (dim-2 "anomaly" was seed scatter, shown by the 16-seed sweep) | All gates use exact references or replicated evidence; single-run continuous gaps are descriptive by rule |
| E10 | Stalled-session migration bug: `observations[:, :dim]` sliced from a [T,1] simulated path (dims>1 broken) | Shape-contract unit tests on every dataset builder (Section 8.1, U-DATA) |
| E11 | Wrong-target comparisons (KSC-surrogate vs exact rows) before Jacobian correction discipline | Leaderboard spec (Section 9) requires same-target references per row and raw-y Jacobian-corrected cross-family columns labeled descriptive |
| E12 | Near-total likelihoods compared across different transform conventions (offset vs exact log-square) | Jacobian-correction unit tests including the offset->0 consistency identity (U-JAC) |

---

## 8. Test program (audit target D)

Naming: U-* unit, I-* integration, G-* phase gates. All tests
TF float64, CPU-deterministic; GPU/XLA variants where stated. Every test
lands in `tests/` with the standard pytest harness; long ladders live in
`docs/benchmarks/` scripts with schema tests.

### 8.1 Unit tests — mathematical identities

- U-GRAM-1: Gram-chain normalizer vs brute-force quadrature of h^2, d in
  {1,2,3}, random cores, both measure conventions. Tol 1e-12 relative.
- U-GRAM-2: marginalization contraction vs brute-force marginal on a grid,
  d = 3 -> keep {1}, {2}, {1,3}. Tol 1e-12.
- U-GRAM-3: normalizer invariance under exact rank-preserving core
  regauging (left-multiply G_i by invertible S, right-divide G_{i+1}).
- U-LSQ-1: `fixed_design_lsq_derivative` vs FD on random smooth targets
  (existing test extended to multi-RHS batched form).
- U-LSQ-2: multi-RHS tangent solve == loop of single-RHS solves, exactly
  (same factorization; bitwise or 1e-15).
- U-EVAL-1: `tt_evaluation_derivative` product rule vs FD (existing,
  extended to batched dot-core stacks).
- U-SQN-1: `squared_tt_log_normalizer_derivative` vs FD (existing).
- U-RET-1: retained squared-TT density tangent vs FD at random points.
- U-RET-2 (new P1 object): TT-native retention output == dense-grid
  retention output for n in {1,2,3} where dense is affordable. Tol 1e-10.
- U-RET-3: retention tangent (new P2 term) vs FD, n in {1,2,3}.
- U-SHIFT-1: max-shift tangent at a non-tie point vs FD; at a constructed
  tie, verify branch-consistent one-sided derivative and status flag.
- U-FLOOR-1: constructed floor activation raises status and invalidates the
  claim (no silent clamp).
- U-JAC-1: raw-y Jacobian corrections: exact log-square vs offset formula
  at offset -> 0 agreement; both vs change-of-variables on a synthetic
  density with known closed form.
- U-DATA-1: every simulated-dataset builder returns declared [T, dim]
  shapes; multi-coordinate panels have per-coordinate independent paths
  (regression test for E10).
- U-BATCH-1: theta-batch equivariance: engine value/score on a permuted
  batch equals permuted value/score (pattern exists for the SV route;
  generalize).
- U-ADAPT-1: adapter-swap purity: engine executes byte-identical graph
  structure across two adapters with equal shape signatures (genericity
  leak detector).
- U-STRUCT-1: structural adapter honors deterministic completion —
  propagated points satisfy `k_t = f(k_{t-1}, m_{t-1}, m_t)` to 1e-12
  (Ch18b constraint-support gate).
- U-TUNE-1: tuning procedure determinism — same inputs and seeds produce
  identical scope artifacts.

### 8.2 Integration tests — per phase

- I-P0-1: engine skeleton refactor leaves entire existing suite green
  (CI run recorded in the P0 result note).
- I-P1-1: LGSSM n-ladder, n in {2,4,8,16,32,64}, T in {8,120}: TT value vs
  exact Kalman within declared per-rung tolerance; records r*(n).
- I-P1-2: nonlinear spot-checks: predator-prey (n=2) and SIR d=18 short
  horizon vs existing lane references (same-target).
- I-P1-3: no-dense-grid assertion: instrumented run at n = 8 proves no
  `O(q^n)` allocation occurs (memory watermark bound).
- I-P2-1: full-path score vs FD (FD-quality-first protocol) on: LGSSM
  rungs, predator-prey, SIR d=18 T>0 (the previously blocked case),
  structural model T=20.
- I-P2-2: p-scaling harness, synthetic model with p in {3,30,300}:
  gradient/value cost ratio <= ~6 at p=300; memory within Section 4
  envelope; score equality vs per-parameter reference implementation at
  p=3 (1e-12).
- I-P2-3: same-scalar test: score is the derivative of the exact scalar the
  value path emits (finite-difference of the actual program output).
- I-P3-1: eager-vs-XLA parity, value and score, 1e-12 relative (pattern of
  the existing SV XLA parity test).
- I-P3-2: GPU memory-growth policy verified in-run per repository rule.
- I-P4-1: SV adapter through generic engine reproduces the admitted batched
  actual-SV route on its frozen T10 program (near-bit; same finite
  program).
- I-P4-2: LGSSM adapter vs exact Kalman value+score at n in {1,2,4}.
- I-P4-3: d=1 parity vs existing scalar TT path (exact-transformed SV,
  KSC SV).
- I-P4-4: SIR/PP adapters vs lane references at matched scopes; same-target
  gaps gated, cross-family gaps recorded descriptively.
- I-P4-5: structural adapter passes the Ch18b gate list: metadata test,
  constraint-support test, linear recovery test (phi, gamma settings that
  linearize), degenerate-transition test.
- I-P5-1: procedure-vs-history dominance: v1-procedure configs reproduce or
  dominate historical hand configs (same-target gap at <= cost) on all five
  legacy models.
- I-P6-1: NAWM-scale synthetic rehearsal (n=100, m=100, T=120, p=300):
  value+gradient wall-clock and memory vs Section 4 budget; multi-seed
  (>= 4 seeds) for timing stability; NO accuracy claim beyond program
  consistency (no exact reference exists at that scale — state this in the
  artifact).
- I-LB-1: leaderboard build integration test — schema, one row family per
  algorithm, all mandatory columns present, claim labels validated against
  the allowed vocabulary.

### 8.3 Statistical/replication tests

- S-1: any continuous cross-family comparison reported on the leaderboard
  at T in {20,40} carries >= 8-seed replication with paired summaries
  (mean, sd, t95), following the 16-seed sweep pattern; single-seed values
  are labeled descriptive.
- S-2: refinement/resolution stability screens use the two-step rule (both
  ladder steps below threshold), never one step.

---

## 9. Final deliverable: the full leaderboard (audit target E)

### 9.1 Model rows

| Model | State n | Obs m | Horizons | Parameters p | Same-target reference |
|---|---|---|---|---|---|
| LGSSM | 2 (ladder: to 64) | 1-2 | 25; ladder T=120 | 2-4 | exact Kalman (value AND score) |
| Actual SV (exact log-chi-square) | 1 (panels 1-3) | 1-3 | 10, 20 | 2 | dense exact-transformed reference |
| KSC SV (mixture surrogate) | 1 (panels 1-3) | 1-3 | 20 | 2 | dense KSC mixture reference |
| Predator-prey | 2 | 2 | **20, 40** | 3-4 | dense quadrature reference (n=2 affordable) |
| Austria SIR | 18 | 9 | **20, 40** | 3 | lane-module references + internal consistency (no dense ref at d=18 — label accordingly) |
| Structural deterministic (Ch18b, implemented in `structural_fixtures.py` / `structural_ukf_neutra_target_design_tf.py`) | 2 (m_t, k_t) | 1 | 20, 100 | 5 (rho, sigma, phi, gamma, R) | dense quadrature over (x_{t-1}, eps_t); STR-UKF row as comparator |
| NAWM-scale synthetic | 100 | 100 | 120 | 300 | none exists — consistency + budget row only, claim-capped |

### 9.2 Algorithm columns (per model row, where applicable)

Generic ZC squared-TT (this program) | SVD-UKF | UKF/CKF/GenUT family |
SGQF | LEDH(-PFPF-OT) where applicable | mixture-Kalman (SV rows) |
dense reference (where affordable) | exact Kalman (LGSSM row).

Mandatory columns per (model, algorithm) cell: value; same-target gap where
a reference exists; score status (analytic / FD-only / none) + FD relative
error; wall time value / full gradient; scope artifact id; status flags;
claim label from the allowed vocabulary (EXACT_ORACLE /
CERTIFIED_APPROXIMATION / SURROGATE_USEFULNESS / DIAGNOSTIC_ONLY / BLOCKED).

### 9.3 Structural model: specific audit attention

The Ch18b model has the structural split `x_t = (m_t, k_t)` with
`k_t = phi k_{t-1} + gamma m_t^2` deterministic given
`(m_{t-1}, k_{t-1}, eps_t)`. Chapter policy (final section) demands:
integration variables are the DECLARED stochastic variables
`(x_{t-1}, eps_t)`, never a padded full-state law; deterministic completion
is computed, not noised.

For the TT engine this means the adjacent-pair target must be assembled with
the transition kernel `p_theta(x_t | x_{t-1})` that is **degenerate**
(a Dirac on the k-coordinate given the m-path). Two admissible designs the
audit should choose between (we propose (a)):

(a) TT over `(m_t, x_{t-1})` with `k_t` completed deterministically inside
the adapter's observation evaluation (integration space = declared
stochastic block; matches the chapter exactly); or
(b) TT over `(x_t, x_{t-1})` with an explicitly labeled numerical
regularization of the degenerate direction (chapter-permitted only with a
written approximation label; weaker).

Design (a) keeps the TT dimension at the stochastic dimension, which is also
the honest scaling story for DSGE: the TT should live on the stochastic
block, not on deterministic completions. This is both a correctness point
(Ch18b gates) and the main rank-containment lever at NAWM scale. Audit
check: confirm the adapter contract in the plan carries the state-partition
metadata needed to declare integration spaces per model (it must after this
memo; flag if plan text lags).

### 9.4 Claim discipline on the final leaderboard

- Same-target gaps: gate-bearing.
- Cross-algorithm gaps: descriptive unless S-1 replication supports an
  interval statement; no "best/beats" language without predeclared
  uncertainty analysis.
- SIR d=18 and NAWM-scale rows carry explicit no-dense-reference caveats.
- No HMC/posterior claims anywhere on the leaderboard; HMC admission is
  P6's separate campaign under NeuTra governance.

---

## 10. Requested audit output format

Return a written verdict with:

1. Per-section findings table: `AGREE` / `DISAGREE(reason)` /
   `INSUFFICIENT(what is missing)` for Sections 3, 4, 5, 7, 8, 9.
2. Any mathematical error found: state the incorrect claim, the corrected
   statement, and the consequence for the plan.
3. Ranked list of the three weakest points of the program (your judgment).
4. Any prior-error ledger row whose prevention you judge non-structural
   (i.e., could recur despite the plan).
5. Missing tests: anything claimable by the program that no U-/I-/S- test
   would catch.
6. Explicit answer to: "Can this program pass all its own gates and still
   mislead the owner about NAWM-scale feasibility?" If yes, what additional
   gate closes that hole?

Per repository policy, review is advisory: material mathematical, numerical,
cost, or privacy findings block; stylistic preferences do not. Please do not
propose governance ceremony beyond the repository's standing rules.

---

## Appendix: key code anchors for the auditor

- Gram normalizer: `bayesfilter/highdim/squared_tt.py:164`
- Differentiable primitives: `bayesfilter/highdim/derivatives.py:521,606,647,657`
- Scalar score chain: `bayesfilter/highdim/filtering.py:1255-1300`
- Dense-grid retention defect: `bayesfilter/highdim/filtering.py:4048`
- Per-parameter score loop defect: `bayesfilter/highdim/filtering.py:1463`
- Admitted batched SV route: `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
- Structural model: `bayesfilter/testing/structural_fixtures.py`,
  `bayesfilter/testing/structural_ukf_neutra_target_design_tf.py`
- Ch18b gates: `docs/chapters/ch18b_structural_deterministic_dynamics.tex`
  (Validation Gates and Final Policy Rule)
- Method A multimodel evidence:
  `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-result-2026-08-04.md`
- Seed-scatter lesson:
  `docs/plans/bayesfilter-actual-sv-cross-family-gap-16-seed-result-2026-08-15.md`
