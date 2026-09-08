# Fable Audit Report: Direct-Factor SR-UKF Execution Plan

Date: 2026-08-15
Reviewer: Fable (Claude, Opus 5), using the local MathDevMCP CLI
Status: `AUDIT_COMPLETE_NO_IMPLEMENTATION_AUTHORITY`
Handoff: `docs/plans/bayesfilter_direct_factor_srukf_fable_mathdevmcp_audit_handoff_2026_08_15.md`
Plan under audit: `docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md`

## 0. Provenance, environment, and tool availability

- BayesFilter commit at audit: `3030d86df9cb00346df82c7c19f015c09c7c6e1f` (matches handoff).
  Dirty worktree contains only the pre-existing unrelated files plus the two
  plan documents and `build-tf221/`; no BayesFilter or MacroFinance source was
  edited by this audit.
- SHA-256 digests of every reviewed file: `source_hashes.txt` (10 files,
  including both plans, the five BayesFilter sources, the focused test file,
  the monograph chapter, and the MacroFinance handoff).
- Environment: `environment.json` (MathDevMCP `doctor`, ok). Available
  backends: sympy, sage, latexml, pandoc, Lean 4.20 toolchain. Unavailable:
  leansearchv2, jixia (irrelevant to this audit; recorded per protocol).
- Commands and elapsed times: `command_timings.txt` (12 CLI invocations,
  0.2–0.9 s each). All runs were local; no network or package mutation.
- Python for TensorFlow was not needed; all numerical counterexample checks
  are dependency-free float arithmetic (`counterexample_checks.txt`). This is
  audit-diagnostic code outside the repository runtime, consistent with the
  CLAUDE.md diagnostic boundary.

MathDevMCP outputs are diagnostic review evidence only; none of the results
below is treated as a proof certificate.

### Tool-run summary and abstentions

| Check | Artifact | Outcome |
|---|---|---|
| `doctor` | environment.json | ok; provenance only |
| `search-latex` (monograph-rooted) | latex_lookup.json | Located `alg:nf_sr_ukf` (l.283) and `rem:srukf_ad_design` (l.351); source navigation only |
| `assumptions-for` QR-gram | assumption_checks.md | `missing_assumptions`: conformability (generic); recorded |
| `assumptions-for` downdate | assumption_checks.md | `inconclusive` → **ABSTAIN**; obligation discharged by human derivation + numeric check |
| `assumptions-for` logdet | assumption_checks.md | `missing_assumptions`: conformability + SPD/valid-logdet domain; matches plan assumptions |
| `derive-or-refute` (3 identities) | identity_checks.md | `unknown`/`missing_assumptions`: the scalar symbolic router cannot represent noncommutative matrix products → **ABSTAIN** on all three; discharged by human derivation (Section A) and bounded numeric checks |
| `audit-kalman-recursion` on `kalman_qr_tf.py` | math_to_code_checks.md | `mismatch` — **tool-vocabulary false negative.** The AST classifier (`ast_operation_graph.py`) has no `qr`/`triangular_solve` token classes at all; the file demonstrably calls `tf.linalg.qr` (l.150) and `tf.linalg.triangular_solve` (l.169–170, 341). Recorded as a tool limitation, not as evidence against the source. |
| `audit-math-to-code` downdate identity vs `srukf_factor_tf.py` | math_to_code_checks.md | `structural_mismatch` — **expected baseline**: the prototype refactorizes the filtered covariance via `cholesky_factor_first_derivatives` (l.381–385) and omits `S_r` from the innovation stack (l.331). This is the documented defect the plan repairs. Re-run after implementation. |
| `code-implements-equation` factor logdet | math_to_code_checks.md | `mismatch`/`scope_limited` — same expected baseline; prototype uses `abs(diag)` logdet (l.235) without an appended `S_r`. |
| `generate-math-tests` | generated_test_obligations.json | Generated `symbolic_identity`, `numeric_fixture`, `shape_property`, `finite_difference`, `expected_failure` nominations. Note: the handoff's requested kinds `random_spd`, `near_singular`, `failure_case` are not supported by this MathDevMCP version (supported kinds recorded); nearest supported kinds were used. The generated cases do **not** include batch permutations, parameter derivatives, center residuals, observation-noise derivatives, or XLA/eager parity — the plan's own matrix (§5–§6) is strictly stronger and already covers all of these. |
| `prepare-review-packet` | review_packet.json | `review_ready`, packet id `bayesfilter-direct-factor-srukf-20260815`; not an approval token |
| Independent numeric counterexamples | counterexample_checks.txt | **Two confirmed plan errors** (FS-1, FS-2 below) |

## A. Derivation verdict (plan §3, questions 3.1.1–3.1.12)

1. **Augmented placement (plan §3.1)** — SUPPORTED.
   `S_a S_a' = blockdiag(S_x S_x', S_q S_q') = blockdiag(P_x, Q)` holds for any
   conformable blocks; blockdiag of lower-triangular positive-diagonal factors
   is again lower triangular with positive diagonal. Block placement is linear,
   so `d_p m_a = [d_p m; 0]` and `d_p S_a = blockdiag(d_p S_x, d_p S_q)` are the
   exact derivatives, and carrying zero blocks for the fixed DZ5 `Q` keeps a
   future parameterized `Q` from being silently dropped.

2. **Stack orientation (plan §2.2, §3.2, §3.4)** — SUPPORTED.
   With row offsets, `χ_i − m = o_i S'`, i.e. column-form offset `S o_i'`; for a
   standardized rule with `Σ w_i o_i' o_i = I` this reproduces `P = S S'`. The
   repo helpers use exactly this convention (`srukf_factor_tf.py` l.265;
   `experimental_batched_svd_sigma_point_tf.py` einsum `"ra,bda->brd"`).
   `A_x A_x' = Σ w_i^{(c)} δx_i δx_i'`, thin QR of `A_x'` (rows `2d+1 ≥ n_x`
   always) with positive-diagonal normalization gives unique `S⁻ = R'` with
   `S⁻S⁻' = A_xA_x'`. Same for `A_y ∥ S_r` with `2d+1+n_y ≥ n_y`. Matches
   `stack_qr_lower_factor_first_derivatives` orientation in
   `qr_factor_tf.py` l.179–195.

3. **Exact DZ5 rule (plan §2.3)** — SUPPORTED.
   `α=1, β=2, κ=0 ⇒ λ = α²(d+κ)−d = 0`, so `w_0^{(m)} = λ/(d+λ) = 0`,
   `w_0^{(c)} = λ/(d+λ) + (1−α²+β) = 2`, `w_i = 1/(2(d+λ)) = 1/(2d)`, spread
   `γ = √d`. Verified against the repo rule generator
   (`sigma_points_tf.py` l.149–176) and the hardcoded
   `unscented_unit_spread` rule (`srukf_factor_tf.py` l.145–160). Second-moment
   condition `Σ w_i^{(c)} o_i'o_i = I` holds (center offset is zero; each axis
   pair contributes `2·(1/2d)·d·e_je_j'`). All covariance weights nonnegative,
   so the pure-QR stack representation is valid with no signed-rank update.

4. **Center residual (plan §2.3, §3.2)** — SUPPORTED.
   The stack runs `i = 0..2d` and the plan states the center column is
   included; `w_0^{(m)}=0` excludes the center from the mean while
   `w_0^{(c)}=2` includes `δx_0 = f(χ_0) − x̄` in the covariance. Phase 3
   fixture 2 requires a nonzero propagated center residual and a measured
   difference from a center-omitting implementation, which pins the behavior.
   Consistent with the monograph's center-residual caveat under
   `alg:nf_sr_ukf` (ch17, l.309–314 and l.330–332).

5. **Process noise exactly once (plan §3.2)** — SUPPORTED.
   In the augmented formulation the innovation coordinates already span `Q`
   through `S_q` inside `S_a`; the prediction stack contains only propagated
   state residuals and no appended `S_q`. For affine `f(x,q)=Fx+Gq+c` the rule
   gives `Σ w_i^{(c)} δx_iδx_i' = F P_x F' + G Q G'` exactly, confirming no
   double count. The explicit "do not append S_q again" sentence closes the
   classic additive-noise/augmented-noise confusion.

6. **QR derivative identities (plan §3.3)** — SUPPORTED.
   From `dA = dQ R + Q dR` and `Q'Q = I`: `E = Q'(dA)R^{-1} = Q'dQ + dRR^{-1}`,
   with `Q'dQ` skew (zero diagonal) and `dRR^{-1}` upper triangular. The split
   is unique because skew ∩ upper = {0}; `Ω = strictLower(E) − strictLower(E)'`
   recovers `Q'dQ` exactly, giving `dR = (E−Ω)R` and
   `dQ = QΩ + (I−QQ')dAR^{-1}` (the thin-QR range-complement term). `dS = dR'`
   and `dSS' + SdS' = d(R'R) = d(A_xA_x')` follows. This matches the existing
   implementation (`qr_factor_tf.py` l.76–89) exactly. The nondifferentiable
   sign branch at a zero pivot is explicitly failed closed (plan §3.3). Note
   MathDevMCP could not decide these matrix identities (ABSTAIN recorded);
   the above is the human derivation the protocol requires, and it is also
   backed by the repo's existing QR-derivative test surface.

7. **Gain formula and solve orientation (plan §3.5)** — **COUNTEREXAMPLE / REVISE (FS-1).**
   The target `K = P_xy (S_yS_y')^{-1}` is correct and matches the likelihood's
   `S_y`. The stated implementation form is wrong:
   `U' = S_y^{-T} P_xy'` gives `U = P_xy S_y^{-1}` and then
   `K = U S_y^{-1} = P_xy S_y^{-1} S_y^{-1} ≠ P_xy S_y^{-T} S_y^{-1}` because a
   lower-triangular `S_y` is not symmetric. Numerical counterexample
   (`counterexample_checks.txt`): `S_y = [[2,0],[1,1]]`, `P_xy = [[1,2],[3,4]]`
   gives plan-`K = [[−1.25,2],[−2.25,4]]` vs true `K = [[−0.5,1.5],[−0.5,2.5]]`
   (max error 1.75); the corrected orientation reproduces `K` to machine zero.
   Correct forms: define `U := K S_y` (note this equals `V` of §3.7, a useful
   simplification), then

   ```text
   U S_y' = P_xy        ⇔  U' = S_y^{-1} P_xy'          (lower solve with S_y)
   K  = U S_y^{-1}      ⇔  K' = S_y^{-T} U'             (upper solve with S_y')
   dU' = S_y^{-1} (dP_xy' − dS_y U')
   dK  = dU S_y^{-1} − K dS_y S_y^{-1}                   (unchanged)
   ```

   The plan's derivative line `dU' = S_y^{-T}(dP_xy' − dS_y'U')` is internally
   consistent with the wrong `U` and must be replaced together with it. The
   `dK` and `dm_f = dx̄ + dKe + Kde`, `de = −dȳ` lines are correct once `U` is
   fixed. Silver lining: with the corrected `U`, the downdate input `V = KS_y`
   is literally `U` — one solve output serves both the gain and §3.7.

8. **Factor-native score (plan §3.6)** — SUPPORTED.
   `dlogdet(P_y) = tr(P_y^{-1}dP_y) = 2tr(S_y^{-1}dS_y) = 2Σ_j (dS_y)_{jj}/(S_y)_{jj}`
   (product of lower triangulars); `z = S_y^{-1}e ⇒ dz = S_y^{-1}(de − dS_y z)`;
   `d(z'z) = 2z'dz`; `de = −dȳ` since `y_t` is data. Expanding
   `2z'dz = 2e'P_y^{-1}de − 2z'S_y^{-1}dS_yz` and
   `e'P_y^{-1}dP_yP_y^{-1}e = 2z'S_y^{-1}dS_yz` shows term-by-term equality with
   the covariance-form score in the plan, so the §3.6 comparator identity is
   exact, and the comparator is correctly quarantined as diagnostic-only.

9. **Sequential downdate equivalence (plan §3.7)** — SUPPORTED, with a
   required addition (FS-4). Telescoping gives
   `L^{(n_y)}L^{(n_y)'} = S⁻S⁻' − Σ_j v_jv_j' = S⁻S⁻' − VV' = P_f` provided all
   intermediate downdates succeed. The plan should state the intermediate
   feasibility lemma: the partial result after `k` columns equals
   `P_f + Σ_{j>k} v_jv_j' ⪰ P_f`, so **if `P_f` is SPD in exact arithmetic,
   every intermediate margin is strictly positive regardless of column
   order**. This is exactly the statement needed to attribute a runtime margin
   failure to either a mathematically indefinite `P_f` or finite-precision
   loss (audit question 3.2.9), and it is currently missing.

10. **Scalar downdate derivative equations (plan §3.7)** — **COUNTEREXAMPLE / REVISE (FS-2).**
    The pivot-scalar block (`r, c, s`, `dr, dc, ds`) is correct. The vector
    update as written is wrong: with the plan's explicit instruction "The old
    `(a,u)` values are used for the second assignment", the update
    `u ← cu − s·a_old` does **not** satisfy `L_{new}L_{new}' = LL' − vv'`.
    Counterexample (`counterexample_checks.txt`): `L = [[2,0],[1,1.5]]`,
    `x = [0.5,0.3]` gives reconstruction error `1.03e−3` for the old-`a` rule
    versus `4.4e−16` for either correct variant. The two correct, equivalent
    forms (using `c² + s² = 1`):

    ```text
    a_new = (a_old − s·u_old)/c
    u_new = c·u_old − s·a_new          (sequential in-place: a first, then u)
          = (u_old − s·a_old)/c        (equivalent old-value form)
    ```

    The plan is also internally inconsistent: its own derivative line
    `du_new = dc·u + c·du − ds·a_new − s·da_new` differentiates the **correct**
    `a_new` form, contradicting the value-path instruction. Repair: fix the
    value recurrence and the prose note ("old `u`, new `a`", or give the
    `/c` form); the stated derivative equations then match and are correct
    (`da_new` is the quotient rule on old values; `du_new` consumes `a_new`
    and `da_new`, so it must be evaluated after them — worth one ordering
    sentence for the batched implementation).

11. **Explicit assumptions (plan §2.2, §3.3, §3.7, §4.3)** — SUPPORTED.
    SPD factors via strictly positive diagonals, fixed QR sign branch with
    fail-closed pivot behavior, strict downdate margins with no nugget,
    finiteness assertions at each step, and `K ≥ N` for the thin QR are all
    stated. The only missing assumption-level statement is the FS-4
    feasibility lemma above. MathDevMCP's assumption inventory adds nothing
    beyond conformability and the SPD/logdet domain, both already covered.

12. **No accidental orientation-equivalence claim** — SUPPORTED.
    Phase 5 restricts principal-root comparisons to fixtures where the
    orientation is intentionally equivalent, and §9 nonclaims plus the
    handoff §10 explicitly disclaim general nonlinear equivalence between the
    lower-QR factor orientation and the symmetric principal root.

## B. Stability verdict

Divisions and square roots with their guards (plan reference in parentheses):

| Operation | Guard | Status |
|---|---|---|
| `R^{-1}` in `E = Q'(dA)R^{-1}` | positive pivots, fail closed (§3.3, §4.3) | adequate; add relative conditioning diagnostic (FS-8) |
| `sign(diag R)` branch | pivot ≠ 0; zero pivot fails closed (§3.3) | adequate; `sign(0)→+1` in repo helper is masked by the positivity gate |
| `log(S_y,jj)` and `(dS_y)_{jj}/(S_y)_{jj}` (§3.6) | strictly positive diagonal | adequate |
| triangular solves with `S_y`, `S_y'` (`z`, `U`, `K`, `dz`, `dU`, `dK`) | positive diagonal | adequate once FS-1 fixes orientation |
| `r = sqrt(L_kk² − x_k²)` (§3.7) | strict margin `> 0`, no nugget | adequate and explicit |
| `c = r/L_kk`, `s = x_k/L_kk`, `/L_kk²` in `dc, ds` | `L_kk > 0` invariant (maintained since each pivot is overwritten by `r > 0`) | adequate |
| `(a − su)/c`, `/c²` in `da_new` | `c > 0 ⇐ r > 0 ∧ L_kk > 0` | adequate |
| `dr = (…)/r` | margin `> 0` | adequate; derivative magnitude blows up as margin → 0 even while finite — this is a conditioning issue, not a guard issue (FS-8) |
| `sqrt(w_i^{(c)})` | rule weights statically nonnegative (§2.3) | closed |

Assessment per audit questions 3.2.1–3.2.10:

1–2. Pivot/margin → 0 is detected before any invalid division: positivity
gates precede the derivative divisions in the stated assert order (§4.3), and
neither a nugget nor a smooth sign replacement is permitted. Adequate.

3. Sequential downdates can accumulate more rounding error than a block
method over `n_y` columns. The plan's per-row min-margin and reconstruction
residual diagnostics are the right monitors. Recommended (advisory FS-5): add
the exact all-additive block comparator — QR of `([A_x, 0_{n_x×n_y}] − K A_y)'` —
which reconstructs `P_f` exactly (`Ã_xA_y' = P_xy`, expansion gives
`P⁻ − KP_yK'`), never materializes `P_f`, and independently validates both
`S_f` and `dS_f` via the already-audited QR derivative. This is precisely the
handoff §4.4 option-2 family, used here as a comparator, never a fallback.

4. Finiteness checks: enumerated at every step (§4.3). Adequate.

5. Sign-normalization determinism: after positive-diagonal normalization the
thin QR factor of a full-column-rank stack is unique, so CPU/GPU and
eager/graph/XLA agree up to roundoff; genuine nondeterminism only arises for
near-zero pivots, which the gate rejects. Phase 7 requalification per mode
plus recorded failures is the right evidence protocol.

6. Parameter-dependent `R(θ)` with a simultaneously varying residual stack is
handled inside one QR derivative because the stack derivative concatenates
`sqrt(w) d δy_i` and `dS_r` columns (§3.4); no covariance-derivative
refactorization occurs. Adequate.

7. Static shapes and batch nativeness: §2.4 forbids Python loops over `B`/`P`;
time recursion and pivot/column loops via `tf.while_loop` with static bounds.
Adequate. (Note: with small static `n_y` and pivot counts, statically unrolled
index loops are also contract-compliant and often XLA-friendlier; the plan's
language already permits this.)

8. Failure distinguishability: per-row minimum pivot, minimum margin,
residuals, and failure counts are required, but the plan does not name a
failure taxonomy. Advisory FS-7: emit distinct codes
(`qr_pivot_nonpositive`, `downdate_margin_nonpositive`,
`nonfinite_derivative`, `invalid_observation_factor`,
`batch_contract_violation`) so Phase 6 artifacts are mechanically
attributable, and add explicit NaN/Inf-input rejection tests for both
primitives.

9. Mathematical vs roundoff SPD failure: currently not distinguishable from
the stated diagnostics alone. Material FS-4: add the intermediate-feasibility
lemma (Section A.9) plus an offline diagnostic procedure (eigenvalues of the
independently assembled comparator `P_f`, diagnostic-only under the CLAUDE.md
boundary) so a failed row is classified as model-indefinite versus
finite-precision, with original evidence retained in both cases.

10. GPU memory growth, dtype, TF32/XLA state, device identity, execution
mode: §4.4 requires all of these in artifacts and matches the repository GPU
memory rule (growth verified before initialization, fail closed, recorded in
the manifest). Adequate.

## C. Robustness verdict

- **Static shapes / batch invariance / duplicates / mixtures / permutations /
  production batch**: §2.4 + Phase 6 cover contexts 1, 2 (duplicate), 2
  (mixed), 36, 480 with permutations and the exact production ordering, and a
  hard row-independence gate with predeclared float64 tolerances. Adequate.
- **Parameter derivatives**: the factor adapter (§4.2) carries explicit
  `d_*` blocks for mean, initial factor, process factor, observation factor;
  the DZ5 adapter provides `S_r, dS_r` directly. Adequate.
- **Lagged observations**: §2.1 requires the active observation contract to
  be declared and differentiated identically; §6 includes it in integration
  coverage when active. Adequate.
- **XLA**: candidate API defaults to `jit_compile=True`; eager/non-XLA are
  reference modes; Phase 7 records per-mode outcomes rather than dropping
  modes. Adequate.
- **Artifact provenance**: Phase 0 manifest (hashes, versions, dirty status)
  and Phase 8 memo (commands, seeds, devices, timings, tolerances, raw and
  aggregated results) are complete.
- **Hidden fallback surfaces**: the fail-closed policy (§4.3) explicitly bans
  floor/principal-root/SVD/NaN-substitution fallbacks. Two gaps:
  1. **FS-3 (material)** — the §7 route guard is specified as a token list, but
     two of its obligations are contextual, not lexical:
     "`cholesky_factor_first_derivatives` **on filtered covariance**" and
     "covariance-to-root helpers **in the time loop**". A substring guard (the
     current `srukf_route_guard.py` mechanism) cannot express either, and the
     §4.2 compatibility factorization boundary legitimately contains a
     Cholesky call that a naive token ban would flag. Required repair: place
     the one-time compatibility conversion in a separate non-admitted module;
     define the admitted runtime file set for the new backend with a strict
     zero-token ban (`tf.linalg.eigh`, `tf.linalg.svd`, `cholesky`,
     `principal_sqrt`, `tf_principal_sqrt_ukf`, SVD sigma-point dispatch
     names); extend `FORBIDDEN_SRUKF_ROUTE_PATTERNS` accordingly (the current
     tuple contains none of the §7 tokens); and keep runtime
     downdate-provenance diagnostics as the semantic complement the plan
     already promises.
  2. The prototype `srukf_factor_tf.py` itself still contains the defect
     pattern (covariance refactorization, no `S_r`); the plan correctly builds
     new files rather than wiring the prototype in, and the MathDevMCP
     baseline mismatch evidence documents this starting point.

## D. Test verdict

Mapping of acceptance criteria (§6, §8) to tests:

| Criterion | Closing test(s) | Status |
|---|---|---|
| QR reconstruction ≤ 1e−12 | Phase 1; `tests/test_linear_qr_factor_tf.py` (exists) | covered |
| QR derivative ≤ 1e−10 (FD) | Phase 1; batched vectorization test (exists) | covered |
| Downdate reconstruction ≤ 1e−12 | Phase 2 `tests/test_cholesky_downdate_tf.py` (new) vs independently assembled `P − vv'` | covered; this is the test that closes FS-2 |
| Downdate derivative ≤ 1e−10 | Phase 2 FD for factor and derivative | covered |
| Score FD agreement ≤ 1e−7/1e−9 | Phase 5 centered FD + independent reverse/cotangent | covered |
| Row invariance ≤ 1e−10/1e−11 | Phase 6 duplicate/mixed/permuted/production contexts | covered |
| Min diagonal / margin > 0 | Phases 2–6 per-row diagnostics | covered |
| Route guard | `tests/test_factor_srukf_route_guard.py` (new) | covered once FS-3 defines the mechanism |
| Execution modes | Phase 7 matrix with recorded failures | covered |
| Artifact schema + hashes | Phase 8 memo; §6 schema validation | covered |
| Center-residual activity | Phase 3 fixture 2 (differs from center-omitting run) | covered; strong design |
| Observation-noise derivative (`dS_r ≠ 0`) | Phase 3 fixture 3 | covered |
| Failed-downdate fail-closed | Phase 3 fixture 4 + Phase 2 indefinite rejection | covered |
| Factor authority over covariance | Phase 4 (carry `mean, S_x, d_mean, dS_x` only) | covered |

Gaps and required additions:

- **Gain-orientation primitive test (closes FS-1)**: add a primitive-level
  check that the two triangular solves reproduce `K` against a dense
  `P_xy P_y^{-1}` oracle on a random SPD fixture. The Phase 3 affine fixture
  would catch the error end-to-end (wrong `K` corrupts the filtered mean and
  factor, though not the one-step likelihood, whose value bypasses `K`), but a
  primitive test localizes it.
- **Independent oracle strength**: Phase 5's independent covariance assembly,
  covariance-form score, FD, and reverse-cotangent checks are adequate and
  non-circular. Adding the FS-5 joint-stack comparator would also
  independently validate `dS_f` without downdates.
- **FD protocol (FS-6, advisory)**: §6 fixes tolerances but not the FD step
  policy; the MacroFinance handoff Stage F requires step-size and
  repeatability evidence. Declare central differences, per-parameter scaled
  steps, and a two-step-size consistency check before Phase 5.
- **Nonfinite-input tests (FS-7, advisory)**: make NaN/Inf rejection an
  explicit listed test for both primitives, not only a runtime assert.
- MathDevMCP `generate-math-tests` nominations are strictly weaker than the
  plan's matrix (no batch/permutation/parameter-derivative/XLA obligations);
  no additional obligations arise from the tool beyond those already listed.

## E. Required changes

```text
issue_id: FS-1
severity: blocking
exact source/plan path: docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md §3.5 (gain implementation form and dU recursion)
failure mechanism or counterexample: U' = S_y^{-T} P_xy' followed by K = U S_y^{-1} yields K = P_xy S_y^{-1} S_y^{-1} ≠ P_xy (S_y S_y')^{-1}; counterexample S_y=[[2,0],[1,1]], P_xy=[[1,2],[3,4]] gives max elementwise error 1.75 (counterexample_checks.txt, check 1). Wrong K corrupts filtered mean, downdate input V, and every subsequent step while remaining finite.
required repair: replace with U' = S_y^{-1} P_xy' (lower solve), K = U S_y^{-1}; note U = K S_y = V of §3.7. Replace derivative line with dU' = S_y^{-1}(dP_xy' − dS_y U'); keep dK = dU S_y^{-1} − K dS_y S_y^{-1}.
test that closes the issue: new primitive gain-orientation test vs dense P_xy P_y^{-1} oracle; Phase 3 affine fixture closed-form filtered-mean/factor match.
```

```text
issue_id: FS-2
severity: blocking
exact source/plan path: execution plan §3.7 (scalar downdate vector update and the "old (a,u)" note)
failure mechanism or counterexample: u ← c·u_old − s·a_old fails L_new L_new' = L L' − x x'; counterexample L=[[2,0],[1,1.5]], x=[0.5,0.3] gives reconstruction error 1.03e−3 vs 4.4e−16 for the correct forms (counterexample_checks.txt, check 2). The plan's own du_new formula uses a_new, contradicting the value-path instruction; as written, implementations following the value text produce silently wrong, finite factors.
required repair: state u_new = c·u_old − s·a_new (sequential in-place order: a then u), or equivalently u_new = (u_old − s·a_old)/c; correct the note to "old u, new a" (or present the two-register form); state evaluation order da_new before du_new. The listed derivative equations are then correct as written.
test that closes the issue: Phase 2 reconstruction against independently assembled P − v v' plus derivative FD (already planned; retain at ≤1e−12/1e−10).
```

```text
issue_id: FS-3
severity: material
exact source/plan path: execution plan §7 and §4.2; bayesfilter/nonlinear/srukf_route_guard.py (current pattern list lacks every §7 token)
failure mechanism or counterexample: substring guard cannot express the contextual bans ("on filtered covariance", "in the time loop"); the §4.2 compatibility factorization legitimately contains a Cholesky call, so a naive token ban either misses in-loop refactorization or false-positives the boundary.
required repair: isolate the compatibility conversion in a non-admitted module; declare the admitted runtime file set with a strict zero-token ban (tf.linalg.eigh, tf.linalg.svd, cholesky, principal_sqrt, tf_principal_sqrt_ukf, SVD dispatch names); extend FORBIDDEN_SRUKF_ROUTE_PATTERNS; keep runtime downdate-provenance diagnostics as the semantic complement.
test that closes the issue: tests/test_factor_srukf_route_guard.py fixtures rejecting each token in admitted files and accepting the boundary module; CI scan of the admitted file list.
```

```text
issue_id: FS-4
severity: material
exact source/plan path: execution plan §3.7 and §4.3 (diagnostics interpretation)
failure mechanism or counterexample: without the intermediate-feasibility lemma, a negative downdate margin cannot be attributed between a mathematically indefinite P_f and finite-precision margin loss, which §3.2.9 and report section B require.
required repair: add to §3.7 the lemma that the partial downdate after k columns equals P_f + Σ_{j>k} v_j v_j' ⪰ P_f, so SPD P_f implies positive margins for every column order in exact arithmetic; add an offline diagnostic-only classification step (eigenvalues of the independently assembled comparator P_f) for failed rows, preserving original evidence for both outcomes.
test that closes the issue: Phase 3 fixture 4 asserting the failure artifact carries the mathematical-vs-roundoff classification for a deliberately indefinite case and for a near-margin roundoff case.
```

```text
issue_id: FS-5
severity: advisory
exact source/plan path: execution plan §5 Phase 2/Phase 5 comparator list
failure mechanism or counterexample: sequential downdates may accumulate more rounding error than a block method; no exact block comparator is currently listed.
required repair: add the exact all-additive comparator S_f_cmp from QR of ([A_x, 0_{n_x×n_y}] − K A_y)' (identity: Ã_x A_y' = P_xy ⇒ (Ã_x − K A_y)(Ã_x − K A_y)' = P⁻ − K P_y K' = P_f), comparator-only, never a fallback; also validates dS_f via the QR derivative.
test that closes the issue: Phase 5 comparison of downdate S_f/dS_f vs joint-stack comparator within declared tolerances.
```

```text
issue_id: FS-6
severity: advisory
exact source/plan path: execution plan §6 (thresholds) vs MacroFinance handoff Stage F
failure mechanism or counterexample: FD tolerances are declared but no step-size selection or repeatability protocol; a rare-row FD failure would lack the required step-size evidence.
required repair: declare centered differences, per-parameter scaled steps, and a two-step-size (h, h/2) consistency check before Phase 5; record both step sizes in artifacts.
test that closes the issue: Phase 5/6 artifact schema fields for FD step sizes and repeatability agreement.
```

```text
issue_id: FS-7
severity: advisory
exact source/plan path: execution plan §4.3 diagnostics and §5 Phase 1–2 test lists
failure mechanism or counterexample: failure classes are asserted but not named; NaN/Inf-input rejection is implied by asserts but not a listed test obligation.
required repair: define distinct failure codes (qr_pivot_nonpositive, downdate_margin_nonpositive, nonfinite_derivative, invalid_observation_factor, batch_contract_violation) in the diagnostics schema; add explicit NaN/Inf-input rejection tests for the QR and downdate primitives.
test that closes the issue: primitive tests asserting the specific failure code per injected defect.
```

```text
issue_id: FS-8
severity: advisory
exact source/plan path: execution plan §6 (minimum diagonal/margin gates)
failure mechanism or counterexample: absolute ">0" gates admit arbitrarily ill-conditioned rows whose derivative divisions (1/r, 1/c) are finite but enormous.
required repair: record relative conditioning diagnostics (pivot relative to stack column norms; margin relative to L_kk²) with declared advisory thresholds; keep the hard gate at strict positivity.
test that closes the issue: conditioning fields present in Phase 6 artifacts and referenced in the tolerance-justification step of §6.
```

```text
issue_id: FS-9
severity: advisory
exact source/plan path: execution plan §3.4 derivative stack line
failure mechanism or counterexample: typographical: "[\sqrt{w_i^{(c)}},d_p\delta y_i]" has a comma where multiplication is meant.
required repair: correct to sqrt(w_i^{(c)}) · d_p δy_i.
test that closes the issue: none needed (editorial).
```

## F. Final decision

The plan's architecture is sound: the augmented block-diagonal placement, the
exact DZ5 rule with the center column, single inclusion of process noise, the
positive-diagonal QR derivative calculus, the factor-native score, the
sequential downdate strategy, the fail-closed policy, the batch/XLA contract,
and the test ladder are all correct and well-bounded, and the nonclaims are
consistently maintained. However, two normative derivation steps are wrong as
written and each is confirmed by a numerical counterexample: the gain
triangular-solve orientation (§3.5, FS-1) and the downdate vector-update
recurrence (§3.7, FS-2). Both are one-line repairs, but they sit exactly on
the implementation-critical path and the plan text is what implementers will
transcribe. Two material specification gaps (route-guard mechanism FS-3,
SPD-vs-roundoff attribution FS-4) must also be closed before the guard and
diagnostics can deliver what the plan promises.

VERDICT: REVISE
