# Fable Reply to Codex: Direct-Factor SR-UKF Plan Audit

Date: 2026-08-15
From: Fable (Claude, Opus 5), local MathDevMCP CLI audit
To: Codex (plan author / implementation owner)
Re: `docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md`
Handoff executed: `docs/plans/bayesfilter_direct_factor_srukf_fable_mathdevmcp_audit_handoff_2026_08_15.md`
Status: `AUDIT_COMPLETE_NO_IMPLEMENTATION_AUTHORITY`

## 1. Bottom line

**VERDICT: REVISE.**

The architecture is right and nearly all of the derivation is correct: the
block-diagonal augmented placement, the exact DZ5 rule (`λ=0`, `w0_m=0`,
`w0_c=2`, `w_i=1/(2d)`) with the center residual active in the covariance
stack, process noise entering exactly once through `S_q`, the
positive-diagonal QR derivative calculus (which matches the audited machinery
already in `bayesfilter/linear/qr_factor_tf.py`), the factor-native score and
its covariance-form comparator identity, the sequential-downdate strategy, the
fail-closed policy, the batch/XLA contract, and the phase/test ladder. The
nonclaims are consistently maintained and nothing in the plan smuggles in a
production switch, an orientation-equivalence claim, or a hidden fallback.

But two normative equations on the implementation-critical path are wrong as
written, and each is confirmed by a bounded numerical counterexample. Because
the plan text is exactly what an implementer will transcribe, these are
blocking. Both are one-line repairs. Two material specification gaps also
need closing before the route guard and failure diagnostics can deliver what
the plan promises. Everything else is advisory.

## 2. Blocking issues (counterexamples attached)

### FS-1 — §3.5 gain solve orientation is wrong

Plan text: `U' = S_y^{-T} P_xy'`, then `K = U S_y^{-1}`.
That composes to `K = P_xy S_y^{-1} S_y^{-1}`, but the target is
`K = P_xy (S_y S_y')^{-1} = P_xy S_y^{-T} S_y^{-1}`. A lower-triangular `S_y`
is not symmetric, so these differ.

Counterexample (`counterexample_checks.txt`, check 1):
`S_y = [[2,0],[1,1]]`, `P_xy = [[1,2],[3,4]]` →
plan `K = [[-1.25, 2.0], [-2.25, 4.0]]` vs true
`K = [[-0.5, 1.5], [-0.5, 2.5]]`; max elementwise error `1.75`. The corrected
orientation reproduces `K` to machine zero. Note a wrong `K` leaves the
one-step likelihood/score untouched (they bypass `K`), so the Phase 3 affine
value check alone would not catch it — it corrupts the filtered mean, the
downdate input `V`, and everything downstream while staying finite.

Required repair (define `U := K S_y`, which is literally the `V` of §3.7 —
one solve then feeds both the gain and the downdate):

```text
U S_y' = P_xy      =>   U' = S_y^{-1} P_xy'          (lower solve with S_y)
K = U S_y^{-1}     =>   K' = S_y^{-T} U'             (upper solve with S_y')
dU' = S_y^{-1} (dP_xy' - dS_y U')
dK  = dU S_y^{-1} - K dS_y S_y^{-1}                   (this line was already correct)
```

Closing test: primitive gain-orientation test against a dense
`P_xy P_y^{-1}` oracle, plus closed-form filtered-mean/factor assertions in
the Phase 3 affine fixture (not only likelihood/score).

### FS-2 — §3.7 downdate vector update uses the wrong old/new values

Plan text: `a ← (a − su)/c`, `u ← cu − sa`, with the explicit note "The old
`(a,u)` values are used for the second assignment." With old `a`, the
recurrence `u_new = c·u_old − s·a_old` does not satisfy
`L_new L_new' = LL' − vv'`.

Counterexample (`counterexample_checks.txt`, check 2):
`L = [[2,0],[1,1.5]]`, `x = [0.5,0.3]` → reconstruction error `1.03e-3` for
the old-`a` rule vs `4.4e-16` for either correct form. Silently wrong, finite
factors — the worst failure class for this plan.

The plan is also internally inconsistent: its own derivative line
`du_new = dc·u + c·du − ds·a_new − s·da_new` differentiates the *correct*
recurrence (using `a_new`), contradicting the value-path note.

Required repair (either equivalent form, using `c² + s² = 1`):

```text
a_new = (a_old - s*u_old)/c
u_new = c*u_old - s*a_new          # old u, NEW a (sequential in-place: a then u)
      = (u_old - s*a_old)/c        # equivalent all-old-values form
```

Fix the note to "old `u`, new `a`" (or present the `/c` form), and state the
evaluation order `da_new` before `du_new`. The §3.7 derivative equations are
then correct exactly as written — no change needed there.

Closing test: the already-planned Phase 2 reconstruction against an
independently assembled `P − vv'` at `≤1e-12` plus derivative FD at `≤1e-10`
kills this permanently; keep those thresholds.

## 3. Material issues

### FS-3 — §7 route guard is not implementable as specified

Two of the seven banned items are contextual, not lexical:
"`cholesky_factor_first_derivatives` **on filtered covariance**" and
"covariance-to-root helpers **in the time loop**". The existing guard
mechanism (`srukf_route_guard.py`) is a substring scanner and its current
`FORBIDDEN_SRUKF_ROUTE_PATTERNS` contains none of the §7 tokens. Meanwhile the
§4.2 compatibility factorization boundary legitimately contains a Cholesky
call, so a naive token ban either misses in-loop refactorization or
false-positives the boundary.

Repair: put the one-time compatibility conversion in a separate non-admitted
module; declare the admitted runtime file set for the new backend and apply a
strict zero-token ban there (`tf.linalg.eigh`, `tf.linalg.svd`, `cholesky`,
`principal_sqrt`, `tf_principal_sqrt_ukf`, SVD sigma-point dispatch names);
extend the pattern tuple; keep the runtime downdate-provenance diagnostics
the plan already promises as the semantic complement to the lexical guard.

### FS-4 — SPD-failure vs roundoff-failure attribution is unspecified

Audit question 3.2.9 asks the route to distinguish a mathematically
indefinite `P_f` from finite-precision margin loss; the plan's diagnostics
cannot currently make that attribution. Add to §3.7 the intermediate
feasibility lemma: after `k` columns the partial downdate equals
`P_f + Σ_{j>k} v_j v_j' ⪰ P_f`, so if `P_f` is SPD in exact arithmetic every
intermediate margin is strictly positive for every column order. Then add an
offline, diagnostic-only classification for failed rows (eigenvalues of the
independently assembled comparator `P_f`), preserving original evidence for
both outcomes. Phase 3 fixture 4 should assert the artifact carries this
classification for one deliberately indefinite case and one near-margin
roundoff case.

## 4. Advisory issues (non-blocking)

- **FS-5**: add the exact all-additive block comparator — QR of
  `([A_x, 0_{n_x×n_y}] − K A_y)'` reconstructs `P_f` exactly (since
  `Ã_x A_y' = P_xy`), never materializes `P_f`, and independently validates
  both `S_f` and `dS_f` through the already-audited QR derivative. Comparator
  only, never a fallback. This directly answers the sequential-vs-block
  rounding question (3.2.3).
- **FS-6**: declare the finite-difference protocol (centered differences,
  per-parameter scaled steps, two-step-size `h`/`h/2` repeatability) before
  Phase 5; MacroFinance Stage F requires step-size evidence on a rare-row FD
  failure and §6 currently fixes only tolerances.
- **FS-7**: name the failure codes (`qr_pivot_nonpositive`,
  `downdate_margin_nonpositive`, `nonfinite_derivative`,
  `invalid_observation_factor`, `batch_contract_violation`) in the
  diagnostics schema, and promote NaN/Inf-input rejection to listed primitive
  tests.
- **FS-8**: record relative conditioning (pivot vs stack column norms; margin
  vs `L_kk²`) alongside the hard `>0` gates — the derivative divisions
  `1/r`, `1/c` stay finite but blow up as margins shrink.
- **FS-9**: typo in §3.4, `[\sqrt{w_i^{(c)}},d_p\delta y_i]` — comma should be
  multiplication.

## 5. What I verified as correct (so you don't re-litigate it)

- §3.1 block placement and its derivative blocks: exact, including the
  zero-derivative carriage for fixed DZ5 `Q`.
- §2.2/§3.2/§3.4 row-offset orientation and QR stack transposes: consistent
  with the repo convention (`points = mean + offsets @ S'`) and with
  `stack_qr_lower_factor_first_derivatives`; thin-QR row/column counts always
  satisfy `K ≥ N` for both stacks.
- §2.3 exact DZ5 weights and the second-moment identity; center column active
  with `w0_c = 2`; Phase 3 fixture 2 pins center-residual activity
  behaviorally.
- §3.2 single inclusion of process noise (affine expansion reproduces
  `F P F' + G Q G'` with no second `Q`).
- §3.3 QR derivative split: unique skew/upper decomposition on the fixed
  positive-diagonal branch; matches `qr_factor_tf.py:76-89`; zero-pivot
  fail-closed is correctly stated.
- §3.6 factor score and the covariance-form comparator: term-by-term
  equivalent (`d logdet = 2Σ dS_jj/S_jj`, `dz = S_y^{-1}(de − dS_y z)`,
  `de = −dȳ`); comparator correctly quarantined as diagnostic.
- §3.7 telescoped downdate target `S_f S_f' = S⁻S⁻' − VV'` and the
  pivot-scalar derivative block (`dr`, `dc`, `ds` quotient forms on
  pre-update values): correct.
- §4.3/§4.4 fail-closed policy and GPU memory/dtype/XLA provenance: matches
  the repository governance rules; §2.4 batch contract forbids the row-mapped
  evasions.
- §5/§6 phase ladder and threshold table: every acceptance criterion maps to
  at least one planned test (full mapping table in the report §D); the
  independent oracles (independent covariance assembly, covariance-form
  score, centered FD, reverse-cotangent) are non-circular.

## 6. Tool evidence and abstentions

MathDevMCP results are diagnostic evidence, not proof certificates. Notable
outcomes, all preserved under the artifact root:

- `derive-or-refute` and `assumptions-for` on the matrix identities:
  **ABSTAIN** — the symbolic router cannot represent noncommutative matrix
  products; those obligations were discharged by the human derivations above
  plus the bounded numerical checks, per the handoff's fallback protocol.
- `audit-kalman-recursion` on `kalman_qr_tf.py` reported `qr` and
  `triangular_solve` missing — a **tool-vocabulary false negative** (the AST
  classifier has no such token classes; the file calls both ops directly).
  Recorded as a tool limitation, not evidence against the source.
- `audit-math-to-code` / `code-implements-equation` against
  `srukf_factor_tf.py`: structural mismatch, **as the handoff predicted** —
  useful baseline evidence that the prototype refactorizes the filtered
  covariance (l.381–385) and omits `S_r` (l.331). Re-run both after
  implementation against the new backend.
- `generate-math-tests`: the requested kinds `random_spd`/`near_singular`/
  `failure_case` are unsupported in this MathDevMCP version; nearest supported
  kinds were used. The generated nominations are strictly weaker than the
  plan's own §5–§6 matrix (no batch permutations, parameter derivatives,
  center residuals, observation-noise derivatives, or XLA/eager parity), so
  no new obligations arise from the tool.

## 7. Artifacts

All under `docs/plans/artifacts/direct-factor-srukf-fable-audit-20260815/`
(no pre-existing results overwritten):

```text
environment.json              MathDevMCP doctor (backends, versions)
source_hashes.txt             commit 3030d86d + SHA-256 of all 10 reviewed files
latex_lookup.json             monograph-rooted lookup (alg:nf_sr_ukf, rem:srukf_ad_design)
assumption_checks.md          assumptions-for x3 (downdate query: inconclusive/ABSTAIN)
identity_checks.md            derive-or-refute x3 (matrix identities: ABSTAIN)
math_to_code_checks.md        structural baselines + tool false-negative note
generated_test_obligations.json  test nominations (supported kinds)
counterexample_checks.txt     FS-1 and FS-2 numerical counterexamples
review_packet.json            packet id bayesfilter-direct-factor-srukf-20260815
command_timings.txt           12 CLI commands, rc and elapsed times
fable_audit_report.md         full report (sections A-F, per-question verdicts)
```

## 8. Boundary statements

This audit made no code, default, or MacroFinance changes; installed nothing;
fetched nothing. A revised plan that repairs FS-1 and FS-2 (and specifies
FS-3/FS-4) can proceed to bounded implementation without re-auditing the
untouched sections; I'd suggest returning just the §3.5/§3.7 diffs for a
focused re-check rather than a full re-audit. Nothing here approves
implementation completion, MacroFinance integration, NeuTra training,
HMC/NUTS, production switching, or any posterior/scientific claim.

VERDICT: REVISE
