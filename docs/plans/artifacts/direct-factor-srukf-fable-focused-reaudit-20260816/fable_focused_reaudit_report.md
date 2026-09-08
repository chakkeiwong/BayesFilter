# Fable Focused Re-audit Report: Revised Direct-Factor SR-UKF Plan

Date: 2026-08-16
Reviewer: Fable (Claude, Opus 5), local MathDevMCP CLI
Status: `FOCUSED_REAUDIT_COMPLETE_NO_IMPLEMENTATION_AUTHORITY`
Scope: closure of FS-1 through FS-9 only, per
`docs/plans/bayesfilter_direct_factor_srukf_fable_focused_reaudit_handoff_2026_08_16.md`.
The architecture and derivations accepted in the 2026-08-15 audit were not
re-reviewed.

## 1. Provenance

- Commit: `3030d86df9cb00346df82c7c19f015c09c7c6e1f` (unchanged). Dirty
  worktree: same pre-existing files plus plan documents; no source edits by
  this re-audit.
- Hashes (`source_hashes.txt`): revised plan
  `307a6bc6…`, this handoff `ec2492e4…`, prior reply `818be811…`, prior
  `counterexample_checks.txt` `10526089…`.
- Environment: `environment.json` (MathDevMCP `doctor` ok). No network, no
  package mutation, no file edits outside this artifact root.
- Per handoff §3, the previously-abstaining broad symbolic calls were not
  repeated. Evidence is bounded human algebra plus an independent NumPy
  diagnostic script under this artifact root
  (`reaudit_numeric_checks.py`, output
  `reaudit_numeric_checks_output.txt`) — explicitly permitted as independent
  reference evidence, run with `tf-gpu-221` env Python, float64,
  deterministic fixtures, centered FD at `h` and `h/2`. 18/18 checks passed.

## 2. Finding-by-finding disposition

### FS-1 (blocking) — §3.5 gain solve orientation: **CLOSED**

The normative route is now `U := K S_y`, `U S_y' = P_xy`,
`U' = S_y^{-1} P_xy'` (lower solve), `K = U S_y^{-1}` (equivalently
`K' = S_y^{-T} U'`, upper solve). Algebra: transposing `U S_y' = P_xy` gives
`S_y U' = P_xy'`, hence the lower solve; then
`K = P_xy S_y^{-T} S_y^{-1} = P_xy (S_y S_y')^{-1}` exactly. The derivative
`dU' = S_y^{-1}(dP_xy' − dS_y U')` follows from differentiating
`S_y U' = P_xy'` (the plan now shows this), and
`dK = dU S_y^{-1} − K dS_y S_y^{-1}` from differentiating `K S_y = U`.

Numerical evidence (check 1): on the prior counterexample fixture
(`S_y = [[2,0],[1,1]]`, `P_xy = [[1,2],[3,4]]`), revised `K` matches the dense
`P_xy P_y^{-1}` oracle to 0.0; the pre-revision orientation still errs by
1.75 (counterexample preserved). Analytic `dU`, `dK` match centered FD at
`h = 1e−6` and `5e−7` (max err ≤ 4.6e−10); the derivative-solve residual
`S_y dU' + dS_y U' − dP_xy'` is 0.0. Closing tests (dense-oracle gain
comparison; affine filtered mean/factor closed-form assertions) are in
Phase 3, so a wrong orientation can no longer pass on likelihood alone.

### FS-2 (blocking) — §3.7 downdate recurrence: **CLOSED**

The normative recurrence is now `a_new = (a_old − s·u_old)/c`,
`u_new = c·u_old − s·a_new`, with explicit sequential order (`a_new` first),
the all-old equivalent `u_new = (u_old − s·a_old)/c` recorded (equivalence:
`c·u − s(a−su)/c = (u(c²+s²) − sa)/c = (u − sa)/c`), the requirement to
document which form is implemented, and the statement that all pivot
derivative right-hand sides use pre-update values with `da_new` computed
before `du_new`.

Numerical evidence (check 2): on the prior counterexample fixture
(`L = [[2,0],[1,1.5]]`, `x = [0.5,0.3]`), both forms reconstruct
`LL' − xx'` to 4.4e−16 and agree with each other to 0.0; the pre-revision
rule still fails at 1.03e−3 (counterexample preserved). The full §3.7
derivative recurrence matches centered FD at both step sizes (≤ 3.9e−10) and
satisfies `dS_f S_f' + S_f dS_f' = dP_f` against the independently assembled
`dP_f` to 3.1e−17. The internal inconsistency flagged in FS-2 is gone: value
text and derivative equations now describe the same recurrence.

### FS-3 (material) — route guard boundary: **CLOSED**

§4.1/§4.2/§7 now specify: a closed admitted file set (`stack_qr_tf.py`,
`lower_rank_downdate_tf.py`, `factor_srukf_tf.py`); the compatibility
conversion isolated in non-admitted `factor_srukf_compat.py`, never imported
by the admitted runtime; a standalone QR kernel forbidden from importing the
legacy mixed-purpose `qr_factor_tf.py`; case-insensitive substring matching
over an exact 14-token list (including `cholesky`, `tf.linalg.eigh`,
`tf.linalg.svd`, principal-root and SVD dispatch names, and
covariance-refactorization helper names); extension of
`FORBIDDEN_SRUKF_ROUTE_PATTERNS`; a closed-file-list assertion; and an
import-boundary test plus runtime downdate provenance as the semantic
complement.

Implementability: the existing scanner model (substring over given paths)
needs only a lowercase extension, both trivially mechanical. The rename from
the original plan's `cholesky_downdate_tf.py` to `lower_rank_downdate_tf.py`
removes the token collision, so the rank-downdate primitive is not falsely
banned; nothing the three admitted files legitimately need
(`tf.linalg.qr`, `tf.linalg.triangular_solve`, `band_part`, `while_loop`)
matches any banned token. The new backend filename `factor_srukf_tf.py` is
distinct from the legacy prototype `srukf_factor_tf.py`, so the legacy file
stays outside the admitted set without a collision. Implementer note (not a
finding): case-insensitive `cholesky` also bans the word in comments and
docstrings of admitted files — intentional per the plan's evasion clause;
prose there should say "lower-factor rank-one downdate". Note also that the
standalone-kernel rule deliberately duplicates the §3.3 QR derivative
machinery rather than importing `qr_factor_tf.py`; this is an isolation
tradeoff, and the Phase 1 tests plus unchanged §3.3 math keep it audited.

### FS-4 (material) — SPD vs roundoff attribution: **CLOSED**

§3.7 now states the feasibility lemma
`P^(k) = (L^(k))(L^(k))' = P_f + Σ_{j>k} v_j v_j' ⪰ P_f`, hence exact SPD
`P_f` makes every intermediate target SPD and every exact scalar-pivot margin
positive for every column order. Verified numerically (check 4): the partial
factor reproduces `P_f + Σ_{j>k} v_jv_j'` to 4.4e−16 after each column under
both column orders, with all scalar margins positive.

Attribution is correctly diagnostic: the original runtime failure is retained
first; the offline eigenvalue comparator (explicitly outside the admitted
runtime, never a fallback) classifies `downdate_target_indefinite` vs
`downdate_roundoff_or_implementation_suspected`; the artifact keeps the
comparator covariance, `λ_min`, first failed time/column/pivot, margin, and
conditioning. The "suspected" naming and the diagnostic-only language keep
floating-point evidence from being promoted to an exact mathematical claim.
Observation (no change required): the comparator's `λ_min` is itself a
floating-point quantity, so classifications with `λ_min` near zero are
boundary evidence in either direction; the retained raw evidence already
supports that reading. Phase 3 fixtures 5 and 6 exercise both classes.

### FS-5 (advisory) — block comparator: **CLOSED**

§3.7 defines the zero-extended stack `A_x^+ = [A_x, 0_{n_x×n_y}]` and
`Ã_f = A_x^+ − K A_y`, dimensionally aligned with the noise-augmented `A_y`
(both have `2d+1+n_y` columns; `A_x^+ A_y' = P_xy` because the zero block
annihilates the `S_r` columns). Its derivative
`dÃ_f = dA_x^+ − dK A_y − K dA_y` is complete, including `dS_r` inside
`dA_y`. Verified (check 3): `Ã_f Ã_f' = P⁻ − K P_y K'` to 1.1e−16;
`dÃ_f` matches centered FD at both step sizes (≤ 1.1e−10); the product-rule
reconstruction against an independently assembled `dP_f` holds to 1.4e−17.
Diagnostic-only role and never-repair language are explicit; Phase 5 wires it
in as a comparator.

### FS-6 (advisory) — FD protocol: **CLOSED**

§6 declares centered differences, per-parameter steps
`h_i = η·max(1,|p_i|)`, recording of `η` and the actual `h_i`, repetition at
`h_i/2` with agreement required at both steps and the inter-estimate change
reported, all frozen from benign-fixture conditioning before any rare-row
inspection. This satisfies the MacroFinance Stage F step-size/repeatability
requirement.

### FS-7 (advisory) — failure codes and nonfinite inputs: **CLOSED**

§4.3 names the five failure codes (plus `none`) exactly as recommended and
requires explicit NaN/Inf rejection of inputs and intermediates; Phase 1 adds
NaN/Inf rejection tests for stacks and stack derivatives, Phase 2 for
factors, vectors, and derivatives; first-failed time/column/pivot is in the
per-row diagnostics.

### FS-8 (advisory) — conditioning diagnostics: **CLOSED**

§4.3 records QR pivots relative to stack column norms and downdate margins
relative to `L_kk²`, advisory only, with strict positivity kept as the hard
gate and an explicit prohibition on conditioning thresholds converting a row
to a different algorithm or excusing a failure.

### FS-9 (advisory) — §3.4 typo: **CLOSED**

The derivative stack now reads `[\sqrt{w_i^{(c)}}\,d_p\delta y_i]_i ∥ d_pS_r`
(multiplication, comma removed).

## 3. Acceptance answers (handoff §4)

1. §3.5 computes exactly `P_xy(S_yS_y')^{-1}`, derivative consistent with the
   value program — **SUPPORTED** (check 1; algebra above).
2. §3.7 is a valid lower-factor rank-one downdate with unambiguous old/new
   value and derivative ordering — **SUPPORTED** (check 2; both forms agree,
   prior counterexample still refutes the old rule).
3. `V = U = K S_y` with `dV = dU` — **SUPPORTED** (same matrix by
   definition; §3.5/§3.7 cross-reference is consistent, one solve feeds both
   the mean update and the downdate).
4. Admitted file boundary mechanically enforceable without banning the
   legitimate compatibility conversion — **SUPPORTED** (closed three-file
   list; compat module non-admitted and unscanned but import-boundary
   tested; renamed downdate file avoids the `cholesky` token; scanner needs
   only a trivial case-insensitivity extension).
5. Feasibility lemma justifies the exact-arithmetic statement for every
   column order; floating-point attribution kept diagnostic — **SUPPORTED**
   (check 4 for both orders; retained-evidence-first ordering; "suspected"
   class naming; comparator confined offline).
6. Block comparator dimensionally valid with observation-noise columns and
   derivative complete — **SUPPORTED** (check 3, including `dS_r` columns).
7. FD, nonfinite-input, conditioning, failure-code, unit, and
   integration-test obligations sufficient for FS-6–FS-9 — **SUPPORTED**
   (§6 protocol; §4.3 codes and rejections; Phase 1/2 test items; Phase 3
   fixtures 5–6).
8. No new hidden inverse, covariance refactorization, runtime eigh/SVD,
   fallback, nonlinear orientation-equivalence claim, or
   implementation/production authority introduced — **SUPPORTED**. The only
   new eigenvalue use is the offline failure classifier (non-admitted,
   diagnostic, never fallback); the comparator's dense `P_y^{-1}` appears
   only in test/diagnostic assembly; Phase −1 grants plan-review gating
   only; §9 nonclaims and the no-switch boundary are unchanged.

## 4. Tool limitations and abstentions

Per handoff instruction, the broad symbolic `derive-or-refute`/
`assumptions-for` calls were not repeated (prior ABSTAIN on noncommutative
matrix products stands). All four required numerical checks ran to
completion; no new abstentions. MathDevMCP output here is diagnostic
evidence, not a proof certificate.

## 5. Artifacts

Under `docs/plans/artifacts/direct-factor-srukf-fable-focused-reaudit-20260816/`
(prior audit root untouched):

```text
environment.json                     doctor output
source_hashes.txt                    commit, dirty status, 4 file hashes
reaudit_numeric_checks.py            diagnostic script (checks 1-4)
reaudit_numeric_checks_output.txt    raw output, 18/18 PASS
fable_focused_reaudit_report.md      this report
```

## 6. Decision

All nine findings are closed at their cited anchors; both blocking
counterexamples now confirm the revised equations and still refute the
pre-revision ones; no new defects or authority were introduced by the
revision. Per the handoff boundary, this verdict authorizes only the bounded
implementation phases in the revised plan — not implementation completion,
MacroFinance integration, NeuTra, HMC/NUTS, a backend switch, production use,
or scientific claims.

VERDICT: AGREE
