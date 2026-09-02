# n=4 Failure Fix Plan — 2026-08-27

> **Retired on 2026-08-28:** Do not execute this plan. Its proposed branch
> normalization and R_shift/R_gram repairs are wrong relative to the traced
> target and Gram mathematics. The tau/sign evidence it used is also
> contradicted by the raw per-step artifacts. It is preserved as historical
> provenance. The active proposal is
> bayesfilter-n4-root-cause-diagnostic-plan-2026-08-28.md.

**Status:** RETIRED_DO_NOT_EXECUTE

**Provenance:** codex review of attempt05_n4_failure_analysis.tex identified a concrete implementation defect (floored-vs-unfloored Gram mismatch) plus four diagnostic gaps. This plan implements the five required repairs in the order specified by the failure-analysis document (§6, Problem A).

## Background

attempt05 produced a clean verdict for n=2 (r*(2)=6) but failed completely at n=4 (r*(4)=null). The n=4 failure exhibits a wrong-sign total (+36.94 vs reference -66.70, ~103 nats over T=20) while all explanatory diagnostics (rms, row ESS, ALS condition) remain healthy. The failure-analysis document identified three candidate mechanisms: floored-vs-unfloored Gram mismatch, branch-axis counting-measure mismatch, and RMS normalization understatement.

The codex review confirmed all three are real defects in the current code.

## The Five Required Fixes (in Order)

### Fix 1: Floored-vs-unfloored Gram mismatch (the gating defect)

**What is wrong:**
- Line 91 of `bayesfilter/highdim/squared_tt_engine_gaussian_xla_tf.py`: computes branch amplitudes from `chol = cholesky(gram + branch_gram_floor*floor_scale*I)` where `branch_gram_floor = 1e-12` relative (default from `EngineConfig`, line 70 of `squared_tt_engine_v0_tf.py`).
- Line 169: stores the **unfloored** `gram` as `suffix_gram` for the next step.
- Line 260 of `bayesfilter/highdim/retained_quadratic_form_tf.py`: evaluates the retained density using that **unfloored** `suffix_gram`.

So step t fits against H(E_{t-1} + δI)H^T but stores and propagates H E_{t-1} H^T into step t+1. This is a genuine inconsistency between what was fitted and what gets retained.

**Fix (preferred route: remove the unpropagated floor):**
1. Remove `branch_gram_floor` from the branch-amplitude Cholesky call (line 91 of `squared_tt_engine_gaussian_xla_tf.py`). Use `chol = tf.linalg.cholesky(gram)` directly.
2. Add a **fail-closed PSD check** before the Cholesky call:
   - Verify `gram` is finite (no NaN, no ±∞).
   - Verify `gram` is symmetric: `||gram - gram^T||_F / ||gram||_F < 1e-14`.
   - Compute eigenvalues via `tf.linalg.eigvalsh(gram)`.
   - Verify all eigenvalues ≥ 0 (PSD).
   - Verify relative eigenvalue margin: `λ_min/λ_max ≥ 1e-12` (same threshold as the removed floor, now a veto).
   - If any check fails, raise `ValueError("invalid Gram matrix: <reason>")`.
3. Record `λ_min(E)`, `λ_max(E)`, `cond(E) = λ_max/λ_min` in the per-step diagnostics dict (see Fix 3 below).
4. Apply the same fix to all other C2 engines: `squared_tt_engine_xla_tf.py` (line 203), `squared_tt_engine_gaussian_tf.py` (line 375), `squared_tt_engine_adapted_xla_tf.py` (line 89), `squared_tt_adjoint_engine_tf.py` (line 212), `squared_tt_engine_adapted_tf.py` (line 226).

**Alternative (if a ridge is deemed necessary after testing):**
Define a new modified program that stores `E_eff = E + δI` and uses it consistently everywhere: branch amplitudes, retained evaluation, z_h, z_c, score/tangent. Treat this as a numerics-altering route requiring a no-harm evaluation per the safety-guardrail policy. Not recommended unless the no-floor route fails the LGSSM cross-check.

**Evidence this is correct:**
The design note `bayesfilter-squared-tt-engine-branch-axis-design-2026-08-16.md` (line 36) specifies a relative-eigenvalue guard and a fail-closed singularity veto, not an unpropagated ridge. The current `branch_gram_floor` implementation violates single-source-of-truth.

### Fix 2: RMS normalization understatement

**What is wrong:**
The branch axis uses counting measure (identity mass matrix), so each physical coordinate contributes `r_left+1` times to the Gram integral. The reported fit residual `rms` is normalized by **branch-averaged** row count (the sum over all repeated branch rows divided by the number of distinct physical coordinates), but if τ is calibrated from `rms^2` while the Gram uses counting measure, ε²_rel = rms²/z_h is understated by factor `(r+1)`.

At rank 6, the counting-measure rms² = 7 × (8.05×10⁻³)² = 4.5×10⁻⁴, which is 4.5× the τ cap of 1e-4 — so τ saturates by construction, independent of fit quality.

**Fix:**
1. In `_fit_als_graph` (line 110–130 of `squared_tt_engine_xla_tf.py`), the reported `rms` is already correct for the ALS objective (it's the Christoffel-weighted residual over the branch-repeated rows). Do NOT change the ALS objective.
2. In the **τ computation** (line 302 of `squared_tt_engine_gaussian_xla_tf.py`), adjust the ε²_rel calculation to use the counting-measure rms² if the Gram uses counting measure:
   ```python
   eps_rel_sq_raw = float(rms.numpy())**2 / max(z_h_new, 1e-30)
   # branch_count from line 95 is available in this scope
   branch_count = retained.boundary_rank + 1
   eps_rel_sq = eps_rel_sq_raw * branch_count  # counting-measure correction
   tau_t = max(config.tau, min(eps_rel_sq, 1e-4))  # clamp as before
   ```
3. Record **both** the branch-averaged rms (current `rms`) and the counting-measure rms² in the per-step diagnostics dict (see Fix 3).

**Why this is correct:**
The Gram contraction (line 170) computes `z_h = einsum("ab,ab->", p_gram, new_gram)` which sums over the branch axis using counting measure (the `DiscreteIndicatorBasis1D` has identity mass matrix per `bases.py`). So the integral ∫ f² η over the full mixed-basis space counts each branch equally, making the effective residual `(r+1) × (branch-averaged rms²)`.

### Fix 3: Add Gram health diagnostics

**What is missing:**
The artifact records `cond_max = 55037`, which is the **ALS design-matrix condition** (line 125 of `squared_tt_engine_xla_tf.py`), not the retained Gram condition. We do not know whether the floored-vs-unfloored mismatch accumulated through near-singularity or remained O(10⁻¹²).

**Fix:**
1. After computing `new_gram` (line 169 of `squared_tt_engine_gaussian_xla_tf.py`), compute its eigenvalues:
   ```python
   gram_eigvals = tf.linalg.eigvalsh(new_gram)  # sorted ascending
   gram_lambda_min = float(gram_eigvals[0].numpy())
   gram_lambda_max = float(gram_eigvals[-1].numpy())
   gram_cond = gram_lambda_max / max(gram_lambda_min, 1e-30)
   ```
2. Add to the per-step diagnostics dict returned at line 178:
   ```python
   "gram_lambda_min": gram_lambda_min,
   "gram_lambda_max": gram_lambda_max,
   "gram_cond": gram_cond,
   "rms_branch_avg": float(rms.numpy()),
   "rms_counting_sq": float(rms.numpy())**2 * branch_count,
   "eps_rel_sq_corrected": eps_rel_sq,  # from Fix 2
   ```
3. These fields are already written to `rows.jsonl` by the runner's cell-append logic (line 136–140 of `run_attempt05_sv_ladder_20260826.py`), so no runner change is needed.

### Fix 4: Per-step instrumentation

**What is needed:**
Dump {shift_t, log z_h_t, log z_c_t, τ_t} for one n=4 d=6 cell and compare the running Σ Δ_t to the reference's per-step log-increments, to localize where the sign/magnitude diverges.

**Fix:**
1. Add a `dump_per_step: bool = False` kwarg to `run_value_filter_branch_axis_gaussian_xla` (line 18 of `squared_tt_engine_gaussian_xla_tf.py`).
2. If `dump_per_step`, write a line to `sys.stderr` at line 310 (after computing `log_increment`):
   ```python
   if dump_per_step:
       sys.stderr.write(
           f"STEP {t}: shift={shift:.10e} log_zh={tf.math.log(z_h_new):.10e} "
           f"log_zc={tf.math.log(zc_new):.10e} tau={tau_t:.10e} "
           f"incr={increment_value:.10e} cumul={cumulative_value:.10e}\n"
       )
       sys.stderr.flush()
   ```
   where `cumulative_value` is a running sum initialized to 0 before the loop.
3. In the runner, add a `--dump-per-step` flag that passes `dump_per_step=True` to the engine call.
4. Run `python docs/benchmarks/run_attempt05_sv_ladder_20260826.py --cell 4 6 6 42 --dump-per-step 2> per_step_n4_d6_r6_s42.log` and compare the cumulative value at each step to the reference's per-step totals.

### Fix 5: LGSSM cross-check

**What is needed:**
Run the same n=4 d=6 r=6 configuration on the LGSSM oracle (where Gate A3 was exact at n=4). If it also produces the wrong-sign total, it is a route defect, not SV-specific. If it succeeds, the SV failure is model-specific.

**Fix:**
1. The LGSSM oracle fixture already exists: `tests/highdim/test_c2_gaussian_engine_oracle.py`.
2. Add a new test function `test_n4_d6_r6_cross_check` that:
   - Calls `_lgssm_fixture(n=4, T=12, seed=44)` (or use T=20 if budget allows; the key is n=4 with exact hints).
   - Runs `run_value_filter_branch_axis_gaussian_xla` with `degree=6, rank=6, row_count=8192, sweeps=32` (matching the SV cell).
   - Compares the engine total to the exact Kalman filter total (from `_exact_hint_factories` which returns the oracle log-increments).
   - Asserts the gap is < 1e-9 (the Gate A3 standard).
3. If the test passes, the n=4 route is correct for LGSSM and the SV failure is model-specific (hint drift, likelihood underflow, or degree starvation). If it fails, the route has a general defect independent of the model.

**Note:** Gate A3 passed at `n=4, d=6, r=3, T=10` in the original A3 run. The cross-check here is at `r=6, T=20`, which is strictly harder, so a pass is the required evidence; a fail at r=6 T=20 may indicate capacity (not enough rank/sweeps), not a bug — so the cross-check should also try r=3 T=10 if r=6 T=20 fails.

## Execution Order

1. **Fix 1 (Gram mismatch):** highest priority, gating defect.
2. **Fix 2 (RMS correction):** required for correct τ, independent of Fix 1.
3. **Fix 3 (Gram diagnostics):** pure observability, no behavior change, safe to add.
4. **Run unit tests** to verify Fixes 1–3 don't break existing passing cases.
5. **Fix 5 (LGSSM cross-check):** diagnostic run to decide SV-specific vs general.
6. **Fix 4 (per-step dump):** final instrumentation if cross-check fails or to localize the remaining error after Fixes 1–2.

## Success Criteria

**After Fixes 1–3:**
- All existing unit tests pass (especially `test_c2_gaussian_engine_oracle.py`).
- The n=4 d=6 r=6 s=42 cell no longer has a wrong-sign total, or the per-step dump (Fix 4) localizes the remaining error to a specific step and mechanism.

**After Fix 5:**
- If LGSSM cross-check passes: SV failure is model-specific, document as capacity or hint-drift issue.
- If LGSSM cross-check fails: route has a general defect, continue debugging with Fix 4.

**Terminal success:**
- n=4 d=6 r≤6 either passes veto-clean (making r*(4) measurable) or fails with a localized, explained cause (making r*(4)=null a capacity result, not a bug).

## Risks and Contingencies

**Risk 1: Removing the floor causes Cholesky failures.**
- **Mitigation:** The fail-closed PSD check (Fix 1, step 2) will veto before Cholesky is called, preserving the Class B safety guarantee. If the veto fires frequently, investigate why `gram` is near-singular (ALS convergence, row starvation, or ill-conditioned target).
- **Fallback:** Implement the alternative (store `E_eff = E + δI` consistently) as a reviewed numerics-altering route.

**Risk 2: The counting-measure correction (Fix 2) over-corrects and makes τ too large.**
- **Mitigation:** The correction is a simple factor-of-(r+1) multiplication; at rank 6, it changes τ from being understated by 7× to being correctly stated. If τ becomes "too large" (i.e., fires more vetoes), that is the **correct** behavior — the current code is under-reporting the residual relative to the normalization integral.
- **Verification:** Compare τ values before and after Fix 2 on the n=2 passing cells (which should be unaffected or slightly tighter).

**Risk 3: The LGSSM cross-check (Fix 5) fails due to insufficient rank/sweeps, not a bug.**
- **Mitigation:** Try the original Gate A3 configuration (n=4 d=6 r=3 T=10) first, which passed at 5.5e-10 gap. If that still passes after Fixes 1–3, the route is correct; if it fails, it's a regression and a blocker.

**Risk 4: Fixes 1–3 are correct but insufficient; the n=4 failure persists.**
- **Response:** This is an expected possibility. Fixes 1–3 remove known defects but do not guarantee the n=4 result becomes interpretable. If the failure persists after Fixes 1–3, Fix 4 (per-step dump) and Fix 5 (LGSSM cross-check) will localize the remaining cause. The failure-analysis document (§6) already lists the next candidates: shift-Gram normalization inconsistency, defensive-mass amplification, hint drift. The per-step dump will separate them.

## Artifacts

**Before Fixes:**
- `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/cell_n4_d6_r6_s42_w32.json` (wrong-sign total: +36.94)
- `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/verdict.json` (r*(4)=null)

**After Fixes 1–3:**
- New cell artifact at the same path (or a timestamped sibling) with corrected τ, added Gram diagnostics, and (hopefully) corrected total.
- Unit test results showing all existing tests still pass.

**After Fix 5:**
- `tests/highdim/test_c2_gaussian_engine_oracle.py::test_n4_d6_r6_cross_check` result (pass/fail).

**After Fix 4 (if needed):**
- `per_step_n4_d6_r6_s42.log` with per-step {shift, log z_h, log z_c, τ, incr, cumul}.

## Review Checklist

- [ ] Fix 1 removes the unpropagated floor and adds fail-closed PSD check with λ_min, λ_max, cond(E) recording.
- [ ] Fix 2 corrects τ calibration with the counting-measure rms² factor.
- [ ] Fix 3 adds Gram health diagnostics to per-step dict.
- [ ] Fix 4 adds per-step dump infrastructure (kwarg + stderr logging).
- [ ] Fix 5 adds LGSSM cross-check test at n=4 d=6 r=6.
- [ ] Execution order is Fix 1 → Fix 2 → Fix 3 → unit tests → Fix 5 → (Fix 4 if needed).
- [ ] Success criteria are clear and testable.
- [ ] Risks have stated mitigations and fallbacks.
- [ ] All five fixes match the failure-analysis document (§6, Problem A, five required repairs).

---

**Plan author:** Claude Code (Opus 4.8)  
**Plan date:** 2026-08-27  
**Plan basis:** codex review of `attempt05_n4_failure_analysis.tex` + failure-analysis document §6
