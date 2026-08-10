# SVX adjacent-state analytic derivative implementation plan

Date: 2026-08-06
Status: `ACTIVE_IMPLEMENTATION_PLAN`

## Context

The diagnostic phase established three decisive facts about the active `SVX-ZC`
serious NeuTra/HMC route.

1. The current active target uses an autodiff score backend through
   `bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf.batched_fixed_tt_likelihood_value_score_status(...)`.
2. The active value program depends materially on the **adjacent UKF-frozen
   cores** beginning at `t=1`; zeroing them drives the `T=2` value to `-inf`,
   while zeroing the one-axis initial core leaves `T=1` and `T=2` unchanged.
3. The strongest existing analytic transformed-SV score candidate,
   `exact_transformed_sv_independent_panel_zhaocui_tt_score(...)`, is not the
   same finite value program: the value gap is small at `T=1` but appears at
   `T=2` and grows by `T=10`.

So the missing serious-route artifact is not an adapter swap. It is the
**analytic adjacent-state derivative for the exact active frozen-core finite
value program**.

## Step 1 — Write the math derivation first

Before changing code, write a mathematical derivation note under `docs/plans/`
that formalizes the active finite program and its required derivative in the
repo's notation.

Recommended new note:

- `docs/plans/bayesfilter-svx-adjacent-state-analytic-derivative-derivation-2026-08-06.md`

This note should explicitly define:

1. the active value program at `t=0` and `t>=1`;
2. the role of the transformed observation
   `z_t = log(y_t^2) - 2 log(beta)` / equivalent target observation form;
3. the one-axis initial fit:
   - target values,
   - weighted least-squares solve,
   - normalizer contribution;
4. the adjacent-state fit:
   - predictive target using the retained previous marginal,
   - transition density,
   - observation density,
   - adjacent-state weighted LSQ fit,
   - two-axis normalizer;
5. the exact derivative terms that must be propagated:
   - derivative of the target log values with respect to each parameter,
   - derivative of the weighted LSQ solution,
   - derivative of the squared-TT normalizer,
   - derivative of the retained marginal values used at the next time step,
   - total accumulated score across time;
6. the exact assumptions / non-claims:
   - same finite deterministic program,
   - fixed branch only,
   - no moving basis,
   - same UKF-frozen adjacent-core identity,
   - no target change relative to the current active value path.

The note must separate clearly:

- what is already implemented in `filtering.py` helper algebra,
- what can be reused verbatim,
- what is genuinely new for the adjacent-state batched TT program.

## Step 2 — Audit the derivation with MathDevMCP CLI

After writing the derivation note, audit it with MathDevMCP CLI before coding.

Use the local MathDevMCP environment under `/home/chakwong/MathDevMCP` and run
at least:

1. `assumptions-for` on the main derivative claims,
2. `debug-derivation` / `derive-from` on the LSQ derivative and normalizer
   derivative steps,
3. `audit-math-to-code` or `prepare-review-packet` on the mapping from the
   derivation note to the intended code hooks.

The goal of this phase is not a proof assistant certificate; it is a rigorous
external-tool-first audit of the derivation before implementation.

The result should record:

- which derivation steps are directly supported,
- which steps need clearer assumptions,
- whether any hidden target change slipped in,
- and whether the planned code reuse from `filtering.py` is mathematically
  faithful.

## Step 3 — Implement the analytic adjacent-state derivative in the active module

Primary file:

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`

Implementation target:

- leave `batched_fixed_tt_likelihood_value_status(...)` as the value authority;
- replace `batched_fixed_tt_likelihood_value_score_status(...)` so it no longer
  uses `tf.GradientTape()` through the whole program;
- compute the score analytically for the same finite program.

Expected implementation strategy:

1. factor the current value path into helpers for:
   - target log values at `t=0`,
   - target log values at `t>=1`,
   - one-axis fit,
   - two-axis fit,
   - one-axis normalizer,
   - two-axis normalizer,
   - previous marginal evaluation;
2. add derivative helpers for:
   - target log values,
   - weighted LSQ solve via `fixed_design_lsq_derivative(...)` style algebra,
   - one-axis and two-axis normalizer derivatives,
   - retained marginal derivative propagation;
3. accumulate score terms by time step without autodiff through the whole
   recursion.

Prefer reusing algebra already in `bayesfilter/highdim/filtering.py` rather than
re-deriving code mechanically when the helper already matches the needed term.

## Step 4 — Expose explicit serious-route backend metadata

Once the analytic backend exists, update the active adapter in:

- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`

to emit and/or verify:

- `score_backend_id`
- `runtime_autodiff_for_hmc = False`
- serious-route diagnostics saying the score is the analytic derivative of the
  same finite program

Do not switch the registry route until the focused tests below pass.

## Step 5 — Add focused tests before any tuning rerun

Add or extend tests such as:

- `tests/test_zhao_cui_actual_sv_neutra_target.py`
- optionally `tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py`

Required checks:

1. finite value and score on 2–3 probe rows;
2. `status_code == 0` and required telemetry keys;
3. backend metadata indicates non-autodiff;
4. value matches the current active value authority;
5. analytic score matches centered finite differences of the **same active finite
   program** on probe points;
6. adjacent frozen-core sensitivity is preserved in the value path.

Also rerun:

- `tests/test_neutra_shared_procedure.py`
- `tests/test_neutra_all_models_end_to_end_contract.py`

once the active adapter is switched.

## Step 6 — Only then rerun SVX-ZC serious tuning

After the derivation note, MathDevMCP audit, implementation, and focused tests
all pass:

- rerun only `SVX-ZC`
- on the 4080-only mask
- under the repaired common tuning route
- with a fresh output root

Do not resume the broader sweep until this isolated rerun produces a valid
viable-set artifact or a new mathematically interpretable failure.

## Skeptical audit of this plan

- **Wrong baseline risk:** avoided. We are not swapping in a different analytic
  lane simply because it exists; diagnostics showed the candidate analytic route
  is a different finite program.
- **Proxy promotion risk:** avoided. OOM and route mismatch are treated as
  engineering/mathematical blockers, not as evidence against SVX itself.
- **Hidden target change risk:** addressed by requiring the derivation note and
  active-value equality checks before the registry swap.
- **Missing stop condition risk:** implementation stops at the first failed
  derivation audit, focused test, or active-value mismatch; no serious tuning
  rerun happens before those gates pass.
- **External-tool omission risk:** addressed explicitly by the MathDevMCP CLI
  audit step before coding.

Audit verdict: `PASS_FOR_TARGETED_SVX_ADJACENT_DERIVATIVE_IMPLEMENTATION`

## Verification

1. Write the derivation note.
2. Run MathDevMCP CLI audit on the derivation note and code mapping.
3. Implement the analytic adjacent-state derivative.
4. Run focused SVX tests until they pass.
5. Re-run shared-procedure / end-to-end contracts.
6. Run a tiny adapter smoke batch.
7. If and only if those pass, rerun `SVX-ZC` serious common tuning.
