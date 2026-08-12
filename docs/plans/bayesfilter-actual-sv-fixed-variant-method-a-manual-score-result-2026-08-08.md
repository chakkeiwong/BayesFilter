# Experiment result: actual-SV fixed-variant Method A manual score backend

## Plan reference
- `docs/plans/bayesfilter-actual-sv-fixed-variant-method-a-manual-score-plan-2026-08-08.md`

## Command actually run
```bash
python -m py_compile \
  bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py \
  bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py \
  tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py \
  tests/test_zhao_cui_actual_sv_neutra_target.py \
  scripts/run_fixed_variant_value_score_multimodel_20260804.py

CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_zhao_cui_actual_sv_neutra_target.py

CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py
```

## Result summary
- The fixed-variant actual-SV campaign row is now rewired to the true frozen-core route again.
- The route under `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py` and `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` is confirmed to be the governing scalar for the current campaign.
- The current score backend for that scalar remains autodiff-backed.
- No existing manual/analytical route in the repo was found that computes the same frozen-core scalar.
- Execution therefore reached a real blocker: a same-scalar manual score backend does not already exist and must be derived/implemented as new code.

## Diagnostics
| Metric | Value | Interpretation |
|---|---:|---|
| Actual-SV fixed-core route tests | pass | The restored campaign path is internally consistent and finite. |
| Multimodel summary rerun | pass | The campaign row is back on the correct fixed-core scalar. |
| Existing same-scalar manual backend | 0 | No drop-in manual/analytical score route exists for this scalar. |
| Current actual-SV Method A status | blocked | The row remains autodiff-backed under the true fixed-core route. |

## Engineering observations
- The earlier analytic actual-SV helper route was the wrong scalar family for the fixed-variant campaign and has been removed from the campaign path.
- The true route is the comparator-generated frozen-core batch TT program in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
- Existing manual score patterns in the repo are structurally useful, but they target different scalar families:
  - source-order/frozen-proposal APF,
  - Austria SIR fixed-variant source-order route,
  - Contract-E / TP actual-SV overcomplete LEDH route,
  - SRUKF / SGQF surrogate routes.
- None of those is a same-scalar manual implementation of the current frozen-core actual-SV batch TT route.

## Empirical evidence
- `tests/test_zhao_cui_actual_sv_neutra_target.py` passes on the restored fixed-core route.
- The CPU-only multimodel summary rerun reports actual SV as the correct fixed-core row and blocks it under Method A because the derivative backend is still `autodiff_same_scalar`.

## Mathematical claims
- Checked and supported: the fixed-variant actual-SV campaign scalar is the comparator-issued frozen-core batch TT program, so any valid Method A backend must differentiate that exact scalar with frozen cores held constant after the one-time fit at `theta_0`.
- Checked and supported: helper routes that refit cores or switch to a different transformed-SV surrogate are wrong relative to the current campaign target.
- Not yet established: a manual derivative formula/backend for the exact frozen-core batch TT scalar.

## Decision
- Do not promote actual SV to Method A yet.
- Keep actual SV blocked under Method A until a new same-scalar manual score backend is derived and implemented for `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.

## Next step
- Derive and implement a manual/analytical score backend for the exact frozen-core actual-SV batch TT scalar in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`, then validate it against centered finite differences on that same scalar before changing campaign metadata.
