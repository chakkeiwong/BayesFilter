# SVX analytic-route diagnostic plan

Date: 2026-08-06
Status: `ACTIVE_DIAGNOSTIC`

## Question

Why did the attempted analytic replacement for `SVX-ZC` fail, and what exact
mathematical/engineering piece is missing to produce an admissible serious
NeuTra/HMC target for the current transformed-SV + UKF-frozen-initializer lane?

This is not yet an implementation plan for the final fix. It is a bounded
source-faithful diagnostic intended to decide which of the following is true:

1. an existing analytic route already computes the same value program and only
   needs correct wiring;
2. an existing analytic route computes a different target and is not a valid
   drop-in replacement;
3. the repo lacks the specific analytic adjacent-state derivative needed for the
   current SVX target, so that derivative must be implemented.

## Current known facts

- Active `SVX-ZC` target uses the UKF-frozen initializer and current transformed
  actual-SV target identity via
  `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`.
- Its score comes from `tf.GradientTape()` over the batched fixed TT program in
  `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
- The obvious nearby `zhao_cui_fixed_adjacent_tt_tf.py` score path is also
  autodiff-based, so it is not the admissible serious replacement.
- The transformed-SV fixed-branch analytic score path in
  `bayesfilter/highdim/sv_mixture_cut4.py` delegates to
  `scalar_nonlinear_fixed_design_tt_score_path(...)`, which uses explicit model
  parameter-score methods and looks admissible.
- The first attempted swap failed because that path expects a
  `FixedBranchFilterConfig`, not a `ScalarAdjacentTTConfig`, and because the
  quick adapter rewrite violated the repo's batch-method source audit.

## Hypotheses to test

### H1. Existing analytic transformed-SV score matches the active value program closely enough

Candidate:
- `exact_transformed_sv_independent_panel_zhaocui_tt_score(...)`

What would make H1 true:
- at small probe points, its **likelihood value** agrees closely with the active
  SVX value program when both are run on the same transformed observations;
- any remaining difference is just prior/Jacobian recomposition or batch/status
  wrapper details.

If H1 is true, the missing piece is likely **only** wiring and batch/status
adapter construction.

### H2. Existing analytic transformed-SV score is target-mismatched

What would make H2 true:
- at small probe points, the active SVX value and the candidate analytic value
  differ materially even before any serious HMC run;
- the difference traces to distinct finite programs (e.g. one-axis scalar
  fixed-design path versus the current adjacent-state frozen-core path).

If H2 is true, then simply swapping in the transformed-SV analytic TT route is
mathematically wrong for the current target.

### H3. The current active target depends essentially on the adjacent-state frozen-core program

What would support H3:
- value differences appear specifically after `t=1`, where the current route
  seeds the two-axis fit with UKF-frozen adjacent cores;
- the current one-axis initial core is identity-bound but not functionally used,
  while the adjacent cores materially affect the value path.

If H3 is true, then the missing analytic piece is probably an **analytic
adjacent-state derivative of the same frozen-core finite program**.

### H4. We can prove the UKF initializer only matters as identity metadata, not numerically

What would make H4 true:
- altering the frozen adjacent cores within admissible test controls does not
  materially change the current value program.

If H4 were true, an easier analytic substitution might be defensible. Right now
I expect H4 to be false, but it should be checked rather than assumed.

## Planned diagnostics

### Diagnostic 1 — active target vs transformed-SV analytic candidate on probe points

For 2–3 theta rows near the validation center:
- evaluate the current active SVX target value
- evaluate the candidate transformed-SV analytic route value using the same
  transformed observations
- compare only the value first, not the score

Artifact answer:
- do the values agree closely enough to treat the analytic candidate as the same
  target family for HMC purposes?

### Diagnostic 2 — isolate where the current value program diverges

Compare horizons/prefixes if needed:
- `T=1` / first observation only
- `T=2`
- full `T=10`

This tells us whether the mismatch begins exactly when the adjacent-state frozen
TT branch activates.

### Diagnostic 3 — verify current frozen-core dependence

Run tiny local probes to check whether current value outputs change when the
adjacent frozen cores change (or at minimum inspect the exact code path to
establish where they enter and whether they seed the ALS updates materially).

### Diagnostic 4 — determine the admissible backend requirement

Confirm that the serious replacement must satisfy:
- no `GradientTape` through the filtering recursion,
- no batch method delegation rejected by `neutra_batching.py`,
- batch-native `neutra_batch_log_prob_and_grad_status(...)`,
- required status telemetry keys.

## Success criterion for the diagnostic phase

At the end of this diagnostic phase we must be able to say exactly one of:

1. **Existing analytic route is valid** — proceed to implement a clean adapter.
2. **Existing analytic route is invalid target-wise** — implement a missing
   analytic adjacent-state derivative for the current frozen-core finite program.
3. **Evidence is still ambiguous** — add one more smallest discriminating value
   comparison before any serious implementation.

## Non-claims

- This phase does not claim HMC readiness for SVX-ZC.
- This phase does not produce a serious tuning artifact.
- This phase does not yet decide whether to change the target signature.
- This phase does not promote any convenient approximate analytic route unless
  value-path evidence supports it.
