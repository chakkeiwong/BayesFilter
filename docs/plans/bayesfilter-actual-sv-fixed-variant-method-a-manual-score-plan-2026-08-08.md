# Actual-SV fixed-variant Method A manual score backend

## Question
Can we replace the current autodiff-backed score in the fixed-variant actual-SV frozen-core TT route with a manual / analytical score backend for the exact same scalar, and thereby admit actual SV under Method A?

## Mechanism being tested
A new manual score backend in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` for the comparator-issued fixed-seed batch TT scalar. The implementation must keep the TT seed cores, quadrature, basis tensors, and scalar contract fixed and change only the score backend.

## Derivation lock and MathDevMCP audit gate
Before any further implementation, write a derivation note for the exact active scalar and audit it with MathDevMCP. This is a hard gate, not an optional follow-up, because earlier attempts failed by drifting between scalar families and implementing before proving the right target.

Required audit steps:
- write the scalar definition in repo notation, explicitly stating:
  - what tensors are frozen after the one-time comparator fit at `theta_0`,
  - what runtime quantities are replayed deterministically at each time step,
  - what depends on `theta`,
  - what the step increments are,
  - what is *not* allowed to vary;
- run MathDevMCP CLI on the derivation using at least:
  - `assumptions-for`,
  - `derive-from` and/or `debug-derivation`,
  - `audit-math-to-code`;
- record the results in a companion audit artifact under `docs/plans/`;
- do not implement the backend until the derivation and math-to-code audit both pass without a target mismatch.

This gate is specifically intended to prevent the earlier procedural mistakes:
- wiring a nearby but different scalar family,
- debugging score mismatches before proving the target contract,
- mixing frozen-seed, frozen-fit, and refit semantics,
- using finite differences to discover route mismatch instead of first proving same-scalar identity.

## Scope
- Variant: fixed-variant actual-SV frozen-seed T10 route (`SVX-ZC-T10-d10-r2-o25-center-frozen-ukf-v1`)
- Objective: same-scalar manual / analytical score
- Seed(s): comparator seed `81101` and the existing campaign probe point
- Training steps: none; the comparator seed cores are frozen after the one-time comparator fit
- HMC/MCMC settings: none in this phase
- XLA/JIT mode: CPU-only validation first; no GPU requirement for the targeted checks
- Expected runtime: focused test/runtime work only

## Success criteria
- A derivation note and MathDevMCP audit artifact exist for the exact active scalar and explicitly confirm same-scalar identity.
- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py` no longer uses `GradientTape` as the runtime score backend for the active scalar.
- The new score matches centered finite differences of the exact same active scalar at the campaign probe point and at least one nearby point.
- The adapter and runner reclassify actual SV as `manual` / `analytical` only after those tests pass.

## Diagnostics
Primary:
- finite value/score on the active actual-SV route
- same-scalar centered finite-difference residuals for the new manual score
- exact equality / near-equality of the manual score and autodiff oracle on the same scalar as a diagnostic check

Secondary:
- permutation / batch equivariance
- status telemetry keys and `status_code == 0`
- campaign runner row status and backend metadata

Sanity checks:
- frozen-seed identity unchanged
- no helper route from `sv_mixture_cut4.py` is used in the campaign path
- no refit-at-each-theta behavior introduced

## Expected failure modes
- The manual score accidentally differentiates a different scalar family.
- The implementation still hides autodiff behind a helper or refit call.
- The score matches the value path at one point but fails nearby, indicating an incomplete derivative.
- The runner/result metadata are updated before the score is proven correct.

## What would change our mind
- If a same-scalar manual score cannot be derived cleanly for the existing active scalar, we would stop and record actual SV as still blocked under Method A rather than shipping a mismatched route.
- If the MathDevMCP audit finds that the derivation silently changes the scalar family, we would revert the attempt before implementation.
- If the tests show the manual backend computes a different scalar than the active value route, we would revert it.

## Command
```bash
# 1. Write the derivation note for the exact active scalar.
# 2. Audit the derivation and the math-to-code mapping with MathDevMCP.
#    Minimum required commands:
#    assumptions-for
#    derive-from and/or debug-derivation
#    audit-math-to-code

python -m py_compile \
  bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py \
  tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py \
  tests/test_zhao_cui_actual_sv_neutra_target.py

CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py \
  tests/test_zhao_cui_actual_sv_neutra_target.py

CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py
```

## Verification
1. Write the derivation note for the exact active scalar.
2. Run the MathDevMCP audit and record the results in a companion artifact under `docs/plans/`.
3. Syntax check the touched Python files.
4. Run the actual-SV tests.
5. Add/run a frozen-seed score-vs-FD check at the campaign probe point.
6. Rerun:
   - `CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py`
7. Confirm actual SV is now manual/analytical only if the manual score matches finite differences on the active scalar.

## Interpretation rule
- If the derivation fails the MathDevMCP audit gate, stop implementation and repair the derivation before coding.
- If the new manual score matches FD on the active scalar, then actual SV can be promoted to Method A manual/analytical.
- If the new manual score fails same-scalar FD checks, then actual SV remains blocked under Method A and the result note must say so plainly.

## Skeptical audit
Before executing, verify that the new score backend computes the same frozen-seed scalar derivative as the existing fixed variant. If the implementation needs to fit per-theta cores again, or if it uses a different transformed-SV helper family, it is the wrong route and must be rejected. Also verify that the derivation has passed the MathDevMCP audit gate before treating any code result as evidence for promotion.
