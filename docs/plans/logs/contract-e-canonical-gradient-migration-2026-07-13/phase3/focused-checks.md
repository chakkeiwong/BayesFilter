# Phase 3 Focused Check Log

Date: 2026-07-13

Execution target: deliberate CPU-only float64 reference checks and a tiny
CPU-XLA compilation/execution smoke. `CUDA_VISIBLE_DEVICES=-1` was set before
TensorFlow import. This is not GPU evidence.

## Final Command

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_ledh_contract_e_cloud_reset_phase3.py \
  tests/test_contract_e_phase1_normative_math.py \
  tests/test_contract_e_cholesky_ridge_reset.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py \
  tests/highdim/test_ledh_contract_e_schema_v2_factory.py

134 passed, 2 warnings in 9.93s
wall time: 12.66 seconds
exit: 0
```

The warnings are the existing TensorFlow Probability `distutils.version`
deprecations.

## Other Checks

```text
python -m py_compile <Phase 3 source and test>
exit: 0

python -m json.tool <each of the three Phase 3 JSON artifacts>
exit: 0

git diff --check -- <Phase 3 paths>
exit: 0
```

## Coverage

- certificate file hashes, schema/status, power-of-two rational denominators,
  53-bit numerator bounds, and exact binary64 representation;
- identity/scalar-factor and exact noncommuting charts;
- nonzero transported-covariance JVP/VJP branch;
- all five input JVPs and VJPs against frozen rationals and TensorFlow autodiff;
- JVP/VJP duality and dense-reference composition;
- every shared dense/cloud forward intermediate;
- independent-versus-concatenated `B=2` execution;
- persisted general-chart diagnostics recomputed exactly;
- source prohibitions and XLA-on public wrapper defaults; and
- Phase 0 revocation plus Phase 1/2 compatibility.

## Interpretation

The bounded exact engineering certificates pass. The frozen general chart is
finite and its diagnostics are reproducible, but it remains inconclusive for
general numerical parity because no justified forward-error or scientific
adequacy requirement exists. Six Phase 3 promotion blockers remain unresolved.
