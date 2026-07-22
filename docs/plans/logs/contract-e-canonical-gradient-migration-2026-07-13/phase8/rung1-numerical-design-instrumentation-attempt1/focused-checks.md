# Rung 1 Numerical-Design Instrumentation Focused Checks

Date: 2026-07-14

All runtime checks used `CUDA_VISIBLE_DEVICES=-1`,
`TF_ENABLE_ONEDNN_OPTS=0`, and `MPLCONFIGDIR=/tmp`. No target-prefix value or
score command ran in this slice.

```text
pytest -q tests/highdim/test_contract_e_phase8_numerical_design_instrumentation.py \
  tests/highdim/test_contract_e_phase8_preparation_telemetry.py \
  tests/highdim/test_ledh_contract_e_streaming_phase4.py
19 passed, 2 warnings in 53.82s
```

The final bundle covers dense tiny row/column marginal parity, signed residual
definitions, exact gap/eigenvalue/trace diagnostics, frozen scalar/score
preservation, and the production O(N) source/AST allocation audit.

Source-bound certificates then passed:

- float64 exact derivative: zero ULP aggregate and per-batch manual JVP versus
  forward autodiff;
- float64 CPU-XLA: frozen Phase 5 v2 objective/score hex, branch hash, center
  replay, endpoint charts, and one-callable checks reproduce exactly.

Python compilation and scoped `git diff --check` passed. The initial three
localized test/reporting defects were repaired without changing the canonical
coupling, quotient, reset, derivative, or target configuration; the final run
was rerun unchanged.

TensorFlow CUDA plugin and `cuInit` messages are expected under intentional CPU
hiding and are not GPU evidence.
