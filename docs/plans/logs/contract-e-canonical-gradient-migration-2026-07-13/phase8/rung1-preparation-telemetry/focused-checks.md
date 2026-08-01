# Rung 1 Preparation And Telemetry Focused Checks

Date: 2026-07-14

All runtime checks used `CUDA_VISIBLE_DEVICES=-1` and
`TF_ENABLE_ONEDNN_OPTS=0`. They are CPU reference/debug evidence, not GPU
evidence.

## Preparation And Canonical Tests

```text
pytest -q tests/highdim/test_contract_e_phase8_preparation_telemetry.py \
  tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py
19 passed, 2 warnings in 102.86s
```

The checks cover PHILOX repeatability and domain separation, residual-design
centering, explicit ridge/mask validation, exact telemetry identities, frozen
float64 value/score hex preservation, and the shared canonical graph tests.

## Source-Bound Certificates

- Float64 exact derivative: all hard checks passed; aggregate and per-batch
  manual JVP versus forward autodiff differ by zero ULP.
- Float64 CPU-XLA: all hard checks passed and the Phase 5 v2 objective, score
  hex values, and branch hash reproduce exactly.
- Float32 CPU-XLA: all hard checks passed; center output is bitwise repeatable,
  endpoints remain branch-identical, and one concrete value-and-score callable
  is used.

## Static Checks

Python compilation and scoped `git diff --check` passed. The preparation
builder contains no ridge generator or hidden reset/transport default, and the
canonical source has no historical raw-route import.

TensorFlow emitted plugin-registration noise despite explicit GPU hiding. No
GPU evidence is inferred from those messages.

## Scientific Boundary

These checks establish preparation identity and telemetry/scalar preservation
only. They do not establish target ridge, reset, Sinkhorn, chunk, residual, or
Kalman-gradient adequacy; formal FD, GPU, HMC, leaderboard, and release gates
remain open.
