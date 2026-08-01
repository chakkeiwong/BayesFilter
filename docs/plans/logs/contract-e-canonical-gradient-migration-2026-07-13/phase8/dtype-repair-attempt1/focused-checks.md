# Rung 0A Dtype Repair Focused Checks

Date: 2026-07-14

All commands used `CUDA_VISIBLE_DEVICES=-1` and `TF_ENABLE_ONEDNN_OPTS=0`.
These are CPU reference/debug checks, not GPU evidence.

## Canonical Full Filter

```text
pytest -q tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py
11 passed, 2 warnings in 105.48s
```

This includes exact float64 preservation and float32 per-batch/aggregate manual
JVP versus forward-autodiff equality.

## Shared Dependencies

```text
pytest -q tests/highdim/test_ledh_contract_e_streaming_phase4.py \
  tests/highdim/test_ledh_contract_e_cloud_reset_phase3.py \
  tests/test_ledh_compact_transport_jvp.py
30 passed, 2 warnings in 45.98s
```

## Source-Bound Certificates

- fresh float64 exact derivative: every hard check passed and maximum ULP is
  zero;
- fresh float64 CPU-XLA: every hard check passed and the Phase 5 v2 objective,
  score hex values, and branch hash reproduce exactly;
- float32 CPU-XLA: finite, center-bitwise repeatable, endpoint branch-identical,
  and one concrete value-and-score callable;
- every checked three-step float32 FD relative screen passes, but the formal
  Phase 1 seven-step plateau/error-bound subgate remains inconclusive.

## Historical Preservation

- six-route payload bitwise identity: true;
- payload SHA-256:
  `e97ef467de3932339ff837b565c029df8363a75195276027e0cc60d65b34a24f`;
- route inventory: zero unclassified hits, all root symbols present, numerical
  kernel AST hashes match the Phase 6 baseline.

## Static Checks

Python compilation and scoped `git diff --check` passed. The static shared-core
audit found no dtype-conditioned canonical algorithm.

TensorFlow emitted CUDA plugin-registration and `cuInit` noise despite explicit
GPU hiding. No GPU evidence is inferred from those messages.
