# Phase 6 Focused Check Log

Date: 2026-07-14

All TensorFlow tests and diagnostic records in this phase deliberately used
`CUDA_VISIBLE_DEVICES=-1`. They are CPU reference/debug evidence, not GPU
evidence. TensorFlow emitted CUDA plugin-registration and `cuInit` noise during
some imports despite the device-hiding variable; the commands completed on the
declared CPU-only path and those messages are not GPU-health evidence.

## Final Test Union

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q <Phase 6 focused union>
427 passed, 2 warnings in 403.26s
exit: 0
```

The union covered central v1 revocation, v2 factory forgery/inert registry,
explicit raw opt-in, all six row score contracts, LGSSM shard arithmetic,
historical forward artifacts, inclusive aggregation, the unified harness, and
legacy raw diagnostics. The warnings are the existing TensorFlow Probability
`distutils.version` deprecations.

## Inventory And Kernel Preservation

```text
CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/audit_contract_e_phase6_historical_routes.py \
  --output docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase6/post-edit-inventory.json \
  --expected-kernel-hashes docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase6/pre-edit-inventory-v2.json
exit: 0
```

Recorded results:

- `zero_unclassified_hits: true`;
- every executable-root symbol present; and
- `numerical_kernel_hashes_match_baseline: true`.

Post-edit inventory SHA-256:
`864645ee11d611881ab9cf8ef1fe97d47697c4b7f3a135ebc8f328b820806637`.

## Bitwise Raw-Diagnostic Preservation

```text
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
MPLCONFIGDIR=/tmp/matplotlib-contract-e-phase6 python \
  docs/benchmarks/emit_contract_e_phase6_raw_diagnostic_baseline.py \
  --output docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase6/post-edit-raw-diagnostic-baseline.json \
  --compare docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase6/pre-edit-raw-diagnostic-baseline.json
exit: 0
```

`bitwise_json_identity: true`; the six-route payload hash remains
`e97ef467de3932339ff837b565c029df8363a75195276027e0cc60d65b34a24f`.
The post-edit wrapper artifact SHA-256 is
`28b966eaa47e9ab2d7d2a15511a23b806a379740ba4d0babae58bf9e63549ff1`.

## Static Checks

Python compilation passed for edited benchmark/test modules. Scoped
`git diff --check` passed. Source discovery found no unclassified executable
root, and the production Contract E v2 factory remains empty.

## Interpretation

Raw-route reachability and classification are fail closed, and the protected
historical numerical functions produced unchanged checked outputs. This does
not establish correctness of the raw target, Contract E numerical adequacy,
Kalman equivalence, nonlinear validity, admission, HMC, leaderboard, default,
or release readiness.

