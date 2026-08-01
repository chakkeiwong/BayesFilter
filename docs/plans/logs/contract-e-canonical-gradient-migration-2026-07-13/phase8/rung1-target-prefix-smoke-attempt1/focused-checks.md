# Rung 1 Target-Prefix Smoke Focused Checks

Date: 2026-07-14

All runtime commands used `CUDA_VISIBLE_DEVICES=-1`,
`TF_ENABLE_ONEDNN_OPTS=0`, and `MPLCONFIGDIR=/tmp` before TensorFlow import.

```text
pytest -q tests/highdim/test_contract_e_phase8_target_prefix_smoke.py
6 passed, 2 warnings in 5.51s
```

Python compilation, scoped `git diff --check`, raw-route source search,
campaign-clock check, and no-overwrite precondition passed.

The one authorized command completed in `15.169998470999417` seconds with
status `TARGET_PREFIX_WIRING_SMOKE_PASSED_DESCRIPTIVE_ONLY`. All 18 serialized
hard-check fields are true. XLA compiled for Host, the only logical device was
`/device:CPU:0`, and no retry ran.

TensorFlow emitted CUDA plugin-registration and `cuInit` noise despite explicit
GPU hiding. This is CPU-hidden reference/debug evidence, not GPU evidence.
