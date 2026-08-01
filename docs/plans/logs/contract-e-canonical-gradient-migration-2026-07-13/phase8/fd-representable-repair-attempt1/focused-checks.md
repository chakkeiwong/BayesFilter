# Representable-Step FD Focused Checks

Date: 2026-07-14

```text
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp \
pytest -q tests/highdim/test_contract_e_phase8_fd_reclassification.py
6 passed, 2 warnings in 3.17s
```

Compilation, scoped diff validation, exact prior-result/fixture/source/prepared
identity, 35-pair nominal-step preflight, and no-overwrite checks passed. The
single CPU-hidden/XLA attempt completed all 35 endpoint pairs and returned
`SEVEN_STEP_FD_HEURISTIC_SCREEN_PASSED`.

All endpoint calls used the same value-and-score concrete function. No target-
prefix, Kalman, GPU, HMC, or primary-shape call ran.
