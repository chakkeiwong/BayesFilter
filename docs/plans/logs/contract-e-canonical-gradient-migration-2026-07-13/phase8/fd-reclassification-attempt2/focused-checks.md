# FD Reclassification Focused Checks

Date: 2026-07-14

The final preflight used deliberate CPU hiding:

```text
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp \
pytest -q tests/highdim/test_contract_e_phase8_fd_reclassification.py
5 passed, 2 warnings in 2.94s
```

Python compilation, scoped `git diff --check`, exact fixture/certificate hashes,
and fresh attempt-2 no-overwrite checks passed.

The retry ran with XLA on Host and only `/device:CPU:0` visible. It emitted all
35 records in `33.982998236999265` seconds. Center, prepared-input, source,
branch, chart, and one-callable identity checks passed. Exact representable
step symmetry passed for 13 pairs and failed for 22, producing the predeclared
inconclusive result.
