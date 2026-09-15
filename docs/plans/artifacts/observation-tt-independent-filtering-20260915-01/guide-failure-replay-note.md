# Exact guide failure replay

The running frozen A06 attempt records all SGQF levels invalid for d4 sequence
8. The wrapper catches the error and preserves it, as prescribed, but the shared
guide builder discards the successful prefix and failing time index on exception.
Replay that exact saved sequence through the unchanged builder with a diagnostic
wrapper around its update, retaining the prefix and failing input. No level,
covariance, data, fitting control or proposal is changed and no filter is rerun.

This is a CPU-only localization check, with CUDA intentionally hidden before
TensorFlow import. It tests whether the same failure reproduces and identifies
its time/input; it cannot rescue the failed candidate or select a replacement.
Budget: one 60-second timeout, charged to the existing five-hour active ledger.
Preserve JSON/log/source hash under this directory. A different result on CPU
would be labeled a backend discrepancy and would not invalidate the trusted GPU
failure. A later scientific repair would require a reviewed amendment and fresh
calibration/holdout data.
