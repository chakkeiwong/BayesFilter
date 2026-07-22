# Phase 2 PP-UKF Canary Attempt 02 Repair Record

Classification: `HARNESS_CONTEXT_GRID_FAILURE`.

After the memory-growth repair, PP-UKF loaded its target, frozen chart, and
historical HNN on the trusted GPU. The generic canary then raised
`KeyError: step_sizes` before constructing or running a chain. Historical
predator-prey code hardcoded its reviewed grid rather than storing it in the
context; SIR/structural contexts store theirs.

This attempt produced no sampler, accuracy, or performance evidence. Repair:
centralize the already reviewed per-cell grids in one comparison helper and
test PP, SIR, and structural values explicitly. Retry in fresh `attempt-03`.
No scientific target, method, threshold, hardware, or budget changed.
