# GPU Harness UUID and Provenance Repair

This repository-scoped plan is governed by the cross-repository plan at
`/home/chakwong/MacroFinance/docs/plans/gpu_harness_uuid_provenance_repair_plan_2026_08_18.md`.

Scope is limited to GPU launcher identity, live device selection, TensorFlow
memory-growth provenance, and bounded infrastructure probes. It does not
authorize filtering, likelihood, HMC, tuning, sampler, or scientific runs.

The active checkout is `/home/chakwong/BayesFilter` at the execution-time git
commit recorded in the result note. Existing user modifications are preserved.
The selector must emit a stable NVIDIA UUID, use the repository's current
trusted utilization/free-memory thresholds, and fail closed when no GPU is
eligible. TensorFlow probes must use `TF_FORCE_GPU_ALLOW_GROWTH=true`, verify
growth before logical-device/tensor initialization, and record the selected
UUID, NVIDIA row, TensorFlow device details, operation placement, allocator
bytes, and nonclaims.

BayesFilter usage audit: no active MacroFinance-local `filters.*` or
`inference.hmc*` imports are permitted; only `bayesfilter.runtime` policy,
selector/probe scripts, shell wrappers, and focused tests may be touched.

Review status: passed local skeptical review. Numeric CUDA ordinals are not
treated as physical identity, driver baseline memory is not treated as a busy
GPU by itself, and dated historical launchers remain out of scope.
