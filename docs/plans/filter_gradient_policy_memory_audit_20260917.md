# Filtering, gradient policy and XLA memory audit

Owner request, 2026-09-17: enumerate repository filtering and gradient algorithms,
review their implementation for Python numerical loops, NumPy and XLA-default
violations, and compare memory with and without XLA. Baseline Git `3582b4ac`;
the working tree was clean. This is an audit and bounded diagnostic, not a
new filtering method, tuning campaign or scientific admission.

## Evidence contract

Inventory repository-owned Python implementations, public exports, model
adapters, direct consumers and local dependencies. Cover covariance/factor
Kalman, sigma-point and SGQF, particle/flow/OT/SQMC, fixed tensor-train routes,
hard-bound filters and independent score estimators. Include historical and
reference implementations with explicit roles. A directory name or absence
from the earlier Kalman guard does not establish exemption or compliance.

Static discovery records all loop syntax, NumPy/import/materialization boundaries,
autodiff/pfor use and compilation declarations. Review numerical versus schema
loops in context and follow actual consumer calls for material findings. Static
inspection cannot prove dynamic callbacks pure or an outer function compilable;
report such gaps. Do not treat a decorator as executed XLA evidence. Preserve
the earlier tests and HLO; do not rerun them without a new concern.

Memory diagnostics use the same deterministic fixture, shapes, outputs and
precision for each mode in separate fresh processes. Compare graph mode without
XLA and XLA, with an eager diagnostic where affordable. Verify that nested
functions do not leave the supposedly non-XLA arm partially compiled. Record
host current RSS/high-water RSS, TensorFlow GPU current/peak allocation, timing,
trace counts, output parity and reachable callback/compiler declarations at
post-import, inputs, trace, cold and warm stages. Saved outputs and histories
must have the same lifetime across modes. Compilation/cache memory and device
tensor allocation are distinct quantities; nvidia-smi reservation is not live
tensor memory. Repeated fixed-shape calls test growth after compilation.

Use GPU by default for memory comparisons, with verified memory growth on the
selected visible idle device before initialization; CPU is a labeled reference
if useful. Use existing `/home/ubuntu/miniforge3/envs/tf-gpu`, float64, TF32 status
recorded, two intra-op threads, one inter-op/OpenBLAS thread, fixed seeds. No
package/environment change. Unique artifacts live under
`docs/plans/artifacts/filter-gradient-policy-memory-20260917/`.

Pass/fail concerns: nonfinite or mismatched valid outputs, hidden compilation
in the reference arm, Python callbacks in measured graphs, repeated retracing,
or unbounded fixed-shape memory growth. Descriptive cold/warm memory differences
alone are not correctness failures. Flag >2x device peak or >256 MiB incremental
host growth for explanation, not automatic rejection. A larger fixture can
clarify allocator granularity, subject to the same budget. No speedup, leak-free
repository, posterior correctness, Zhao-Cui source-faithfulness or canonical
LEDH admission claim follows from this diagnostic.

## Budget and skeptical review

At most 20 GPU process-minutes and 30 CPU process-minutes for fresh memory
workers and targeted checks, one GPU worker at a time, each worker capped at
180 seconds, at most three attempts per unchanged fixture/mode. Stop on
uncontrolled memory, device contention, semantic mismatch or exhausted budget.
Harness or sandbox failures may be repaired and retried within the contract.

Pre-execution review: the main risk is a false non-XLA comparison caused by
inner default-XLA wrappers, followed by confusing allocator reservation with
live tensors or charging TensorFlow imports only to one arm. Fresh processes,
explicit inner flags, reachable graph inspection and phase-separated metrics
address those risks. Small fixtures may underexercise memory and remain limited
mechanics evidence. Audit conclusions will cite concrete code paths; candidate,
historical and reference roles cannot be inferred solely from filenames. No
new mathematical behavior or canonical route claim is proposed. The plan passes
for this bounded engineering question.

## Execution and recovery record

The realized comparison uses four deterministic FP64 fixtures: fixed full-rank
rectangular SRUKF and direct-factor SRUKF (B=8, P=3, N=2, T=96), covariance
Kalman with analytical score (B=8, P=32, N=24, M=12, T=96), and the batch
analytical finite-Sinkhorn JVP primitive (B=4, particles=128, state=4, P=4,
16 solver iterations, K=N). The first three cover repaired complete filtering
calculations; the fourth covers transport-gradient memory without claiming a
complete LEDH route. Sizes and iteration counts are engineering fixtures, not
tuned research defaults. Numerical parity and finite/graph checks are the
pass/fail criteria; timings and memory ratios are descriptive.

All eight corrected GPU workers completed, taking 82.48 seconds inside the
workers in total, and each completed 20 warm calls with one trace. Preliminary
initialization/measurement and static-discovery defects were repaired within
the same budget; their artifacts are explicitly superseded. No environment
installation or algorithmic runtime rewrite was performed.

Final findings, limitations and repair order are recorded in
`filter_gradient_policy_memory_result_20260917.md`; the algorithm catalogue is
`filter_gradient_algorithm_inventory_20260917.md`. Primary machine evidence is
`artifacts/filter-gradient-policy-memory-20260917/static-complete.json.gz` and
`artifacts/filter-gradient-policy-memory-20260917/memory-comparison.json`.
