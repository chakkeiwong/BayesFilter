# Gaussian analytical model-binding repair result

The Gaussian Kalman/UKF public filter and non-iAPF Gaussian oracle now bind
theta, model matrices and initial-law tangents inside one enclosing XLA owner.
`gaussian_execution_tf.make_gaussian_execution` calls the existing
`gaussian_tf.parameterized_model` and `make_gaussian_kernel` authorities; no
filter equation, analytical score formula, precision or validity threshold
changes. The owner has a stable theta/observation signature and defaults to
JIT. Adaptive iAPF and its oracle path remain unchanged and deferred.

Numerical workers04883--04888 qualify CPU/GPU, dimensions1/3, observation
dimensions1/2, Kalman/UKF and FP32/FP64. All five FP64 outputs match the archived
original exactly on both devices. The largest complete FP32 difference is
7.15256e-7 on CPU and zero on GPU, inside the declared1e-5 bound. Two five-point
step sizes give maximum analytical-score error1.19567e-11, inside the1e-7
bound. Changed theta/observations, exact replay and cache identity pass.

Both actual public endpoints pass with live dataset seeds, changed theta,
identical complete non-runtime records and replay. Original/candidate invalid
inputs raise the same existing exception type; nonfinite scores are not
accepted. Runtime timing/device fields are explicitly outside record equality.
Default XLA, one trace, enclosing model transforms, no pfor/host callbacks and
actual GPU output placement pass. The explicit non-XLA CPU reference also
passes; it does not redefine the default. GPU workers use non-display GPU2
UUID `GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, TF32 enabled and verified memory
growth. CPU workers intentionally hide GPUs and are reference checks.

Fresh-process cost workers04889--04892 measure the complete ordinary UKF
endpoint, including unchanged data preparation, oracle and reporting. Each
arm has30 synchronized warm calls and exact replay. Both GPU arms pass the
same-device uncontended preflight. Inputs, source, environment, seed and CPU
affinity match within each pair.

The campaign worker configures TensorFlow before these checks; both adapters'
runtime-configuration callbacks are replaced by the same recorded preconfigured
runtime descriptor. Thus cold time includes endpoint compilation/execution but
excludes TensorFlow import and `configure_runtime` initialization overhead.
This boundary also applies to the public numerical tests; numerical calls,
data generation, host validity checks and formatting execute normally.

| Backend | Before warm ms | Repaired warm ms | Cold ratio | Extra sampled warm RSS MiB | Allocator peak before/after bytes |
|---|---:|---:|---:|---:|---:|
| CPU |19.431307|3.405178|0.994441|6.031250|N/A|
| GPU |37.882603|9.014423|0.908296|-17.578125|33792 /26368|

These are one fresh process per arm and are descriptive. They do not establish
statistical performance ranking, native executable eviction, exact continuous
peak memory or terminal cost acceptance. The screen shows no new64MiB host
increase or cold-time investigation trigger in this scope. Earlier enclosing
direction/fitted-APF compiler residency and the streaming CPU slowdown remain
separate open findings; this table does not resolve them.

Initial readback04893 passes161 checks. Seven affected sibling callers
(six analytical-direction cases and the fixed fitted-APF Gaussian endpoint)
also pass CPU04894 and GPU04895. Final source-bound readback04896 passes161
combined evidence/policy checks and verifies the repository worker's recorded
trusted placement and physical-device memory-growth receipts.
It rechecks full outputs and derivatives, real endpoint fields, HLO/result/run
hashes, source identities and cost comparability. The guard now covers314
sources with1457 existing exact metadata/schema/reporting/reference exceptions;
no allowance was added. All new modules/tests pass Ruff; whitespace checks pass.
The14-worker unit uses200.548542 CPU /247.859791 GPU process-seconds without a
failed worker. Remaining global budget is24.507187 CPU /24.701521 GPU hours.

Skeptical result review: the original compiled filter is preserved and only its
previously eager input construction is enclosed. Agreement with that archived
route and independent finite differences supports this binding repair. It does
not close all callers: shared random-cloud preparation, optional KDM pilot
arithmetic and other endpoint numerical boundaries remain open. The seven directly affected sibling callers now have renewed numerical
checks; other caller obligations and affected saved costs still require their
final source applicability review. No independent reviewer, adaptive iAPF compliance,
canonical LEDH admission, full repository completion or main merge is claimed.

Next repair, specified in `filter_gradient_score_input_execution_20260929.md`:
migrate shared non-iAPF input/RNG and optional KDM-pilot numerical
preparation using the existing compatible Philox authority and actual live-seed
consumer tests. Preserve the nonlinear FP64 CPU physical dataset and all
streams. Keep the separate4539 fitted-record leaves, angle/status reporting,
historical DZ5 local arithmetic investigation and terminal audit visible.
