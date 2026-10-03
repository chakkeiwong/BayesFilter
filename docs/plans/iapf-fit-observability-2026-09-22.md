# Preserve guide-fitting failure diagnostics in the actual consumer

Status: COMPLETE; skeptical review PASS. Routine Class-A observability repair
within the authorized CPU/GPU campaign. No numerical algorithm/default change.
Result: [330 checks, 24 CPU tests and exact four-case GPU replay](artifacts/iapf-fit-observability-20260922-01/result.md).
New fields agree with independent references within4.27e-14. No default changed.

Question: do the adaptive consumer's saved records expose target-squared
concentration and density-energy escape without a separate capture/reconstruction?
Append five named diagnostic columns to the existing18, preserving their order:
target_squared_effective_count, target_squared_max_weight,
initial_log_density_energy, log_density_energy, initial_optimization_loss.
For normalized log target ell, use pi=softmax(2ell), ESS=1/sum pi².
Density energy is log(mean p_i²), where p is the standardized Gaussian density
with the same fixed normalization omitted by the original objective. Compute
with log-sum-exp so disappearance is visible before/after numerical underflow.
These are explanatory fields, never acceptance or continuation gates.

Baseline: preserved phase7 actual adaptive consumer inputs, guide coefficients,
all original18 diagnostics, histories, stopping decisions, value and score.
Primary engineering criterion: all original fields replay at1e-9, with identical
statuses and discrete decisions. New fields on16 actual final fitting targets
must match independent R phase7 energy/loss and phase9 concentration results at
1e-8. Require correct column wiring through the recursive TensorArray and the
real execute_iapf endpoint. The new diagnostics must be invariant to multiplying
all targets by a constant. A known single-point-dominated target and a uniform
target must report the correct counts, including extreme log scales.

Run focused existing initialization, fit and multivariate-consumer tests on CPU
with GPU hidden, and a new mathematical-observability regression. Replay all
four preserved adaptive cases on the selected RTX4080SUPER through actual
FP64/XLA kernels, memory growth configured and verified before initialization.
Reuse the existing seed namespace/data; no new experimental observations. GPU
commands are escalated. This is an explicit FP64 reference exception, not TF32
clearance. The previous d5 rejection must remain a rejection, not be hidden.

Vetoes: changed accepted computation/status, missing/misordered diagnostic,
incorrect independent value, wrong device/memory policy, corrupted input or
budget exhaustion. Localized harness failures may be repaired and retried in
fresh directories. Prior filtering/TF32 vetoes remain. No filter ranking,
paper replication, original-author identity, model-score or HMC promotion.
Baseline/default audit: formulas follow the checked loss decomposition and
target-centered Hessian; thresholds test numerical equality, not guide quality.
The existing fit tolerance/objective/bounds/floor and acceptance rules stay fixed.

Skeptical audit PASS: new output fields can change compiler fusion, so replay
the actual consumer and all original diagnostics rather than merely unit-testing
the added formulas. Column order is append-only, self-described by the consumer.
All new values remain observability; no posthoc target/weight threshold is added.

Transfer exact phase10 budget:45.848 CPU /47.803 GPU hours remaining. Cap this
phase at one CPU hour and one GPU hour, four launches <=600s each including
repairs. Root `docs/plans/artifacts/iapf-fit-observability-20260922-01/`.
Harness `docs/benchmarks/diagnose_iapf_fit_observability.py --mode cpu_tests|gpu_replay --attempt <unique>`.
Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; preserve exact commands,
environment, sources/inputs, logs, seeds, wall time, verification and result.
Follow-on research: a fitting repair must address the objective and sampling
design as well as initialization, and be tested on fresh downstream likelihood
controls; these diagnostics cannot themselves promote such a repair.
