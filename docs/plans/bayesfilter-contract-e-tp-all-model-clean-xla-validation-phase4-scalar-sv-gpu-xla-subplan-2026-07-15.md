# Phase 4: Scalar-SV Trusted Full-Horizon GPU/XLA Subplan

Date: 2026-07-15

Status: `REVIEWED_ACTIVE_EXECUTION`

Review record: the requested Claude one-path review was blocked by the platform
external-data policy after a successful tiny health probe. A fresh independent
Codex reviewer returned `REVISE` on CPU/GPU agreement, exact commands, GPU
memory policy, FD details, comparison outputs, manifest fields, executable
budget enforcement, provenance, and invalid-control classification. The same
plan is patched below; Codex remains supervisor/executor. The focused re-review
found no material issue and returned `VERDICT: AGREE`; the execution boundary
is therefore open under the reviewed campaign budget.

## Objective And Research Question

Determine independently for Actual SV and KSC-SV whether the Phase 3
loop-native center-scoped finite program can be prepared at `T=1000`, retain
the same scalar and total score, and execute through the exact trusted
GPU/XLA factory with bounded graph topology and fail-closed behavior.

This phase is center-scoped. The registry supplies no reviewed nonlinear
parameter region. Full-box and HMC readiness remain deferred.

## Entry Conditions

- Phase 3 Actual/KSC controlling results are
  `PASS_SCALAR_SV_LOOP_NATIVE_CPU_XLA_PREFIX` through `T=100`.
- Actual SV uses the exact transformed log-square/log-chi-square target,
  order 25, continuation order 129/radius 10, and lookahead 16 at `T=1000`.
- KSC-SV uses the offset `1e-8` KSC mixture target, order 41, continuation
  order 129/radius 10, and lookahead 8 at `T=1000`.
- Both use `transition_before_first_observation=false`, float64, fixed-square
  offline charts, and the loop factory with XLA enabled.
- Generalized SV remains a scientific negative result and is excluded.
- Remaining budget is `95.62 CPU core-hours`, `31.99 trusted GPU-hours`, and
  three full-horizon attempts per eligible row.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Actual order 25/lookahead 16 | prior Contract E--TP gradient-comparison Phase 5 `T=100` target-specific repair hypothesis; retained by current Phase 3; not a default | long-horizon chart invalidity or approximation drift | offline chart summary, CPU `T=1000` value/score/FD |
| KSC order 41/lookahead 8 | prior Contract E--TP gradient-comparison Phase 5 adjacent-refinement hypothesis; retained by current Phase 3; not a default | fresh chart conditioning or long-horizon invalidity | same |
| continuation order 129/radius 10 | inherited prefix numerical baseline; convenience hypothesis | tail truncation or quadrature error | descriptive prior dense-reference refinements; no equivalence claim |
| float64 | numerical reference arm | slower GPU and no TF32 production claim | device/timing/memory manifest |
| center theta only | owner-approved center-scoped path; region design absent | cannot support HMC/box claims | explicit region nonclaim and finite invalid control |
| FD step `1e-5` and `0.05*sqrt(2)` | inherited float64 scalar diagnostic and owner FD-only policy | step cancellation or endpoint invalidity | record endpoints and step-sensitivity on failure |
| invalid theta `[4,center_log_beta]` | both exact Phase 3 `T=3,100` factories returned false and poisoned all claim-bearing outputs | chart may remain valid at `T=1000`, making the control nondiscriminating | classify as invalid-control-design repair, not center-algorithm failure; do not silently choose a new point |
| CPU/GPU comparison | descriptive componentwise comparison only; no justified cross-device equivalence margin exists | a large difference may reveal a device-specific defect but cannot be classified from magnitude alone | each device independently passes same-scalar FD/validity/fail-closed; report drift without ranking or equivalence |

No setting becomes a repository or cross-model default from this phase.

## Skeptical Pre-Execution Audit

Status: `PASS_REVIEWED_FOR_EXECUTION`.

- Same finite row scalar is the engineering baseline; older dense references
  remain descriptive scientific diagnostics.
- `T=1000` preparation occurs before GPU launch and may reject a row without
  consuming a full-horizon GPU attempt.
- CPU `T=1000` execution is a reference/preflight exception, not production
  evidence; GPU/XLA is the target.
- Short/full graph comparison uses `T=100/T=1000`, which shares the same
  target-specific order, lookahead, factory, and loop topology per row.
- Exact full-horizon score and FD endpoints execute through the same compiled
  factory/configuration as the GPU result.
- Center success cannot establish a parameter region, filtering accuracy,
  equivalence, canonical/default status, leaderboard status, or HMC validity.

## Required Implementation And Artifacts

1. Extend preparation/harness horizon validation to `T=1000` without adding
   Python loops to the compiled route. Offline Python chart selection remains a
   permitted preparation role.
2. Generate fresh immutable per-row `T=1000` preparations under
   `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/`.
3. Add one dedicated GPU harness that binds exactly one preparation and emits:
   target/preparation hashes, graph inventory, compile-first/warm timings,
   value, score, full increment history, final particles, final log weights,
   chart diagnostics, same-scalar FD, finite device placement, GPU
   identity/memory, XLA/TF32/dtype, deterministic comparison summaries and
   hashes, and a run manifest.
4. The exact same factory receives center theta, FD endpoints, and finite
   off-center theta `[4, center_log_beta]`; invalidity must return false and
   poison objective, score, every increment, particles, and log weights.
5. Keep Actual and KSC artifacts, logs, attempt counts, and classifications
   separate.

Fresh attempt roots must refuse overwrite. Failed attempts remain preserved.

The trusted harness must set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow
import, call
`bayesfilter.runtime.gpu_memory_policy.configure_tensorflow_gpu_memory_growth`
before logical-device initialization or tensor creation, fail closed on any
configuration/verification error, and record every physical GPU's verified
growth status. Memory growth is not a hard memory cap; the phase relies on the
wall-time/attempt budget and records allocator peak bytes.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | can each center-scoped `T=1000` finite program pass its own scalar/FD/validity/fail-closed gates on trusted GPU/XLA with bounded graph growth? |
| Baseline | exact same row-specific loop core/factory on intentional CPU-hidden reference plus Phase 3 `T=100` graph |
| Primary criterion | `T=1000` preparation valid; CPU and GPU each independently pass same-scalar FD, finite value/score/state, validity, graph, warm replay, and fail-closed gates; CPU/GPU numerical differences are descriptive only |
| Promotion veto | chart invalid at center/FD endpoint, partial/nonfinite score, CPU fallback, XLA failure, finite invalid state, wrong target/preparation, or graph ratio failure |
| Explanatory only | compile/warm time, memory, condition number, minimum weight, feature residual, prior dense-reference gap |
| Not concluded | nonzero-radius region, filtering accuracy/equivalence, superiority, float32/TF32 production readiness, canonical/default/HMC/leaderboard readiness |
| Artifacts | preparations, CPU preflights, trusted GPU results/logs, hashes, result record, next subplan |

## Quantitative Gates

- source guard approves the exact loop factory and rejects the historical root;
- `T=100` and `T=1000` contain functional loops;
- full/short top-node and function-node ratios are `<=1.10`;
- full/short GraphDef-byte ratio is `<=1.25`;
- center, both `1e-5` FD endpoints, and warm replay are valid and finite;
- maximum individual-coordinate same-scalar FD relative error is at most
  `0.05*sqrt(2)`. For coordinate `j`, FD is
  `[L(theta+1e-5 e_j)-L(theta-1e-5 e_j)]/(2e-5)` and relative error is
  `|score_j-FD_j|/max(|score_j|,|FD_j|,1e-12)`. Two coordinates require four
  endpoint calls through the exact factory. The `1e-12` floor is a numerical
  denominator guard only; there is no cross-method interpretation. If the
  primary step fails, fixed explanatory steps `3e-6` and `3e-5` may localize
  cancellation/truncation, but cannot overturn the primary failure;
- trusted GPU is visible, output placement records GPU execution, and no CPU
  fallback occurs;
- CPU/GPU objective, each increment, each score coordinate, final particles,
  and final log weights are compared componentwise with absolute differences,
  symmetric relative differences
  `|cpu-gpu|/max(|cpu|,|gpu|,1e-12)`, and array hashes. These are explanatory
  diagnostics only because no checked operation-count derivation covers the
  device-specific transcendental libraries and XLA reductions. No magnitude
  establishes equivalence or failure. A hard device-consistency veto fires
  only if one device fails its independent finite/validity/same-scalar FD gate,
  the row/target/preparation identities differ, or the GPU falls back to CPU;
- finite off-center control returns `valid=false` and all claim-bearing
  numerical outputs, including every increment, nonfinite through the exact
  compiled factory. If the
  Phase 3-proven point is valid at `T=1000`, classify
  `INVALID_CONTROL_DESIGN_REPAIR`; do not reject the center algorithm or choose
  a replacement point without a visible plan amendment.

## Execution Ladder

For each row independently:

1. Generate and hash `T=1000` preparation.
2. Run CPU-hidden center and FD preflight plus `T=100/T=1000` graph inventory.
3. If preflight passes, run an escalated trusted-GPU device/memory-policy probe.
4. Launch one fresh-process trusted GPU/XLA `T=1000` attempt.
5. On localized compiler/harness/resource failure, repair and retry within the
   same target/configuration and remaining budget. A retry uses a fresh root.
6. Stop the row after three full-horizon GPU attempts or a scientific/chart
   veto; continue the other row unless a shared-core defect is proved.

All CPU commands set `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. All GPU
device and XLA commands use trusted/escalated permissions and record the owner-
designated managed-session trust basis. The exact commands and paths are frozen
in the following section before implementation/execution.

## Exact Commands And Timeouts

Preparation CLI implementation first adds `1000` to the explicit horizon
choices. The CPU/GPU harness is added at
`docs/benchmarks/run_contract_e_tp_clean_xla_phase4_scalar_sv.py`; the structured
probe is added at `docs/benchmarks/probe_contract_e_tp_phase4_gpu.py`. Both must
pass focused unit tests before these commands run.

```bash
mkdir -p docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 3600 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_actual_nongaussian_T1000 --time-steps 1000 --teacher-order 25 --continuation-order 129 --continuation-radius 10 --lookahead-steps 16 --chart-mode fixed_square --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/actual_t1000_preparation.json
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 3600 python docs/benchmarks/prepare_contract_e_tp_scalar_sv_charts.py --row-id zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 --time-steps 1000 --teacher-order 41 --continuation-order 129 --continuation-radius 10 --lookahead-steps 8 --chart-mode fixed_square --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/ksc_t1000_preparation.json
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-cpu-20260715
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 7200 python docs/benchmarks/run_contract_e_tp_clean_xla_phase4_scalar_sv.py --device cpu --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/actual_t1000_preparation.json --phase3-short-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/actual_result.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-cpu-20260715/actual_result.json
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 timeout 7200 python docs/benchmarks/run_contract_e_tp_clean_xla_phase4_scalar_sv.py --device cpu --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/ksc_t1000_preparation.json --phase3-short-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/ksc_result.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-cpu-20260715/ksc_result.json
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-probe-20260715
TF_FORCE_GPU_ALLOW_GROWTH=true timeout 120 python docs/benchmarks/probe_contract_e_tp_phase4_gpu.py --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-probe-20260715/probe.json
mkdir docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-20260715
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mplconfig timeout 10800 python docs/benchmarks/run_contract_e_tp_clean_xla_phase4_scalar_sv.py --device gpu --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/actual_t1000_preparation.json --phase3-short-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/actual_result.json --cpu-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-cpu-20260715/actual_result.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-20260715/actual_result.json
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mplconfig timeout 10800 python docs/benchmarks/run_contract_e_tp_clean_xla_phase4_scalar_sv.py --device gpu --preparation docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/ksc_t1000_preparation.json --phase3-short-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/ksc_result.json --cpu-result docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-cpu-20260715/ksc_result.json --output docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-20260715/ksc_result.json
```

Each command redirects verbose output to a sibling `.log` in actual execution;
the structured manifest records the unredirected semantic command. Before any
retry, the result/ledger sums recorded CPU/GPU wall time and confirms the next
command timeout cannot exceed the remaining phase cap. `timeout` termination is
an infrastructure failure and consumes recorded elapsed budget and the GPU
full-horizon attempt if compilation/execution began.

## Required Checks

- preparation schema/hash/target identity and chart positivity/rank checks;
- `T=1000` loop/unrolled diagnostic only if it remains computationally
  tractable without graph unrolling; otherwise CPU loop-core/FD plus Phase 3
  exact `T<=100` parity is the declared baseline and the omission is explicit;
- exact CPU/GPU factory comparison;
- source and graph inventories;
- same-factory invalid control;
- scalar/adaptor/LGSSM/structural focused regressions;
- compileall, JSON/hash validation, and `git diff --check`.

The CPU and GPU artifacts must serialize or hash the complete increment
history, final particles, and final log weights and must emit componentwise
absolute/symmetric-relative differences and array hashes. Cross-device
differences have no equivalence pass/fail field. The invalid control checks
objective, score, every increment, final particles, and final log weights
individually.

Every serious-run manifest records git commit and dirty status, exact command,
Python/platform/conda environment, TensorFlow version, target data and
preparation hashes, seed or `N/A`, CPU/GPU visibility and output placement,
XLA/TF32/dtype, memory-policy verification for every physical GPU, allocator
current/peak bytes, wall time, attempt number, artifact paths, plan, and result.

## Budget And Repair Reserve

Phase cap: `24 CPU core-hours`, `16 trusted GPU-hours`, and at most three
full-horizon GPU attempts per row.

Minimum entry budget: `12 CPU core-hours`, `6.1 GPU-hours`, and one attempt per
row. Repair reserve: `12 CPU core-hours`, `6.1 GPU-hours`, and one additional
attempt per row. Available resources exceed minimum plus reserve
componentwise: CPU `95.62 >= 24`, GPU `31.99 >= 12.2`, attempts/row `3 >= 2`.

Initial per-command maximums are two one-hour preparations, two two-hour CPU
preflights, a two-minute GPU probe, and two three-hour GPU attempts. All CPU
commands bind BLAS/OpenMP and TensorFlow intra/inter-op thread counts to one;
accounting nevertheless uses a conservative two CPU core-hours per wall-hour
to include the main/inter-op overlap. Thus the initial CPU ladder fits within
12 CPU core-hours and the GPU ladder within 6.1 GPU-hours. The equal CPU/GPU
repair reserves remain available. No retry starts
unless recorded cumulative use plus its timeout remains within the phase and
campaign caps; otherwise the row is budget-blocked and the other legal row or
Phase 5 proceeds.

## Forbidden Actions And Claims

- Do not invent or infer a parameter box.
- Do not switch Actual/KSC target laws, transforms, orders, lookaheads, charts,
  or preparations after observing GPU results.
- Do not disable XLA, stop gradients, fall back to CPU, or use NumPy/SciPy in
  the gradient-bearing runtime.
- Do not relax topology or FD gates after seeing results.
- Do not call center-only success HMC, parameter-region, accuracy,
  equivalence, canonical/default, leaderboard, or production readiness.

## Exact Phase 5 Handoff

Before close, write the Phase 4 result, draft or refresh
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase5-predator-prey-loop-core-subplan-2026-07-15.md`,
and review it for real-plane support, initial-observation time order, target
continuation, RK4 functional loops, finite-program parity, fail-closed behavior,
artifacts, budget, and nonclaims. Phase 5 proceeds automatically only if its
`NEXT_PHASE_READINESS` table is entirely `PASS`.

## Stop Conditions

Stop a row on invalid probability law/target identity, no positive fixed chart
at center under the frozen preparation rule, infeasible fixed-shape state,
three exhausted full-horizon GPU attempts, or row budget exhaustion. Repair
localized shape, XLA, autodiff, memory, serialization, or harness failures.
Stop the program only for a shared invalid harness/evidence path, total budget
exhaustion, or another runbook human-required boundary; row-local failure does
not block the other row or later legal phases.
