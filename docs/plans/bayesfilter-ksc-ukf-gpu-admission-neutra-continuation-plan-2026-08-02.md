# KSC-UKF GPU Admission and NeuTra Continuation Plan

Date: 2026-08-02  
Status: `REVIEWED_READY_TO_EXECUTE`

## Objective

Advance `KSC-UKF` only if the repaired mass-preserving clustered Gaussian-sum
route passes a trusted GPU/XLA cap-32 parity canary. If it passes, perform
scope-specific batched NeuTra training, statistical HMC tuning, and shared
sequential NeuTra HMC. If it fails, preserve the canary as terminal evidence
and do not train or sample.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Does the CPU-admitted KSC Gaussian-sum UKF route retain its value, score, and status behavior under the repository GPU/XLA execution policy, and if so can it support a KSC-specific NeuTra/HMC run? |
| Candidate | `bounded_deterministic_mass_preserving_clustered_gaussian_sum_ukf`, component cap 32 for the canary; later controls are tuned within the KSC scope. |
| Baseline | The same repaired route on `/CPU:0`, using the frozen KSC dataset, transformed observations, seven-component KSC mixture, and audit points. No SVX-ZC setting is a KSC default. |
| Primary promotion criterion | GPU/XLA cap-32 value gap to CPU `<=1e-10`, score gap `<=1e-8`, finite status, and the CPU dense-reference admission row remains passing. |
| Promotion veto | Missing/failed GPU, XLA disabled, memory-growth failure, nonfinite values/scores/status, CPU/GPU parity over threshold, frozen-data hash drift, or artifact collision. |
| Continuation veto | Target/data corruption, unavailable trusted GPU, harness invalidity, or campaign budget exhaustion. A numerical canary failure blocks KSC promotion but does not invalidate the CPU diagnostic. |
| Repair trigger | Localized XLA/static-shape, device, serialization, or harness failure with target, thresholds, and budget unchanged. |
| Explanatory diagnostics | Runtime, component counts, retained mass, dense-reference residuals, acceptance, energy error, and score differences after the canary. They do not independently establish posterior correctness or superiority. |
| Nonclaims | No exact likelihood claim, source-faithful claim, statistical superiority, default readiness, posterior correctness, or HMC convergence before the corresponding later gates pass. |

## Evidence Contract

The canary answers only CPU/GPU/XLA implementation parity for the already
admitted approximate KSC target. It uses the existing runner
`docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py` with a
fresh output root:

```text
docs/plans/artifacts/bayesfilter-ksc-ukf-gpu-admission-neutra-20260802/
  canary-attempt01/
```

The runner preserves the frozen state and raw-observation hashes, CPU dense
orders `(401, 601)`, audit points, cap ladder, value/score thresholds,
permutation checks, and target transform. It must record the git commit,
TensorFlow/TFP versions, visible GPU, TF32/XLA setting, memory-growth
verification, command, wall time, and result hashes.

Training and HMC are conditional phases, not implied by a successful canary.
They require a fresh KSC scope identity and a separate plan/artifact ledger.

## Scope And Default Audit

| Choice | Provenance | Failure mode | Early diagnostic | Promotion status |
| --- | --- | --- | --- | --- |
| Cap 32 canary | CPU repair result and prior failed XLA attempt | A larger or smaller cap may have different behavior; cap 32 is not globally optimal | exact CPU/GPU value/score comparison | reviewed admission check |
| Float64 target tensors | Existing KSC repair and dense reference | GPU/XLA precision or unsupported op can alter results | finite/status and parity thresholds | binding canary choice |
| TF32 enabled | repository GPU policy | TF32 may affect low-order parity; this is measured, not assumed | recorded parity gaps | execution-policy default |
| GPU memory growth | repository owner directive | initialization after device creation can fail or reserve memory | fail-closed helper before logical device access | binding execution gate |
| No inherited NeuTra controls | per-scope LEDH/NeuTra tuning policy | transfer can bias training or HMC | fresh KSC scope artifact | binding policy |

## Execution Phases

1. Run focused KSC repair tests and inspect the prior CPU artifact.
2. Run the trusted GPU/XLA cap-32 canary in the fresh root above.
3. If the canary passes, write a KSC-specific tuning/training plan before any
   optimizer or HMC command. Train only with the shared batch-native GPU/XLA
   NeuTra route, then tune and run sequential HMC under its own gates.
4. If the canary fails, write a terminal KSC result/reset note and stop this
   continuation; do not use CPU evidence to authorize NeuTra or HMC.

## Budget And Stop Conditions

- Focused tests: routine local check.
- GPU/XLA canary: one fresh attempt, at most 30 minutes and one GPU process.
- No NeuTra/HMC budget is consumed or launched unless the canary passes and a
  reviewed KSC-specific downstream plan exists.
- Never overwrite prior KSC repair attempts.

## Skeptical Plan Audit

- **Baseline:** CPU and GPU evaluate the same repaired program and frozen data;
  no unrelated model is used as truth.
- **Proxy promotion:** parity is the only canary promotion criterion;
  residuals, runtime, and diagnostics cannot promote NeuTra/HMC.
- **Missing stops:** the fresh root, one-attempt budget, GPU/XLA and memory
  growth vetoes, and no-training-on-failure stop are explicit.
- **Unfair comparison:** audit points, event order, transform, mixture, and
  thresholds are inherited unchanged from the CPU admission artifact.
- **Hidden assumptions:** cap 32, float64, TF32, and memory growth are recorded
  hypotheses/policy choices with direct diagnostics.
- **Artifact validity:** the runner writes structured result, manifest, and
  hashes and refuses an existing output root.

Audit verdict: `PASS_FOR_ONE_BOUNDED_GPU_CANARY`. A passing canary does not
authorize downstream training/HMC without a separate target-specific plan.

## Planned Artifact Root

`docs/plans/artifacts/bayesfilter-ksc-ukf-gpu-admission-neutra-20260802/`

