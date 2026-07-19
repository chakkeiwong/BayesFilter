# Phase A3 Result: Forecast Oracle And Predictive Statistics

Date: 2026-07-13 (Asia/Shanghai)

Status: `PASSED_FOR_A4_DESIGN_ONLY`

## Outcome

The scalar-LGSSM forecast oracle and predictive-statistics machinery passed the
bounded A3 engineering contract on a CPU-hidden reference path and the trusted
GPU/XLA path. The focused suite passed `65/65`; both structured runtime
artifacts passed all 21 contract checks and independent numerical replay.

This is an engineering result, not a predictive-equivalence result. The actual
independent identical-law fixture returned `INCONCLUSIVE_UNDERPOWERED`: its MMD
branch passed, but its co-primary feature intervals were too wide to fit inside
the provisional test-only margins. Of the four valid controlled alternatives,
only the mean perturbation emitted `MATERIAL_DIFFERENCE`; variance, skew, and
cross-horizon dependence remained `INCONCLUSIVE_UNDERPOWERED`. Those misses are
calibration repair signals for A4, not hard vetoes and not evidence against the
oracle, implementation, or predictive-validation research direction.

No ordinary HMC, NeuTra training, NeuTra-HMC, SSL-LSTM sampler comparison, or
A4 calibration run occurred.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the scalar-LGSSM oracle and predictive-statistics machinery for A4 design | `PASS`: `65/65` focused tests; CPU analytic/direct checks and independent replay passed; trusted GPU/XLA placement and persisted-input parity passed | No A3 engineering or continuation veto remains open; both artifacts passed 21/21 checks and fresh replay | A3 used one provisional fixture design; the identical-law co-primary interval was underpowered and three of four controlled alternatives were not detected | Design A4 calibration with multi-seed null coverage and power ladders; do not execute it under A3 authority | SSL-LSTM equivalence, calibrated margins or weights, posterior correctness, sampler validity, HMC/NeuTra readiness, model adequacy, superiority, or default readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the reviewed scalar-LGSSM formulas, direct replay, statistical hierarchy, fail-closed decisions, CPU artifact, and trusted GPU/XLA parity |
| Statistically supported ranking | None; A3 did not compare methods, and one-fixture alternative outcomes do not support a ranking |
| Descriptive-only differences | Residual magnitudes below thresholds, runtime, compiler/HLO details, high moments, quantiles, covariance diagnostics, quadratic MMD values, and controlled-alternative point outcomes |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | Calibration-only multi-seed coverage and power curves for margins, weights, bandwidths, alpha allocation, block/bootstrap settings, and sample sizes, followed by a separately authorized confirmatory design |

## Separate Evidence Ledgers

| Ledger | Status | Evidence and boundary |
| --- | --- | --- |
| Engineering correctness | `passed_a3_bounded_oracle_statistics_surface` | Focused tests, analytic scalar-LGSSM formulas, direct equation replay, summary/statistical contracts, materialized resampling indices, fail-closed decision branches, and two independent artifact receipts |
| Numerical validity | `passed_for_reviewed_fixtures` | Maximum analytic residual `4.440892098500626e-16` versus tolerance up to `1.3688827562622899e-13`; exact direct replay; CPU/GPU residual `2.886579864025407e-15` versus `8.510642180260583e-12` |
| Statistical validation machinery | `passed_fixture_screen_with_power_repairs_required` | Simultaneous mean diagnostic `2.602982573556344 < 3.023341439739154`; exact-binomial null-coverage lower bound `0.9280102451052912 >= 0.75`; correct fail-closed statuses; actual equivalence fixture remained underpowered |
| Sampler/posterior validity | `not_assessed` | No HMC, NeuTra, posterior-reference, convergence, or parameter comparison ran |
| SSL-LSTM predictive equivalence | `not_assessed` | No estimated SSL-LSTM arm was compared with a reference arm |
| Scientific/model adequacy | `not_assessed` | No empirical data, rolling-origin forecast, calibration audit, or scientific comparison ran |

## Research Intent Disposition

| Field | A3 result |
| --- | --- |
| Main question | Can the current TensorFlow/TFP machinery correctly and reproducibly compute joint 1-to-10-step forecast-law diagnostics on CPU and trusted GPU/XLA? |
| Baseline | Derived analytic scalar-LGSSM forecast law plus direct equation simulation from materialized innovations |
| Candidate mechanism | TensorFlow `float64` oracle, forecast summaries, standardized paths, descriptive quadratic MMD, cross-chain linear MMD, hierarchical intervals, and fail-closed classification |
| Promotion criterion | Focused tests, independently replayed CPU analytic/direct evidence, and trusted persisted-input GPU/XLA parity pass conjunctively |
| Promotion veto | Formula/direct mismatch, invalid covariance or hierarchy, nonfinite values, invalid evidence emitting `PASS`, artifact/replay failure, missing GPU/XLA placement, or CPU/GPU parity failure |
| Continuation veto | None fired; the target, analytic assumptions, artifacts, device route, and inferential hierarchy remained valid |
| Repair trigger | Harness/schema/provenance defects were repaired locally; limited null/alternative power is handed to A4 calibration |
| Explanatory only | Runtime, HLO identity, residuals below hard thresholds, quadratic MMD, high moments, quantiles, covariance entries, and one-fixture alternative detection |
| Must not be concluded | Predictive equivalence, calibration, posterior correctness, sampler readiness, superiority, model adequacy, or production/default readiness |

## Focused Checks And Numerical Evidence

| Check | Result |
| --- | --- |
| CPU-hidden focused suite | `65 passed, 955 warnings in 13.92s` |
| CPU artifact contract | `A3_CPU_REFERENCE_PASSED`; 21/21 checks |
| CPU independent replay | `A3_CPU_REFERENCE_VERIFIED`; all checks and numerical replay passed |
| Trusted GPU/XLA contract | `A3_GPU_XLA_PARITY_PASSED`; 21/21 checks |
| GPU independent replay | `A3_GPU_XLA_PARITY_VERIFIED`; all checks and numerical replay passed |
| CPU maximum analytic residual | `2.220446049250313e-16` |
| GPU maximum analytic residual | `4.440892098500626e-16` |
| Largest relevant analytic tolerance | `1.3688827562622899e-13` |
| Direct simulation replay | Maximum residual `0.0` on CPU and GPU |
| Monte Carlo mean screen | Maximum z-score `2.602982573556344`; simultaneous critical value `3.023341439739154` |
| Repeated null coverage | `63/64`; exact Clopper-Pearson lower bound `0.9280102451052912`; required lower bound `0.75` |
| CPU/GPU parity | Maximum residual `2.886579864025407e-15`; maximum scale-aware threshold `8.510642180260583e-12`; passed |

The coverage requirement is an A3 fixture screen, not a calibrated A4
confirmatory guarantee. The 64 replicates and one seed family do not establish
general coverage across SSL-LSTM posterior geometry, chain dependence, or
candidate calibration settings.

## Actual Decision And Alternative Diagnostics

| Fixture | Decision | Interpretation |
| --- | --- | --- |
| Independent identical-law pair | `INCONCLUSIVE_UNDERPOWERED` | MMD upper-bound branch passed; co-primary feature intervals did not fit inside provisional test margins; not equivalence evidence |
| Mean perturbation | `MATERIAL_DIFFERENCE` | Valid construction and inferential objects; descriptive evidence that this fixture detected its mean shift |
| Variance perturbation | `INCONCLUSIVE_UNDERPOWERED` | Valid positive log-variance change; A4 repair trigger |
| Skew perturbation | `INCONCLUSIVE_UNDERPOWERED` | Valid finite centered third-moment change; higher moments remain explanatory; A4 repair trigger |
| Dependence perturbation | `INCONCLUSIVE_UNDERPOWERED` | Valid covariance change with preserved marginal target; A4 repair trigger |

All four alternative constructions were finite and valid. A3 deliberately did
not turn provisional alternative power into an engineering veto. No stochastic
ranking is supported: observed feature maxima, MMD estimates, and interval
widths are descriptive single-fixture quantities.

## Runtime Artifacts And Integrity

| Artifact | Status | SHA-256 | Evidence signature |
| --- | --- | --- | --- |
| `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json` | `A3_CPU_REFERENCE_PASSED` | `f8252b9a0f6bba1bc5350b0516ceaddca04006bfe489acc74ac7f13d7846d82b` | `5b271aaa1395b3fde2bd2b79beb846f4abf73d40c280e13c85bc84512553325c` |
| `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference-verify.json` | `A3_CPU_REFERENCE_VERIFIED` | `b5ccfbb4fb285a95a66a6434c272113141457a9a342b34d30ae7468cbd81151a` | Binds the CPU signature |
| `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary.json` | `A3_GPU_XLA_PARITY_PASSED` | `5c31b26fbf20a10b754ad3e99bb8dc1481b12c74669c3b60e8e7cae8e080b693` | `5322ac24a3bc4618cb396be8fd13ce41c68f72e5acf7ba5835a64ad9e4ee2c30` |
| `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary-verify.json` | `A3_GPU_XLA_PARITY_VERIFIED` | `7e39a273ab4979d51bdfb4a2db876fcc0a96ac53f84ce9a3bba5946b6220eb6c` | Binds the GPU signature |

The CPU receipt binds the exact CPU artifact hash and independently replays the
formulas, materialized innovations and resampling indices, statistical
decisions, compiler evidence, source/configuration bindings, and schema. The
GPU receipt performs the corresponding replay and also binds the verified CPU
input. Both receipts report `independent_numerical_replay_passed=true`.

The two generation artifacts intentionally bind successive adapter versions.
The CPU artifact binds adapter v1 SHA-256
`047449702edad16a0db7316ac7daf2d8a1b8b587bd6fc7ea4a8d0f85c952ab28`.
After CPU verification, the first GPU attempt exposed the legacy three-row
loader; adapter v2 added only the exact four-row persisted-index loader and its
assignment and has SHA-256
`9d614b69b1535278994eb1027a3048824faa137e6fbfc60768cb6ce2ec17a36a`.
The GPU artifact binds v2. Removing those two recorded v2 hunks from the current
file reconstructs a compiling v1 byte stream with the CPU-bound hash exactly.
This version transition does not change CPU numerical generation, and the GPU
artifact consumes the independently verified CPU values. The exact transition
is preserved at
`docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/tier2-generation-adapter-v1-to-v2.patch`.

## Run Manifests

Both artifacts record Git commit
`3d353253dc93a102722e00cbca8803a1b3fce7fa` and `git_dirty=true`. The
environment was conda `tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, TensorFlow
Probability `0.25.0`, `float64`, XLA JIT enabled, TF32 enabled, and root seed
`[20260713, 1303]`. Materialized tensor contents, not root-seed regeneration,
are replay authority.

| Field | CPU reference | Trusted GPU/XLA |
| --- | --- | --- |
| Execution role | `cpu_hidden_xla_reference` | `trusted_gpu_xla_oracle` |
| Device inventory | One logical/physical CPU; `CUDA_VISIBLE_DEVICES=-1` | CPU plus two NVIDIA GeForce RTX 4080 SUPER logical/physical GPUs |
| Trust basis | `cpu_hidden_reference_exception_not_gpu_evidence` | `owner_designated_managed_session_visible_gpu_trusted` |
| Started (UTC) | `2026-07-13T15:28:31.717692+00:00` | `2026-07-13T15:47:30.528230+00:00` |
| Completed (UTC) | `2026-07-13T15:32:23.223370+00:00` | `2026-07-13T15:55:58.357912+00:00` |
| Wall time | `231.50568554899655` seconds | `507.8297090129927` seconds |
| Data version | A3 scalar-LGSSM fixture numeric configuration SHA-256 `5eb087faefaf40c1393dd844b0d037ac02c0ebee161fa3d949ce432ef7a016d2` | Same persisted CPU numeric fixture and tensor inputs |
| Plan/result | `docs/plans/bayesfilter-ssl-lstm-predictive-validation-live-plan-2026-07-13.md`; this result | Same |

Exact CPU invocation reconstructed from the artifact's command and recorded
environment:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  TMPDIR=/tmp/bayesfilter-a3-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a3-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --mode cpu-reference \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json
```

CPU independent-verification command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  TMPDIR=/tmp/bayesfilter-a3-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a3-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference-verify.json
```

Exact trusted GPU/XLA invocation reconstructed from the artifact's command and
recorded environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  TMPDIR=/tmp/bayesfilter-a3-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a3-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --mode gpu-xla \
  --cpu-reference docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-cpu-reference.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary.json
```

Trusted GPU/XLA independent-verification command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  TMPDIR=/tmp/bayesfilter-a3-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a3-tmp/cuda-cache \
  XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/verify_ssl_lstm_predictive_validation_a3_2026_07_13.py \
  --artifact docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary.json \
  --output docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3/oracle-gpu-xla-canary-verify.json
```

Focused-suite command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a3-pycache \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_scalar_lgssm_forecast_oracle.py \
  tests/test_predictive_equivalence.py
```

## Repair Record

| Event | Classification | Repair and evidence |
| --- | --- | --- |
| Provisional alternative power was required as an aggregate pass | Evidence-role defect in reused harness | Required alternative construction and inferential validity, while restoring one-fixture detection power to explanatory/repair-trigger status |
| TensorFlow Probability distribution metadata absent | Manifest compatibility defect | Recorded `tensorflow_probability.__version__` from the imported module |
| Verifier expected three resampling rows and replaced the persisted seed | Independent-verifier schema defect | Required chain, draw, forecast-replication, and seed rows; replayed the exact persisted seed |
| Historical artifact-role label leaked into seed attestation | Adapter compatibility defect | Validated the Tier 2 role directly and used a shallow historical compatibility view only inside the unchanged helper |
| Dependence reconstruction differed by `4.440892098500626e-16` | Floating evaluation-order difference | Applied the predeclared `8192 * eps * scale` tolerance to continuous replay while retaining exact identity for categorical and integer fields |
| Raw HLO differed in process-local TensorFlow function IDs | Compiler-evidence normalization defect | Retained raw HLO/hash self-consistency and normalized only process-local function IDs for fresh-versus-persisted comparison; all 284 normalized lines matched |
| Generator and verifier provenance were coupled | Provenance design defect | Generation binds production/tests/core/runner; receipts separately bind independent replay core/verifier |
| GPU loader repeated the three-row schema | GPU adapter schema defect | Consumed all four verified CPU rows, including persisted arm seeds, without regenerating random indices |

These were harness, adapter, verifier, or provenance defects. None invalidated
the analytic oracle, production statistics, target, data, or research
direction. The repair loop stopped after both final artifacts and receipts
passed; no failed intermediate artifact is used as evidence.

## Candidate Versus Research Direction

The A3 machinery candidate passes its bounded engineering screen. The
provisional A3 calibration candidate does not pass as a confirmatory design:
its identical-law interval was underpowered and variance, skew, and dependence
alternatives were missed. That rejects reuse of the A3 fixture constants as A4
defaults; it does not reject moment-based predictive validation, MMD, the
state-space LSTM, HMC, NeuTra, or the broader research direction.

A4 should test whether additional chains/draws/forecast replications, calibrated
block/bootstrap choices, scientifically grounded margins, and robust weighting
can separate null variation from material alternatives. If prospective
multi-seed calibration cannot achieve that separation within its resource
budget, the correct result is
`BLOCK_PREDICTIVE_EQUIVALENCE_NOT_IDENTIFIABLE_AT_AVAILABLE_BUDGET`, not wider
post-hoc margins.

## Post-Run Red Team

| Challenge | Assessment |
| --- | --- |
| Strongest alternative explanation for agreement | Analytic and TensorFlow paths may still share an indexing or convention error, and CPU/GPU parity preserves any common error. Direct materialized-innovation equation replay reduces but does not eliminate that shared-lineage risk. |
| Strongest explanation for alternative misses | The provisional `4`-draw block length, `128` bootstrap replicates, path count, alpha allocation, scales, and test margins were fixture values, not calibrated choices; wide intervals and a diffuse bandwidth mixture can mask valid variance/skew/dependence perturbations. |
| What would overturn the A3 result | A reproducible analytic/direct mismatch, invalid covariance or hierarchy, source-bound replay failure, fail-open classification, trusted GPU/XLA placement failure, or parity residual above the predeclared threshold on the bound fixture. |
| Weakest evidence | Alternative detection and null behavior come from a small, single fixture family. The `63/64` coverage result is a bounded regression screen, not broad statistical validation. |
| Permitted A4 use | A3 supplies validated interfaces, algebra, fixture-generation patterns, and fail-closed machinery. Its provisional margins, weights, bandwidths, block length, bootstrap count, alpha split, sample counts, and seeds are explicitly not calibration inputs to be frozen without fresh A4 evidence. |

## A4 Handoff

A4 design drafting is authorized because the focused tests, CPU artifact,
independent CPU replay, trusted GPU/XLA artifact, independent GPU replay,
hierarchy checks, fail-closed decision branches, and artifact integrity checks
passed with no continuation veto. The design must own and prospectively
calibrate all margins, scales, weights, MMD settings, alpha allocation,
resampling settings, sample sizes, and confirmation seeds.

This status authorizes A4 design and focused review only. A4 execution requires
its own Tier 2 pre-execution audit and evidence contract. It does not authorize
HMC, NeuTra, model-file changes, a predictive-equivalence decision, or a
scientific/default claim.

## Nonclaims

- Not SSL-LSTM predictive equivalence or calibration.
- Not posterior correctness, parameter agreement, or identification.
- Not HMC convergence, validity, readiness, or an ordinary-HMC baseline.
- Not NeuTra training, NeuTra-HMC validity, readiness, or superiority.
- Not empirical model adequacy, forecasting superiority, or scientific validity.
- Not a calibrated MGF, characteristic-function, GMM, inverse-variance, or
  kernel weighting rule.
- Not production, public API, package, release, or default readiness.
