# BayesFilter V1 Nonlinear Performance NP4 XLA Gate Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP2 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`
- NP3 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`
- NP4 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-jit-gate-plan-2026-05-15.md`

## Phase purpose

Convert NP4 from benchmark-only timing hints into a focused support-matrix artifact for CPU-only XLA value paths, with explicit non-claims for dynamic horizon, GPU XLA, and score cells not covered by a new focused certification test.

## Skeptical plan audit

Audit target: whether the planned NP4 commands and artifact would answer the actual support question without over-claiming from static-shape CPU-only evidence.

Findings:

- Wrong-baseline risk: older compiled-parity tests covered graph mode and a small subset of value cells, but did not separate `return_filtered=False` from `return_filtered=True` or certify all required backends. NP4 therefore needed a new focused value-path test artifact instead of reusing prior graph-only evidence.
- Graph-fallback risk: an NP4 test must call `tf.function(jit_compile=True)` directly. Otherwise graph success could be mistaken for XLA support.
- Return-contract risk: value support could be overstated if `return_filtered=True` were inferred from `return_filtered=False`. NP4 therefore tested those modes separately and required filtered-means and filtered-covariances parity when returned.
- Dynamic-shape overclaim risk: all current value tests use fixed observation horizons and existing implementations loop over horizon with Python `range`, so static-shape success cannot justify dynamic-horizon support. Dynamic horizon remains `not_claimed` for every row.
- GPU conflation risk: this worker was instructed to run CPU-only and not to probe trusted GPU devices, so GPU XLA cannot be inferred from CPU XLA success. Every GPU row remains `skipped_no_trusted_device`.
- Score overclaim risk: NP4 allows score XLA claims only with focused branch-certified value+score XLA parity. This worker did not add such score tests, so score rows remain `untested` rather than being upgraded from existing graph or finite-difference evidence.
- Environment-mismatch risk: because the local environment may expose a GPU-enabled TensorFlow install, the test file and commands had to hide GPUs before TensorFlow import. The focused test file sets `CUDA_VISIBLE_DEVICES=-1` before importing TensorFlow, and the executed pytest commands also exported `CUDA_VISIBLE_DEVICES=-1`.

Audit outcome: pass for the restricted NP4 scope. The commands and artifact answer the CPU-only fixed-static-shape value support question and preserve the planned non-claim boundaries.

## Evidence contract

Question:

- Which current nonlinear value paths compile and run under `tf.function(jit_compile=True)` on CPU for fixed static shapes, and which other XLA support claims remain unsupported or intentionally out of scope for this NP4 run?

Baseline:

- Existing graph parity coverage in `tests/test_compiled_filter_parity_tf.py`.
- Existing nonlinear value behavior tests in `tests/test_nonlinear_sigma_point_values_tf.py`.
- NP2 and NP3 deferral results, which explicitly forbid treating missing NP4 support certification as already established.

Primary criterion:

- A backend/cell may be marked `supported` for CPU XLA only if a focused CPU-only test validates eager vs `tf.function(jit_compile=True)` parity for the same static shape, with `return_filtered=False` and `return_filtered=True` handled separately, and with filtered means/covariances checked when returned.

Veto diagnostics:

- XLA success inferred from graph mode without `jit_compile=True`.
- `return_filtered=True` inferred from `return_filtered=False`.
- Dynamic-horizon support inferred from fixed-shape evidence.
- GPU XLA inferred from CPU-only evidence.
- Score support inferred from non-XLA or non-branch-certified tests.
- Retracing boundaries omitted for rows claimed as supported.

Explanatory diagnostics only:

- pytest pass counts;
- warning-only output from TensorFlow Probability import;
- prior graph parity tests not using XLA JIT;
- existing finite-difference score tests.

What is not concluded:

- any GPU XLA support or speedup claim;
- any dynamic-horizon XLA support claim;
- any score-path XLA support claim;
- any default-policy change;
- any HMC readiness statement;
- any broad performance statement from this compile/parity artifact alone.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`

## Implemented focused test coverage

New focused file:

- `tests/test_nonlinear_xla_parity_tf.py`

Added coverage:

- CPU-only device hiding before TensorFlow runtime probe.
- CPU XLA eager-vs-XLA parity for Model B value paths with:
  - cubature, `return_filtered=False`;
  - cubature, `return_filtered=True`;
  - UKF, `return_filtered=False`;
  - UKF, `return_filtered=True`;
  - CUT4, `return_filtered=False`;
  - CUT4, `return_filtered=True`.
- CPU XLA eager-vs-XLA parity for Model C value paths with the same six backend/mode cells.
- Same-static-shape no-retracing check for every supported CPU value cell by asserting one concrete function after repeated calls on the same static shape.

Not added in this NP4 run:

- dynamic-horizon XLA tests;
- trusted-GPU XLA tests;
- branch-certified score XLA parity tests.

## Commands run

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_xla_parity_tf.py
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_xla_parity_tf.py -k 'model_b_value_cpu_xla_parity_return_filtered_true or model_c_value_cpu_xla_parity_return_filtered_true'
git rev-parse HEAD && git status --short tests/test_nonlinear_xla_parity_tf.py docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md
```

Observed outcomes:

- `tests/test_nonlinear_xla_parity_tf.py`: `13 passed, 2 warnings in 14.67s`
- targeted `return_filtered=True` subset: `6 passed, 7 deselected, 2 warnings in 8.21s`
- relevant git status after test creation: new untracked test file only before writing this result artifact

## Required support matrix

| Model | Backend/path | Value or score | Static shape | Dynamic horizon | `return_filtered=False` | `return_filtered=True` | Parity tolerance | Concrete-function/retracing boundary | CPU XLA | GPU XLA | Unsupported ops/control-flow notes | Authoritative non-claim text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model B | cubature value (`tf_svd_cubature`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_nonlinear_accumulation_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model B | UKF value (`tf_svd_ukf`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_nonlinear_accumulation_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model B | CUT4 value (`tf_svd_cut4`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_nonlinear_accumulation_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model C | cubature value (`tf_svd_cubature`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_univariate_nonlinear_growth_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model C | UKF value (`tf_svd_ukf`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_univariate_nonlinear_growth_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model C | CUT4 value (`tf_svd_cut4`) | value | observations `(3, 1)`; fixed state/innovation dims from `make_univariate_nonlinear_growth_model_tf()` | `not_claimed` | `supported` | `supported` | eager vs XLA `atol=1e-12` | one concrete function across repeated same-shape calls | `supported` | `skipped_no_trusted_device` | static horizon only; Python loop over horizon remains fixed-shape evidence | dynamic horizon, GPU XLA, and speed claims are not established by this row |
| Model B | cubature score (`tf_svd_cubature_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model B | UKF score (`tf_svd_ukf_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model B | CUT4 score (`tf_svd_cut4_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model C smooth branch | cubature score (`tf_svd_cubature_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model C smooth branch | UKF score (`tf_svd_ukf_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model C smooth branch | CUT4 score (`tf_svd_cut4_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | score XLA parity not added in this NP4 run | no score XLA support claim is made |
| Model C structural fixed-support branch | cubature score (`tf_svd_cubature_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | branch-certified score XLA parity not added in this NP4 run | no structural fixed-support score XLA claim is made |
| Model C structural fixed-support branch | UKF score (`tf_svd_ukf_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | branch-certified score XLA parity not added in this NP4 run | no structural fixed-support score XLA claim is made |
| Model C structural fixed-support branch | CUT4 score (`tf_svd_cut4_score`) | score | N/A for NP4 | `not_claimed` | `not_claimed` | `not_claimed` | N/A | N/A | `untested` | `skipped_no_trusted_device` | branch-certified score XLA parity not added in this NP4 run | no structural fixed-support score XLA claim is made |

## Interpretation

This NP4 run certifies a narrow claim boundary only: on CPU, with GPU hidden before TensorFlow import, the focused static-shape Model B and Model C value cells for cubature, UKF, and CUT4 match eager execution under `tf.function(jit_compile=True)` for both `return_filtered=False` and `return_filtered=True`, and the `return_filtered=True` cells also match filtered means and filtered covariances.

This is enough to mark those six value rows as CPU-XLA `supported` under the NP4 status vocabulary. It is not enough to claim anything about dynamic horizon, GPU XLA, score XLA, runtime improvement, or default policy.

## Run manifest

| Field | Value |
| --- | --- |
| Git commit | `81c647b157612a233dde1a9fbd847647a5b03b8f` |
| Dirty worktree | `true` |
| Command | `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_xla_parity_tf.py`; `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_nonlinear_xla_parity_tf.py -k 'model_b_value_cpu_xla_parity_return_filtered_true or model_c_value_cpu_xla_parity_return_filtered_true'` |
| Environment | `local CLI session; conda env path indicates tf-gpu environment` |
| Python | `pytest interpreter from local environment; exact version not probed in this bounded worker run` |
| TensorFlow | `imported during pytest; exact version not separately probed in this bounded worker run` |
| TensorFlow Probability | `imported during pytest; exact version not separately probed in this bounded worker run` |
| CPU/GPU status | `CPU-only run` |
| GPU intentionally hidden | `yes; CUDA_VISIBLE_DEVICES=-1 in test file and commands` |
| Device visibility | `focused test asserts tf.config.list_physical_devices("GPU") == []` |
| Dtype | `tf.float64` |
| Model/backend/shape/horizon/parameter dimension/point count | `Model B and Model C value paths; backends tf_svd_cubature/tf_svd_ukf/tf_svd_cut4; observations shape (3, 1); fixed static horizon 3; parameter dimension N/A; point count backend-dependent and not separately emitted in this artifact` |
| Random seeds | `N/A` |
| Warmup policy and wall time | `pytest-driven compile/parity checks only; no benchmark warmup policy; wall times 14.67s and 8.21s for the two executed commands` |
| Output artifacts | `tests/test_nonlinear_xla_parity_tf.py`; `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md` |
| Governing phase plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-jit-gate-plan-2026-05-15.md` |
| Result file | `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md` |
| Derivation/proof-obligation artifact | `N/A; NP4 run added compile/parity certification only and did not change value algebra or derivative structure` |

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Certify the listed CPU static-shape value rows as XLA-supported; keep all dynamic-horizon, GPU, and score rows non-claims or untested | passed for the six listed value rows: focused eager-vs-XLA parity exists for `return_filtered=False` and `return_filtered=True`, with filtered-state parity checked when returned | vetoes respected: no graph fallback used for supported rows, no dynamic-horizon inference, no GPU inference, no score inference | whether the same backends remain XLA-supported for other horizons, other model shapes, trusted GPU execution, and branch-certified score paths | NP5 may use only the `supported` CPU value cells from this matrix; any GPU or score continuation requires its own trusted and focused NP4-grade certification artifact first | no GPU claim, no dynamic-horizon claim, no score XLA claim, no speed claim, no default-policy claim |

## Non-implication text

This NP4 result does not mean that all nonlinear TensorFlow paths are XLA-ready. It certifies only the explicit static-shape CPU value cells listed above. It does not certify dynamic-horizon behavior, GPU XLA behavior, any analytic score branch, Hessian behavior, or any runtime improvement. Existing graph-mode or finite-difference evidence remains distinct from XLA certification and was not upgraded by implication.

## Files changed

- `tests/test_nonlinear_xla_parity_tf.py`
- `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`

## Phase exit label

`NP4_CPU_VALUE_XLA_MATRIX_CERTIFIED_SCORE_AND_GPU_NONCLAIMS_RETAINED`
