# Kalman QR Dtype And Batched Score Visible Execution Ledger

Date: 2026-07-09

## Status

`DRAFT_LEDGER_CREATED`

## Ledger

### 2026-07-09T18:52:21+08:00 - Phase 0 - PRECHECK

Evidence contract:

- Question: Can the QR Kalman dtype/batched-score program be launched with
  explicit dtype, review, artifact, and claim boundaries?
- Baseline/comparator: current FP64-only QR Kalman code paths and 2026-07-09
  FP64 CPU/GPU XLA benchmark artifacts.
- Primary criterion: governance docs, Phase 0 subplan, and review bundle exist,
  pass local text checks, and receive read-only review or documented substitute
  review before Phase 0 result accepts the gate.
- Veto diagnostics: hidden source edit before inventory, unsupported FP32/GPU
  claim, missing review, missing phase result, or unapproved GPU/runtime action.
- Nonclaims: no FP32 support, no batched analytical score implementation, no
  speed ranking, no HMC/posterior/default-readiness claim.

Actions:

- Created draft master program, visible runbook, subplans, ledger, stop
  handoff, and review bundle.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-gated-execution-runbook-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-subplan-2026-07-09.md`

Gate status:

- `PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Launch Phase 1 dtype infrastructure precheck and implementation.

### 2026-07-09T19:02:19+08:00 - Phase 0 - PASS_REVIEW

Evidence contract:

- Question: Are the governance docs and Phase 0/1 handoff safe after the
  Claude review gate was rejected as external disclosure risk?
- Baseline/comparator: Phase 0 local checks, dtype inventory, and Round 1
  substitute-review findings.
- Primary criterion: Fresh Codex substitute review returns `AGREE` after
  repair, with weaker-than-Claude review status recorded.
- Veto diagnostics: pretending Claude reviewed the artifacts, retrying around
  approval rejection, missing artifact coverage, or unsupported FP32/batched
  score claim.
- Nonclaims: no FP32 support, no source cleanup, no batched analytical score,
  no runtime or scientific claim.

Actions:

- Patched review protocol in master/runbook/subplan/result.
- Ran fresh Codex substitute review round 2.

Artifacts:

- `docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-codex-substitute-review-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-result-2026-07-09.md`

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Begin Phase 1.

### 2026-07-09T19:12:00+08:00 - Phase 1 - ASSESS_GATE

Evidence contract:

- Question: Do the new helpers infer and preserve floating dtype without
  silently mixing dtypes?
- Baseline/comparator: current QR helpers that coerce tensors to `tf.float64`.
- Primary criterion: focused helper tests pass for FP32/FP64 preservation,
  literal default, mixed dtype rejection, unsupported dtype rejection, and one
  CPU/XLA trace.
- Veto diagnostics: helper silently upcasts/downcasts mixed dtype, ambiguous
  mixed dtype policy, or eager-only test coverage.
- Nonclaims: no QR value/score kernel has been migrated; no FP32 QR support or
  batched analytical score exists.

Actions:

- Added `bayesfilter/linear/dtypes_tf.py`.
- Added `tests/test_linear_qr_dtype_contracts.py`.
- Ran CPU-hidden pytest and `git diff --check`.
- Completed fresh Codex substitute review with `VERDICT: AGREE`.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-result-2026-07-09.md`

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Begin Phase 2 QR value dtype cleanup.

### 2026-07-09T19:32:19+08:00 - Phase 2 - PASS_REVIEW

Evidence contract:

- Question: Do QR value kernels preserve explicit FP32/FP64 dtype while
  retaining existing FP64 value behavior?
- Baseline/comparator: existing FP64 QR value parity tests and FP32 values
  compared against FP64 references at FP32 tolerance.
- Primary criterion: observed FP32 outputs for compact, while-loop,
  batched-static, masked, filtered, and dispatcher value paths; FP64 tests still
  pass.
- Veto diagnostics: hidden FP64 value-path coercion, value parity failure,
  nonfinite output, CPU/XLA compile failure, or unsupported analytical-score or
  benchmark claim.
- Nonclaims: no analytical-score dtype support, no benchmark dtype controls, no
  batch-native score, no GPU/default-readiness claim, and no speed ranking.

Actions:

- Migrated QR value paths, touched QR factor helpers, model container, and
  result envelopes to preserve explicit floating dtype where required.
- Added FP32/FP64 value and container tests.
- Repaired Phase 3 subplan after substitute review identified missing
  derivative-container scope.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-result-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-subplan-2026-07-09.md`

Checks:

- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_qr_compact_loglik_tf.py` -> `26 passed`
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_factor_tf.py` -> `4 passed`
- `git diff --check -- bayesfilter/linear bayesfilter/results_tf.py tests docs/plans docs/reviews` -> passed

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Begin Phase 3 analytical-score dtype cleanup.

### 2026-07-09T20:03:50+08:00 - Phase 3 - PASS_REVIEW

Evidence contract:

- Question: Does analytical QR score preserve requested dtype and remain
  correct against scalar/autodiff references?
- Baseline/comparator: existing FP64 analytical score and FP32 autodiff through
  QR value on small fixtures.
- Primary criterion: FP64 existing parity remains; FP32 analytical value/score
  outputs are FP32 and match FP32 autodiff within declared tolerance.
- Veto diagnostics: hidden FP64 coercion in score kernels, derivative payload
  containers, or derivative result envelopes; score dtype mismatch; parity
  failure; nonfinite score; or XLA compile failure.
- Nonclaims: no benchmark dtype controls, no batch-native analytical score, no
  GPU/default-readiness claim, and no speed ranking.

Actions:

- Migrated analytical QR score, score/Hessian helper paths, derivative payload
  containers, and derivative result envelopes to preserve explicit FP32/FP64.
- Added FP32/FP64 derivative container, wrapper, masked score/Hessian, and
  CPU/XLA dynamic-score smoke tests.
- Completed fresh Codex substitute review with `VERDICT: AGREE`.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-result-2026-07-09.md`

Checks:

- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_kalman_qr_derivatives_tf.py` -> `29 passed`
- `git diff --check -- bayesfilter/linear bayesfilter/results_tf.py tests docs/plans docs/reviews` -> passed
- `python -m py_compile bayesfilter/linear/kalman_qr_derivatives_tf.py bayesfilter/linear/types_tf.py bayesfilter/results_tf.py` -> passed

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Begin Phase 4 benchmark dtype controls.

### 2026-07-09T20:13:58+08:00 - Phase 4 - CPU_HIDDEN_XLA_SMOKE_PASS

Evidence contract:

- Question: Can benchmark artifacts fail closed if requested dtype differs from
  observed output dtype?
- Baseline/comparator: existing FP64-only benchmark harness and artifacts.
- Primary criterion: FP32 and FP64 CPU-hidden smoke artifacts record matching
  requested/observed dtype for both analytical and autodiff arms.
- Veto diagnostics: missing dtype field, dtype mismatch not failing, TF32
  conflated with dtype, or non-JIT run mislabeled as XLA evidence.
- Nonclaims: no full performance ladder, no GPU performance, no statistical
  speed ranking, no batch-native score implementation.

Actions:

- Added `--dtype float32|float64` benchmark controls.
- Threaded requested dtype through fixture construction and compiled analytical
  and autodiff score arms.
- Added JSON/Markdown requested and observed dtype fields and fail-closed row
  parity behavior.
- Ran FP32 and FP64 CPU-hidden XLA smoke artifacts.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-result-2026-07-09.md`
- `docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.json`
- `docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.md`
- `docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.json`
- `docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.md`

Checks:

- `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` -> passed
- `CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float32_cpu_xla_2026-07-09.md` -> passed
- `CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --device cpu --jit-compile --dtype float64 --output-json docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_dtype_smoke_float64_cpu_xla_2026-07-09.md` -> passed
- `git diff --check -- scripts docs/benchmarks docs/plans` -> passed

Gate status:

- `PASSED_CPU_HIDDEN_XLA_SMOKE`

Next action:

- Review and execute Phase 5 batched analytical score contract.

### 2026-07-09T20:17:04+08:00 - Phase 5 - CONTRACT_PASS_REVIEW

Evidence contract:

- Question: Is the batch-native analytical score contract precise and
  implementable without confusing batch size `B` and parameter dimension `P`?
- Baseline/comparator: existing scalar analytical score and batched-static QR
  value/autodiff gradients.
- Primary criterion: contract states inputs `[B, ...]`, derivative tensors
  `[B, P, ...]`, outputs `[B]` and `[B, P]`, dtype behavior, time-invariant
  limitation, and scalar/autodiff references.
- Veto diagnostics: contract permits `tf.vectorized_map` scalar wrapper as
  final kernel, leaves dtype ambiguous, or lacks parity baseline.
- Nonclaims: no implementation correctness or performance claim.

Actions:

- Wrote exact batch-native analytical QR score contract.
- Named Phase 6 source, test, scalar comparator, batched-autodiff comparator,
  dtype, shape, and source-contract paths.
- Completed fresh Codex substitute review with `VERDICT: AGREE`.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-result-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-subplan-2026-07-09.md`

Checks:

- `git diff --check -- docs/plans tests bayesfilter/linear` -> passed

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Execute Phase 6 batch-native analytical score implementation.

### 2026-07-09T20:34:51+08:00 - Phase 6 - IMPLEMENTATION_PASS_REVIEW

Evidence contract:

- Question: Does the batch-native analytical score return `[B]` values and
  `[B, P]` scores matching scalar analytical and autodiff references?
- Baseline/comparator: scalar `tf_qr_sqrt_kalman_score` row loop and autodiff
  through batched-static QR value on small fixtures.
- Primary criterion: FP32/FP64 batch-native outputs have requested dtype and
  match references within declared tolerance under CPU/XLA.
- Veto diagnostics: vectorized/scalar row wrapper as final kernel, shape
  mismatch, dtype mismatch, parity failure, nonfinite output, or XLA compile
  failure.
- Nonclaims: no runtime superiority, no full benchmark ladder, no GPU evidence,
  no HMC/posterior/default-readiness claim.

Actions:

- Implemented `tf_qr_sqrt_kalman_score_batched_static`.
- Added batched derivative helpers in the derivative module.
- Added dedicated batched analytical score tests.
- Repaired review-identified no-jitter update derivative branch and improved
  `B != P`, multidimensional, and source-contract coverage.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-result-2026-07-09.md`
- `tests/test_linear_qr_batched_analytical_score_tf.py`

Checks:

- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py` -> `7 passed`
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py` -> `19 passed`
- `python -m py_compile bayesfilter/linear/kalman_qr_derivatives_tf.py tests/test_linear_qr_batched_analytical_score_tf.py` -> passed
- `git diff --check -- bayesfilter/linear tests docs/plans` -> passed

Gate status:

- `PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Refresh and execute Phase 7 correctness and benchmark ladder.

### 2026-07-09T21:03:24+08:00 - Phase 7 - SMOKE_GATE_PASS

Evidence contract:

- Question: Can the refreshed benchmark harness time the batch-native
  analytical score path with honest scalar analytical and autodiff comparators
  while preserving dtype/device/JIT provenance?
- Baseline/comparator: batch-native analytical score, scalar analytical row
  loop, and scalar-value autodiff row loop across independent parameter
  proposals.
- Primary criterion: CPU-hidden FP32 XLA smoke writes complete JSON/Markdown
  artifacts and all applicable rows pass dtype, shape, finite-output, and
  parity checks.
- Veto diagnostics: correctness failure, dtype mismatch, missing device
  provenance, nonfinite timed output, unapproved GPU run, or missing
  compile/warm split.
- Nonclaims: no speed ranking, GPU/default readiness, HMC readiness, posterior
  correctness, or scientific validity.

Actions:

- Refreshed `scripts/benchmark_kalman_qr_parameter_count_scaling.py` with
  `--batch-size`, batch-native analytical timing, scalar analytical row-loop
  timing, autodiff row-loop timing, and a non-timed batch-static autodiff
  diagnostic.
- Repaired the Phase 7 subplan after the first smoke showed the batch-static
  autodiff value-gradient route returned nonfinite scores for the benchmark
  fixture.
- Ran focused CPU-hidden tests, py-compile, CPU-hidden FP32 XLA smoke, and diff
  hygiene.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-result-2026-07-09.md`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json`
- `docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md`
- `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.log`
- `docs/benchmarks/logs/kalman_qr_phase7_cpu_hidden_tests_2026-07-09.log`

Checks:

- `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` -> passed
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py` -> `19 passed`
- `CUDA_VISIBLE_DEVICES=-1 python scripts/benchmark_kalman_qr_parameter_count_scaling.py --dimensions 10 --parameter-counts 50 --timesteps 8 --repeats 1 --batch-size 2 --device cpu --jit-compile --dtype float32 --output-json docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.json --output-md docs/benchmarks/kalman_qr_batched_score_smoke_float32_cpu_xla_2026-07-09.md` -> passed
- `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` -> passed

Gate status:

- `PASSED_CPU_HIDDEN_XLA_SMOKE_FULL_LADDER_DEFERRED`

Next action:

- Refresh exact full-ladder CPU/GPU commands and obtain/confirm GPU approval
  and provenance before launching GPU runtime, or proceed to Phase 8 closeout
  documenting the full ladder as deferred.

### 2026-07-09T21:12:00+08:00 - Phase 7B - BLOCKER

Evidence contract:

- Question: Can the full CPU/GPU descriptive ladder be launched with trusted
  GPU provenance?
- Baseline/comparator: Phase 7 CPU-hidden smoke plus Phase 7B TensorFlow GPU
  provenance gate.
- Primary criterion: TensorFlow reports at least one logical GPU and benchmark
  JSON can record `/GPU:0`, physical/logical GPU lists, JIT, dtype, TF32 flag,
  and managed-session trust basis before GPU ladders run.
- Veto diagnostics: TensorFlow physical/logical GPU list empty, `--device gpu`
  benchmark raises before artifact row, or GPU provenance cannot be recorded.
- Nonclaims: no GPU benchmark, CPU/GPU comparison, speed ranking,
  production/default readiness, HMC readiness, or posterior correctness.

Actions:

- Drafted Phase 7B full ladder subplan.
- Ran fresh bounded Codex substitute reviews of the subplan and harness.
- Repaired subplan log/provenance/handoff/comparator wording findings.
- Removed stale `descriptive_batched_autodiff...` JSON key from the harness.
- Reran CPU-hidden FP32 smoke after the stale-key repair.
- Ran `nvidia-smi`, TensorFlow GPU provenance probe, and a tiny GPU smoke.

Artifacts:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7b-full-ladder-subplan-2026-07-09.md`
- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7b-full-ladder-blocker-result-2026-07-09.md`
- `docs/benchmarks/logs/kalman_qr_batched_score_smoke_float32_gpu_xla_2026-07-09.log`

Checks:

- `python -m py_compile scripts/benchmark_kalman_qr_parameter_count_scaling.py` -> passed
- `git diff --check -- scripts docs/benchmarks docs/plans bayesfilter/linear tests` -> passed
- `nvidia-smi` -> driver sees two NVIDIA GPUs
- TensorFlow probe -> `physical_gpu=[]`, `logical_gpu=[]`
- GPU smoke -> failed before benchmark rows with no logical TensorFlow GPU

Gate status:

- `BLOCKED_GPU_TENSORFLOW_VISIBILITY`

Next action:

- Phase 8 blocker closeout, or rerun Phase 7B after TensorFlow GPU visibility
  is repaired in a trusted environment.
