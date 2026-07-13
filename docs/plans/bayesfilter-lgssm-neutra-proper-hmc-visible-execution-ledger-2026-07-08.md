# BayesFilter LGSSM NeuTra Proper HMC Visible Execution Ledger

Date: 2026-07-08

## Ledger

### 2026-07-08 - Program Launch - PRECHECK

Evidence contract:

- Question: Can BayesFilter close the packaging, mechanics, CPU-harness, and
  LGSSM-reference gaps needed before proper HMC testing of the Phase 16 NeuTra
  path?
- Baseline/comparator: Phase 16 GPU/XLA training artifact, current
  manual-score LGSSM target signatures, frozen transport loader, and exact
  LGSSM reference posterior.
- Primary criterion: each phase either produces a reviewed artifact for its
  boundary or records an exact blocker without overclaiming.
- Veto diagnostics: stale Phase 10/11 artifact use, `jit_compile=false`
  fallback in runtime evidence, runtime autodiff in admitted route, hidden
  training, hidden HMC before mechanics gate, hidden sample generation, missing
  hashes/signatures, malformed artifacts, unsupported posterior/HMC/product/
  scientific claims.
- Nonclaims: no HMC convergence, posterior correctness, sampler superiority,
  production readiness, default readiness, nonlinear SSM validity, DSGE/c603
  validity, or scientific validity until a specific reviewed phase proves that
  narrower claim.

Skeptical audit:

- Wrong baseline blocked: Phase 17 can use only the Phase 16 GPU/XLA training
  artifact, not stale Phase 10/11 non-XLA artifacts.
- Proxy promotion blocked: packaging/finite mechanics cannot imply HMC
  readiness.
- Hidden assumption named: GPU training and CPU sampling/HMC remain separate
  phases.
- Environment boundary: Phase 17 packaging is CPU-hidden; future GPU/HMC phases
  require reviewed trusted-runtime gates.
- Artifact match: master program, runbook, ledger, and Phase 17 subplan paths
  are declared.

Actions:

- Created master program:
  `docs/plans/bayesfilter-lgssm-neutra-proper-hmc-gap-closure-master-program-2026-07-08.md`.
- Created visible gated runbook:
  `docs/plans/bayesfilter-lgssm-neutra-proper-hmc-visible-gated-execution-runbook-2026-07-08.md`.
- Created this ledger.

Gate status:

- `PHASE17_REVIEW_PENDING`

Next action:

- Run bounded read-only Claude review for the master/runbook and Phase 17
  subplan before executing packaging.

### 2026-07-08 - Phase 17 - REPAIR_AND_REVIEW

Evidence contract:

- Question: Can BayesFilter package the Phase 16 GPU/XLA-trained affine state
  as a frozen payload and reload it against the current manual-score LGSSM
  target signatures without running fixed-transport HMC mechanics?
- Baseline/comparator: Phase 16 GPU/XLA training-state artifact and current
  frozen affine artifact loader.
- Primary criterion: write parseable payload and validation JSON from the exact
  Phase 16 source file hash, load the payload with matching target signature,
  pass finite forward/base value/score checks, and record that training,
  mechanics, HMC sampling/tuning, external sample generation, and
  `jit_compile=false` did not run.
- Veto diagnostics: stale Phase 10/11 source, source hash/signature mismatch,
  malformed payload, nonfinite loader/reference diagnostics, hidden mechanics
  or HMC, hidden training, hidden samples, unsupported readiness/scientific/
  product claims.
- Nonclaims: no fixed-transport mechanics, HMC readiness, posterior
  correctness, XLA compile readiness, production readiness, default readiness,
  or scientific validity.

Skeptical audit:

- Wrong baseline was repaired by adding an exact Phase 16 file SHA-256 guard.
- Proxy promotion was repaired by moving fixed-transport HMC mechanics out of
  Phase 17 and into Phase 18.
- Hidden fallback is blocked by explicit `jit_compile_runtime_executed=false`
  and `jit_compile_false_runtime_executed=false` fields.
- Artifact mismatch is covered by exact payload/validation filenames and
  post-packaging JSON field validation.

Actions:

- Patched `bayesfilter/testing/neutra_gpu_affine_payload_tf.py` to use Phase
  16/17 names, exact Phase 16 file-hash validation, and packaging-only
  loader/reference validation.
- Patched `tests/test_neutra_gpu_affine_payload_tf.py` to assert the exact
  Phase 17 filenames, source hash, nonactions, and stale Phase 10 rejection.
- Patched the Phase 17 subplan with exact local check, packaging, JSON, and
  field-validation commands.
- Attempted Claude review gate using
  `/home/chakwong/python/claudecodex/scripts/claude_review_gate.sh`; the
  sandbox approval reviewer rejected it as external-service disclosure risk.
  No workaround was attempted.
- Wrote same-foreground Codex substitute review:
  `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase17-subplan-codex-substitute-review-2026-07-08.md`.

Local checks:

- `python -m py_compile bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py`: passed.
- Source/config scan for Phase 16/17 required tokens and stale Phase 10/11
  source tokens: passed with `missing=[]`, `violations=[]`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `5 passed, 2 warnings`.
- `git diff --check -- bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md`: passed.

Artifacts:

- `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase17-subplan-review-bundle-2026-07-08.md`
- `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase17-subplan-codex-substitute-review-2026-07-08.md`

Gate status:

- `PHASE17_SUBPLAN_CODEX_SUBSTITUTE_REVIEW_AGREE`

Next action:

- Execute Phase 17 CPU-hidden packaging and validate the produced payload and
  validation JSON.

### 2026-07-08 - Phase 17 - PASSED

Evidence contract:

- Question: Can BayesFilter package the Phase 16 GPU/XLA-trained affine state
  as a frozen payload and reload it against the current manual-score LGSSM
  target signatures without running fixed-transport HMC mechanics?
- Primary criterion: exact Phase 16 source hash recorded, payload and
  validation JSON written and parseable, loader accepts the payload, finite
  forward/base value/score checks pass, no stale Phase 10/11 source, and no
  hidden training/mechanics/HMC/samples/`jit_compile=false`.

Actions:

- Ran Phase 17 packaging command:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_affine_payload_tf --phase16-training-state-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json --artifact-dir docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07 --seed 20260707`.
- Validated both JSON artifacts with `python -m json.tool`.
- Ran Phase 17 field-validation script with `failed=[]`.
- Wrote Phase 17 result:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-result-2026-07-08.md`.

Artifacts:

- Payload:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json`.
- Validation:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json`.

Gate status:

- `PASS_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD`

Next action:

- Draft and review Phase 18 fixed-transport HMC mechanics XLA compile subplan.

### 2026-07-08 - Phase 18 - SUBPLAN_AND_LOCAL_CHECKS

Evidence contract:

- Question: Can the Phase 17 frozen affine payload be bound to the current
  LGSSM generic SSM adapter as an accepted fixed-transport HMC mechanics target
  and compiled with trusted GPU `jit_compile=True` without running HMC chains?
- Baseline/comparator: Phase 17 payload/validation, current LGSSM signatures,
  Phase 15 manual-score trusted GPU/XLA compile gate, and current fail-closed
  fixed-transport authority policy.
- Primary criterion: either a pass artifact records accepted base/fixed
  transport XLA authority, trusted GPU `jit_compile=True` mechanics compile
  success, finite mechanics value/score, timing/size proxies, and no forbidden
  runtime actions; or a blocker records exact error/provenance without fallback.
- Veto diagnostics: `jit_compile=false`, CPU runtime evidence for compile
  success, hidden training, hidden HMC sampling/tuning, hidden external sample
  generation, fallback/GradientTape authority promotion, signature mismatch,
  nonfinite mechanics, malformed artifacts, unsupported readiness/scientific/
  product claims.

Skeptical audit:

- Wrong baseline blocked: Phase 18 can use only the Phase 17 payload and
  current signatures.
- Proxy promotion blocked: mechanics compile success cannot imply HMC
  convergence or posterior correctness.
- Hidden assumption named: XLA-HMC authority is an explicit evidence-backed
  opt-in, not a global generic-adapter default.
- Environment boundary: runtime evidence requires trusted GPU/XLA execution.

Actions:

- Drafted Phase 18 subplan:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md`.
- Wrote same-foreground substitute review:
  `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase18-subplan-codex-substitute-review-2026-07-08.md`.
- Patched `bayesfilter/ssm/target_builder.py` to add explicit
  evidence-backed `xla_hmc_ready` and `full_chain_xla_diagnostic_ready` opt-in
  fields that default to false.
- Patched the LGSSM generic target fixture to opt into XLA-HMC value/score
  authority using the Phase 15 result as evidence.
- Added Phase 18 diagnostic helper:
  `bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py`.
- Added Phase 18 focused tests:
  `tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py`.

Local checks:

- `python -m py_compile bayesfilter/ssm/target_builder.py bayesfilter/ssm/__init__.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py`: passed.
- Source scan for forbidden runtime tokens in Phase 18 helper path: passed with
  `missing=[]`, `violations=[]`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_gpu_affine_payload_tf.py tests/test_neutra_artifact_loader.py tests/test_fixed_transport_hmc_binding.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q`: passed, `41 passed, 1 skipped, 2 warnings`.
- `git diff --check` on touched Phase 18 code/tests: passed.

Gate status:

- `PHASE18_LOCAL_CHECKS_PASSED_TRUSTED_GPU_RUNTIME_PENDING`

Next action:

- Run trusted `nvidia-smi`, then run the Phase 18 trusted GPU/XLA
  `jit_compile=True` mechanics compile diagnostic.

### 2026-07-08 - Phase 18 - PASSED

Actions:

- Ran trusted `nvidia-smi`: passed.
- Ran trusted Phase 18 GPU/XLA command:
  `TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_fixed_transport_hmc_mechanics_xla_tf --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json --seed 20260707 --device /GPU:0`.
- Validated diagnostic JSON with `python -m json.tool`.
- Ran Phase 18 field-validation script with `failed=[]`.
- Wrote Phase 18 result:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-result-2026-07-08.md`.

Artifacts:

- Diagnostic JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json`.

Key diagnostics:

- `compile_time_proxy_seconds = 2.3157755790743977`.
- `concrete_graph_serialized_bytes = 3087116`.
- `compiler_ir_hlo_text_bytes = 1262491`.
- Base and fixed-transport value/score capabilities both recorded
  `accepted_xla_hmc_authority = true`.
- Finite mechanics value/score checks passed.
- Outputs were on `/GPU:0`.
- No training, HMC sampling/tuning, external sample generation, or
  `jit_compile=false` was run.

Gate status:

- `PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE`

Next action:

- Draft Phase 19 CPU-hidden multicore HMC chain harness subplan. Do not run
  chains until that subplan is reviewed and accepted.

### 2026-07-08 - Phases 20/21 - SUBPLANS_DRAFTED

Evidence contract:

- Phase 20 owns bounded CPU-hidden LGSSM reference posterior validation with
  `jit_compile=True`; exact LGSSM posterior residuals are the pass/block
  comparator.
- Phase 21 owns readiness classification from Phase 20 evidence. It is not a
  runtime shortcut and cannot promote product, default, broad HMC, nonlinear
  SSM, DSGE/c603, or scientific claims.

Skeptical audit:

- Wrong baseline blocked: Phase 20 comparator is the exact LGSSM reference
  posterior, not Phase 18 mechanics compile or Phase 19 worker metadata.
- Proxy promotion blocked: acceptance, R-hat/ESS, runtime, and compile success
  cannot pass Phase 20 without posterior residual checks and clean vetoes.
- Hidden boundary blocked: Phase 19 remains harness-only and cannot run or
  claim Phase 20 validation.
- Environment mismatch named: all chain/sample generation remains
  CPU-hidden; training/compile evidence remains separate.

Actions:

- Created Phase 20 subplan:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md`.
- Created Phase 21 subplan:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-subplan-2026-07-08.md`.
- Updated the runbook/master status to
  `PHASE18_PASSED_PHASE19_READY_PHASE20_21_SUBPLANS_DRAFTED`.

Gate status:

- `PHASE20_21_SUBPLANS_CODEX_SUBSTITUTE_REVIEW_AGREE`

Next action:

- Continue Phase 19 implementation under the already-reviewed Phase 19
  harness boundary.

Review artifacts:

- `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase20-subplan-codex-substitute-review-2026-07-08.md`
- `docs/reviews/bayesfilter-lgssm-neutra-hmc-phase21-subplan-codex-substitute-review-2026-07-08.md`

### 2026-07-08 - Phase 19 - PASSED

Evidence contract:

- Question: Can BayesFilter build and run a CPU-hidden multicore harness for
  fixed-transport LGSSM NeuTra chain workers with deterministic metadata and no
  forbidden fallback before reference validation?
- Primary criterion: CPU-hidden worker metadata, deterministic seeds,
  `jit_compile=True`, no forbidden actions, and a bounded worker smoke or
  exact blocker.
- Veto diagnostics: no `jit_compile=false`, no training, no GPU sample
  generation, no HMC transition/sampling/tuning, and no posterior validation.

Skeptical audit:

- Wrong baseline blocked: Phase 19 used only the Phase 17 payload, Phase 18
  diagnostic, and current signatures.
- Proxy promotion blocked: the pass result is harness evidence only.
- Hidden boundary blocked: full-chain HMC and posterior validation remain
  Phase 20 responsibilities.

Actions:

- Added Phase 19 helper:
  `bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Added focused tests:
  `tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Ran the exact Phase 19 CPU-hidden `jit_compile=True` smoke command.
- Wrote Phase 19 result:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-result-2026-07-08.md`.

Artifacts:

- Harness JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json`.
- Harness JSON SHA-256:
  `38c0400b04ce7438f3bc70236ccfd42916c11c1b9c05d95bd76260b64f8c10b4`.

Key diagnostics:

- Two workers passed with return code `0`.
- Worker compile-time proxies were `12.7924878988415` and
  `12.870044000912458` seconds.
- Worker second-call wall times were about `0.0010` seconds.
- CPU-hidden environment recorded `CUDA_VISIBLE_DEVICES=-1`.
- `accepted_full_chain_xla_diagnostic_authority=false`, as required for
  Phase 19.

Local checks:

- `python -m py_compile bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_neutra_gpu_affine_payload_tf.py -q`: passed, `15 passed, 2 warnings`.
- Phase 19 source scan: passed with `violations=[]`.
- `python -m json.tool` on the Phase 19 harness JSON: passed.
- Phase 19 JSON field validation: passed with `failed=[]`.
- `git diff --check` on Phase 19 helper/tests and Phase 19-21 docs: passed.

Gate status:

- `PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS`

Next action:

- Continue to Phase 20 LGSSM reference HMC validation under the Phase 20
  subplan.

### 2026-07-08 - Phase 20 - PASSED

Evidence contract:

- Question: Do bounded CPU-hidden multicore fixed-transport LGSSM NeuTra HMC
  chains, run with `jit_compile=True`, agree with a deterministic quadrature
  reference posterior over the exact LGSSM likelihood target closely enough to
  pass the predeclared fixture gate?
- Primary criterion: finite retained samples, no worker errors, no
  `jit_compile=false`, no GPU sample generation, no hidden training, and
  posterior mean/covariance residuals within tolerance.
- Veto diagnostics: nonfinite samples or target values, worker failure,
  CPU-hidden violation, `jit_compile=false`, hidden training, GPU sample
  generation, missing reference posterior, provenance mismatch, malformed
  artifact, or posterior residual above tolerance.

Skeptical audit:

- Wrong baseline blocked: Phase 20 used deterministic 2D quadrature over the
  exact LGSSM likelihood target, not a mechanics compile proxy.
- Proxy promotion blocked: acceptance and XLA compile logs did not by
  themselves pass the phase; posterior residuals and veto checks were required.
- Hidden assumption repaired: parent-process TensorFlow work before worker
  multiprocessing was replaced by spawned reference/HMC workers.
- Boundary preserved: no `jit_compile=false`, no training, no GPU sample
  generation, no DSGE/c603, and no broad HMC or scientific claim.

Actions:

- Added Phase 20 helper:
  `bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py`.
- Added focused tests:
  `tests/test_neutra_lgssm_reference_hmc_validation_tf.py`.
- Ran the exact bounded CPU-hidden `jit_compile=True` Phase 20 validation
  command after repairing process isolation.
- Wrote Phase 20 result:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-result-2026-07-08.md`.

Artifacts:

- Validation JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json`.
- Validation JSON SHA-256:
  `094962f3fd8dbd5002ef5d92e42e23ae34cc52dc234ad836d78fe4edd768e188`.
- Stable artifact hash:
  `sha256:07404fd92a4b5e69449088c8852392241fdbbc9cb61eea91ceb4b2b235f6d553`.

Key diagnostics:

- Decision: `PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION`.
- Retained sample count: 256 total.
- Worker acceptance rates: `[1.0, 1.0]`.
- Mean max absolute residual: `0.23550553833312193` versus tolerance `0.35`.
- Covariance max absolute residual: `0.6497737415124094` versus tolerance
  `0.65`.
- R-hat/ESS: unavailable in this bounded helper; not used as promotion
  evidence.
- CPU-hidden environment recorded `CUDA_VISIBLE_DEVICES=-1`.

Local checks:

- `python -m py_compile bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_lgssm_reference_hmc_validation_tf.py`: passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q`: passed, `19 passed, 2 warnings`.
- Phase 20 source scan: passed with `violations=[]`.
- `python -m json.tool` on the Phase 20 validation JSON: passed.
- Phase 20 JSON field validation: passed with `failed=[]`.
- `git diff --check` on Phase 20 helper/tests/subplan: passed.

Gate status:

- `PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION`

Next action:

- Continue to Phase 21 readiness decision classification under the Phase 21
  subplan. Preserve that any readiness decision is fixture-local and
  artifact-local.

### 2026-07-08 - Phase 21 - COMPLETE

Evidence contract:

- Question: Given Phase 20 evidence, what is the narrow readiness
  classification for the LGSSM fixed-transport NeuTra-HMC path?
- Primary criterion: emit exactly one of `LGSSM_REFERENCE_HMC_READY`,
  `BLOCKED_FOR_REPAIR`, or `INSUFFICIENT_EVIDENCE_NO_PROMOTION`.
- Veto diagnostics: any Phase 17-20 `jit_compile=false` runtime, hidden
  training, GPU sample generation, missing reference posterior, posterior
  residual failure, nonfinite samples, worker errors, malformed artifacts, or
  unsupported product/default/scientific claims.

Skeptical audit:

- Wrong baseline blocked: the decision used Phase 20 reference validation and
  Phase 17-19 provenance, not compile or acceptance proxies alone.
- Proxy promotion blocked: no method ranking, product readiness, default
  readiness, or scientific claim was emitted.
- Hidden assumption named: readiness is fixture-local and artifact-local.
- Environment boundary preserved: Phase 21 ran no new training, HMC, GPU job,
  tuning, DSGE/c603 work, or non-LGSSM expansion.

Actions:

- Parsed Phase 17-20 artifacts with the Phase 21 consistency checker.
- Wrote Phase 21 decision JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase21_readiness_decision_seed20260707.json`.
- Wrote Phase 21 result:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-result-2026-07-08.md`.

Gate status:

- `LGSSM_REFERENCE_HMC_READY`

Scope:

- Static QR LGSSM fixture and exact Phase 17-20 artifacts only.
- Not production readiness, default readiness, sampler superiority, nonlinear
  SSM validity, DSGE/c603 validity, broad NeuTra validity, or scientific
  validity.

Next action:

- Start a new reviewed program for longer LGSSM replication or first non-LGSSM
  target. Do not run additional HMC or broader targets without that new plan.
