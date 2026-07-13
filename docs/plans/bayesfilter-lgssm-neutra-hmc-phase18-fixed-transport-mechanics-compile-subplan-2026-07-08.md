# Phase 18 Subplan: Fixed-Transport HMC Mechanics XLA Compile Gate

Date: 2026-07-08

## Phase Objective

Repair or validate the fixed-transport HMC mechanics authority boundary for the
Phase 17 frozen affine payload, then run a trusted `jit_compile=True`
mechanics-only compile diagnostic that records compile time and compilation-size
proxies.

This is an HMC mechanics compile gate only. It is not HMC sampling/tuning,
chain execution, external sample generation, posterior validation, production
promotion, or scientific promotion.

## Entry Conditions Inherited From Previous Phase

- Phase 17 passed frozen payload packaging and loader/reference validation.
- Phase 17 payload:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json`.
- Phase 17 validation:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json`.
- Phase 17 frozen artifact signature:
  `5e36c60aaca37facb3e110138e1b2da2ebe758ace1efb6a8845650553dc3d7e0`.
- Phase 17 transport hash:
  `f1780d9eb8ae0f6d5e6865da6dbb3d1d1a22c4c2e5c89beb60c1f887c5f48fc7`.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Phase 17 did not run fixed-transport HMC mechanics. A local attempt during
  plan repair showed the current fixed-transport mechanics path rejects
  `use_xla=True` because the base generic SSM adapter advertises
  `xla_hmc_ready=False`.

## Required Artifacts

- A narrowly reviewed implementation patch, expected scope:
  - `bayesfilter/ssm/target_builder.py`;
  - `bayesfilter/testing/lgssm_generic_target_adapter_tf.py`;
  - focused tests under `tests/`;
  - a Phase 18 diagnostic helper under `bayesfilter/testing/` if needed.
- Trusted GPU/XLA mechanics compile diagnostic JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json`.
- Phase 18 result or blocker:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- Source/config guard checks proving:
  - generic SSM adapters default to `xla_hmc_ready=False`;
  - XLA-HMC readiness is an explicit opt-in, not a global default;
  - the LGSSM fixture opts in only because the Phase 15 trusted GPU/XLA compile
    gate passed for the same target and adapter signatures;
  - fallback/GradientTape authorities still cannot be promoted by the fixed
    transport wrapper.
- `python -m py_compile` for changed helpers/tests.
- CPU-hidden focused pytest for source/config/loader/mechanics guards only.
- Source scan preserving no `GradientTape`, `batch_jacobian`, `tape.`, or
  `jit_compile=False` in the admitted Phase 18 runtime helper path.
- Trusted `nvidia-smi` before GPU/XLA execution.
- Trusted GPU/XLA command must run with `jit_compile=True` only.
- Diagnostic JSON must record:
  - payload path and payload file SHA-256;
  - payload stable hash and frozen artifact signature;
  - target, adapter, and fixed-transport adapter signatures;
  - value/score capability before and after fixed-transport wrapping;
  - `use_xla=true`;
  - `jit_compile=true`;
  - first-call wall time;
  - second-call wall time;
  - compile-time proxy;
  - concrete graph serialized byte size when available;
  - compiler IR/HLO text byte size when available;
  - finite mechanics value and score checks;
  - no training;
  - no HMC sampling/tuning;
  - no external sample generation;
  - no `jit_compile=false` runtime fallback.
- `python -m json.tool` on the diagnostic JSON if produced.
- Field-validation script for the required JSON fields.
- `git diff --check` for touched files.
- Bounded read-only review of the Phase 18 result and Phase 19 subplan before
  CPU multicore chain harness work.

Exact local check commands before trusted GPU execution:

```bash
python -m py_compile bayesfilter/ssm/target_builder.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_gpu_affine_payload_tf.py tests/test_neutra_artifact_loader.py tests/test_fixed_transport_hmc_binding.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q
python - <<'PY'
from pathlib import Path

paths = [
    Path("bayesfilter/ssm/target_builder.py"),
    Path("bayesfilter/testing/lgssm_generic_target_adapter_tf.py"),
    Path("bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py"),
]
combined = "\n".join(path.read_text() for path in paths if path.exists())
forbidden = ["GradientTape", "batch_jacobian", "tape.", "jit_compile=False"]
violations = [token for token in forbidden if token in combined]
required = [
    "xla_hmc_ready",
    "full_chain_xla_diagnostic_ready",
    "phase18_fixed_transport_hmc_mechanics_xla_compile",
]
missing = [token for token in required if token not in combined]
print({"missing": missing, "violations": violations})
raise SystemExit(1 if missing or violations else 0)
PY
```

Exact trusted GPU command, after local checks and review:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_fixed_transport_hmc_mechanics_xla_tf \
  --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json \
  --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json \
  --seed 20260707 \
  --device /GPU:0
```

Exact post-run validation commands:

```bash
python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json >/tmp/bayesfilter-phase18-mechanics-json-tool.out
python - <<'PY'
import json
from pathlib import Path

path = Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json")
payload = json.loads(path.read_text())
checks = {
    "size_lt_20mb": path.stat().st_size < 20_000_000,
    "decision": payload.get("decision") == "PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE",
    "jit_compile_true": payload.get("jit_compile") is True,
    "jit_compile_false_not_run": payload.get("jit_compile_false_runtime_executed") is False,
    "use_xla": payload.get("use_xla") is True,
    "training_not_run": payload.get("training_executed") is False,
    "hmc_sampling_not_run": payload.get("hmc_sampling_or_tuning_executed") is False,
    "samples_not_run": payload.get("external_sample_generation_executed") is False,
    "finite_value": payload.get("finite_checks", {}).get("mechanics_value_finite") is True,
    "finite_score": payload.get("finite_checks", {}).get("mechanics_score_finite") is True,
    "compile_time_recorded": payload.get("compile_time_proxy_seconds", -1.0) >= 0.0,
}
failed = [name for name, passed in checks.items() if not passed]
print({"checks": checks, "failed": failed})
raise SystemExit(1 if failed else 0)
PY
git diff --check -- bayesfilter/ssm/target_builder.py bayesfilter/testing/lgssm_generic_target_adapter_tf.py bayesfilter/testing/neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_general_ssm_target_builder.py tests/test_lgssm_generic_target_adapter_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the Phase 17 frozen affine payload be bound to the current LGSSM generic SSM adapter as an accepted fixed-transport HMC mechanics target and compiled with trusted GPU `jit_compile=True` without running HMC chains? |
| Baseline/comparator | Phase 17 payload/validation, current LGSSM target signatures, Phase 15 manual-score trusted GPU/XLA compile gate, and current fail-closed fixed-transport mechanics authority policy. |
| Primary criterion | Either a pass artifact records accepted base and fixed-transport value/score XLA authority, trusted GPU `jit_compile=True` mechanics compile success, finite mechanics value/score, timing/size proxies, and no forbidden runtime actions; or a blocker records exact error/provenance without fallback. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence for compile success, hidden training, hidden HMC sampling/tuning, hidden external sample generation, fallback/GradientTape authority promotion, target/adapter/payload signature mismatch, nonfinite mechanics, malformed/oversized artifact, unsupported readiness/scientific/product claim. |
| Explanatory diagnostics | First/second call timing, compile-time proxy, concrete graph bytes, HLO text bytes, value/score capability manifests, mechanics manifest, payload hashes. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad nonlinear SSM validity, CPU multicore harness readiness, or scientific validity. |
| Artifact | Phase 18 diagnostic JSON, result/blocker, changed helper/tests, and Phase 19 subplan if Phase 18 passes. |

## Forbidden Claims/Actions

- Do not run `jit_compile=false` runtime diagnostics.
- Do not run NeuTra training or optimizer updates.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not use stale Phase 10/11 non-XLA artifacts.
- Do not use DSGE/c603.
- Do not change default policy.
- Do not claim HMC, posterior, production, default, broad XLA, or scientific
  readiness.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 18 writes a pass/blocker result;
- any pass records trusted GPU execution, `jit_compile=True`, `use_xla=True`,
  accepted value/score authority, finite mechanics values/scores, compile timing
  and size proxies;
- any blocker records exact error/provenance and no fallback run;
- no training, HMC sampling/tuning, or external sample generation occurred;
- Phase 19 subplan states the CPU-hidden multicore chain harness boundary and
  keeps chain execution separate from GPU training.

## Stop Conditions

Stop if:

- trusted GPU access is unavailable;
- the authority repair would require promoting a fallback/GradientTape route;
- any runtime evidence would require `jit_compile=false`;
- target, adapter, payload, or transport signatures change unexpectedly;
- mechanics diagnostics are nonfinite;
- artifact JSON is malformed or too large;
- the phase would need HMC sampling/tuning, training, or external sample
  generation;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 18 can use only the Phase 17 payload and current signatures. |
| Proxy promotion | Mechanics compile success cannot imply chain convergence or posterior correctness. |
| Hidden assumption | XLA authority must be explicit opt-in and backed by Phase 15 evidence, not a global generic-adapter default. |
| Environment mismatch | Runtime compile evidence requires trusted GPU execution. |
| Boundary leak | HMC sampling/tuning, training, samples, and `jit_compile=false` are forbidden. |
| Artifact mismatch | Diagnostic JSON must preserve payload hashes, signatures, timing, size proxies, and nonclaims. |

Audit status: draft ready for review before implementation/runtime execution.
