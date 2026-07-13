# Phase 17 Subplan: Frozen GPU/XLA-Trained Affine Payload

Date: 2026-07-08

## Phase Objective

Package the Phase 16 GPU/XLA-trained affine NeuTra state into the frozen affine
payload schema and validate loader/reference behavior against the current
manual-score LGSSM target signatures.

This is packaging and loader validation only. It is not training, HMC
sampling/tuning, external sample generation, posterior validation, or production
promotion.

## Entry Conditions Inherited From Previous Phase

- Phase 16 passed bounded trusted GPU/XLA training with `jit_compile=True`.
- Phase 16 training-state artifact:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json`.
- Phase 16 file SHA-256:
  `727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b`.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Old Phase 10/11 non-XLA artifacts remain stale diagnostic history only.

## Required Artifacts

- Frozen affine payload JSON under
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/`.
  Exact path:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json`.
- Validation JSON under the same artifact directory.
  Exact path:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json`.
- Phase 17 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-result-2026-07-08.md`.

## Required Checks/Tests/Reviews

- Source/config guard checks proving packaging consumes the Phase 16
  `gpu_xla_training_state` filename by default.
- `python -m py_compile` for packaging helper and tests.
- CPU-hidden pytest for source/config/loader guards only.
- Packaging command must set `CUDA_VISIBLE_DEVICES=-1`. It must not bind
  fixed-transport HMC mechanics in Phase 17. The fixed-transport XLA-HMC
  mechanics authority and compile/timing/size gate is deferred to Phase 18.
- `python -m json.tool` on payload and validation JSON if produced.
- Validation JSON must record:
  - source Phase 16 state path;
  - source file SHA-256;
  - target and adapter signatures;
  - payload hash;
  - loader signature;
  - finite forward/base value/score checks;
  - no training;
  - no fixed-transport HMC mechanics;
  - no HMC sampling/tuning;
  - no external sample generation;
  - no `jit_compile=false` runtime fallback.
- Bounded read-only review of the Phase 17 result and any next HMC/mechanics
  subplan before HMC work.

Exact local check commands before packaging:

```bash
python -m py_compile bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_gpu_affine_payload_tf.py -q
python - <<'PY'
from pathlib import Path

source = Path("bayesfilter/testing/neutra_gpu_affine_payload_tf.py").read_text()
tests = Path("tests/test_neutra_gpu_affine_payload_tf.py").read_text()
required = [
    "PHASE16_TRAINING_STATE_PATH",
    "EXPECTED_PHASE16_TRAINING_STATE_FILE_SHA256",
    "PHASE17_PAYLOAD_FILENAME",
    "PHASE17_VALIDATION_FILENAME",
    "phase17_frozen_gpu_xla_trained_affine_payload",
]
missing = [token for token in required if token not in source + tests]
forbidden = [
    "EXPECTED_PHASE10_TARGET_SIGNATURE",
    "EXPECTED_PHASE10_ADAPTER_SIGNATURE",
    "PHASE10_TRAINING_STATE_PATH",
    "PHASE11_PAYLOAD_FILENAME",
    "PHASE11_VALIDATION_FILENAME",
]
violations = [token for token in forbidden if token in source]
print({"missing": missing, "violations": violations})
raise SystemExit(1 if missing or violations else 0)
PY
```

Exact packaging command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_gpu_affine_payload_tf \
  --phase16-training-state-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json \
  --artifact-dir docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07 \
  --seed 20260707
```

Exact post-packaging validation commands:

```bash
python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json >/tmp/bayesfilter-phase17-payload-json-tool.out
python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json >/tmp/bayesfilter-phase17-validation-json-tool.out
python - <<'PY'
import json
from pathlib import Path

payload_path = Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json")
validation_path = Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_payload_validation_seed20260707.json")
expected_source = "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_training_state_seed20260707.json"
expected_sha = "727fea040502e4fcb1af2203b9a490d03ab00dca63fd756f501a5bc3c936af7b"
expected_target = "275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038"
expected_adapter = "d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900"
payload = json.loads(payload_path.read_text())
validation = json.loads(validation_path.read_text())
checks = {
    "payload_size_lt_20mb": payload_path.stat().st_size < 20_000_000,
    "validation_size_lt_20mb": validation_path.stat().st_size < 20_000_000,
    "payload_phase": payload.get("phase") == "phase17_frozen_gpu_xla_trained_affine_payload",
    "validation_decision": validation.get("decision") == "PASS_PHASE17_FROZEN_GPU_XLA_AFFINE_PAYLOAD",
    "source_path": validation.get("source_training_state_path") == expected_source,
    "source_sha": validation.get("source_training_state_file_sha256") == expected_sha,
    "target_signature": validation.get("target_signature") == expected_target,
    "adapter_signature": validation.get("adapter_signature") == expected_adapter,
    "training_not_run": validation.get("training_executed") is False,
    "hmc_not_run": validation.get("hmc_executed") is False,
    "samples_not_run": validation.get("external_sample_generation_executed") is False,
    "mechanics_not_run": validation.get("fixed_transport_hmc_mechanics_executed") is False,
    "mechanics_deferred": validation.get("fixed_transport_hmc_mechanics_deferred_to_phase18") is True,
    "mechanics_manifest_absent": validation.get("mechanics_manifest") is None,
    "jit_compile_runtime_not_run": validation.get("jit_compile_runtime_executed") is False,
    "jit_compile_false_not_run": validation.get("jit_compile_false_runtime_executed") is False,
    "finite_checks": all(validation.get("finite_checks", {}).values()),
}
failed = [name for name, passed in checks.items() if not passed]
print({"checks": checks, "failed": failed})
raise SystemExit(1 if failed else 0)
PY
git diff --check -- bayesfilter/testing/neutra_gpu_affine_payload_tf.py tests/test_neutra_gpu_affine_payload_tf.py docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase17-frozen-gpu-xla-affine-payload-subplan-2026-07-08.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter package the Phase 16 GPU/XLA-trained affine state as a frozen payload and reload it against the current manual-score LGSSM target signatures without running HMC mechanics? |
| Baseline/comparator | Phase 16 training-state artifact and current frozen affine artifact loader. |
| Primary criterion | Packaging writes payload and validation JSON, loader accepts the payload with matching target signature, finite forward/base value/score checks pass, exact Phase 16 source hash is recorded, fixed-transport HMC mechanics are not run, and stale Phase 10/11 artifacts are not used. |
| Veto diagnostics | Source state not Phase 16/XLA, target or adapter signature mismatch, malformed payload, nonfinite loader/reference diagnostics, hidden training, hidden fixed-transport HMC mechanics, hidden HMC sampling/tuning, hidden sample generation, `jit_compile=false` fallback, oversized artifacts, unsupported readiness/scientific/product claims. |
| Explanatory diagnostics | Payload hash, loader signature, forward/logdet probes, base value/score checks, reference residuals. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad nonlinear SSM validity, or scientific validity. |
| Artifact | Phase 17 payload JSON, validation JSON, and result/blocker note. |

## Forbidden Claims/Actions

- Do not train NeuTra.
- Do not bind fixed-transport HMC mechanics.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not use stale Phase 10/11 non-XLA artifacts.
- Do not run or endorse `jit_compile=false` fallback.
- Do not use DSGE/c603.
- Do not claim HMC, posterior, production, default, or scientific readiness.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 17 writes a pass/blocker result;
- payload and validation JSON are parseable and below `20 MB`;
- validation records the Phase 16 source artifact hash;
- target and adapter signatures match the current Phase 16 signatures;
- no training, fixed-transport HMC mechanics, HMC sampling/tuning, or external
  sample generation occurred;
- next subplan states whether it will repair/validate fixed-transport HMC
  mechanics authority, run a reference validation gate, or stop.

## Stop Conditions

Stop if:

- Phase 16 source artifact is missing, malformed, oversized, or stale;
- target or adapter signatures change unexpectedly;
- payload loader rejects the artifact;
- finite checks fail;
- packaging would need training, fixed-transport HMC mechanics, HMC, or
  external sample generation;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Only the Phase 16 XLA-trained state is an admissible source. |
| Proxy promotion | Loader/reference finite checks do not imply posterior/HMC readiness. |
| Hidden fallback | Validation must record no `jit_compile=false` fallback and no stale artifact use. |
| Artifact mismatch | Payload and validation JSON must preserve hashes, signatures, and source path. |
| Boundary leak | Training, HMC sampling/tuning, and external sample generation are forbidden. |

Audit status: ready for review before execution.
