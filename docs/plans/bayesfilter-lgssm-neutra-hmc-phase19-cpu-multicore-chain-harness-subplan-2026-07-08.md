# Phase 19 Subplan: CPU Multicore HMC Chain Harness

Date: 2026-07-08

## Phase Objective

Build a CPU-hidden multicore harness for running fixed-transport LGSSM NeuTra
HMC chains in separate worker processes, with deterministic seeds, worker
metadata, return codes, environment records, artifact paths, and fail-closed
boundary checks.

This phase is a harness and boundary phase. It may run only a bounded
mechanics/worker smoke that is explicitly marked as non-posterior evidence. It
must not claim posterior correctness, HMC convergence, sampler quality,
production readiness, or scientific validity. Phase 20 owns LGSSM reference
posterior validation.

## Entry Conditions Inherited From Previous Phase

- Phase 17 passed frozen payload packaging and loader/reference validation.
- Phase 18 passed trusted GPU/XLA fixed-transport mechanics compile gate.
- Phase 18 diagnostic JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json`.
- Phase 17 payload:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json`.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Fixed-transport adapter signature from Phase 18:
  `db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9`.
- All chain or worker execution in this phase must hide GPU with
  `CUDA_VISIBLE_DEVICES=-1`.
- No runtime fallback to `jit_compile=false` is allowed.

## Required Artifacts

- A CPU multicore harness helper, expected path:
  `bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Focused tests, expected path:
  `tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py`.
- Harness smoke JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json`.
- Phase 19 result or blocker:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-result-2026-07-08.md`.

The harness artifact must record:

- payload path and payload SHA-256;
- Phase 18 diagnostic path and SHA-256;
- target, adapter, fixed-transport adapter, payload, and transport signatures;
- CPU-hidden environment per worker;
- worker count, worker indexes, process ids, return codes, and seeds;
- `jit_compile=true` policy and `jit_compile_false_runtime_executed=false`;
- no training;
- no GPU sample generation;
- whether any HMC transition was run;
- if any bounded HMC transition is run, the exact number of chains/results/
  burnin/leapfrog steps and that the run is smoke-only;
- no posterior/readiness/scientific/product claims.

## Required Checks/Tests/Reviews

- Source/config guard checks proving:
  - CPU-hidden execution is required for worker runs;
  - `jit_compile=false` is rejected;
  - worker seeds are deterministic and distinct;
  - worker metadata includes return codes and environment;
  - GPU sample generation is forbidden;
  - Phase 19 cannot write a Phase 20 reference-validation decision.
- `python -m py_compile` for the helper and tests.
- CPU-hidden focused pytest for harness config/source behavior.
- Source scan preserving no `GradientTape`, `batch_jacobian`, `tape.`, or
  `jit_compile=False` in the admitted Phase 19 helper path.
- A bounded CPU-hidden harness smoke may be run only if it uses
  `jit_compile=True` and writes the required smoke JSON. If XLA-on-CPU HMC
  cannot compile, write a blocker; do not rerun with `jit_compile=false`.
- `python -m json.tool` on the harness JSON if produced.
- Field-validation script for the required JSON fields.
- `git diff --check`.
- Read-only review of the Phase 19 result and Phase 20 subplan before LGSSM
  reference validation.

Exact local check commands before any worker smoke:

```bash
python -m py_compile bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py tests/test_neutra_gpu_affine_payload_tf.py -q
python - <<'PY'
from pathlib import Path

paths = [
    Path("bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py"),
]
combined = "\n".join(path.read_text() for path in paths if path.exists())
forbidden = ["GradientTape", "batch_jacobian", "tape.", "jit_compile=False"]
violations = [token for token in forbidden if token in combined]
required = [
    "CUDA_VISIBLE_DEVICES",
    "jit_compile",
    "worker_count",
    "phase19_cpu_multicore_hmc_chain_harness",
]
missing = [token for token in required if token not in combined]
print({"missing": missing, "violations": violations})
raise SystemExit(1 if missing or violations else 0)
PY
```

Exact bounded smoke command, after review:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_cpu_multicore_hmc_chain_harness_tf \
  --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json \
  --phase18-diagnostic-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json \
  --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json \
  --seed 20260707 \
  --worker-count 2 \
  --chain-count 2 \
  --num-results 1 \
  --num-burnin-steps 0 \
  --num-leapfrog-steps 2 \
  --step-size 0.1 \
  --jit-compile true
```

Exact post-smoke validation commands:

```bash
python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json >/tmp/bayesfilter-phase19-harness-json-tool.out
python - <<'PY'
import json
from pathlib import Path

path = Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json")
payload = json.loads(path.read_text())
checks = {
    "size_lt_20mb": path.stat().st_size < 20_000_000,
    "decision": payload.get("decision") in {
        "PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS",
        "BLOCK_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS",
    },
    "cpu_hidden": payload.get("cuda_visible_devices") == "-1",
    "jit_compile_true": payload.get("jit_compile") is True,
    "jit_compile_false_not_run": payload.get("jit_compile_false_runtime_executed") is False,
    "training_not_run": payload.get("training_executed") is False,
    "gpu_sample_generation_not_run": payload.get("gpu_sample_generation_executed") is False,
    "posterior_validation_not_run": payload.get("posterior_validation_executed") is False,
    "worker_metadata": bool(payload.get("workers")),
}
failed = [name for name, passed in checks.items() if not passed]
print({"checks": checks, "failed": failed})
raise SystemExit(1 if failed else 0)
PY
git diff --check -- bayesfilter/testing/neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-subplan-2026-07-08.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter build a CPU-hidden multicore harness for fixed-transport LGSSM NeuTra HMC chains with deterministic worker metadata and no forbidden fallback before reference validation? |
| Baseline/comparator | Phase 17 payload, Phase 18 mechanics compile diagnostic, current LGSSM signatures, and repository policy that sample/chain generation is CPU multicore while NeuTra training is GPU. |
| Primary criterion | Either a pass artifact records CPU-hidden worker metadata, deterministic seeds, `jit_compile=True`, no forbidden actions, and a bounded smoke result or no-transition dry-run as declared; or a blocker records exact error/provenance without fallback. |
| Veto diagnostics | Any `jit_compile=false` runtime run, GPU-visible worker run, hidden training, hidden GPU sample generation, missing worker return codes/seeds/env, missing payload/Phase18 provenance, treating a smoke as posterior validation, malformed/oversized artifact, unsupported readiness/scientific/product claim. |
| Explanatory diagnostics | Worker wall times, return codes, seed ledger, tiny smoke transition outputs if run, CPU environment, command manifest. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, LGSSM reference agreement, broad nonlinear SSM validity, or scientific validity. |
| Artifact | Phase 19 helper/tests, harness JSON, result/blocker, and Phase 20 subplan if Phase 19 passes. |

## Forbidden Claims/Actions

- Do not run `jit_compile=false`.
- Do not run NeuTra training or optimizer updates.
- Do not use GPU for sample or chain generation.
- Do not run Phase 20 reference validation in this phase.
- Do not claim HMC convergence, posterior correctness, production readiness,
  default readiness, or scientific validity.
- Do not use DSGE/c603.
- Do not change default policy.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 19 writes a pass/blocker result;
- any pass records CPU-hidden worker metadata, deterministic seeds, return
  codes, and required provenance;
- no `jit_compile=false`, GPU sample generation, training, or posterior
  validation occurred;
- Phase 20 subplan declares exact LGSSM reference posterior validation
  diagnostics, chain lengths, acceptance/R-hat/ESS checks, posterior mean/cov
  residuals, uncertainty limitations, and nonclaims.

## Stop Conditions

Stop if:

- CPU-hidden worker execution is unavailable;
- worker HMC smoke requires `jit_compile=false`;
- worker metadata cannot record seeds, return codes, environment, and artifacts;
- payload or Phase 18 diagnostic provenance is missing or mismatched;
- the phase would need serious HMC validation rather than harness smoke;
- artifacts are malformed or too large;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Harness can use only Phase 17/18 artifacts and current signatures. |
| Proxy promotion | Harness smoke cannot become posterior/HMC readiness. |
| Hidden assumption | CPU multicore is the execution target for chain/sample generation; GPU remains for training/compile gates. |
| Environment mismatch | Workers must record `CUDA_VISIBLE_DEVICES=-1`; GPU-visible workers are vetoes. |
| Boundary leak | Phase 20 reference validation is forbidden in Phase 19. |
| Artifact mismatch | Worker metadata, seeds, return codes, command, and provenance are mandatory. |

Audit status: draft ready for review before implementation/runtime execution.
