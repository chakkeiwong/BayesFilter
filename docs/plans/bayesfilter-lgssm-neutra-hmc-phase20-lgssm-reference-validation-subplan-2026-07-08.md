# Phase 20 Subplan: LGSSM Reference HMC Validation

Date: 2026-07-08

## Phase Objective

Run bounded CPU-hidden multicore fixed-transport LGSSM NeuTra HMC chains and
compare retained samples against a deterministic LGSSM quadrature reference
posterior under
predeclared diagnostics.

This is the first proper posterior-reference validation phase in the Phase
17-21 program. It may classify the LGSSM reference target as passing or blocked
for repair under this narrow fixture only. It must not claim sampler
superiority, production readiness, default readiness, nonlinear SSM validity,
DSGE/c603 validity, or scientific promotion.

## Entry Conditions Inherited From Previous Phase

- Phase 17 passed frozen GPU/XLA-trained affine payload packaging.
- Phase 18 passed trusted GPU/XLA fixed-transport mechanics compile with
  `jit_compile=True`.
- Phase 19 passed CPU-hidden multicore worker-harness boundary checks and
  wrote:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-result-2026-07-08.md`.
- Phase 19 harness JSON exists and records:
  - `cuda_visible_devices = "-1"`;
  - deterministic worker seeds and return codes;
  - no training;
  - no GPU sample generation;
  - no `jit_compile=false` runtime;
  - no posterior validation.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Fixed-transport adapter signature from Phase 18:
  `db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9`.
- Any HMC sampling or retained-chain generation in this phase must run with
  `CUDA_VISIBLE_DEVICES=-1`.
- No runtime fallback to `jit_compile=false` is allowed.

## Required Artifacts

- Phase 20 validation helper or runner, expected path:
  `bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py`.
- Focused tests, expected path:
  `tests/test_neutra_lgssm_reference_hmc_validation_tf.py`.
- Phase 20 HMC validation JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json`.
- Private retained sample shards, if retained samples exceed metadata-only
  reporting. Public result artifacts may contain sample summaries and hashes
  only.
- Phase 20 result or blocker:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-result-2026-07-08.md`.
- Refreshed Phase 21 subplan if Phase 20 changes the readiness decision inputs.

The validation JSON must record:

- payload path and payload SHA-256;
- Phase 18 diagnostic path and SHA-256;
- Phase 19 harness path and SHA-256;
- target, adapter, fixed-transport adapter, payload, and transport signatures;
- deterministic LGSSM quadrature reference posterior mean/covariance source
  and hash;
- HMC command, worker count, chain count, seed ledger, process ids, return
  codes, and CPU-hidden environment;
- `jit_compile=true` policy and `jit_compile_false_runtime_executed=false`;
- retained sample count per chain and total retained sample count;
- acceptance, finite-value, finite-sample, and error/divergence diagnostics;
- R-hat and ESS only when enough chain/sample evidence exists; otherwise mark
  them unavailable and explain why;
- posterior mean and covariance residuals against the exact LGSSM reference;
- explicit uncertainty limitations;
- nonclaims.

## Required Checks/Tests/Reviews

- Source/config guard checks proving:
  - CPU-hidden execution is required for chain generation;
  - `jit_compile=false` is rejected;
  - full-chain diagnostic authority is explicitly declared for this phase only;
  - worker seeds are deterministic and distinct;
  - worker metadata includes return codes and environment;
  - posterior-reference diagnostics are separated from sampler/product/science
    claims;
  - no `GradientTape`, `batch_jacobian`, or runtime tape path exists in the
    admitted Phase 20 helper.
- `python -m py_compile` for the helper and tests.
- CPU-hidden focused pytest for validation helper/source behavior.
- Source scan preserving no `GradientTape`, `batch_jacobian`, `tape.`, or
  `jit_compile=False` in the admitted Phase 20 helper path.
- Bounded CPU-hidden HMC validation run with `jit_compile=True`; if XLA-on-CPU
  full-chain HMC cannot compile, write a blocker and do not rerun with
  `jit_compile=false`.
- `python -m json.tool` on the validation JSON if produced.
- Field-validation script for the required JSON fields.
- `git diff --check`.
- Read-only review of the Phase 20 result and Phase 21 readiness-decision
  subplan before any readiness decision is finalized.

Exact local check commands before the validation run:

```bash
python -m py_compile bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_lgssm_reference_hmc_validation_tf.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_cpu_multicore_hmc_chain_harness_tf.py tests/test_neutra_fixed_transport_hmc_mechanics_xla_tf.py -q
python - <<'PY'
from pathlib import Path

paths = [
    Path("bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py"),
]
combined = "\n".join(path.read_text() for path in paths if path.exists())
forbidden = ["GradientTape", "batch_jacobian", "tape.", "jit_compile=False"]
violations = [token for token in forbidden if token in combined]
required = [
    "CUDA_VISIBLE_DEVICES",
    "jit_compile",
    "full_chain",
    "reference_posterior",
    "posterior_validation_executed",
]
missing = [token for token in required if token not in combined]
print({"missing": missing, "violations": violations})
raise SystemExit(1 if missing or violations else 0)
PY
```

Exact bounded validation command, after review:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m bayesfilter.testing.neutra_lgssm_reference_hmc_validation_tf \
  --payload-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json \
  --phase18-diagnostic-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json \
  --phase19-harness-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json \
  --output-path docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json \
  --seed 20260707 \
  --worker-count 2 \
  --chain-count 4 \
  --num-results 64 \
  --num-burnin-steps 64 \
  --num-leapfrog-steps 4 \
  --step-size 0.05 \
  --jit-compile true
```

Exact post-run validation commands:

```bash
python -m json.tool docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json >/tmp/bayesfilter-phase20-validation-json-tool.out
python - <<'PY'
import json
from pathlib import Path

path = Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json")
payload = json.loads(path.read_text())
checks = {
    "size_lt_20mb": path.stat().st_size < 20_000_000,
    "decision": payload.get("decision") in {
        "PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION",
        "BLOCK_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION",
    },
    "cpu_hidden": payload.get("cuda_visible_devices") == "-1",
    "jit_compile_true": payload.get("jit_compile") is True,
    "jit_compile_false_not_run": payload.get("jit_compile_false_runtime_executed") is False,
    "training_not_run": payload.get("training_executed") is False,
    "gpu_sample_generation_not_run": payload.get("gpu_sample_generation_executed") is False,
    "posterior_validation_run": payload.get("posterior_validation_executed") is True,
    "reference_present": bool(payload.get("reference_posterior")),
    "sample_diagnostics_present": bool(payload.get("sample_diagnostics")),
}
failed = [name for name, passed in checks.items() if not passed]
print({"checks": checks, "failed": failed})
raise SystemExit(1 if failed else 0)
PY
git diff --check -- bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py tests/test_neutra_lgssm_reference_hmc_validation_tf.py docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do bounded CPU-hidden multicore fixed-transport LGSSM NeuTra HMC chains, run with `jit_compile=True`, agree with the deterministic quadrature reference posterior closely enough to pass the predeclared LGSSM fixture gate? |
| Baseline/comparator | Deterministic 2D quadrature reference posterior for the current static QR LGSSM fixture, plus Phase 17 payload, Phase 18 mechanics compile diagnostic, Phase 19 CPU worker harness, and current signatures. |
| Primary criterion | A pass requires finite retained samples, no worker errors, no `jit_compile=false`, no GPU sample generation, acceptable acceptance/error diagnostics, and posterior mean/covariance residuals within the predeclared tolerance recorded by the helper. Otherwise write a blocker with exact diagnostics. |
| Promotion veto diagnostics | Nonfinite samples or target values, worker failure, CPU-hidden violation, `jit_compile=false` runtime, hidden training, GPU sample generation, missing reference posterior, signature/provenance mismatch, malformed/oversized artifact, or posterior residual above tolerance. |
| Explanatory diagnostics | Acceptance, R-hat/ESS when available, MCSE or bootstrap intervals if implemented, chain wall times, compile/runtime split if observable, posterior mean/cov residuals, sample covariance eigenvalues, worker return codes. |
| Not concluded | Sampler superiority, optimal tuning, production readiness, default readiness, nonlinear SSM validity, DSGE/c603 validity, broad NeuTra validity, or scientific validity. |
| Artifact | Phase 20 helper/tests, validation JSON, result/blocker, and refreshed Phase 21 subplan. |

## Forbidden Claims/Actions

- Do not run `jit_compile=false`.
- Do not run NeuTra training or optimizer updates.
- Do not use GPU for sample or chain generation.
- Do not use DSGE/c603.
- Do not change default policy.
- Do not claim sampler superiority, product readiness, default readiness,
  nonlinear SSM validity, or scientific validity.
- Do not rank methods using descriptive diagnostics alone.
- Do not treat a short/few-chain pass as broad HMC convergence beyond this
  LGSSM reference fixture.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 20 writes a pass/blocker result;
- any pass records CPU-hidden worker metadata, deterministic seeds, retained
  sample count, reference posterior, and residual diagnostics;
- no `jit_compile=false`, GPU sample generation, or training occurred;
- the result states whether R-hat/ESS and uncertainty diagnostics were
  available or not available;
- Phase 21 subplan declares exact decision categories:
  `LGSSM_REFERENCE_HMC_READY`, `BLOCKED_FOR_REPAIR`, or
  `INSUFFICIENT_EVIDENCE_NO_PROMOTION`;
- Phase 21 subplan preserves all nonclaims and boundary conditions.

## Stop Conditions

Stop if:

- Phase 19 did not pass or did not produce the required harness artifact;
- CPU-hidden full-chain execution is unavailable;
- HMC validation requires `jit_compile=false`;
- full-chain diagnostic authority cannot be explicitly scoped to Phase 20;
- the deterministic LGSSM quadrature reference posterior cannot be computed or
  hashed;
- worker metadata cannot record seeds, return codes, environment, and artifacts;
- payload, Phase 18, or Phase 19 provenance is missing or mismatched;
- posterior diagnostics are malformed or ambiguous;
- artifacts are malformed or too large;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Deterministic LGSSM quadrature reference posterior is the comparator; Phase 17/18/19 artifacts provide only provenance and mechanics/harness gates. |
| Proxy promotion | Acceptance, R-hat/ESS, and runtime are not promotion criteria unless posterior residuals also pass and vetoes remain clear. |
| Missing stop condition | Any need for `jit_compile=false`, GPU sampling, hidden training, or missing reference posterior is a blocker. |
| Hidden assumption | Full-chain XLA diagnostic authority must be explicit and scoped to Phase 20 only. |
| Environment mismatch | Workers must run with `CUDA_VISIBLE_DEVICES=-1`; GPU-visible chain generation is a veto. |
| Artifact mismatch | Hashes/signatures for payload, Phase 18, Phase 19, target, adapter, and transport are mandatory. |

Audit status: repaired after skeptical audit. The original draft used "exact
LGSSM reference posterior" language. That was too strong for the nonlinear
parameter posterior. Phase 20 now uses a deterministic 2D quadrature reference
over the exact LGSSM likelihood target and must not call that analytic/exact
posterior evidence.
