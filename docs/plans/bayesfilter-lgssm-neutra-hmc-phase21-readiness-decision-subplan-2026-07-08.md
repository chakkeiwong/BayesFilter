# Phase 21 Subplan: HMC Readiness Decision Gate

Date: 2026-07-08

## Phase Objective

Classify the Phase 20 LGSSM reference-validation evidence and decide whether
the fixed-transport LGSSM NeuTra-HMC path is ready for this reference target,
blocked for repair, or insufficiently evidenced for promotion.

This phase is primarily a decision and evidence-classification phase. It does
not run new training, GPU jobs, broad HMC tuning, nonlinear SSM validation,
DSGE/c603 work, or product/scientific promotion. A tiny local parser/check may
run only to verify artifact consistency.

## Entry Conditions Inherited From Previous Phase

- Phase 17 passed frozen GPU/XLA-trained affine payload packaging.
- Phase 18 passed trusted GPU/XLA fixed-transport mechanics compile.
- Phase 19 passed CPU-hidden multicore worker-harness boundary checks.
- Phase 20 wrote:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-result-2026-07-08.md`.
- Phase 20 validation JSON exists or Phase 20 blocker result explains why it
  could not be produced:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json`.
- No Phase 17-20 artifact records `jit_compile=false`, hidden training, GPU
  sample generation, or unsupported readiness/scientific/product claims.

## Required Artifacts

- Phase 21 readiness result:
  `docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-result-2026-07-08.md`.
- Optional machine-readable decision JSON:
  `docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase21_readiness_decision_seed20260707.json`.
- Updated master program, runbook, and ledger statuses.
- If blocked, a repair subplan or explicit stop handoff naming the smallest
  next discriminating repair.

The Phase 21 result must include:

- decision table;
- inference-status table;
- hard veto screen;
- statistically supported ranking status;
- descriptive-only differences;
- default-readiness status;
- next evidence needed;
- exact list of artifacts inspected, with hashes where applicable;
- what is and is not being concluded.

## Required Checks/Tests/Reviews

- Parse Phase 17, Phase 18, Phase 19, and Phase 20 JSON artifacts if present.
- Verify all required nonclaim and boundary fields:
  - no `jit_compile=false`;
  - no training after Phase 16;
  - no GPU sample generation;
  - CPU-hidden chain/sample generation;
  - exact LGSSM reference comparator present for any readiness pass.
- Verify the Phase 20 result decision matches the Phase 20 JSON decision.
- Verify Phase 21 result does not claim product/default/scientific/broad HMC
  readiness.
- `git diff --check` on Phase 21 docs.
- Read-only review of the Phase 21 decision result before final closeout.

Exact local consistency check:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = {
    "phase17_payload": Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_gpu_xla_frozen_payload_seed20260707.json"),
    "phase18": Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json"),
    "phase19": Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase19_cpu_multicore_hmc_chain_harness_seed20260707.json"),
    "phase20": Path("docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07/lgssm_static_qr_exact_kalman_affine_neutra_phase20_lgssm_reference_hmc_validation_seed20260707.json"),
}
loaded = {name: json.loads(path.read_text()) for name, path in paths.items() if path.exists()}
checks = {
    "phase17_present": "phase17_payload" in loaded,
    "phase18_present": "phase18" in loaded,
    "phase19_present": "phase19" in loaded,
    "phase20_present_or_blocker_expected": "phase20" in loaded,
}
if "phase18" in loaded:
    checks["phase18_jit_true"] = loaded["phase18"].get("jit_compile") is True
    checks["phase18_no_jit_false"] = loaded["phase18"].get("jit_compile_false_runtime_executed") is False
if "phase19" in loaded:
    checks["phase19_cpu_hidden"] = loaded["phase19"].get("cuda_visible_devices") == "-1"
    checks["phase19_no_training"] = loaded["phase19"].get("training_executed") is False
    checks["phase19_no_gpu_samples"] = loaded["phase19"].get("gpu_sample_generation_executed") is False
if "phase20" in loaded:
    checks["phase20_cpu_hidden"] = loaded["phase20"].get("cuda_visible_devices") == "-1"
    checks["phase20_jit_true"] = loaded["phase20"].get("jit_compile") is True
    checks["phase20_no_jit_false"] = loaded["phase20"].get("jit_compile_false_runtime_executed") is False
    checks["phase20_reference_present"] = bool(loaded["phase20"].get("reference_posterior"))
failed = [name for name, passed in checks.items() if not passed]
print({"checks": checks, "failed": failed})
raise SystemExit(1 if failed else 0)
PY
git diff --check -- docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-subplan-2026-07-08.md docs/plans/bayesfilter-lgssm-neutra-hmc-phase21-readiness-decision-result-2026-07-08.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Given Phase 20 evidence, what is the narrow readiness classification for the LGSSM fixed-transport NeuTra-HMC path? |
| Baseline/comparator | Phase 20 validation JSON and result, exact LGSSM reference posterior, and Phase 17-19 provenance artifacts. |
| Primary criterion | Emit exactly one of `LGSSM_REFERENCE_HMC_READY`, `BLOCKED_FOR_REPAIR`, or `INSUFFICIENT_EVIDENCE_NO_PROMOTION`, with veto status and next action. |
| Veto diagnostics | Any Phase 17-20 `jit_compile=false` runtime, hidden training, GPU sample generation, missing reference posterior for a pass, posterior residual failure, nonfinite samples, worker errors, malformed artifacts, or unsupported product/default/scientific claims. |
| Explanatory diagnostics | Acceptance, R-hat/ESS when available, posterior mean/cov residuals, uncertainty limitations, compile/runtime notes, worker metadata. |
| Not concluded | Sampler superiority, production readiness, default readiness, nonlinear SSM validity, DSGE/c603 validity, broad NeuTra validity, or scientific validity. |
| Artifact | Phase 21 result, optional decision JSON, updated ledger/runbook/master, and repair subplan or stop handoff if blocked. |

## Forbidden Claims/Actions

- Do not run NeuTra training.
- Do not run new HMC sampling/tuning unless Phase 20 explicitly left a
  documented parser-only ambiguity that cannot be resolved from artifacts.
- Do not run GPU jobs.
- Do not use DSGE/c603.
- Do not change default policy.
- Do not rank methods using descriptive diagnostics alone.
- Do not claim production, default, product, broad HMC, nonlinear SSM, or
  scientific readiness.

## Exact Next-Phase Handoff Conditions

If the decision is `LGSSM_REFERENCE_HMC_READY`:

- hand off to a new human-reviewed program for optional longer-chain LGSSM
  replication or a first non-LGSSM target;
- preserve that readiness applies only to the named LGSSM fixture and exact
  Phase 17-20 artifacts.

If the decision is `BLOCKED_FOR_REPAIR`:

- write a repair subplan naming the smallest failing mechanism:
  authority/compile, worker harness, tuning, posterior residuals, reference
  comparator, or artifact integrity;
- do not continue to broader targets.

If the decision is `INSUFFICIENT_EVIDENCE_NO_PROMOTION`:

- write the missing-evidence list and the smallest next validation ladder;
- do not claim readiness.

## Stop Conditions

Stop if:

- Phase 20 result is missing;
- Phase 20 JSON and result disagree;
- any required artifact is malformed;
- the evidence supports no unambiguous one-of-three decision;
- a decision would require new runtime outside this subplan;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Decision is based only on Phase 20 vs exact LGSSM reference posterior and Phase 17-19 provenance. |
| Proxy promotion | Acceptance, ESS, runtime, and compile success cannot by themselves yield readiness. |
| Missing stop condition | Ambiguous or missing artifacts force blocker/insufficient-evidence status. |
| Hidden assumption | Readiness, if any, is fixture-local and artifact-local. |
| Environment mismatch | No new GPU or sample-generation runtime is planned. |
| Artifact mismatch | JSON/result decisions must agree before closeout. |

Audit status: draft ready for review after Phase 20 result exists.
