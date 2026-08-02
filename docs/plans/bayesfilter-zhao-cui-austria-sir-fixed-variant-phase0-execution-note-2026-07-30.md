# Zhao-Cui Austria SIR Fixed-Variant Phase 0 Execution Note

Date: 2026-07-30

Status: `AUTHORIZED_BOUNDED_CPU_ONLY_RECONSTRUCTION_AUDIT`

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.

## Question

Can the exact committed P88 T1 artifact be reconstructed as the same
TensorFlow squared-TT density and then admitted as a complete fixed-TTSIRT T1
retained branch with a T2 previous-marginal boundary?

## Evidence Contract

| Field | Contract |
|---|---|
| Baseline | P88 JSON with SHA-256 `ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e` |
| Primary criterion | Exact core hashes/shapes, basis/measure/tau/floors, and reconstructed normalizers match; complete transport identity exists for T1 retained-object construction and T2-boundary evaluation. |
| Hard veto | P88 hash mismatch, density mismatch, missing frame/transport/reference-sample identity, estimator substitution, or fabricated T2 cores. |
| Explanatory only | Fit/holdout residuals, historical P90 fixture, and rank/degree labels. |
| Nonclaim | No active-data value, score, T2 fit, T20, GPU, HMC, correctness, or production readiness. |
| Result artifacts | Attempt 1: `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-2026-07-30.json`; repaired attempt 2: `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-attempt02-2026-07-30.json`; matching Markdown result. |

## Skeptical Audit

The P88 artifact visibly serializes density cores and density configuration but
does not visibly serialize the affine frame arrays, complete TTSIRT transport
configuration, frozen reference samples, or a retained branch identity. The
audit therefore executes only a fail-closed reconstruction check. It must emit
`BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE` if those fields are absent;
it must not select replacement defaults.

Audit verdict: `PASS_FOR_FAIL_CLOSED_PHASE0_EXECUTION_ONLY`.

## Command And Budget

Intentional CPU-only execution, with GPU devices hidden before TensorFlow
import:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_zhao_cui_austria_sir_fixed_variant_phase0.py -q
CUDA_VISIBLE_DEVICES=-1 python -m scripts.run_zhao_cui_austria_sir_fixed_variant_phase0 --repository-root . --output docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-attempt02-2026-07-30.json
```

Budget: one focused test process, expected under five minutes and 4 GiB. No
training, network, package mutation, GPU, HMC, or long experiment is allowed.

Launch 0 used direct script execution and failed before the audit because the
repository root was absent from Python's import path. No artifact was written.
The module-form command above is the localized harness repair; the target,
method, evidence gates, and budget are unchanged.

Structured attempt 1 wrote the unambiguous scientific fields but used one
`artifact_path` key for both the P88 input and result output. That JSON remains
preserved as structured attempt-1 evidence. Structured attempt 2 uses a
repaired schema with distinct
`p88_artifact_path` and `result_artifact_path` fields to the fresh attempt-2
path above; no numerical method or criterion changed.
