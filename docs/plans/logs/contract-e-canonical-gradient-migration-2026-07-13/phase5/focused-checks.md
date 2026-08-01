# Phase 5 Focused Check Log

Date: 2026-07-14

All TensorFlow checks in this file deliberately set `CUDA_VISIBLE_DEVICES=-1`
before import. They are CPU reference/debug evidence, not GPU evidence.

## Tests

```text
python -m pytest -q tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py
8 passed, 2 warnings in 58.89s
exit: 0
```

After the final `B=1,T=1` active-reset test was added, the complete Phase 0-5
compatibility union was rerun:

```text
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 python -m pytest -q \
  tests/highdim/test_ledh_contract_e_cloud_reset_phase3.py \
  tests/test_contract_e_phase1_normative_math.py \
  tests/test_contract_e_cholesky_ridge_reset.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py \
  tests/highdim/test_ledh_contract_e_schema_v2_factory.py \
  tests/highdim/test_ledh_contract_e_streaming_phase4.py \
  tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py

150 passed, 2 warnings in 100.20s
exit: 0
```

Warnings were the existing TensorFlow Probability `distutils.version`
deprecations.

## Structured Certificates

Exact same-private-primal derivative:

```text
status: ZERO_ULP_SAME_PRIVATE_PRIMAL_CORE_PASSED
maximum_per_batch_ulp_distance: 0
maximum_aggregate_ulp_distance: 0
all_charts_valid: true
parameter_direction_axis_final_and_size_five: true
```

CPU-XLA same-callable v2:

```text
status: EXECUTED_ENGINEERING_CERTIFICATE
jit_compile: true
one_concrete_value_and_score_callable: true
center_bitwise_identity: true
all_endpoint_charts_valid: true
all_endpoint_branches_match_center: true
```

The v1 CPU-XLA artifact is intentionally retained with
`all_endpoint_branches_match_center: false`. The branch-only v2 fixture repair
is documented separately.

## Static Checks

Python compilation passed for the owned module, focused tests, and both
certificate harnesses. All Phase 5 JSON files parsed, recorded hashes matched,
source-prohibition tests passed, and scoped `git diff --check` passed.

The first direct certificate-runner launch failed before output because the
repository root was not on the script import path. The harness was repaired to
insert its resolved repository root before importing `bayesfilter`; no failed
numerical artifact was produced. The later v1 execution was the first
numerical same-callable artifact and is preserved as the branch-instability
failure.

## Interpretation

The tiny same-core engineering derivative gate passes exactly. General
numerical adequacy, full-time GPU feasibility, Kalman equivalence, nonlinear
validity, admission, HMC, leaderboard, and release claims remain blocked.
