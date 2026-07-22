# Phase 2 Focused Check Log

Date: 2026-07-13

Execution target: deliberate CPU-only schema/reference checks with
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. No GPU result is claimed.

## Commands And Results

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_contract_e_phase0_emergency_revocation.py \
  tests/highdim/test_ledh_contract_e_schema_v2_factory.py

104 passed, 2 warnings in 4.03s
exit: 0

python -m py_compile \
  bayesfilter/highdim/ledh_contract_e_identity.py \
  bayesfilter/highdim/ledh_forward_contract_v2.py \
  bayesfilter/highdim/ledh_score_contract_v2.py \
  bayesfilter/highdim/_contract_e_phase2_test_fixture.py \
  tests/highdim/test_ledh_contract_e_schema_v2_factory.py
exit: 0

git diff --check -- <Phase 2 source, test, and subplan paths>
exit: 0
```

The two warnings are pre-existing TensorFlow Probability deprecations of
`distutils.version`.

## What Was Checked

- deterministic identity across mapping order and repeated issuance;
- prepared-input dtype, rank, shape, value, finiteness, positivity, missing,
  and extra-field binding;
- callable source, current code object, defaults, globals, dependency closure,
  TensorFlow wrapper/JIT settings, and external package version binding;
- immutable identity records and write-once route/allowlist registries;
- inert public factory and isolation of the private test candidate factory;
- raw, lambda, partial, monkeypatched, stale, unregistered, and incomplete
  callable rejection;
- v1 forgery/revocation compatibility;
- forward/score identity, exact-forward digest, semantic, gate, and admission
  failure paths;
- canonical reconstruction without mutable input/output aliases; and
- inclusive leaderboard rejection of an unadmitted v2 candidate.

## Interpretation

The checks establish schema/factory mechanics on the tested Python process.
They do not establish a production Contract E implementation, complete v2-aware
consumer migration, numerical validity, same-scalar gradient correctness,
Kalman agreement, GPU/XLA behavior, HMC readiness, or leaderboard readiness.
