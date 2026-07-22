# Phase 8 Subplan: Cross-Model Score Wiring Regression Gate

Date: 2026-07-10

## Phase Objective

Add and run a focused cross-model regression gate proving that all six current
LEDH score adapters expose compact no-time-history score wiring, preserve their
row-specific target semantics and parameter contracts, default to production
precision, and quarantine historical/manual routes from full admission.

This is a wiring and smoke-test phase. Full-row GPU score-memory evidence is
reserved for Phase 9.

## Entry Conditions

- LGSSM, fixed-SIR, predator-prey, actual-SV, generalized-SV, and KSC-SV have
  model-specific focused contract tests.
- The shared score contract admits only row-matched compact provenance with
  production precision, same-scalar correctness, and trusted memory evidence.
- Generalized-SV and KSC-SV preserve sequential per-seed compact component
  evaluation to avoid unreviewed peak-memory multiplication.
- Existing inclusive leaderboard integration already validates admitted score
  artifacts with `require_admitted=True`; Phase 8 must not duplicate or weaken
  that policy.

## Required Artifacts

- Cross-model tests:
  `tests/highdim/test_ledh_score_wiring_phase8_cross_model.py`
- Phase 8 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-result-2026-07-10.md`
- Phase 9 subplan:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md`
- Review bundle for the Phase 8 result and Phase 9 subplan.

## Required Checks

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  tests/highdim/test_ledh_score_wiring_phase8_cross_model.py
```

Run four bounded CPU-hidden shards, stopping on the first failure:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_wiring_phase8_cross_model.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py
```

The cross-model file must verify at least:

- exact row-to-compact-provenance mapping for all six rows;
- exact target policy and parameter order inherited from admitted value
  artifacts;
- production dtype/TF32 defaults in every score CLI or default path;
- default diagnostic source does not call historical/manual score routes;
- all five nonlinear rows use sequential compact per-seed wrappers;
- tiny or synthetic fixtures are never represented as runtime full admission;
- KSC exact-native actual-SV nonclaim and actual-SV transformed-target boundary.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do all six adapters satisfy one coherent compact-score wiring contract before trusted GPU memory runs? |
| Baseline/comparator | Current model-specific adapters/tests, admitted forward artifacts, shared score contract, and existing validator-based leaderboard integration. |
| Primary criterion | Cross-model invariants and all model-specific focused suites pass without changing target semantics, parameter contracts, admission policy, or sequential seed schedule. |
| Veto diagnostics | Any default score path calls a historical/manual route; wrong row compact provenance; non-production default; target/parameter mismatch; KSC exact-native overclaim; any nonlinear compact component batches all seeds; tiny fixture promoted as runtime evidence. |
| Explanatory diagnostics | Per-file test counts and CPU-hidden tiny smoke behavior. |
| Not concluded | Full-row GPU memory, score admission, runtime ranking, HMC readiness, posterior correctness, leaderboard completion, or scientific validity. |
| Artifact | Cross-model test file and Phase 8 result. |

## Forbidden Actions And Claims

- Do not launch GPU/full-row score commands in Phase 8.
- Do not edit target density math, parameter transformations, or value
  artifacts to make cross-model tests pass.
- Do not promote synthetic contract fixtures or tiny runs to score admission.
- Do not rebuild or claim completion of the leaderboard in Phase 8.
- Do not rank stochastic or runtime results.

## Exact Handoff Conditions

Advance to Phase 9 only if:

- cross-model and all model-specific focused tests pass;
- no target, provenance, precision, sequential-seed, or admission veto fires;
- Phase 8 result records CPU-only scope and explicit nonclaims;
- Phase 9 trusted GPU score-memory subplan exists and is reviewed.

## Stop Conditions

- A cross-model invariant conflicts with an admitted row's target or parameter
  contract.
- Passing requires changing score math or admission criteria after seeing
  results.
- Any historical route remains default/full-admissible and cannot be repaired
  locally.
- Any individual shard is expected to exceed five minutes; split it further
  before execution.
- Review fails to converge after five rounds.

## Skeptical Plan Audit

- Wrong baseline risk: use current adapters and admitted value artifacts, not
  July 8 raw/tiny score candidates.
- Proxy risk: passing cross-model tests is not full-row memory or score
  admission evidence.
- Hidden assumption risk: the six rows have distinct target and coordinate
  contracts; uniform wiring must not imply uniform likelihood semantics.
- Environment risk: GPUs are intentionally hidden; CUDA initialization noise
  is not GPU evidence.
- Artifact sufficiency: the test suite answers wiring consistency only and
  hands full-scale memory to Phase 9.

Audit result: allowed after fresh review of the repaired Phase 6 result, Phase
7 result, and this subplan.

### Phase 8 execution-policy repair audit

The Phase 9 readiness audit found that fixed-SIR, predator-prey, and actual-SV
still passed every full-row seed into one compact component invocation. That
could multiply peak GPU memory and conflicts with the sequential-seed baseline
already enforced for generalized-SV and KSC-SV. This is a repair trigger for
execution policy, not evidence against the score recurrence or target scalar.

The repair is allowed only if it preserves seed order, concatenates the same
per-seed log-likelihoods, averages the same per-seed gradients, evaluates
fixed-SIR finite-difference values sequentially, and adds two-seed invocation
tests. GPU execution and full admission remain vetoed until all Phase 8 shards
pass again.
