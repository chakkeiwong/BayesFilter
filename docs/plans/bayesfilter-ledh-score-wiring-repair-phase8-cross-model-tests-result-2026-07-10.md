# Phase 8 Result: Cross-Model Compact Score Wiring Gate

Date: 2026-07-10

Status: `PASSED_AFTER_ALL_NONLINEAR_SEQUENTIAL_SEED_REPAIR`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| All six LEDH rows satisfy the reviewed compact-score wiring contract. | Passed four CPU-hidden shards after repairing fixed-SIR, predator-prey, and actual-SV to evaluate compact score components sequentially per seed. | No historical/default-route, target, parameter-order, production-precision, seed-schedule, tiny-admission, or exact-native-overclaim veto fired. | The nonlinear score CLIs do not yet provide XLA-JIT, trusted-device, reset memory, or seed-shard provenance required for production GPU evidence. | Review this result and the Phase 9 harness-first GPU subplan. | No new GPU memory evidence, full score admission, runtime ranking, HMC readiness, posterior correctness, exact nonlinear likelihood correctness, leaderboard completion, or scientific validity. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Do all six adapters satisfy one coherent compact-score wiring contract before trusted GPU memory runs? |
| Baseline/comparator | Current row adapters, admitted forward artifacts, the shared score contract, and model-specific focused tests. |
| Primary criterion | Passed. Each default score path is compact; FD perturbations use value-only same-scalar routes; every nonlinear compact component is called sequentially per seed; row targets and parameter orders remain unchanged. |
| Veto diagnostics | None fired. Historical/manual routes remain outside default/full-admissible paths, production defaults are `float32` plus TF32, actual-SV and KSC targets remain distinct, and runtime full admission is not constructed by Phase 8 tests. |
| Explanatory diagnostics | Test counts and tiny CPU-hidden numerical checks only. |
| Artifact | This result, the cross-model test, model-specific tests, and the visible ledger. |

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed target | Wiring of the compact forward-sensitivity score for each row's admitted realized finite-`N` LEDH log-likelihood estimator. |
| Quantity computed | CPU-hidden tiny numerical checks plus structural and monkeypatch route tests. |
| Relationship | The tests support wiring equivalence and seed-wise aggregation at tiny scale. They do not measure full-row GPU memory or establish full-row numerical validity. |
| Support | `tests/highdim/test_ledh_score_wiring_phase8_cross_model.py` and the four post-repair shard results below. |

## Repair Trigger And Resolution

The initial Phase 8 shards passed, but the Phase 9 readiness audit found that
fixed-SIR, predator-prey, and actual-SV passed all full-row seeds into a single
compact component call. That could multiply peak GPU memory and conflicted
with the reviewed sequential-seed behavior in generalized-SV and KSC-SV.

This was an execution-policy defect, not evidence against the compact score
recurrence. The repair:

- added explicit sequential compact across-seed wrappers for fixed-SIR,
  predator-prey, and actual-SV;
- added a sequential value-only across-seed wrapper for fixed-SIR;
- routed all default coordinate-FD diagnostics through the wrappers;
- routed actual-SV `value-score-only` mode through its wrapper;
- preserved seed order, per-seed log-likelihoods, arithmetic-mean objective,
  and arithmetic-mean score;
- added two-seed invocation tests that require singleton component calls and
  forbid historical routes;
- expanded the cross-model schedule assertion to all five nonlinear rows.

No target density, parameter transformation, finite-difference tolerance,
admission schema, or historical implementation was changed by this repair.

## Local Checks

Compile check:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_wiring_phase8_cross_model.py
```

Result: passed.

Focused new schedule tests: `9 passed, 2 warnings in 5.81s`.

Required post-repair shards:

| Shard | Result |
| --- | --- |
| Shared score contract plus cross-model gate | `73 passed, 2 warnings in 2.74s` |
| LGSSM plus fixed-SIR | `39 passed, 2 warnings in 77.47s` |
| Predator-prey plus actual-SV | `45 passed, 2 warnings in 175.38s` |
| Generalized-SV plus KSC-SV | `38 passed, 2 warnings in 57.46s` |

All commands intentionally used `CUDA_VISIBLE_DEVICES=-1` and
`MPLCONFIGDIR=/tmp` before TensorFlow import.

## Engineering Correctness Ledger

| Ledger | Status |
| --- | --- |
| Route wiring | Compact wrapper is the default score base for every nonlinear row. |
| Seed execution | One compact component call per seed; aggregate is disclosed arithmetic mean. |
| FD execution | Value-only same-scalar route; fixed-SIR now also evaluates it one seed at a time. |
| Target identity | Preserved per admitted forward artifact. |
| Admission boundary | CPU/tiny tests do not create runtime full admission. |
| GPU/XLA readiness | Not yet established for nonlinear score harnesses. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus current scoped uncommitted changes |
| Commands | Compile command and four pytest shards shown above |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; TensorFlow `2.19.1`; host `DESKTOP-RF1Q5IJ` |
| CPU/GPU status | CPU-only wiring diagnostics; GPUs intentionally hidden before framework import |
| Data version | Admitted forward-scalar artifacts dated 2026-07-07 |
| Random seeds | Behavioral fixtures `81120,81121`; full-row metadata fixtures `81120` through `81124` without full execution |
| Wall time | Per-shard times shown above |
| Output artifacts | This result, tests, and ledger; no score admission JSON |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | No Phase 8 wiring/admission veto fired after repair. |
| Statistically supported ranking | None; no stochastic candidates or runtimes were ranked. |
| Descriptive-only differences | Test wall times are execution diagnostics only. |
| Default-readiness | Wiring defaults pass; nonlinear production GPU/XLA evidence harnesses do not yet pass. |
| Next evidence needed | Reviewed XLA/trust/memory harness gate, then trusted per-seed GPU prefix and full-row evidence. |

## Post-Run Red Team

- Strongest alternative explanation: source and tiny tests may pass while an
  XLA-compiled full-row score kernel fails to compile, becomes nonfinite, or
  exceeds memory.
- Result that would overturn the wiring gate: a behavioral test showing a
  default path reaches a historical route, a component receives multiple
  seeds, or the aggregate differs from the mean of singleton results.
- Weakest evidence: no nonlinear compact score kernel has yet emitted a
  trusted XLA GPU artifact with reset memory statistics.

## Phase 9 Boundary

Phase 9 must not launch a nonlinear full-row GPU command from the current CLIs.
Unlike the LGSSM runner, those CLIs do not yet record `jit_compile=True`, the
managed-session GPU trust basis, output-device validation, reset score-memory
statistics, terminal progress artifacts, or separate score-only and FD-only
shards. Those missing fields are a harness continuation veto, not a score-math
failure. Phase 9 begins by repairing and testing that evidence harness.

