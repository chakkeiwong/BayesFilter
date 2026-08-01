# Phase 1: Shared Clean-Loop Guardrails Result

Date: 2026-07-15

Status: `PASS_CLOSED_HANDOFF_READY`

## Verdict

The shared source and graph guardrails pass without changing any model
algorithm. The audit resolves reachable same-module helpers, requires an
explicit role for every reachable Python loop, rejects declared dynamic loops,
fails closed on unresolved local calls, and inventories functional loops from
both top-level and function-library GraphDef nodes.

This is an engineering guard. It does not infer scientific semantics from
function names, and it does not establish that a nonlinear model is clean XLA.

## Controlling Evidence

- Final JSON after nested-function reachability repair:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-02-nested-reachability-repair-20260715/guardrail_audit.json`;
  SHA-256 `344d8480621affbb89c048bdaf61d4e9660ebc2a4ac002c82e41594b38cd370a`.
- Final log:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-01/shared-guardrails/attempt-02-nested-reachability-repair-20260715/phase1_checks.log`;
  SHA-256 `fa829d1ce280ef00e0cb249db54fe54aaacb749bc89d2e9c5b4c82d8b2c504e2`.

The original attempt is preserved but superseded because nested `cond/body`
definitions were absent from its reachable closure. The repaired focused suite
is `23 passed, 2 warnings`; graph counts remain unchanged.

| Control | Expected | Result |
| --- | --- | --- |
| LGSSM loop-native root | pass | pass |
| historical LGSSM unrolled root | reject | reject |
| current scalar-SV root | reject | reject |
| current predator--prey root | reject | reject |
| reachable predator RK4 | reject | reject |
| neutral `q -> z`, aliased `range(len(seq))` | reject | reject |
| explicitly declared fixed-small loop | pass | pass |

## Graph And Regression Evidence

Fresh CPU-hidden graph construction reproduced the controlling topology:

| Horizon | Top nodes | Function nodes | Functional loops | GraphDef bytes |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 4,014 | 3,712 | 4 | 1,234,390 |
| 50 | 4,014 | 3,712 | 4 | 1,233,359 |

The node/function ratios are exactly 1.0 and the byte ratio is
`0.9991647696`. All topology gates pass. The focused permanent suite was
`22 passed, 2 dependency deprecation warnings in 14.58s`. Compileall and
repository-wide `git diff --check` passed.

TensorFlow loaded CUDA plugin libraries and logged a failed `cuInit` despite
`CUDA_VISIBLE_DEVICES=-1`; the variable was set before import and no GPU device
or GPU computation was used. This is CPU-hidden evidence only and makes no GPU
claim.

## Decision And Red Team

| Decision | Criterion | Veto status | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| retain shared audit utility | all positive/negative fixtures classify correctly | none | use it in later loop repairs | not semantic proof without route specs |
| do not build shared model loop driver | Phase 0 showed incompatible state/feature contracts | premature abstraction veto avoided | exact structural fixture next | no cross-model algorithm abstraction |
| preserve nonlinear reject states | current roots contain declared dynamic loops | expected repair trigger | Phases 3 and 5 | not candidate/scientific rejection |

The strongest risk is incomplete cross-module reachability. The utility is
conservative: cross-module algorithm dependencies must be explicit route units,
and unresolved same-module calls reject. Later phases must declare each imported
model/solver unit rather than treating a clean outer module as closure proof.

Phase 1 used approximately 0.02 CPU core-hours, zero GPU hours, and zero
full-horizon attempts. Remaining budget is conservatively 95.97 CPU core-hours,
32 trusted GPU-hours, and three full-horizon attempts per eligible model.

## NEXT_PHASE_READINESS

| Clause | Status | Evidence |
| --- | --- | --- |
| result/check consistency | `PASS` | JSON/log hashes and `22 passed` |
| legal inherited classifications | `PASS` | Phase 0 registry unchanged |
| exact identities | `PASS` | route specifications and GraphDef preparations |
| assumptions audited | `PASS` | explicit role declarations; unresolved calls reject |
| criteria/vetoes/nonclaims/fresh paths | `PASS` | Phase 2 subplan |
| executable next commands | `PASS` | Phase 2 local and trusted tiny XLA ladder |
| no unresolved material boundary | `PASS` | Phase 1 focused re-review `AGREE` |
| budget | `PASS` | available `95.97 CPU,32 GPU`; Phase 2 minimum+reserve declared below |

Overall: `READY` for
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase2-structural-support-regression-subplan-2026-07-15.md`.
