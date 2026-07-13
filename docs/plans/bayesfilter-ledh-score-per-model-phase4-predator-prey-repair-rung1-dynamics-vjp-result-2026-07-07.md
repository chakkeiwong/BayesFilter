# Phase 4 Repair Rung 1 Result: Predator-Prey Dynamics VJP

metadata_date: 2026-07-07
status: `RUNG1_DYNAMICS_VJP_PASSED_FULL_SCORE_NOT_YET_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair-rung1

## Objective

Implement and test the first no-tape manual VJP layer needed by the
predator-prey LEDH total-score adapter:

- predator-prey RHS VJP;
- one-step RK4 VJP;
- full transition-mean VJP over the value-runner RK4 substeps.

These are only dynamics-adjoint ingredients. They do not by themselves admit a
score.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Rung 1 passed. Continue to the full filter reverse-scan repair. |
| Primary criterion status | Met for this rung: RHS and transition-mean VJPs match finite differences for all six physical parameters and state components. |
| Veto diagnostic status | No tape/autodiff route was added; no score artifact was admitted. |
| Main uncertainty | Full LEDH score still needs reverse scan through transport, log-weight normalization, LEDH flow, pre-flow process-noise push, and the filtering time loop. |
| Next justified action | Implement the next repair rung: fixed-randomness forward replay and reverse-scan skeleton using these dynamics VJPs. |
| What is not concluded | No predator-prey score admission; no full LEDH total derivative; no N=10000 score/memory gate; no HMC, posterior, runtime, source-faithfulness, exact-likelihood, or scientific claim. |

## Code And Tests

Added:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`

Updated:

- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Test coverage added:

- RHS VJP versus central finite differences for all six parameters and state
  components.
- Full transition-mean VJP versus central finite differences for all six
  parameters and state components.
- Existing guards still reject value-only and directional-only score
  promotion.

## Local Checks

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  bayesfilter/highdim/models.py \
  bayesfilter/highdim/ledh_score_contract.py
```

Result: passed.

Focused test:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Result:

```text
4 passed, 2 warnings
```

Broader Phase 4 repair check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
25 passed, 2 warnings
```

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the predator-prey dynamics VJP pieces needed for a no-tape total-score adapter be implemented and checked? |
| Answer | Yes for RHS and transition mean. |
| Baseline/comparator | Central finite differences with fixed deterministic inputs. |
| Primary criterion | Passed for Rung 1 only. |
| Veto diagnostics | Full score still not implemented; no score admission. |
| Artifact | This result and the new score helper module/tests. |

## Next Repair Rung

Implement fixed-randomness forward replay and reverse-scan skeleton for the
predator-prey value path:

- match the value runner's initial particle and process-noise seeds;
- store LEDH flow aux and scalar fields needed for reverse scan;
- reuse existing no-tape transport/log-weight/flow VJP primitives;
- use the passed RK4 transition VJP to propagate all state cotangents to the
  physical theta score.

## Nonclaims

- Rung 1 is not score admission.
- Dynamics VJP correctness is not full LEDH score correctness.
- No full-row `N=10000,T=20` score/memory evidence exists.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.
