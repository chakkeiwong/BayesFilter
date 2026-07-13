# Phase 2 Subplan: LGSSM Compact Reference Freeze

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Freeze LGSSM as the reference implementation of the compact forward-sensitivity
LEDH score style and quarantine its historical manual-reverse route as
diagnostic only.

LGSSM is the reference because it already carries particles, log weights,
particle tangents, log-weight tangents, log likelihood, and log-likelihood
tangents through a single forward loop and uses streaming transport
value+JVP.

## Entry Conditions Inherited From Previous Phase

- Phase 1 validator rejects `manual_total_vjp*` full admission.
- Existing LGSSM compact provenance remains validator-compatible.
- No non-LGSSM compact route is admitted yet.

## Required Artifacts

- Phase 2 result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-result-2026-07-08.md`
- Updated tests if needed:
  - `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  - `tests/test_ledh_lgssm_manual_score_phase4.py`
  - `tests/test_ledh_score_memory_n10000.py`
- Phase 3 actual-SV compact subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase2-lgssm-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Local checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Optional trusted GPU check only if the phase result needs fresh memory
evidence:

```bash
BAYESFILTER_RUN_LEDHD_SCORE_MEMORY_N10000=1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_score_memory_n10000.py::test_lgssm_ledh_compact_score_float64_correctness_and_memory_n10000 -q
```

Review:

- Claude read-only review of the Phase 2 result and Phase 3 actual-SV subplan.
- Use the probe ladder if Claude times out. If Claude is unavailable, use a
  Codex substitute review and record the limitation.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is LGSSM frozen as the compact score reference, with historical manual reverse blocked from current admission? |
| Baseline/comparator | LGSSM compact implementation in `benchmark_ledh_same_target_lgssm_m3_t50_value.py`, Phase 1 validator, historical LGSSM manual reverse diagnostic. |
| Primary criterion | Tests confirm compact LGSSM is the only full-admissible route; historical manual reverse remains blocked; source code contains no `GradientTape` or `ForwardAccumulator` in compact score helpers. |
| Veto diagnostics | Manual reverse admitted; compact route fails validator; route labels are ambiguous; old `T=2` memory artifact is treated as current `T=50` admission; target scalar drift. |
| Explanatory diagnostics | Existing memory artifact status, optional GPU memory rerun, compact-vs-historical parity at tiny scale. |
| Not concluded | No non-LGSSM compact port, no HMC readiness, no posterior correctness, no runtime ranking. |
| Artifact | Phase 2 result and Phase 3 actual-SV subplan. |

## Required Implementation Steps

1. Inspect LGSSM compact route constants and helper names.
2. Confirm compact source helpers are no-tape and use streaming transport
   value+JVP.
3. Confirm historical manual reverse route is diagnostic only and cannot pass
   Phase 1 validator as full admission.
4. Confirm existing LGSSM full-row artifact path is either current and
   validator-compatible or explicitly stale/blocked.
5. Add or adjust tests only if a guard is missing.
6. Write Phase 2 result and Phase 3 actual-SV compact subplan.

## Forbidden Claims And Actions

- Do not claim current LGSSM memory artifact is fresh if it is stale.
- Do not admit any non-LGSSM score.
- Do not run full GPU memory checks unless the phase result needs fresh memory
  evidence and the command is trusted/escalated.
- Do not change LGSSM target scalar, row ID, or parameter coordinate system.
- Do not use tape/autodiff or stopped partial derivatives.

## Exact Next-Phase Handoff Conditions

Phase 3 actual-SV may start only if:

- LGSSM compact route is documented as the reference style;
- historical LGSSM reverse route is blocked from current full admission;
- focused LGSSM/score-contract tests pass;
- Phase 3 subplan explicitly ports actual-SV away from `records.append(...)`
  and `reversed(records)`;
- Codex records read-only review findings, resolves any material blockers, and
  confirms no unresolved boundary issue remains. Claude is advisory reviewer
  only and is not execution authority.

## Stop Conditions

Stop and ask for direction if:

- LGSSM compact route fails the Phase 1 validator;
- historical reverse route can still full-admit;
- current artifacts cannot distinguish old `T=2` evidence from the intended
  row;
- fresh GPU memory evidence is required but trusted GPU checks are unavailable;
- Claude and Codex review do not converge after five rounds on the same
  material blocker.
