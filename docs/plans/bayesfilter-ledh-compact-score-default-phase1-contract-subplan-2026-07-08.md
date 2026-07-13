# Phase 1 Subplan: Shared Compact Score Contract

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Make compact forward sensitivity the only score route that can be admitted by
the shared LEDH score contract, and block old reverse-record/manual-total-VJP
routes from full leaderboard score admission.

Phase 1 is the code-enforcement phase for the Phase 0 policy boundary.

## Entry Conditions Inherited From Previous Phase

- Phase 0 recorded that reverse-record/manual-total-VJP routes are
  `historical_wrong_for_leaderboard_score_admission`.
- Phase 0 inventory found current code-level enforcement is incomplete because
  `bayesfilter/highdim/ledh_score_contract.py` still allowlists
  `manual_total_vjp*` provenance.
- No full-admission score command may run until this phase lands validator and
  static guards.

## Required Artifacts

- Updated validator:
  `bayesfilter/highdim/ledh_score_contract.py`
- Updated focused tests:
  `tests/highdim/test_ledh_score_contract_phase1.py`
- Optional route-specific tests if validator changes require them:
  - `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  - `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
  - `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
  - `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- Phase 1 result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-result-2026-07-08.md`
- Phase 2 subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase1-contract-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Local checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Static checks:

```bash
rg -n "manual_total_vjp_no_autodiff_same_scalar_.*LEDH_SCORE_ADMISSION_STATUS_FULL|score_admission_status.*LEDH_SCORE_ADMISSION_STATUS_FULL|records\\.append|reversed\\(records\\)" \
  bayesfilter/highdim docs/benchmarks tests/highdim
```

Review:

- Claude read-only review of validator/test changes, Phase 1 result, and Phase
  2 subplan.
- Use the probe ladder if Claude times out. If Claude is unavailable or
  policy-blocked, write a fresh Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the shared score contract now prevent `manual_total_vjp*` and reverse-record routes from full LEDH leaderboard score admission? |
| Baseline/comparator | Phase 0 inventory, current validator allowlist, LGSSM compact route, and existing score artifact tests. |
| Primary criterion | `validate_ledh_score_artifact(..., require_admitted=True)` accepts compact provenance only and rejects `manual_total_vjp*`; focused tests prove old-route full admission fails while tiny/historical diagnostics remain representable as blocked/not admitted if needed. |
| Veto diagnostics | Old route still admitted; compact route rejected for LGSSM; tests promote directional-only FD; model scripts can still emit admitted artifacts with reverse-record route; hidden target-scalar changes. |
| Explanatory diagnostics | Static hits for reverse-record code that remains historical, expected test updates, and route ID migration notes. |
| Not concluded | No model compact port beyond validator wiring, no new full score admission, no performance or HMC claim. |
| Artifact | Phase 1 result with test output and validator diff summary. |

## Required Implementation Steps

1. Split score provenance into compact-admissible and historical/diagnostic
   route groups.
2. Keep LGSSM compact provenance admitted.
3. Reject `manual_total_vjp*` when `score_admission_status` is
   `n10000_same_target_no_tape_score_admitted` or `require_admitted=True`.
4. Decide whether tiny diagnostic artifacts may still use `manual_total_vjp*`
   with `tiny_score_diagnostic_not_admitted`; if yes, tests must prove they
   cannot be admitted.
5. Add compact route ID constants/placeholders for future model ports without
   making unsupported model-score claims.
6. Add static or AST tests that active leaderboard score scripts with
   `records.append(...)` plus reverse scans cannot be treated as compact.
7. Update affected focused tests so their assertions match the new demotion
   policy.
8. Write the Phase 1 result and draft/refresh Phase 2 LGSSM reference subplan.

## Forbidden Claims And Actions

- Do not claim any non-LGSSM compact score exists before implementation.
- Do not remove historical diagnostic code unless a reviewed plan explicitly
  authorizes removal.
- Do not run any full `N=10000` score command in this phase.
- Do not change target scalar, row IDs, observation policies, or parameter
  coordinate systems.
- Do not use `GradientTape`, `ForwardAccumulator`, stopped partial derivative,
  or hidden autodiff.
- Do not treat tiny directional FD as score admission.

## Exact Next-Phase Handoff Conditions

Phase 2 may start only if:

- validator tests prove full-admission old routes are rejected;
- LGSSM compact admitted artifact tests still pass or are explicitly updated to
  the compact-only contract;
- any old-route model score scripts are blocked from full admission or
  explicitly documented as impossible to admit under the validator;
- Phase 1 result records local checks and nonclaims;
- Phase 2 subplan exists and is reviewed for consistency and boundary safety.

## Stop Conditions

Stop and ask for direction if:

- rejecting `manual_total_vjp*` full admission would break a user-required
  admitted score artifact that must remain current;
- the validator cannot distinguish compact provenance from old provenance;
- focused tests reveal target-scalar mismatches unrelated to route demotion;
- code changes require modifying unrelated dirty user work;
- Claude and Codex review do not converge after five rounds on the same
  material blocker.
