# Phase 1 Subplan: Shared Admitted Artifact Emitter

Date: 2026-07-09

Status: `DRAFT_READY_FOR_PACKET_REVIEW`

## Phase Objective

Add or certify a shared score-artifact emission path so every future full
`N=10000` LEDH score run produces a Phase 1 schema-valid artifact validated
against the admitted value artifact before it can be used by leaderboard
integration.

## Entry Conditions Inherited From Previous Phase

- Phase 0 located all six main admitted value artifacts.
- LGSSM raw compact evidence lacks Phase 1 score schema.
- Fixed-SIR raw full evidence is historical `manual_total_vjp*`.
- Remaining rows have tiny compact diagnostics only.
- No full `N=10000` score command has been launched by this repair runbook.

## Required Artifacts

- Preferred helper: `bayesfilter/highdim/ledh_score_artifact.py`, or a scoped
  addition to `bayesfilter/highdim/ledh_score_contract.py`.
- Tests, preferably `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`.
- Phase 1 result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase1-shared-emitter-result-2026-07-09.md`.
- Phase 2 LGSSM subplan:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-subplan-2026-07-09.md`.

## Required Checks, Tests, And Reviews

Run precheck:

```bash
rg -n "_score_artifact_from|score_admission_status|memory_diagnostics|source_value_artifact|validate_ledh_score_artifact" \
  docs/benchmarks/benchmark_ledh_same_target_*_score.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  bayesfilter/highdim tests/highdim
```

Run focused tests after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/test_two_lane_highdim_ledh_leaderboard.py -q
```

Review Phase 1 result and Phase 2 subplan with Claude if available, otherwise
packet-only Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is there a shared, tested way to emit full-admission score artifacts so future full runs cannot accidentally produce raw legacy evidence? |
| Baseline/comparator | Existing per-model artifact builders, shared score validator, Phase 0 inventory, and July 8 Phase 8 blocker. |
| Primary criterion | A shared helper or certified wrapper builds artifacts that pass `validate_ledh_score_artifact(..., require_admitted=True)` only when all full-admission inputs are present and compact; tests prove raw legacy, tiny, and historical route cases do not pass. |
| Veto diagnostics | Helper admits historical `manual_total_vjp*`; helper admits raw legacy JSON; helper admits missing memory pass; helper allows value/score row mismatch; helper changes target scalar; per-model builders bypass validation. |
| Not concluded | No full `N=10000` score run, no score leaderboard completion, no HMC readiness, no posterior correctness, no runtime ranking. |

## Forbidden Claims And Actions

Do not run a full `N=10000` score command in Phase 1. Do not weaken
`validate_ledh_score_artifact`. Do not treat tiny diagnostics or raw
`primary_pass` memory JSONs as full score artifacts. Do not change target
scalar, observation policy, row id, theta coordinates, or parameter order.

## Exact Next-Phase Handoff Conditions

Phase 2 may begin only if the shared emitter/wrapper exists or current builders
are explicitly certified, focused tests pass, the Phase 2 subplan defines
LGSSM normalization versus rerun criteria, and review returns `VERDICT: AGREE`
or a recorded fallback review agrees.

## Stop Conditions

Stop if a shared emitter would require broad risky refactoring, current
builders cannot validate consistently, historical routes or raw JSONs can
full-admit, source value artifacts fail validation, or continuing requires
target changes, package installation, network/data fetches, destructive
actions, or unrelated dirty-worktree edits.
