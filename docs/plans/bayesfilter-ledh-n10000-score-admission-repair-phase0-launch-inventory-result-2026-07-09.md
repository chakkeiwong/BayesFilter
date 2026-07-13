# Phase 0 Result: Launch Inventory And Artifact Map

Date: 2026-07-09

Status: `PASSED_INVENTORY_GATE_PENDING_PACKET_REVIEW`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Start repair from artifact emission and validation, not score-method redesign. | Passed: every main LEDH row has a located admitted value artifact and current score evidence classification. | No raw memory JSON, tiny diagnostic, or historical `manual_total_vjp*` route was treated as admitted. No full `N=10000` score command was run. | Per-runner artifact builders are uneven; full runs could still emit nonuniform evidence unless Phase 1 centralizes or certifies the schema emitter. | Execute Phase 1 shared admitted-artifact emitter/certification before any full score run. | No score admission, no new memory pass, no full score correctness, no HMC readiness, no posterior correctness, no runtime ranking. |

## Row Inventory

| Row | Value artifact | Current score evidence | Current admission status | Smallest next action |
| --- | --- | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json` | `docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json` | `legacy_raw_score_memory_not_admitted`; compact raw evidence exists but lacks Phase 1 score schema. | Phase 1 emitter certification, then Phase 2 normalize if complete or rerun LGSSM compact `N=10000`. |
| `zhao_cui_spatial_sir_austria_j9_T20` | `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json` | `docs/plans/ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json` plus tiny compact artifact | Raw `N=10000` evidence is `historical_diagnostic_not_admitted`; tiny compact evidence is not full admission. | Phase 3 compact full rerun after Phase 1 emitter gate. |
| `zhao_cui_sv_actual_nongaussian_T1000` | `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact actual-SV score artifact | `tiny_score_diagnostic_not_admitted`. | Phase 4 compact full run. |
| `zhao_cui_predator_prey_T20` | `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json` | Tiny compact predator-prey score artifact | `tiny_score_diagnostic_not_admitted`. | Phase 5 compact full run. |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact generalized-SV score artifact | `tiny_score_diagnostic_not_admitted`. | Phase 6 compact full run without target substitution. |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json` | Tiny compact KSC-SV score artifact | `tiny_score_diagnostic_not_admitted`. | Phase 7 compact full run with exact-native actual-SV overclaim forbidden. |

## Local Checks

Passed:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/test_two_lane_highdim_ledh_leaderboard.py -q
```

Result: `34 passed, 2 warnings`.

## Handoff

Phase 1 should add or certify a shared score-artifact emitter/wrapper before
any full score run so computation evidence cannot again be stranded as raw
legacy JSON or historical-route evidence.
