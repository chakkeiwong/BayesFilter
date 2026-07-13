# Phase 0 Result: Baseline And Score Governance

metadata_date: 2026-07-07
status: `PASSED_BASELINE_GOVERNANCE_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 0

## Phase Objective

Freeze the score program baseline before score implementation.

Score in this program means the no-tape total derivative of the realized
finite-`N` LEDH estimator:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

It does not mean the derivative of the true latent-model likelihood unless a
separate exact-reference comparison proves equality for a specific row.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 0 baseline/governance is locally frozen. |
| Primary criterion status | Passed locally: exactly six value rows are eligible for score repair, no score is admitted, diagnostic SIR is excluded, and tape/autodiff/stopped partial derivatives remain banned for admitted score evidence. |
| Veto diagnostic status | No score admission in Phase 0, no implementation, no diagnostic SIR promotion, no KSC exact native actual-SV claim, and no HMC/runtime/scientific claim. |
| Main uncertainty | Existing LGSSM/fixed-SIR score diagnostics remain inventory evidence until Phase 1 defines a replayable score schema and model phases rerun or normalize them. |
| Next justified action | Review this Phase 0 result and the Phase 1 score schema subplan; if review agrees, execute Phase 1. |
| What is not concluded | No score row is admitted; no score correctness, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or all-algorithm comparison is concluded. |

## Frozen Eligible Row Set

Source artifact:

- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`

Eligible main rows:

| Row | Theta Coordinate | Target Observation Policy |
| --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | `physical_benchmark_exact_oracle` | `lgssm_gaussian_observation_density` |
| `zhao_cui_spatial_sir_austria_j9_T20` | `sir_log_scale_theta` | `fixed_sir_infectious_components_gaussian_observation_density` |
| `zhao_cui_predator_prey_T20` | `physical` | `additive_gaussian_predator_prey` |
| `zhao_cui_sv_actual_nongaussian_T1000` | `synthetic_unconstrained` | `transformed_actual_sv_log_y_square` |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | `source_route_active_transformed_prior_mean` | `source_route_prior_mean_generalized_sv` |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | `synthetic_unconstrained` | `ksc_log_chi_square_gaussian_mixture_surrogate` |

Excluded diagnostic row:

```text
zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale
```

Reason: legacy scoped parameterized SIR diagnostic row; no separate Phase 8
main-row admission artifact.

## Existing Score Evidence Inventory

| Row | Current evidence | Phase 0 classification |
| --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | Existing compact/no-tape same-scalar score helpers and tests; earlier score diagnostics. | Prior diagnostic/implementation evidence only; not admitted by Phase 0. |
| `zhao_cui_spatial_sir_austria_j9_T20` | Existing manual no-tape fixed-SIR tiny score tests and same-scalar FD diagnostics. | Prior diagnostic/implementation evidence only; not admitted by Phase 0. |
| `zhao_cui_predator_prey_T20` | Value artifact admitted; score not implemented/admitted under this program. | Blocked pending model phase. |
| `zhao_cui_sv_actual_nongaussian_T1000` | Value artifact admitted; score not implemented/admitted under this program. | Blocked pending model phase. |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | Value artifact admitted; score not implemented/admitted under this program. | Blocked pending model phase. |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | Value artifact admitted for finite-mixture KSC target; score not implemented/admitted under this program. | Blocked pending model phase. |

## Governance Freeze

For admitted LEDH score evidence:

- `GradientTape` is forbidden.
- `ForwardAccumulator` is forbidden.
- hidden autodiff is forbidden.
- stopped partial derivatives are forbidden.
- value and score must come from the same realized finite-`N` scalar route.
- finite differences must perturb the same scalar whose value is reported.
- every score artifact must name parameter coordinates and preserve target
  policy labels.
- tiny checks must precede `N=10000` correctness and memory checks.
- memory/runtime success alone cannot admit a score.

## Local Checks

JSON syntax check:

```text
python -m json.tool \
  docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json
```

Result: passed.

Existing value/score diagnostic replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/test_ledh_fixed_sir_manual_score_phase4.py -q
```

Result:

```text
14 passed, 2 warnings in 18.74s
```

Static no-tape inventory:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python - <<'PY'
...
PY
```

Result:

```text
PHASE0_NO_TAPE_INVENTORY_OK
```

Repair note: the first static check was too broad and flagged guard strings in
the test files. The Phase 0 subplan was repaired to inspect named helper
functions by AST instead.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | What rows are eligible for LEDH score repair, and what score evidence is currently admitted, diagnostic, or blocked? |
| Answer | Exactly six Phase 8 value rows are eligible. No score is admitted by Phase 0. LGSSM/fixed-SIR evidence is prior diagnostic/implementation evidence only. Four rows remain score-blocked pending model phases. |
| Baseline/comparator | Phase 8 value integration artifact and existing LGSSM/fixed-SIR score tests. |
| Primary criterion | Passed locally: baseline result freezes six eligible rows, excludes diagnostic SIR, forbids tape/autodiff/stopped partial derivatives, and drafts Phase 1 schema/guard plan. |
| Veto diagnostics | No diagnostic SIR promotion, no KSC exact-SV claim, no score admission, no implementation, no HMC/runtime/scientific claim. |
| Explanatory diagnostics | Existing tiny score tests, AST no-tape helper inventory, and Phase 8 value replay. |
| Not concluded | No score admission, score correctness beyond scoped diagnostics, leaderboard rebuild, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Phase 1 Handoff

Phase 1 may begin only after read-only review agrees with this Phase 0 result
and the Phase 1 score schema subplan.

Phase 1 must define a replayable score artifact schema and guards before any
model score can be admitted. The schema must preserve the admitted value
artifact's row id, target scalar, output field, target observation policy,
theta coordinate system, and parameter names/order.

## Nonclaims

- No score row is admitted.
- No score correctness is claimed for the six-row score program.
- No HMC readiness, posterior correctness, scientific superiority, runtime
  ranking, or all-algorithm comparison is claimed.
