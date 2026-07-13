# Phase 0 Result: Route Demotion And Policy Gate

Date: 2026-07-08

Status: `PASSED_PHASE0_PHASE1_MAY_START_AFTER_REVIEW`

## Question

Does the repository have a clear, enforceable boundary that the old
reverse-record/manual-total-VJP score routes are historical and wrong for
leaderboard score admission?

## Decision

The policy boundary is now recorded:

```text
historical_wrong_for_leaderboard_score_admission
```

for reverse-record/manual-total-VJP score routes.

However, this boundary is not yet code-enforced. The current validator still
allowlists `manual_total_vjp_no_autodiff_same_scalar_*` provenance strings, so
Phase 1 must land validator and static guard changes before any score
full-admission command may run.

## Inventory Evidence

Static inventory command:

```bash
rg -n "records\\.append|reversed\\(records\\)|manual_total_vjp_no_autodiff|historical_diagnostic_manual_reverse|compact_forward_sensitivity|_ALLOWED_NO_TAPE_PROVENANCE|--admit-full|LEDH_SCORE_ADMISSION_STATUS_FULL" \
  bayesfilter/highdim docs/benchmarks tests/highdim -g '*.py'
```

Key findings:

- `bayesfilter/highdim/ledh_score_contract.py` still allowlists:
  - `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`;
  - `manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`;
  - `manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot`;
  - `manual_total_vjp_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot`;
  - `manual_total_vjp_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot`.
- LGSSM already has compact provenance:
  - `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`.
- Actual-SV has reverse-record score storage:
  - `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
    contains `records.append(...)` and `reversed(records)`.
  - It exposes `--admit-full`.
- Predator-prey has reverse-record score storage:
  - `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
    contains `records.append(...)` and `reversed(records)`.
  - It exposes `--admit-full`.
- Fixed-SIR delegates to the older parameterized SIR manual route and uses
  `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`.
- Historical diagnostic scripts also contain `records.append(...)`; Phase 1
  must distinguish diagnostic scripts from leaderboard score admission paths.

## Local Checks

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Result:

```text
50 passed, 2 warnings
```

These tests are pre-demotion compatibility checks. Their old phase-number
suffixes come from earlier runbooks; Phase 1 must map them by model content and
row ID, not filename suffix.

## Evidence Table

| Field | Status |
| --- | --- |
| Target scalar identified | `observed_data_log_likelihood_estimator` / `log_likelihood` |
| Old route demotion recorded | Passed |
| Code-level enforcement landed | Not yet; Phase 1 required |
| Full-admission commands run | None |
| New score admitted | None |
| Review status | Launch review returned `VERDICT: AGREE`; Phase 0/Phase 1 review still required |

## Forbidden Until Phase 1 Lands

- No `--admit-full` command may be run for any score script whose route can
  still validate as `manual_total_vjp*`.
- No leaderboard integration may collect `manual_total_vjp*` as admitted score.
- No result may describe one tested-point finite-difference agreement as broad
  mathematical correctness.

## Required Phase 1 Work

Phase 1 must:

1. tighten `bayesfilter/highdim/ledh_score_contract.py` so full admission
   rejects `manual_total_vjp*` provenance;
2. add compact route IDs for every model row before those rows can be admitted;
3. add tests proving admitted artifacts with old provenance fail;
4. add static guards for active leaderboard score scripts containing
   `records.append(...)` plus reverse scans;
5. update model score scripts so old routes emit blocked/historical artifacts
   unless and until compact implementations replace them.

## Nonclaims

- No compact score port was implemented in Phase 0.
- No score artifact was newly admitted.
- No full `N=10000` score run was launched.
- No HMC readiness, posterior correctness, runtime ranking, or scientific
  superiority claim is made.

## Handoff

Phase 1 may start after read-only review of this result and the Phase 1 subplan
returns `VERDICT: AGREE`, or after a documented Codex substitute review if
Claude is unavailable.
