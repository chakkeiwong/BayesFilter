# Phase 1 Result: Shared Compact Score Contract

Date: 2026-07-08

Status: `PASSED_PHASE1_PHASE2_MAY_START_AFTER_REVIEW`

## Question

Does the shared score contract now prevent `manual_total_vjp*` and
reverse-record routes from full LEDH leaderboard score admission?

## Decision

Passed.

The shared score validator now separates compact-admissible provenance from
historical diagnostic provenance. Full score admission rejects every
`manual_total_vjp_no_autodiff_same_scalar_*` route, even if an artifact claims
`n10000_same_target_no_tape_score_admitted` and memory pass.

Historical diagnostics may remain as raw diagnostic outputs outside full score
admission. A historical route may validate through the score artifact contract
only when it is not full admission and still satisfies the contract's
correctness requirements, such as `same_scalar_finite_difference` or
`exact_reference` with status `pass`. Directional-only diagnostics and
blocked/not-run diagnostics are not valid admitted score artifacts and are
rejected by the validator.

## Code Changes

Validator:

- `bayesfilter/highdim/ledh_score_contract.py`

Added:

- `LEDH_SCORE_COMPACT_LGSSM_PROVENANCE`;
- `_COMPACT_ADMISSIBLE_NO_TAPE_PROVENANCE`;
- `_HISTORICAL_DIAGNOSTIC_NO_TAPE_PROVENANCE`.

Full admission now raises:

```text
historical manual_total_vjp score routes cannot be full LEDH score admission;
compact forward sensitivity is required
```

when the provenance is historical/manual-total-VJP.

Tests:

- `tests/highdim/test_ledh_score_contract_phase1.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Added coverage that:

- every old `manual_total_vjp*` provenance fails full admission;
- old routes cannot full-admit even with apparent memory pass;
- directional-only or blocked historical diagnostics remain raw diagnostics,
  not validator-admitted score artifacts;
- fixed-SIR old route cannot build a full-admitted artifact;
- actual-SV old route cannot full-admit even with synthetic memory pass;
- predator-prey old route cannot full-admit even with synthetic memory pass.

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
58 passed, 2 warnings
```

Static inventory:

```bash
rg -n "manual_total_vjp_no_autodiff_same_scalar_.*LEDH_SCORE_ADMISSION_STATUS_FULL|score_admission_status.*LEDH_SCORE_ADMISSION_STATUS_FULL|records\\.append|reversed\\(records\\)" \
  bayesfilter/highdim docs/benchmarks tests/highdim
```

Result classification:

- LGSSM compact admitted test remains the only focused full-admission hit.
- Actual-SV and predator-prey still contain reverse-record implementation code,
  but the validator now blocks those routes from full admission.
- Historical diagnostic scripts still contain `records.append(...)`; they are
  not leaderboard score admission paths.

## Evidence Table

| Field | Status |
| --- | --- |
| Compact LGSSM full admission still validates | Passed |
| `manual_total_vjp*` full admission rejected | Passed |
| Old route raw diagnostics remain available outside full admission | Passed |
| Full score commands run | None |
| Model compact ports implemented | None in this phase |

## Remaining Work

- Phase 2 must freeze LGSSM as the compact reference and ensure its current
  score artifact path remains validator-compatible.
- Phase 3+ must port actual-SV, fixed-SIR, predator-prey, generalized-SV, and
  KSC-SV to compact forward sensitivity before those rows can be admitted.
- Reverse-record code still exists and is now historical/diagnostic or blocked
  until replaced.

## Nonclaims

- No non-LGSSM compact route was implemented.
- No new full score artifact was admitted.
- No `N=10000` score run was launched.
- No HMC readiness, posterior correctness, runtime ranking, public benchmark,
  or scientific superiority claim is made.

## Handoff

Codex may start Phase 2 only after consuming the read-only review findings,
resolving any material blockers, confirming local gates remain passed, and
recording that no unresolved boundary issue remains. Claude is advisory
reviewer only and is not execution authority. If Claude is unavailable, Codex
may use a documented substitute review.
