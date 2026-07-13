# Phase 2 Result: LGSSM Compact Reference Freeze

Date: 2026-07-08

Status: `PASSED_PHASE2_PHASE3_MAY_START_AFTER_REVIEW`

## Question

Is LGSSM frozen as the compact score reference, with historical manual reverse
blocked from current admission?

## Decision

Passed.

LGSSM is the reference implementation for the compact forward-sensitivity score
style.

The reference route is:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot
```

The historical route is:

```text
historical_diagnostic_manual_reverse_scan_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot
```

and remains diagnostic only.

## Source Anchors

- Route constants:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  lines 86-92.
- Compact transport JVP:
  `_compact_forward_transport_jvp_tf`, lines 685-747, calls
  `_filterflow_manual_streaming_finite_transport_value_and_jvp_total`.
- Compact score loop:
  `_compact_value_and_score_from_components`, lines 1391-1662.
- The compact loop carries:
  - `running_particles`;
  - `running_log_weights`;
  - `running_d_particles`;
  - `running_d_log_weights`;
  - `running_log_likelihood`;
  - `running_d_log_likelihood`.
- The compact route returns `score_route = COMPACT_SCORE_ROUTE_ID`.

## Local Checks

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
38 passed, 2 warnings
```

## Evidence Table

| Field | Status |
| --- | --- |
| Compact LGSSM validator compatibility | Passed |
| Historical manual reverse blocked from default admission | Passed |
| Compact helper no-tape/no-forward-accumulator tests | Passed |
| Stale `T=2` memory artifact rejected as current score artifact | Passed |
| Fresh GPU `N=10000` rerun | Not run in Phase 2; existing gate preserved |

## Interpretation

LGSSM now serves as the implementation template for later model ports:

1. propagate parameter tangents forward with particles and log weights;
2. compute same finite-`N` `log_likelihood`;
3. use streaming transport value+JVP, not reverse transport pullback;
4. avoid all-time reverse records;
5. emit compact provenance only for full score admission.

## Nonclaims

- This phase does not admit or implement actual-SV, fixed-SIR, predator-prey,
  generalized-SV, or KSC-SV compact scores.
- This phase does not rerun fresh GPU `N=10000` memory evidence.
- This phase does not establish HMC readiness, posterior correctness, runtime
  ranking, or scientific superiority.

## Handoff

Phase 3 actual-SV may start after Codex consumes the read-only review findings,
resolves any material blockers, confirms local gates remain passed, and records
that no unresolved boundary issue remains. Claude is advisory reviewer only and
is not execution authority.
