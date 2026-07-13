# Phase 2 Result: Fixed-SIR Memory-Style Reclassification

Date: 2026-07-09

Status: `PASSED_FOCUSED_GATE`

## Objective

Introduce a fixed-SIR memory-style route ID distinct from the old
`manual_total_vjp*` historical alias, and allow full admission only for the new
route when all-parameter same-scalar finite-difference correctness and numeric
`N=10000` memory evidence exist.

## Implementation

Updated:

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- `bayesfilter/highdim/ledh_score_contract.py`

The fixed-SIR memory-style route is now:

```text
memory_style_reverse_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot
```

The old fixed-SIR route:

```text
manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot
```

is preserved as `historical_manual_score_route` and is rejected by
`validate_ledh_score_artifact(..., require_admitted=True)`.

## Checks

Focused model checks were included in the combined gate:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

Result:

```text
79 passed, 2 warnings
```

## Evidence Contract Status

| Requirement | Status |
| --- | --- |
| New fixed-SIR memory-style route is recognized by shared contract | Passed |
| Old fixed-SIR `manual_total_vjp*` alias is historical only | Passed |
| Directional-only evidence cannot promote to full admission | Passed |
| Parameterized diagnostic row cannot be admitted as main fixed-SIR row | Passed |
| Numeric memory source/peak/budget gate remains required | Passed |

## Nonclaims

No new fixed-SIR full score artifact was admitted in this phase. This phase
does not claim `N=10000` memory pass beyond existing diagnostic records, does
not establish all-model score readiness, and does not make HMC/posterior or
scientific-superiority claims.

## Handoff

The route/default repair phases are complete for LGSSM, fixed-SIR,
predator-prey, and actual-SV. The next executable blocker is not route
taxonomy; it is the remaining full Sinkhorn-step memory failure in the LGSSM
memory-style reverse/VJP score path.
