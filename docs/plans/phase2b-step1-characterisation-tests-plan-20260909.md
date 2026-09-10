# Phase 2B Step 1 Characterisation Result

**Date:** 2026-09-11  
**Branch:** `surrogate-hmc`  
**Parent:** [LEDH Surrogate-Force HMC Unified Program](ledh-surrogate-hmc-unified-program-2026-09-06.md)  
**Status:** COMPLETE AFTER REPAIR

## Purpose

Freeze supported pre-refactor behavior before the single-cloud and fused-batch
LEDH programs are unified, including feature-off variants and B/K isolation.

## Binding evidence

### Parameter-sensitive configuration matrix

`tests/highdim/test_ledh_configuration_regression.py` uses a nonlinear model
whose transition depends on theta and freezes eight representative programs:

- reset-none, one stage;
- reset-none, annealed;
- Contract-E with all higher-moment controls off;
- diagonal correction without trust radius;
- diagonal correction with trust radius;
- pairwise correction;
- coordinate cap (full reference configuration);
- annealed full composition.

It also proves:

- `with_score=False` leaves the value bitwise unchanged;
- nested controls are exact no-ops when their parent mechanism is disabled;
- adjacent active configurations execute observably different programs; and
- disabled dual-cap/trust configurations cannot be labeled production.

Result: **19 passed** on 2026-09-11.

### Fused B/K row isolation

`tests/highdim/test_ledh_row_independence.py` exercises B in `{1,2,4}` and K in
`{1,2,5}` through the actual fused API. It checks identical-row equality and
that perturbing one theta row leaves every other row's value and K scores
bitwise unchanged.

This test exposed a pre-existing defect: `[B,K,N,P]` directions were directly
reshaped to `[K,B*N,P]`, interleaving B and K when both exceeded one. The repair
transposes to `[K,B,N,P]` before flattening and fixes rank-2 per-row validity.

`tests/highdim/test_ledh_canonical_batch_fused.py` continues to cover B=1
single-cloud parity, graph tracing, K-vs-swept parity, and rank-2 compatibility.

Combined result after repair: **40 passed** across configuration regression,
row isolation, and existing fused tests.

## Supplemental legacy fixture campaign

The local `tests/highdim/fixtures/ledh_golden_master_20260909/` campaign is not a
binding gate:

- 327 of the declared 360 files exist; 33 generation cases failed or are absent;
- JSON fixtures are ignored by Git and therefore are not portable CI evidence;
- the fixture transition ignores theta, so score sensitivity is not exercised;
- its `direction_count` axis passes a rank-2 tensor to the single-cloud API and
  does not represent the fused API's K-direction contract; and
- it stores trace key names rather than the declared full diagnostic payload.

The existing files may still detect incidental value drift, but they cannot be
used to claim Step 1 completeness. The compact nonlinear matrix supersedes them
as the binding pre-refactor baseline.

## Tolerance policy

- Same-lane float64 pre/post parity: rtol `1e-12`, atol `0`.
- Float32 checks must use a recorded tolerance appropriate to the route.
- Exact row isolation and inactive-control identity: bitwise equality.
- Cross-lane parity may use rtol `5e-4` only for operation-order differences;
  it never replaces same-lane baselines.

## Exit status

- [x] Parameter-sensitive supported-configuration baselines
- [x] Feature-off and inactive-control semantics
- [x] Production-label fail-closed behavior
- [x] Fused B/K row isolation
- [x] Existing fused parity and graph tests
- [x] Legacy fixture limitations classified
- [x] Refactor contract written
- [x] Per-row reduction derivation written

Full-union B/K row isolation remains a mandatory implementation gate after the
shared engine first carries Contract-E and higher-moment controls.
