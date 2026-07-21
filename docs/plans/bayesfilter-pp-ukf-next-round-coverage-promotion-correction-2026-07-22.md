# PP-UKF Next-Round Coverage Promotion Correction

Date: 2026-07-22

Status: `IMPLEMENTED_AND_TESTED`

## Finding

The prior PP-UKF compatibility result classified one-hop evaluations as
"guards" and the surrounding interpretation treated failed one-hop points as
local-suitability vetoes for their parent primary. That is not the intended
protocol. The one-hop evaluations fill holes in the coarse primary `L` grid.
Each compatible one-hop point is itself eligible for the next round, while a
failed one-hop point removes only that point. It does not remove a statistically
compatible primary.

## Active Protocol

- Primary grid: `(3, 5, 9, 13, 18, 25)`.
- Primary epsilon: tuned independently for each primary.
- One-hop coverage points: generated from compatible primaries only,
  non-recursively, with the parent epsilon inherited exactly and no retuning.
- Next-round set: the union of all compatible primaries and all compatible
  one-hop coverage points.
- Ranking: none. Acceptance means, interval widths, epsilon, and runtime remain
  descriptive or explanatory; they do not select a winner.
- Required barriers: primary and coverage execution must be complete before a
  next-round set is emitted. A completed but incompatible coverage point is
  excluded individually, not converted into a parent veto.

For the preserved 2026-07-21 evidence, this gives:

```text
compatible primaries: (5, 9, 13, 18, 25)
compatible coverage:  (12, 14, 17, 19, 24)
next round L:         (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)
```

The failed coverage points `(4, 6, 8, 10)` are excluded. `L=5` and `L=9`
remain in the next-round set because their primary evidence is compatible.

## Implementation

`OperationalBroadGridResult` now exposes:

- `viable_coverage_candidates`;
- `next_round_candidates`, preserving primary/coverage provenance and epsilon
  policy; and
- `next_round_l_values`, the sorted unique `L` union.

The changed result and driver payloads use schema `v2`; prior `v1` artifacts
remain historical and are not silently upgraded.

Coverage records retain the historical v1 request identity for artifact
lineage, and now explicitly report `scientific_role` as
`same_epsilon_neighbor_coverage` and `parent_promotion_veto: false`.
The PP-UKF statistical compatibility driver records and checks the ten-value
union in both private and public output payloads. Existing GPU artifacts are
not rerun or overwritten; this correction changes promotion interpretation and
serialization, not the numerical evidence.

## Verification

Focused verification passed:

```text
pytest -q tests/test_hmc_operational_broad_grid.py \
  tests/test_pp_ukf_statistical_compatibility_guard_repair_driver.py \
  tests/test_pp_ukf_operational_broad_grid_driver.py
27 passed
```

The tests assert the ten-value union, exclusion of `(4, 6, 8, 10)`, no
stochastic ranking, exact inherited-epsilon coverage semantics, and preservation
of a compatible parent when a coverage point fails.

## Nonclaims

This correction does not rank candidates, establish acceptance in-band, claim
posterior convergence or correctness, authorize retained sampling, or imply
that the PP-UKF research direction is validated. The next round still requires
its own frozen-kernel validation and declared statistical/numerical gates.
