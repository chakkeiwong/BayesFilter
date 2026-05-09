# Student DPF baseline nonlinear-smoke result

## Date

2026-05-10

## Outcomes

| Implementation | Fixture | Status | Runtime seconds | Classification / reason |
| --- | --- | --- | ---: | --- |
| advanced_particle_filter | advanced_range_bearing_student_t_short | ok | 0.565016 | advanced_nonlinear_smoke_runnable |
| 2026MLCOE | mlcoe_range_bearing_short | ok | 2.89493 | mlcoe_nonlinear_smoke_runnable |

## Interpretation

H3 is supported as a smoke result: both student snapshots expose nonlinear range-bearing paths that can run through quarantined wrappers.  This is not a reference-consistency result.

The smoke paths are useful for deciding whether nonlinear adapter work
is feasible.  They do not provide correctness, convergence, or HMC
target evidence.
