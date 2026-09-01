# Phase 37 Repair and Refresh Note

Date: 2026-08-25  
Active version: `v2.1-training-measure-bound`

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| aggregate attempt 1 | harness/schema mismatch: reporter read `calibration.particles` | inspect the pilot schema and bind the reporter to `calibration.particle_count`; preserve the failed root | repaired; no scientific interpretation made |
| aggregate attempt 2 | none | size-aware aggregation under the fixed target/protocol signatures | `PASS_THETA_SUPPORT_LADDER_HARD_GATES_DESCRIPTIVE` |
| N=64 fresh pilot | none | unchanged theta-measure protocol | `PASS_THETA_MEASURE_PILOT` |
| N=128 fresh pilot | none | unchanged theta-measure protocol | `PASS_THETA_MEASURE_PILOT` |
| N=256 fresh pilot | none | unchanged theta-measure protocol | `PASS_THETA_MEASURE_PILOT` |
| N=256 identity boundary | none | GPU/XLA batch-native screen | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| N=256 affine boundary | none | exact train-split weighted affine factor; oracle checked before training | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |

The pilot ladder increased retained roots (25/47/122 for M0) and passed all
common theta-measure gates, but the result is descriptive because size and
seed changed together. The downstream N=256 screens remained far from an
IID-Gaussian held-out diagnostic. The affine training-measure oracle passed to
about `1e-15`, while validation mean and covariance residuals remained large.

This is not a continuation veto. It is a repair trigger for a disjoint,
validation-selected checkpoint/objective diagnostic. The next subplan must
freeze the N=256 M0 bank, preserve the same target and measure, select any
checkpoint using calibration/validation only, and evaluate the untouched audit
partition without promoting the selected checkpoint or its moments to a
transport, posterior, or HMC claim.

The failed aggregate root remains preserved at
`phase37-support-ladder/aggregate/`; the valid aggregate is under
`phase37-support-ladder/aggregate-attempt2/`. No prior artifact was overwritten.
