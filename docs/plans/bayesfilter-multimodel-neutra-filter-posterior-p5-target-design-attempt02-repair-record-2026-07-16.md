# P5 Target-Design Attempt 02 Repair Record

Date: 2026-07-16

Attempt root:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/target-design/attempt-02`

Classification: `HARNESS_CPU_XLA_TENSORLIST_VARIANT_UNSUPPORTED`.

The repaired runner persisted `attempt_status.json`, then failed while compiling
the first design-seed likelihood-information graph. CPU XLA rejected
`TensorListReserve` with a variant element type created by autodifferentiating
the fixed-horizon innovation-history `TensorArray`. No design row,
prior-predictive result, negative-control result, dataset, signature, or
scientific decision was produced.

The repair replaces the two fixed-horizon `TensorArray` histories with
preallocated `[T,B]` tensors updated by `tf.tensor_scatter_nd_update` inside the
same `tf.while_loop`. The computed predictive means, innovation variances,
derivatives, information formula, seeds, points, thresholds, hardware class,
and budget are unchanged. Focused eager and CPU-XLA information regressions
must pass before attempt 03 uses a fresh root.
