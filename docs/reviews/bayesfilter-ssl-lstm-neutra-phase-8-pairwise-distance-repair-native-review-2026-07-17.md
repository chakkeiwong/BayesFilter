# Phase 8 Pairwise-Distance Repair Native Review

Date: 2026-07-17

Verdict: `AGREE_EXACT_SHAPE_CANARY_ONLY`

Target-pilot repair 02 passed the validated chunked G/H forecasts, then failed
inside `pooled_pairwise_distance_scale` on pooled shape `[8,64,2,10]`. XLA
reported a runtime buffer-size mismatch from `tf.boolean_mask`. No receipt was
written and no G/H difference was computed.

The original statistic selects strict upper-triangle distances, excludes exact
zeros, sorts the positive values, and takes their median. The repair preserves
that definition without a data-dependent tensor shape:

1. compute the full fixed `[N,N]` distance matrix;
2. mark only strict-upper-triangle positive entries as candidates;
3. replace every non-candidate with positive infinity;
4. sort the fixed `N*N` vector; and
5. index the median using the scalar candidate count.

The finite candidates sort before infinity, so the selected order statistics
are exactly those of the original positive upper triangle. A zero candidate
count uses clamped gather indices and then returns NaN, preserving the public
duplicate-degenerate fail-closed behavior.

Focused checks pass: manual median formula, eager/XLA parity, exact duplicate
exclusion, all-duplicate veto, absence of `boolean_mask`, 59 predictive tests,
three exact-shape-canary contracts, Python compilation, and `git diff --check`.

The bounded GPU canary uses deterministic TensorFlow paths at the exact failed
shape `[8,64,2,10]`, compares compiled and eager values/counts, requires one XLA
trace and GPU placement, and reads no retained sample or forecast artifact. It
cannot authorize target-pilot repair 03 or calibration by itself.
