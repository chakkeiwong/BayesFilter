# Filter and gradient repair resume checkpoint

Current checkpoint through 04058, September 26. The prior pushed commit is
`271051c90`; the latent SIR execution unit is committed on the same repair
branch. Read `filter_gradient_latent_sir_result_20260926.md` for its exact
result, and `filter_gradient_transitive_genut_result_20260926.md` plus
`filter_gradient_transitive_import_audit_20260926.md` for the preceding GenUT
continuation and source coverage.

The latent pre-clipping SIR simulator now uses a retained TensorFlow program
with native `tf.while_loop` time control and default XLA compilation. CPU/GPU
path, boundary, pullback, consumer and policy checks pass in 04038--04050 and
04057--04058; six fresh original/graph/XLA cost arms pass in 04051--04056.
The target remains explicitly `extension_or_invention`; no canonical LEDH or
Zhao--Cui filtering admission was made.

Two GenUT correction recurrences and the Austria callback matrix construction
are native. The audit guard now covers 269 sources / 1,422 exact allowances;
129 policy tests pass. GPU investigation found a compiler fusion abort in the
original and initial candidate, plus TF32 moment drift. The final tensor
contractions compile with TF32 enabled and pass CPU/GPU FP64/FP32 numerical
checks, independent moments, replay and consumer checks. Original FP32 GPU
XLA remains unavailable; its graph comparator is explicitly labeled.

Strict cross-mode report tests 04036/04037 still fail by one cap-active count
out of 216 near the 1e-7 displacement predicate. No tolerance was waived.
CPU repaired XLA uses about 54 MiB less RSS than original XLA in the fresh
fixture cohort; cold time is 0.720 versus 1.557 seconds and warm time is similar.
GPU costs are descriptive because sharing was observed. New N*d*d reductions
still need target-scale capacity qualification. All failed attempts, final
costs and source snapshots r1--r11 are preserved in the 04037 evidence archive.

Charges through 04058: 84740.235162 CPU / 77050.122941 GPU seconds, leaving
32.461046 CPU / 30.597188 GPU process-hours under unchanged 56/52-hour caps.
No campaign worker is active. The added 24 CPU hours are already counted.
Main remains unmerged. Canonical LEDH rebuild is excluded by user direction.

Next execute the registered mixed KR transport closure under the same source and
budget rules. Existing original precision disposition, repeated XLA native
retention, DZ5 graph replay/GPU graph finite-difference failures, external
callback autodiff/pfor, public LEDH/reset integration, initializer/supervisor,
reporting/isotropic cases, target capacity and F01--F20 terminal dispositions
remain open. Pending reporting proposals are not approved by elapsed time.
