# Shared small LM product precision repair

This is a localized continuation of `filter_gradient_ledh_seeded_public_20260929.md`
within its existing24-worker/5400 CPU/3600 GPU second allocation, not extra
compute. Runs04669--04677 have used9 workers. The frozen failed GPU endpoint
04672 remains evidence; costs and promotion are held until this repair and
endpoint renewal pass.

The engineering question is whether the shared two-column LM solver can retain
float32 product accuracy with TF32 enabled, so eager/reference and XLA owners
compute the same declared scaled normal equations. On identical captured
inputs04677, TF32 creates normal-matrix errors up to5.04e-4 and coefficient
error8.59e-3 relative to FP64. Disabling TF32 only inside the LM helper reduces
first trust-output error from2.23e-4 to5.68e-7. Full-reset FP64 eager/graph/XLA
agree within4.9e-15. Conditions8--196 do not establish severe ill-conditioning.
This is a localized arithmetic precision defect in the old comparator as well
as a non-XLA runtime helper, not a seeded-stream mismatch or failed reset.

Repair `genut_shape_lm_tf.py`, the shared value/analytical-JVP authority. Replace
the tiny Gram and matrix-vector contractions with explicit TensorFlow element
products and reductions, including the corresponding derivative contractions.
The equations remain S=J_scaled^T J_scaled+lambda I, b=J_scaled^T r and
d a=S^-1(db-dS a). Column scaling, scale floor, damping, strengths, dimensions,
solve, dtype, correction count, trust radii and reset/tuning identity stay fixed.
No global TF32 toggle, FP64 runtime promotion, compiler flag, tolerance increase
or value/score fork is installed. The small product tensors have bounded extra
extent (two coefficient columns); they are not a particle-pair dense matrix.

Before running, qualify saved offending matrices and changed nonzero directions
against independently computed NumPy FP64 scaled normal equations and finite
differences. Matrix bounds are atol=rtol3e-7 for FP32,1e-12 for FP64. Coefficient
bounds are atol2e-6/rtol1e-6 for FP32 and1e-10 for FP64; these are new primitive
accuracy gates, not relaxed endpoint tolerances. Analytical JVP must satisfy
the existing FP64 automatic-oracle test and independent finite differences;
check nonzero multiple directions and graph/XLA agreement on CPU/GPU.
FP32 analytical directional accuracy uses atol2e-5/rtol3e-5 against converged
FP64 five-point differences at2e-4 and1e-4; the relative bound corresponds to
float32 epsilon times the observed upper condition201. FP64 directional bounds
are2e-8, with the existing automatic-oracle gate unchanged at1e-11. Preserved
pre-repair records demonstrate that a defect existed before the fix. Compare
old/new XLA results at unchanged1e-6, ensuring the repair does not degrade the
already more accurate compiled path. The old TF32 eager error remains a
reported before/after difference, never renamed numerical equivalence.

Renew registered endpoint CPU/GPU healthy and rejected fixtures with the
repaired shared helper and unchanged1e-6 complete-record gate. The frozen
filter/flow comparator deliberately imports the current shared reset authority;
this is execution equivalence after the localized arithmetic repair. It cannot
prove bitwise equivalence to the erroneous pre-repair eager helper. Keep the
old failing records and explain that distinction in the result. Also renew
native seeded/fixed-input/analytical-JVP regressions and enforce source coverage.
Only then execute the planned isolated matched costs, using the healthy seeded
dual-trust fixture to exercise the repaired helper (N=8,d=2,T=3,seed123). The
frozen eager filter imports the repaired reset in this execution comparison;
the erroneous original TF32 result remains separately archived. Record this dependency
change for terminal source applicability; old full-reset measurements do not
automatically describe the repaired helper.

Use the existing campaign runner with registered `ledh_seeded_lm_cpu|gpu` groups,
explicit CPU references, and trusted GPU with verified growth/TF32 enabled.
Unique run artifacts retain captured input/source identities and all failures.
Stop qualification on primitive accuracy, healthy record, analytical derivative,
replay/trace, allocation or device failures. A rejected candidate triggers
localization under the remaining allocation, not a gate waiver.

Skeptical review: requiring the repaired XLA owner to imitate an inaccurate
TF32 eager matrix multiplication would test the wrong numerical authority.
Conversely, switching the reference silently would hide the defect. The saved
identical-input FP64 comparisons, isolated-LM intervention and retained failing
endpoint address those risks. Explicit products can alter reduction ordering;
old-XLA comparisons, independent derivatives and current full-record gates
remain mandatory. This repair improves implementation arithmetic, not method
calibration or canonical LEDH admission. No trained-map, posterior, HMC,
statistical ranking or whole-program completion is claimed. Primary-agent
review completed; no independent review asserted.
