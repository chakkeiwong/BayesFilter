# Execution review
CPU-only implementation checks: initial 19 pass / 1 failure (23.72s); the new
cap tail-width constant had been rounded through FP32 by tf.cast. Replaced it
with a dtype-specific tf.constant. Then 20 passed (22.39s). Added an independent
complete-map comparison with the uncapped intended correction on a healthy
fixture: 21 passed (26.42s), complete final log cpu-checks-02.log.
No GPU was used by these CPU checks; CUDA_VISIBLE_DEVICES=-1 was explicit.
Initial failure was an implementation regression, not a scientific candidate
failure or a GPU infrastructure retry.

Radius calibration: tested 2/4/8; 8 is smallest covering all declared healthy
fixtures (maximum 5.492 standard deviations). Exact cap identity and derivatives,
complete-map non-harm within roundoff, and pre-recolor stress bound pass.
The safety fixture set is narrow and cannot establish general safe behavior.

Reference unit: all fifteen data/horizon references pass refinement and
independent grid checks. Four GPU check units: all sixteen analytical score
coordinate checks pass branch-matched finite differences. Trace and ordinary
executor agree. Device/memory-growth/XLA/FP64 provenance is in each manifest.

Stage unit: all five arms, both coordinate traces valid. New richer design
and calibrated protection together avoid the observed reset/cap kurtosis loss.
The actual stage case is fresh calibration data 241001; plan corrected to
match it. Every stage comparison is within a trajectory; different arms have
different subsequent clouds, so cross-arm source moments are not held fixed.
Reversal is a sensitivity diagnostic, not a tuned design choice.

Before calibration: all checks pass, no failed GPU attempt. Only 414.773004s
of the new allowance has been charged. Seven arms x four calibration pairs
per route/horizon remain balanced. Observed compilation and T10 cost fit the
4200s provisional calibration allocation; supervisor retains the aggregate
and elapsed hard stops. No numerical/code changes are needed before launch.

Calibration completed: eight route/horizon units, 224 evaluations, all valid.
The positive-iteration numerical program remains unchanged after calibration.
A static terminal call-chain check found missing stage keys in the general
zero-iteration early return. Added the unchanged point/tangent stage keys.
This changes observability only and preserves the existing no-op semantics.
The initial edge-case test omitted required strength; its two failures were
test-setup failures, not executable confirmation of the missing-key finding.
Corrected test passes (zero-stage-after.log, 4.61s); other 25 focused checks
pass in cpu-checks-04.log. A separate command used a nonexistent test path and
ran no tests (cpu-checks-03.log). These CPU checks intentionally hid GPUs.
No GPU unit failed and no GPU retry has been used.

Validation readiness review: sources changed only in the unused zero-iteration
branch; the new snapshot will record that difference. Frozen choices are below.
Validation failure will veto nomination, not stop the prescribed untouched
comparison or trigger tuning on validation data. There is enough allowance for
validation and the four-design comparison; evaluate measured costs before the
balanced extension.

Frozen selections: {"calibration__iid_dual_cap__10-attempt-01": "steps8", "calibration__iid_dual_cap__120-attempt-01": "epsilon25", "calibration__previous_inverse_cdf__10-attempt-01": "epsilon25", "calibration__previous_inverse_cdf__120-attempt-01": "steps8", "calibration__repaired_permutation__10-attempt-01": "epsilon25", "calibration__repaired_permutation__120-attempt-01": "steps8", "calibration__repaired_permutation_ablation__10-attempt-01": "epsilon25", "calibration__repaired_permutation_ablation__120-attempt-01": "steps8"}

GPU seconds charged 33936.335711; remaining 9263.664289.

Untouched four-design phase complete, all eight units finished in 1023.464570s. Balanced extension preflight: remaining 7778.196388s exceeds 1.5 times measured phase cost plus 300s reserve. Extension uses only the prespecified four additional designs in every cell; choices and criteria stay frozen.
