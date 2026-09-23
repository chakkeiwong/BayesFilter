# Dense attempt and archive checkpoint

The native cloud-to-fitting attempt now converts the raw center score to scaled
coordinates before fitting, matching the external initializer. The draft had
omitted this multiplication; zero scores and unit scales hid it. The GPU test
counter is also corrected: `dtype=tf.int64` must be a keyword because the second
positional argument of `tf.Variable` is `trainable`. Int32 resource placement on
the CPU had stopped GPU compilation before any numerical comparison.

The recovered test no longer treats a source hash as a numerical comparison or
catches an unexpected fitter ValueError as a successful record. It executes the
exact external cloud loop and original3582b4ac fitter, checks all cloud archives
and fitted fields, counts real fitter calls, verifies unusable outputs, changes
and restores inputs without retracing/HLO changes, and checks collection plus
the original disconnected derivative boundary. The policy guard includes the
whole new numerical module without an exception.

| Evidence | Result |
| --- | --- |
| GPU03446--03449 | Centered, moved, invalid and fit-rejected D1 records pass. |
| CPU03455--03458 | Same four strengthened D1 cases pass. |
| CPU03461/03462 | D3 moved and invalid cases pass; actual fitter call count is zero. |
| GPU03450 / CPU03459 | D3 accepted geometry and all other records agree, but nearly singular prediction-Jacobian condition scalars fail the original numerical comparison. Complete attempt qualification remains open. |
| GPU03454 | Current standalone and composed fitting match every field. Same-state diagnostic crosses localize a condition-report discrepancy; it is not a passing full-record qualification. |
| CPU03460 | All21 NumPy-reader compatibility checks pass for the standard-library NPZ writer. |
| CPU03463 | All129 campaign/policy checks pass. Guard234 sources/1335 exact exceptions, two new host file-serialization entries only. |

All earlier failures are preserved.03443/03444 are the int32 resource mistake
and ineffective placement-only fix;03445 is the diagnostic device-name assertion.
03451's rounded-product trial did not alter the failed scalar and was removed.
03452/03453 are diagnostic harness errors, repaired without changing the source
authority. Earlier CPU fixture failures03427/03434 remain in their manifests.
The original weak CPU tests are historical, not replacements for the strengthened
qualification. Fresh GPU follow-up was declined by the selector under foreign
load; it launched no worker and establishes no GPU failure or cost evidence.

The NPZ writer uses TensorFlow for completed tensor materialization and the
standard library for NPY headers, numeric packing and ZIP compression. It covers
the original archive naming scheme, numeric/bool dtypes, empty/scalar/large
arrays, signed zero, subnormal values, nonfinite diagnostic floats and integer
limits. NumPy appears only in the independent reader tests. The two exact
allow-list additions cover traversal of archive members and byte chunks; no
numerical computation or runtime feedback is allowed in those loops.

The new [condition-report proposal](filter_gradient_dense_condition_reporting_proposal_20260923.md)
and the separate [E2 clipping-count proposal](filter_gradient_initializer_clip_reporting_proposal_20260923.md)
are concrete and pending an asynchronous owner decision. Neither criterion is
installed. Healthy geometry, rank, status, callbacks and selection remain fully
checked. The user-directed treatment of rejected unusable precision from02622
does not automatically cover this accepted-fit diagnostic.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain the coordinate-score repair | Complete D1 CPU/GPU original records with nonunit scales/nonzero scores | No tested D1 mismatch | D3 ill-conditioned diagnostic and wider controller remain | Decide reporting scope, then complete matrices | No full initializer admission |
| Retain NPZ serialization helper |21 independent exact reader checks pass | Unsupported dtypes/names reject | Actual completed initializer archive path remains unwired | Integrate after outer-controller qualification | No numerical or performance claim |
| Keep main unmerged | F01--F20 and public/actual-consumer gates remain open | Terminal acceptance is false | Pending comparisons, RNG and public memory/costs | Continue prepared controller and independent gates | No repository-wide policy closure |

Review: the original smoke could pass despite a missing coordinate conversion
and without executing its claimed reference. The strengthened evidence removes
those blind spots. A condition diagnostic near1e14 cannot be called accurate
merely because the downstream fit is usable; the failed scalar remains visible
and no new comparison is silently installed. Same-state attribution bounds this
finding to the tested fixture and does not rule out unrelated Jacobian defects.

The source/run receipt is
`artifacts/filter-gradient-repair-20260917/dense-attempt-checkpoint-03463.json`.
It records exact manifests, failures, commands, source hashes, environments and
charges. Main was refreshed to origin/e9fee5847 and remains unmerged. Outer
controller drafts are prepared for the next bounded unit; seeded RNG, actual DZ5
target/transition, public initializer/staged wiring, terminal costs and final
endpoint audit/integration still remain. No completed CDF campaign was restarted.
