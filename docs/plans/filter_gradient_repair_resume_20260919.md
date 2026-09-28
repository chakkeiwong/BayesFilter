# Filter and gradient repair resume checkpoint

Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Qualified numerical source checkpoint:4bf50d914 (pushed). Main remains unmerged; origin/main is
integrated. Check git HEAD for subsequent documentation/source checkpoints.

Active question: finish actual-consumer source renewal, repair reference-import
isolation, localize the first locator discrepancy, and reconcile endpoint evidence.
Through 04631; active: none.
Global charged/reserved CPU 103609.790750s / GPU 96071.347964s.
Remaining CPU 27.219503h / GPU 25.313514h.
Caps are56CPU/52GPU process-hours; extra24CPU hours are already counted.
Symmetry unit closed: 7/12 workers,
481.384047/2400 combined seconds, no failures.
One numerical worker at a time. CPU is an explicit reference; GPU requires
trusted eligible device and verified memory growth. GPU0 serves remote desktop
and GPU1 has active display. Recheck non-desktop occupancy before any GPU job.

Checked findings and evidence:
- Symmetry result: filter_gradient_factor_precision_symmetry_result_20260928.md.
  CPU04611/GPU04612 trials and derivatives pass. Exact GPU04613 fit is usable
  (before04607 rejected with error2), CPU04614 remains unchanged. GPU04615 and
  CPU04616 each pass52 regressions; readback/policy04617 passes161 checks.
  Optimizer, covariance, anchors and validity fields are preserved; factor
  precisions are exactly symmetric. Strict CPU/GPU record differences4539
  remain recorded. GPU allocator peak2306304bytes is unchanged.
- SVD result: filter_gradient_principal_angle_precision_repair_20260928.md.
  CPU04579--04583 and GPU04599--04601 qualify accuracy; GPU error1.55e-15,
  48 regressions pass, peak70400bytes unchanged. Shared-device warm time+21.87%
  remains descriptive; uncontended cost attribution and strict angles are open.
- Anchor result: filter_gradient_factor_clipped_anchor_repair_20260928.md.
  Initialization/regressions pass both backends. Exact-input04606--04610
  separated the final precision defect from anchor changes; the earlier
  regenerated-cloud GPU failure04604 cannot attribute an anchor regression.
- Locator: filter_gradient_dz5_locator_trajectory_result_20260928.md and
  filter_gradient_dz5_callback_boundary_result_20260928.md. Original/current
  callbacks474/504, first score difference1.53e-13 at identical row1;121 strict
  record leaves differ. All14 saved-point graph/XLA target checks pass04590.
  Identical-output controllers reproduce the original exactly04593. Explicit
  input binding and external barriers do not resolve the difference. Both
  optimizers remain unconverged. No further full trajectory without a smaller
  graph-context diagnostic; no tolerance changes.

Next: Execute original/candidate first-objective locator-context diagnostic and readback within its separate1800-second CPU allocation.
Target cohort04618--04624 and fresh r2 admission pass;
archive dz5-source-renewal-target-04624-evidence.tar.gz has816 verified members.
Renewal unit (closed_renewed_evidence_passed): 11/14 workers,
4955.284692/15000 combined seconds.
Plan: filter_gradient_dz5_source_renewal_20260928.md.
Adapter repair: qualified_real_import_prior_and_policy_checks;
2/4 workers, 33.546684/1200 CPU seconds.
Locator first-objective diagnostic: allocated_after_adapter_qualification;
0/6 workers, 0.000000/1800 CPU seconds.
Endpoint evidence index: readback_and_integrity_tests_passed;
5.906675/600 CPU seconds.
Old consumer snapshots qualify only their own bytes. Strict precision and
isotropic reporting, locator rounding, matched current-source cost attribution,
registered LEDH consumer migration and F01--F20 terminal dispositions remain
open. Do not reuse old admission or classify numerical mismatches as equivalence.

Preserve live MacroFinance files and other campaigns. No subagents, training,
HMC, package/environment mutation, global cache changes, system-limit changes
or tolerance relaxation. Canonical NeuTra remains author-profile IAF; unsupported
LEDH claims remain blocked. Do not merge main until all master gates pass.
