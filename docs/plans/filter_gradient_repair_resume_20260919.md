# Filter and gradient repair resume checkpoint

Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Last pushed source
checkpoint: 5d398a45b. Main remains unmerged; origin/main was integrated.

Active questions: qualify the principal-angle SVD and exact clipped-norm
anchor repairs, then localize actual CDF optimizer trajectory sensitivity.
Plans/results: filter_gradient_principal_angle_precision_repair_20260928.md,
filter_gradient_factor_clipped_anchor_repair_20260928.md and
filter_gradient_dz5_locator_trajectory_result_20260928.md.

Through 04595; active: none.
SVD/locator allocation: 12/16 workers,
1657.617389/3600 combined seconds;
failed runs: none.
Anchor allocation: 5/8 workers,
164.802170/1800 seconds.
Preserved anchor-phase failures: [{'run': 4592, 'group': 'factor_clipped_anchor_graph_fit_cpu'}].
Global charged/reserved: CPU 99723.720225s / GPU 92636.048194s.
Remaining: CPU 28.298967h / GPU 26.267764h.
Caps remain 56 CPU / 52 GPU hours; the extra 24 CPU hours are already included.
One numerical worker at a time; explicit CPU reference or trusted eligible
GPU with verified memory growth. GPU 0 serves remote desktop and GPU 1 has an
active display; recheck occupancy before selecting a non-desktop GPU.

Checked findings:
- CPU SVD accuracy/regressions and saved-fit replay pass (04579--04583).
  The repaired maximum singular-value error is 8.99e-15. Eleven angle-report
  differences remain; no comparator waiver. GPU qualification is pending.
- Original/current locator traces use 474/504 callback rows (04584/04585).
  Current full locator record equals uninstrumented 04572 exactly. Readback
  04586 finds a 1.53e-13 score difference at identical input row 1; positions
  diverge at row 2. Both optimizers report unconverged. Full record mismatches
  remain (121 leaves), independent of their localized/accepted status.
- Replay 04590 passes all 14 saved-point target comparisons against graph;
  maximum error/target-bound ratio is 0.015264. Four stricter record-score
  comparisons fail. Fixed-output replay 04593 reproduces all 474 original
  positions and the full result exactly in both controllers. Original operand
  binding 04594 still reproduces the original, so input binding alone does
  not explain the difference. Readback/policy 04595 passes 162 checks. The
  specific compiler-context mechanism remains open; do not change tolerances.
- Clipped-anchor trial 04587 passes 22 fixture/factor combinations in graph
  and XLA, maximum initial covariance change 6.94e-17. The shared runtime now
  uses the exact capped norm for first-anchor argmax. Installed repair 04588
  and factor regressions 04589 pass. CDF XLA fit 04591 exactly matches the
  pre-anchor-repair 04582 result. Graph fit 04592 still hits its loading-domain
  assertion. Both original and XLA report the first one-factor fit as invalid;
  the clipping-anchor repair does not resolve this separate graph-reference
  interruption. No unchanged retry is planned.

Next: Archive and commit through 04595. GPU SVD/anchor checks remain queued. Next CPU unit: isolate callback compilation context; input binding alone is ruled out. Then refresh actual-consumer source/target evidence, qualify lifetime, and complete terminal F01-F20 review. Preserve graph failure 04592; no unchanged retry.
GPU queue: principal_angle_before_gpu, principal_angle_after_gpu,
principal_angle_regression_gpu, factor_clipped_anchor_trial_gpu,
factor_clipped_anchor_regression_gpu, factor_clipped_anchor_fit_gpu.
Use the existing runner with 300-second limits. No desktop fallback while
busy non-desktop GPUs have ample memory. The old actual-consumer snapshot
qualifies its own source bytes only. Complete consumer renewal/lifetime,
isotropic/precision reporting and F01--F20 terminal dispositions remain open.

Preserve live MacroFinance dirty files and do not interfere with independent
campaigns. No subagents, training, HMC, packages/environment mutation,
global cache changes, system-limit changes or tolerance relaxation. Canonical
NeuTra remains author-profile IAF; unsupported LEDH claims remain blocked.
