# Filter and gradient repair resume checkpoint

Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Last pushed source
checkpoint: 23f681a14. Main remains unmerged; freshly fetched origin/main is
already integrated (131 repair commits ahead, zero behind).

Active questions: qualify the principal-angle SVD and exact clipped-norm
anchor repairs, then localize actual CDF optimizer trajectory sensitivity.
Plans/results: filter_gradient_principal_angle_precision_repair_20260928.md,
filter_gradient_factor_clipped_anchor_repair_20260928.md and
filter_gradient_dz5_locator_trajectory_result_20260928.md.
Current CPU plan: filter_gradient_dz5_callback_boundary_20260928.md.
Active fit plan: filter_gradient_dz5_exact_fit_inputs_20260928.md.

Through 04610; active: none.
SVD/locator allocation: 16/16 workers,
1785.877851/3600 combined seconds;
failed runs: none.
Anchor allocation: 8/8 workers,
354.936557/1800 seconds.
Preserved anchor-phase failures: [{'run': 4592, 'group': 'factor_clipped_anchor_graph_fit_cpu'}, {'run': 4604, 'group': 'factor_clipped_anchor_fit_gpu'}].
New callback-boundary allocation: 3/8 workers,
1020.696333/3600 CPU seconds.
Boundary failures: none.
The SVD, anchor and callback-boundary units are closed. New exact-byte fit
allocation: 5/8 workers,
252.169650/1800 combined seconds.
Global charged/reserved: CPU 100867.729700s / GPU 93083.299551s.
Remaining: CPU 27.981186h / GPU 26.143528h.
Caps remain 56 CPU / 52 GPU hours; the extra 24 CPU hours are already included.
One numerical worker at a time; explicit CPU reference or trusted eligible
GPU with verified memory growth. GPU 0 serves remote desktop and GPU 1 has an
active display; recheck occupancy before selecting a non-desktop GPU.

Checked findings:
- SVD accuracy/regressions and saved-fit replay pass CPU04579--04583.
  GPU04599/04600 reproduces the defect and passes repaired accuracy (max
  error1.55e-15); all48 GPU regressions pass04601. Device peak70400bytes is
  unchanged. Shared-device warm time is21.87% higher; matched uncontended
  attribution remains open. Strict angle-record differences are not waived.
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
- External callback barriers 04596/04597 preserve both original trajectories
  exactly, so that placement is not a repair. Readback/policy04598 passes161
  checks; do not repeat a full trajectory without a smaller graph diagnosis.
- Clipped-anchor trial 04587 passes 22 fixture/factor combinations in graph
  and XLA, maximum initial covariance change 6.94e-17. The shared runtime now
  uses the exact capped norm for first-anchor argmax. Installed repair 04588
  and factor regressions 04589 pass. CDF XLA fit 04591 exactly matches the
  pre-anchor-repair 04582 result. Graph fit 04592 still hits its loading-domain
  assertion. Both original and XLA report the first one-factor fit as invalid;
  the clipping-anchor repair does not resolve this separate graph-reference
  interruption. No unchanged retry is planned.
- GPU anchor trial04602 and12 regressions04603 pass. GPU fit04604 fails with
  code2 (right matrix fails symmetry/finiteness in family stability). Saved
  input ZIP hashes show differing offset bytes versusCPU04591; other three
  operands are identical. Exact CPU04591 input before/after replay must resolve
  this confound. Raw diagnostics are now saved before assertions. Policy04605
  passes160 checks. Source/harness additions after it need final renewal.
- Exact byte inputs04606-04609: both GPU fits reject (codes1/2), both CPU fits
  are usable with identical raw/public records. Readback/policy04610 passes161
  checks. GPU factor precision skew is at most7.63e-12, relative norm5.47e-16;
  condition numbers11175--12246, inverse backward errors below2e-17. Small
  off-diagonal entries fail the unchanged symmetry check (up to2.448x). Evaluate
  symmetric final Cholesky-inverse representation; do not widen tolerances.

Archive through 04595 is committed/pushed: 84 verified members, SHA-256
a756686f6164ddeca63ac4e17f654a84d9d3540ad16bc7f8791c096baec293ff.

Next: Exact-input diagnosis04606-04610 isolates tiny GPU Cholesky-inverse asymmetry triggering unchanged elementwise symmetry checks. Evaluate explicit final symmetric inverse representation with raw residual diagnostics; preserve conditioning/optimizer/gates. Commit/archive checkpoint, then continue full-consumer/lifetime and terminal F01-F20 work.
Next numerical queue must be specified in the symmetric-inverse repair plan.
Use the existing runner with 300-second limits. No desktop fallback while
busy non-desktop GPUs have ample memory. The old actual-consumer snapshot
qualifies its own source bytes only. Complete consumer renewal/lifetime,
isotropic/precision reporting and F01--F20 terminal dispositions remain open.

Preserve live MacroFinance dirty files and do not interfere with independent
campaigns. No subagents, training, HMC, packages/environment mutation,
global cache changes, system-limit changes or tolerance relaxation. Canonical
NeuTra remains author-profile IAF; unsupported LEDH claims remain blocked.
