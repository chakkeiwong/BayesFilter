# Exact saved-input CDF fit localization

GPU04604 returned fit error2 after the clipped-anchor repair. The original
harness regenerated offsets on the executing device, and ZIP hashes show
different offset bytes from CPU04591. Center, center-score and score bytes
match. Its result cannot attribute the GPU failure to the runtime repair.
Also, raw failure diagnostics were discarded before the assertion. Preserve
04604 and correct both evidence defects before further interpretation.

The question is whether the installed fitter and the pre-anchor fitter return
the same decision on **identical** saved CPU04591 operands, on CPU and GPU.
Load `center`, `center_score`, `offsets` and `scores` from that run's NPZ;
verify each array's bytes against its saved operand hashes. Copy those exact
values to the executing device; never regenerate a cloud in this comparison.
Use the unchanged accepted-r2 recipe and thresholds. The before baseline is
5d398a45b (already includes the principal-angle SVD repair); the after source
is 23f681a14's shared numerical implementation. Preserve the complete baseline
dependency hashes. No target, cloud, optimizer, threshold or RNG is changed.

Run before/after CPU and GPU in separate fresh workers through registered
`dz5_exact_fit_{before,after}_{cpu,gpu}` groups. Each gets 300 seconds; reserve
at most eight workers and 1800 combined CPU/GPU seconds inside the unchanged
56 CPU / 52 GPU-hour totals. This is a distinct allocation after the closed
eight-worker anchor unit, not additional global time. One numerical worker at
a time, CPU explicitly hidden GPUs, trusted eligible GPU with verified growth,
unique numbered output directories and no environment mutation. Up to two
localized harness retries fit inside these limits. Stop unchanged numerical
retries; conserve enough allocation for saved-state readback.

Save every raw fitter output before checking the error status, per-array input
hashes, one-trace metadata, placement, environment, elapsed time and process/
allocator memory. The test's pass condition is a valid, complete diagnostic
artifact with the exact source/operand contract. A nonzero fit error remains
an explicitly failed numerical outcome, never a successful fit. When error=0,
decode and save the full public record and original1e-10 record differences
against04591. Readback must localize nonzero errors to family/replicate/pair,
inspect precision symmetry/eigenvalues and compare the exact before/after
matrices. Error2 means the right precision is invalid in a stability comparison;
it is not by itself proof of which operation created it.

Review: exact bytes remove the cross-device cloud confound but do not remove
backend arithmetic differences. A common before/after failure would weaken
the anchor-change explanation; after-only failure would require a repair of
that change. Preserve old first-anchor behavior only as the pinned comparator;
do not restore a known erroneous clipping tie to manufacture agreement.
Ill-conditioned/invalid precision should be reported and rejected, not made
equivalent through larger tolerances or an unreviewed ridge. Component diagnosis
does not admit the current source closure for the actual target, a whole
initializer, HMC, training or scientific/default readiness. Costs are descriptive
only until matched, uncontended repeated evidence exists.
