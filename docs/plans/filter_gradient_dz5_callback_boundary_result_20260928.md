# CDF callback-boundary result

The [plan](filter_gradient_dz5_callback_boundary_20260928.md) tests diagnostic
XLA optimization barriers around callback inputs and outputs. Neither shared
runtime nor scientific target is changed. Earlier controlled transcript 04593
isolated target-output differences as sufficient to reproduce the two locator
paths; input binding alone was ruled out by 04594/04595.

Original-barrier run 04596 passed execution checks in 497.474 seconds. It
returned the same complete record as original 04584, with 474 callback rows.
Locator elapsed time was 457.653 seconds and peak process RSS 11,479,228 KiB.
These are descriptive single-run measurements; no performance ranking follows.

Candidate-barrier run 04597 passed in 512.883 seconds, with 504 callback rows,
471.141 locator seconds and peak RSS 11,485,708 KiB. Readback 04598 passed all
161 comparison/policy checks. Each barrier arm reproduces every unmodified
callback array by byte representation and its complete unmodified result.
The cross-controller discrepancy remains. Thus these input/output barriers
are not a repair and do not isolate the compiler cause. This diagnostic cannot
exclude optimization inside the callback or another enclosing-graph effect.
Both old optimizers reported unconverged; neither this experiment nor a
passing target replay establishes convergence.

| Decision | Criterion / veto status | Main uncertainty and next action | Not concluded |
|---|---|---|---|
| Reject this barrier placement as a sufficient repair | Both execution and bytewise readback checks pass; old discrepancy persists | Inspect the smaller first-score graph/context difference before another full trajectory | Compiler cause or runtime repair |
| Preserve original mismatches | No tolerance or comparator changed | Pointwise reference accuracy does not imply identical finite trajectory | Whole-consumer equivalence |
| Keep main unmerged | GPU/source-consumer/terminal work remains | Complete the existing master queue | Whole-program completion |

Local review: both negative controls are useful. Operand binding and external
callback barriers leave each trajectory unchanged, whereas identical callback
outputs make the two controllers exactly equal. Target arithmetic remains the
localized source of the difference, but a specific optimizer/fusion attribution
would exceed this evidence. Do not install the diagnostic barriers in runtime
or spend another full-trajectory retry without a smaller discriminating test.

Evidence is archived with the GPU repair continuation through04605 in
`artifacts/filter-gradient-repair-20260917/callback-boundary-gpu-repairs-04605-evidence.tar.gz`:
3,267,013 bytes, 64 reopened/hash-verified members, SHA-256
`88f352b5d727897d681ae84610ab18c434ed49da2c5063b4731652689f3231ed`.
The verification JSON identifies failed GPU04604 and every successful worker;
the archive does not turn that failure into qualification.
