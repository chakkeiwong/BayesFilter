# CDF locator trajectory localization

Original-locator run04584 completed and passed source, start replay, callback
accounting and enclosing-XLA checks. It returned localized/accepted after474
callback rows. This is the complete pinned d6a568384 locator with the declared
JIT reference adjustment, using the frozen actual CDF value/analytical-score
callback and accepted-r2 start. Every callback row is retained in
`run-04584/locator-callbacks.npz`; no training or HMC was executed.

The reusable locator 04585 passed in 495.404 seconds and returned
localized/accepted after 504 callback rows. Its complete locator record is
exactly equal to the uninstrumented enclosing initializer record from 04572,
including every position, score, status and counter. For this candidate run,
the observer did not change the saved terminal behavior. The snapshot predates
the SVD and anchor repairs and qualifies its own source bytes only.

Original locator elapsed seconds: 422.6814495159779. Source SHA-256: `575e044f8e33f20b6fc13be1f48b3356110e0ec77799378d0cc738e89ddfc40f`.

The observed process peaks and compile-plus-search times are descriptive;
these two searches take different trajectories and are not a matched warm
performance comparison. Both use XLA, so this table also does not compare
graph-only memory with XLA memory.

| Locator | Callback rows | Locator seconds | Process peak RSS (KiB) | Graph bytes |
|---|---:|---:|---:|---:|
| Pinned original, 04584 | 474 | 422.681 | 11,490,672 | 2,951,177 |
| Reusable candidate, 04585 | 504 | 455.811 | 11,488,084 | 2,958,169 |

Readback 04586 passed in 5.227 seconds. The first score difference occurs at
identical input row 1, with maximum absolute difference 1.53e-13. Positions
first differ bitwise at row 2, then exceed the unchanged 1e-10 record bounds
at row 88. There are 121 complete-record mismatches between original and
candidate locators. Both report `optimizer_converged=False` and
`optimizer_failed=False` under the frozen 240-iteration recipe; acceptance
must not be described as a converged optimum.

Replay 04590 passed in 76.554 seconds, evaluating 14 saved input occurrences:
both arms at the first-difference indices and their two selected centers.
Every saved and standalone-XLA value/score agrees with the explicit graph
reference at the existing target bounds (1e-8 absolute, 1e-7 relative).
The largest error/target-bound ratio is 0.015264. All validity and branch
checks pass, each program traces once, and no host callbacks are present.
Four score comparisons at candidate row 88 and its selected center exceed
the separate 1e-10 record bounds; those failures remain explicitly recorded.

| Decision | Primary criterion | Veto status | Main uncertainty / next action | Not concluded |
|---|---|---|---|---|
| Retain the first-divergence diagnosis | Score differs before positions at the same input | Target-bound replay and validity pass | Compiler-context rounding is observed; later optimization amplification still needs a controlled sensitivity check | A wrong analytical score or equivalent optimizer trajectory |
| Preserve terminal mismatch | 121 old full-record comparisons fail | Both optimizers are unconverged | Keep whole-consumer equivalence open; do not retune or relax the comparator | Full initializer admission |
| Preserve cost observations only | Separate isolated original/current workers completed | Instrumented candidate exactly matches its uninstrumented record | No repeat uncertainty or matched performance experiment | Runtime or memory superiority |

The strongest alternative explanation is sensitivity of the finite L-BFGS
trajectory to rounding, rather than a defective pointwise analytical gradient.
This replay weakens a large pointwise defect explanation only at the sampled
positions. A controlled identical-output callback experiment or an independent
derivative failure at these points would distinguish the alternatives further.
No training, HMC, live external source edit or tolerance change was performed.

Controlled transcript run 04593 passed in 13.445 seconds. With identical saved
callback outputs, both controllers reproduced all 474 original CDF positions
bitwise and every field of the real original locator result from 04584. Neither
exhausted the transcript, and each traced once. This isolates target-output
rounding as sufficient to explain the observed difference in this diagnostic;
no recurrence discrepancy remains on this transcript. The callback deliberately
ignores its positions, so it is never a scientific target or a runtime option.

The next controlled check changes the original controller's initial/scale
capture into explicit function operands. This directly tests the source-level
compilation-context distinction; the raw original comparison remains preserved.

That check, 04594, passed its execution checks in 470.098 seconds but **does
not support the input-binding explanation**. The otherwise original controller
still produced exactly the original 474 callback rows and full result, rather
than the candidate's 504 rows. Readback 04595 verifies every callback array by
byte representation, including the fixed-output transcript. All 162 readback
and policy checks pass. These test passes verify the diagnosis and evidence;
they do not convert the failed hypothesis into a successful repair.

The transformed original source SHA-256 is
`1fa881257e3102fab26244296f6c2c7a9a3a75d7f1a9ab6717adda6047b14f7c`.
The optimizer/target-source/settings remain frozen. The next discriminating
test should isolate the callback's compilation boundary or compare the target
arithmetic in the enclosing graphs, preserving the initial identical input
and 1.53e-13 score discrepancy. Do not repeat an unchanged full trajectory,
alter the target or thresholds, or infer a specific compiler optimization
without evidence. Source-frozen consumer equality remains open.

All runs 04584--04595, including failed graph run 04592, are preserved in
`artifacts/filter-gradient-repair-20260917/locator-anchor-cpu-04595-evidence.tar.gz`.
The archive is 4,728,238 bytes, has 84 reopened/hash-verified members, and
SHA-256 `a756686f6164ddeca63ac4e17f654a84d9d3540ad16bc7f8791c096baec293ff`.
Its verification JSON lists every member, run and checksum. The prior
`dz5-initializer-adapter-04578-evidence.tar.gz` supplies the frozen source/input
dependency. The new archive includes the repaired source and current diagnostic
harnesses; each run retains its actual source hashes and isolated child source.
