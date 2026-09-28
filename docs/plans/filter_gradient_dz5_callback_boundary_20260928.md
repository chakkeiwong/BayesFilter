# CDF callback compilation-boundary localization

The locator investigation through 04595 distinguishes callback arithmetic from
controller arithmetic. Both controllers exactly reproduce the original trace
when given its identical callback outputs. Fourteen real saved positions pass
the existing graph/XLA target bounds. At the first identical optimizer input,
however, the actual compiled callbacks differ by 1.53e-13. Changing original
input capture to explicit operands did not change that result. The precise
compiler mechanism remains unresolved; a wrong analytical gradient has not
been demonstrated at these positions.

This diagnostic tests whether fusion across the callback boundary contributes
to the difference. Add `XlaOptimizationBarrier` around callback input positions
and its value/score/validity outputs in the **observer harness only**, for both
the pinned original and reusable locators. The operation is an identity; it
does not round, clip, quantize or change the score formula. Do not change the
shared runtime or claim that a compiler barrier is already a qualified repair.
Keep the same frozen target/source snapshot, accepted-r2 start, observer,
configuration, precision, data and validation checks as 04584/04585. Archive
every callback, complete locator record, source hash, trace/graph information,
host peak RSS and elapsed time. No HLO export is required for this first test.

Use `dz5_locator_boundary_original_cpu` and
`dz5_locator_boundary_candidate_cpu`, each with a 900-second parent and
840-second child deadline. The actual target remains the previously qualified
read-only snapshot; this cannot admit the newer SVD/anchor source closure.
Use the existing runner, CPU hiding and one numerical worker at a time. This
new allocation permits at most eight workers and 3600 combined CPU seconds
from the unchanged 56 CPU / 52 GPU-hour caps. The owner's extra 24 CPU hours
were already counted. This is not an extension of the prior SVD/anchor units.
At the start, 28.298967 CPU / 26.267764 GPU hours remain; the pending GPU queue
retains its own earlier allocation. No packages, environment or live external
sources may be changed.

Primary diagnosis: compare both new transcripts with each other and with the
unmodified traces, preserving exact representations, the 1e-10 record bound,
and the 1e-8/1e-7 target-value/score bound on identical input positions. If both
controllers agree with barriers, that supports cross-boundary optimization as
the mechanism and motivates a separately tested shared repair. If they still
differ, reject this mechanism as a sufficient explanation and inspect a smaller
graph/operation boundary before another full trajectory. Stop the affected arm
on missing source, invalid start, nonfinite target, host callback, overflow,
multiple traces or deadline; preserve failures, and do not retry unchanged.
An isolated harness defect may be repaired at most twice within this budget.

Review: barriers may change the finite optimizer path even when each score
remains accurate. Matching acceptance or callback counts alone is insufficient.
Agreement between barrier variants does not make the original comparison pass,
prove convergence or establish performance. Target errors must be checked at
matching positions; callback row indices cease to identify equal inputs after
positions diverge. Any favorable cost numbers are descriptive single-run
observations. The experiment is an explanatory diagnostic, not a new method,
source admission, HMC or training run. The smallest justified next step is
saved-array readback and, only for changed common-position outputs, a bounded
graph-reference replay.
