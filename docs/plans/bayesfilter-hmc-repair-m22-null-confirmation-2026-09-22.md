# M22 frozen null confirmation

Source and launch details are superseded by [the merged-source continuation](bayesfilter-hmc-merged-source-continuation-2026-09-22.md). This original inventory was stopped before numerical launch; preserve it as planning history. The targets, denominators and screens carry forward to the new inventory.

The current-source pilot completed two independent experiments each for the
baseline and no-op controls. All four were valid with no rejection. The measured
worker charge was 26.38678574701771 seconds. This tests execution and cost;
four observations cannot establish the null rejection rate.

Freeze 512 fresh complete experiments per arm (1024 total) at root seed
2026092232. Keep M22's source snapshot, target, transition, kernel power,
anchor count, rank count, independent-look schedule, statistics, thresholds
and multiplicity. No pilot observation or older-source trial enters the new
denominator. At the measured pilot rate, 1024 experiments project about 6755
CPU seconds, including repeated attribution of process startup overhead.
Reserve the already planned 10800 seconds within M22's 43200-second ceiling.

Every experiment must be valid. The primary screen remains an exact pointwise
95% Clopper--Pearson upper bound <=0.10 for each arm's rejection rate. Invalid
experiments are not counted as detections and cannot be silently dropped;
they trigger diagnosis. A complete but wider interval is inconclusive. The
study does not prove exact nominal size, kernel correctness for all targets,
or full-pipeline defect power.

Run the existing power engine with its frozen invariance calibration design
from m22-r1/source-r1. Use a fresh null-confirmation-cpu-r1 directory, hidden
GPUs, one TensorFlow intra/inter-op thread, and a single numerical worker.
The shared queue permits one other independent worker. Preserve trial-level
results, look inventories, exact commands, seeds, source hashes, wall time
and all failures. No count grows in response to a rejection rate.

Skeptical audit: the current-source pilot is valid; the cost bound is adequate
at measured cost; the fixed study size is sufficient to tighten M16's interval
without conditional replication. A passing screen remains an operating-
characteristic result for this exact test and kernel, not evidence for the
separate complete-fit mutation question.
