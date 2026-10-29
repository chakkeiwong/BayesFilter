# Isolate Linux RSS accounting from TensorFlow compilation

Question: can the rusage-below-resident discrepancy from04822–04825 occur in
a plain allocation-only process, and does distributing first-touch allocation
across CPUs enlarge it? Matching installed kernel headers distinguish the
approximate global per-CPU-counter read from summing all local counts. This
is a specific mechanism hypothesis, not yet a complete syscall-source proof.

Use two sequential fresh Python standard-library processes, with no TensorFlow,
NumPy, compiler, model or GPU imported. Each allocates32 independent private
anonymous mappings of17MiB, writes one byte per OS page to materialize all
pages, retains them through the final snapshot, then closes them. Both arms
touch544MiB in the same order. The pinned arm uses the lowest available CPU
for every mapping; the spread arm cycles over the lowest32 available CPUs.
Record original/chosen affinity and restore process affinity in finally.
This changes only each diagnostic process's own scheduling. No package,
system setting, cache flush or other process is changed.

Collect status VmRSS/VmHWM/components, smaps_rollup RSS/PSS/Anonymous/huge pages,
and getrusage before allocation, after each touched mapping and after release.
Record counter read order, duration, allocation sizes, Python/kernel/hardware,
command, source hashes, matching installed-header extracts/hashes and wall time.
Reads remain non-atomic. The largest observed sample is a lower bound on the
true peak, never an exact maximum. After release, counter differences must be
reported without mistaking retained high water for current live memory.

Pass/fail concerns are complete artifacts, no prohibited imports, correct
allocation/cleanup, and growth measured by the page-map walk within2MiB of
the544MiB touched pages. Missing counters, allocation failures or a larger
unexplained discrepancy invalidate that diagnostic. A reproduced rusage deficit
without TensorFlow weakens an XLA-specific explanation. A larger deficit in
the spread arm supports local-counter batching; its absence fails to support
that mechanism and does not waive the original observation. No requirement
that a proposed explanation succeed. No statistical performance comparison,
filter cost acceptance, exact syscall attribution or peak-capacity claim.

Bound this unit to two60-second children and120 CPU process-seconds total,
zero GPU seconds, one child at a time; no automatic numerical retry. The
driver reserves/charges120 seconds conservatively via an ordinary supplemental
compute receipt before execution, replacing it with measured driver wall time
only after terminal artifacts exist. Use unique root
`artifacts/filter-gradient-repair-20260917/os-memory-accounting-20260929-r1`.
Command: existing tf-gpu Python runs
`scripts/filter_repair_os_memory_diagnostic.py --output <root>` with CUDA hidden
before child launch. This standard-library diagnostic is outside TensorFlow
algorithm policy and cannot substitute for a GPU/default measurement.

Skeptical review: unequal total mappings, inherited rusage floors, hidden
TensorFlow imports, touching only virtual pages and allocator reuse could
defeat the question. A fresh standard-library supervisor, exact equal private
mapped bytes/page touches, explicit imports and independent smaps observations
address those risks. Kernel headers may not expose the full running syscall
path; preserve that uncertainty. The prior numerical cost cohort is frozen
and remains valid. No runtime numerical source, tolerance or scope changes.
Self-review passes for this bounded explanatory diagnostic.
