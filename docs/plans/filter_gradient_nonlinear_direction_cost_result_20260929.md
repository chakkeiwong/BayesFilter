# Nonlinear direction costs and memory-counter result

The four fresh-process CPU-reference arms04822–04825 pass the declared
complete-output comparison. UKF outputs match exactly; LEDH diagnostics differ
by at most1.11e-16. Run04826 passes163 combined cost, qualification and policy
checks. Source/input/affinity/environment identities match within each pair;
the LEDH fixture binds validated nomination04806 and unchanged eight/eight
counts. These are descriptive measurements, not accepted performance rankings.

| Scope | Prior warm median | Enclosing warm median | Enclosing/prior cold total | Added warm RSS |
|---|---:|---:|---:|---:|
| UKF d1,T2 | 2.390787 ms | 0.455612 ms | 1.003563 | 14.285156 MiB |
| LEDH diagnostics d1,N8,T2 | 6.005153 ms | 3.541901 ms | 1.101679 | 105.546875 MiB |

Cold totals include construction plus first synchronized evaluation:
3.941514/3.955558 seconds for prior/enclosing UKF and9.537191/10.506922 seconds
for prior/enclosing LEDH. Each process had three conditioning and30 timed
calls, synchronized all shared outputs, and traced once. Prior auxiliary
stacking occurred after timing. Compilation of a comparator and HLO export
did not contaminate primary measurements. Host RSS changed by0/0MiB for UKF
and0.230469/0.386719MiB for LEDH between cold and warm snapshots. Python
collection did not reduce retained RSS; this is not evidence of native
executable eviction or a universal leak/capacity bound.

The peak-counter discrepancy reproduces on kernel6.8.0-138-generic:

| Arm | Warm VmRSS = VmHWM = smaps_rollup RSS | getrusage peak | Underreport relative to sampled RSS |
|---|---:|---:|---:|
| UKF prior | 766.507812 MiB | 712 MiB | 54.507812 MiB |
| UKF enclosing | 780.792969 MiB | 724 MiB | 56.792969 MiB |
| LEDH prior | 1107.386719 MiB | 1068 MiB | 39.386719 MiB |
| LEDH enclosing | 1212.933594 MiB | 1180 MiB | 32.933594 MiB |

All status RSS component sums are exact. Status and smaps RSS differ by at
most8KiB across all stages, and agree at every cold/warm/collection snapshot.
Thus rusage is unsuitable as a guaranteed peak-capacity authority here; the
observed enclosing-memory increment is corroborated by two independent OS
interfaces. Reads are sequential and non-atomic. They neither observe every
transient peak nor identify live TensorFlow allocator bytes.

Read-only inspection of installed matching-kernel headers finds
`include/linux/mm.h:2659` uses `percpu_counter_read_positive` for ordinary
`get_mm_counter`, while line2664 provides a summed alternative.
`get_mm_hiwater_rss` at2714 uses the ordinary RSS counter. In
`include/linux/percpu_counter.h:82–89`, local updates are documented as absent
from the global count until synchronization; the sum includes them. This is
a concrete accounting mechanism to test, not a proved syscall call-chain
attribution: full matching `kernel/sys.c` and `fs/proc/task_mmu.c` are absent
locally. A bounded allocation-only diagnostic should separate this mechanism
from TensorFlow/compiler residency without system mutation.

The cohort closes after5 workers/63.357384 CPU seconds, zero GPU seconds,
within its6-worker/1800-second cap. Each numbered manifest in the established
artifact root preserves command, sources, hardware/software/environment,
fixture identity and wall time. CPU is an explicit reference, not the default
GPU production target. No runtime source or numerical threshold changed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Accept bounded measurement readback | Complete parity and comparable provenance pass | None for the recorded screen | One process per arm, tiny scope | Replicate matched costs before cost acceptance | Statistical speed ranking |
| Reject rusage as a strict peak authority | Below corroborated sampled RSS | Peak-capacity claim blocked | Exact kernel syscall accounting; unseen transients | Allocation-only diagnostic and conservative reporting | XLA fabricated RSS or caused a leak |
| Preserve memory/cost tradeoff | Higher observed enclosing RSS; LEDH cold ratio1.102 | Broad cost gate remains open | Native/compiler retention, scale and GPU behavior | Continue master queue | Main merge, GPU capacity or whole-program completion |

Hard numerical/provenance screens pass. Descriptive warm medians are lower;
no statistically supported ranking or default readiness follows. The strongest
alternative explanation is tiny-kernel host-dispatch overhead, with compiler
resident memory as an unresolved cost. Matched replications and uncontended GPU
measurements are still required. Terminal self-review also retains the prior
streaming regression and all unrelated master gaps.

The combined nonlinear archive through04826 contains229 reopened and verified
members,14,183,681 bytes, SHA256
`a3f038cf5fe93d442f781fa345e441ba2f17c0a99192e4d681d6670ccaeba819`.
It includes preserved failed attempts, full qualification/cost records, HLO,
frozen numerical sources and all four fresh fixture files. Integrity is not a
waiver of outstanding numerical, cost or terminal gates.
