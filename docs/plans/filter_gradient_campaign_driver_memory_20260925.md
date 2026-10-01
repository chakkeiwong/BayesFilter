# Campaign driver record-retention diagnosis

Queue this localized harness repair after the current DZ5 oracle cohort. Do not
change its executing source while any worker is active. This work fits the
existing 56 CPU / 52 GPU hour caps and changes no scientific or comparison gate.

During CPU oracle 03863, `ps -C python -C python3 -o pid,ppid,etime,pcpu,rss,args`
reported 2,635,936 KiB RSS for campaign parent PID307070, versus 1,091,196 KiB for
its target child PID307136 at that observation. These are single observations,
not live tensor bytes or proof of the cause. The parent does not evaluate the
TensorFlow target. Source inspection finds `run_job -> records()` loading all
3,863 full manifests, including every historical source-hash dictionary, and
retaining the list throughout the child. Test and status commands need only
charges/counts and selected exact-attempt identities. This creates a concrete
host-memory hypothesis independent of XLA compilation.

Question: can the driver retain bounded metadata while preserving every budget,
retry, provenance and evidence decision? First measure the original loader in
fresh isolated CPU processes on a frozen manifest-path set. Replace retention
with compact metadata and source hashes loaded only when needed, or a streaming
scan; choose the smallest design that preserves existing call semantics. Do not
weaken source equality to timestamps or omit unfinished-run/supplemental charges.
Keep all raw historical manifests unchanged.

Before/after checks must compare exact aggregate charges, run numbering, exact
source-sensitive retry counts, latest passed/failed/unclosed states, and current
provenance decisions. Mutation tests must reject changed/deleted sources and
preserve the three-attempt rule. Run the existing policy suite after changes.
Record fresh-process wall time, starting/current/peak RSS and path-set identity;
three repeats per loader are descriptive evidence. A different decision vetoes
the candidate. Missing measurements or source drift triggers harness repair.
No conclusion about TensorFlow compiler memory, device memory, executable eviction
or target performance follows from reducing this parent's bookkeeping memory.

Reserve at most ten CPU workers / 1800 charged seconds, including localized
retries, within the campaign caps. Use the stable runner and unique numbered
artifacts; one numerical worker at a time. No additional package or hardware
access is needed. The old parent holds historical dictionaries while the child
runs, so whole-machine memory and isolated child measurements must stay distinct.

Skeptical review: the observed parent RSS may include allocator retention beyond
live record dictionaries. Measure before/after in fresh processes rather than
asserting all memory is due to live objects. Keeping the raw manifests and exact
source checks protects retry/provenance semantics. This is a local infrastructure
repair under the existing campaign authorization, with unchanged gates/budget.

Source review narrows the implementation: `run_job` needs only cumulative charges,
next run count and exact key/source attempt count from old manifests. Change the
record iterator to stream, compute those summaries without retaining hash maps,
and retain a full record only at a selected comparison/provenance boundary.
The matrix's `records()[-1]` must become an explicit latest-record read; repeated
passes (terminal gate/status) must be reviewed so iterators are not exhausted or
large manifests reloaded once per test group. Existing monkeypatched record
fixtures must continue to exercise budget/interrupt/selection semantics. The
independent baseline loader comes from committed4c37f9f40, not a rewritten proxy.

03871 passes budget/source/retry semantics;03872 passes all129 policy checks.
Initial cost03873prior /03874streamed preserve identical3870-record counts,
charges and exact attempts. RSS after loading is2,701,922,304 versus37,388,288
bytes. The streamed helper's lifetime ru_maxrss is532,676,608 bytes even though
its pre/post current RSS is32,657,408/37,388,288. That maximum includes process
history before the timed loader and cannot establish this operation's peak.
Preserve the initial cohort including the already-running03875. Add sampling
of current RSS during the timed operation plus the lifetime peak before timing,
then run three fresh matched repeats without further source edits. This repairs
the measurement contract, not the loader or numerical target. Extend this local
unit from10 to12 workers under the same1800-second/global budget; finish with
one policy renewal after the instrumentation change. No extra compute category,
comparison gate or scientific claim is introduced. The original retained-RSS
observation remains valid; the initial lifetime peak is not loader attribution.
