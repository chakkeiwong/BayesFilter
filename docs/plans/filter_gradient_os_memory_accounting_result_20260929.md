# Linux memory-counter discrepancy isolated without TensorFlow

The same rusage-below-resident discrepancy occurs in two fresh standard-library
Python processes with no TensorFlow, NumPy, XLA or other compiler imported.
Spreading otherwise identical page touches over32 CPUs increases the deficit
from2.262MiB to34.375MiB. This supports per-CPU accounting as the explanation
for the misleading peak counter; an XLA-specific mechanism is unnecessary to
produce it. The actual numerical owners' measured RSS increments remain valid
observations and still require cost/capacity disposition.

Both processes retained32 private anonymous17MiB mappings, touched every OS
page (544MiB total), then closed them. The pinned arm used one CPU; the spread
arm used32. The page-map walk measured544.055/544.113MiB growth, within the
declared2MiB bookkeeping tolerance. Counter records, affinity, kernel, Python,
exact commands, sources and installed-header excerpts are preserved under
`os-memory-accounting-20260929-r1` in the campaign artifact root.

| Arm | Final status RSS = smaps RSS | Rusage peak | Deficit | RSS after closing mappings |
|---|---:|---:|---:|---:|
| Pinned | 562.261719 MiB | 560 MiB | 2.261719 MiB | 18.269531 MiB |
| Spread over32 CPUs | 562.375000 MiB | 528 MiB | 34.375000 MiB | 18.382812 MiB |

All artifacts pass independent readback: source/header/excerpt hashes,
commands and hidden GPU state, absent numerical-library imports,32 increasing
allocation records, correct touched bytes and both growth checks. The
supervisor took1.186639 CPU process-seconds, zero GPU time; the initial120-second
reservation was replaced by this measured elapsed charge in
`supplemental-compute-os-memory-20260929-r1.json`. No numerical campaign worker
overlapped; the prior cohort was complete. No package/cache/system/runtime
numerical source changed.

The matching kernel6.8.0-138-generic headers provide a concrete mechanism:
`include/linux/mm.h:2659` calls the approximate per-CPU-counter read;
line2664 has a separately summed counter, and `get_mm_hiwater_rss` at2714
uses the ordinary RSS read. `include/linux/percpu_counter.h:82–89` explicitly
states that local updates can remain absent from the global read until a sum
accounts for them. The spread allocation leaves a much larger deficit with
the same touched bytes, consistent with that distinction. Full syscall
definitions are unavailable locally, so exact source-level attribution of
every syscall remains unproved. The no-TensorFlow reproduction itself is
direct evidence, independent of that source interpretation.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Close bounded counter diagnostic | Equal allocations and independent page-walk records pass | None | Two deterministic placements, non-atomic reads | Carry measurement qualification into remaining cost review | Statistical ranking or exact syscall proof |
| Disallow rusage-only peak guarantees | Counter can underreport known touched resident pages | Exact peak/capacity claims remain blocked | Unobserved transients and process-specific accounting | Keep current/page-map RSS and device allocator metrics distinct | Compilation alone caused the discrepancy |
| Retain real compiler-residency gap | Numerical-owner RSS growth was corroborated separately | Cost/scale/GPU gates remain open | Native retention and concurrent capacity | Continue master cost/performance queue | This accounting diagnosis removes the XLA memory cost |

Terminal self-review: pinning and CPU migration are explanatory interventions,
not performance measurements or operating recommendations. We do not correct
rusage by adding a universal offset, equate process RSS with live tensors, or
upgrade sampled maxima to true peaks. Old rusage fields remain historical
measurements. Reports relying on them alone need qualified interpretation;
unaffected value/gradient evidence does not need numerical reruns.
