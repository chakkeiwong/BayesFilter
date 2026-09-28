# Localize streaming CPU cost before accepting it

The five paired fresh-process comparisons through04749 retain exact complete
shared records. Geometric mean streaming/buffered warm ratios are1.08239 atT32
and1.09678 atT128. Conditional log-t95% intervals extend to1.11687 and1.15608;
both activate the predeclared profiling trigger. CPU RSS is about11MiB lower,
but that saving does not waive a runtime regression. This phase localizes the
cost; it does not change a threshold or accept the regression.

First compare existing optimized HLO from paired runs04729/04730 (T32) and
04731/04732 (T128), using the already tested compiler-reader utilities. Verify
the raw HLO hashes, exact measurement-source identity and matching input records.
Report operation counts, computation/callee structure and the placement of
Philox/Box–Muller work relative to the observation loop, prediction-validity
condition and QR/reset computations. Generated names/metadata alone do not
prove arithmetic differences; distinguish body shape, call multiplicity,
constants and operands. Archive full machine-readable findings. No statistical
or causal performance claim follows from opcode counts alone.

Then run a small native-XLA RNG component diagnostic at N64,d2,float64 and
T32/T128, process seed123. Compare existing `seeded_value_inputs` with direct
stepwise calls to the same `philox_box_muller_normal` authority in a TensorFlow
loop. Both must preserve original initial/per-step draws, per-call state skips,
dynamic seed operands and original float precision. Return per-step cloud
summaries/final state to prevent dead-code elimination, and independently check
the complete noise clouds outside timing. Keep this diagnostic in tests, not a
second runtime RNG implementation. It is a component-cost explanation only;
the full filter remains the performance gate. No old pfor or invalidated LEDH
result/fixture is a comparator.

Use fixed-signature XLA functions, two changed seeds, no host callbacks, HLO,
three unmeasured calls and30 timed synchronized calls. Cold compilation, warm
execution and RSS samples remain separate. A compiler trace or profiling event
collection may help only after checking local tooling availability; no package
installation is authorized or required. Preserve fresh process isolation and
explicit CPU-reference labels. Any inferred speed cause must survive a bounded
full-filter intervention with unchanged records before a runtime repair is
accepted. This plan does not pre-authorize changing RNG, prefetching beyond the
original validity gate, precision, tolerances or filter equations.

At most8 workers/900 CPU process-seconds (zero GPU seconds) inside the remaining
global allocation. Initial300-second timeouts, one numerical worker at a time,
two localized harness retries inside the8-worker cap. Preserve unique numbered
artifacts using the stable campaign runner. Source/input mismatch, missing HLO,
unexpected draws or invalid component outputs stop that unit for localization.
No training/HMC, subagent, package/system/cache, live MacroFinance or main-merge
change. GPU remains default; its uncontended cost/capacity gate stays open.

Skeptical review: a tiny RNG benchmark can be optimized differently and omits
interaction with the filter; it cannot establish the full cause by itself.
Compiler structure is explanatory, not a substitute for an intervention. The
plan reuses saved HLO and qualified controls rather than rerunning a large
matrix. It preserves the memory improvement without redefining success to
accept slower execution. Primary-agent review passes; no independent review
asserted. The separate score-study direction plan remains actionable after
this bounded localization; this phase cannot absorb the remaining campaign.
