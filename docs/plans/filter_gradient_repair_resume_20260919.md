# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Evidence base54ba5c538; use git HEAD for the latest durable checkpoint.
Main remains unmerged. Active plan: filter_gradient_score_direction_cost_20260929.md.

Through 04782; active workers: none.
Charged/reserved CPU 110491.548159s / GPU 97352.844807s.
Remaining CPU 25.307903h / GPU 24.957543h.
Global caps56 CPU/52 GPU hours include the extra24 CPU hours. Active allocation
6 workers/1800 CPU/0 GPU seconds; used/reserved 5 workers,
80.150181 CPU/0.000000 GPU seconds. One numerical worker at a time.

Gaussian score direction repair and cost screen complete through04782. All six actual consumers pass CPU/GPU;162 final readback/policy checks pass. Complete records are exact; maximum FD error9.49e-12. CPU warm ratios0.638016 (diagnosticLEDH)/0.515920 (resamplingKDM), cold ratios1.075618/1.065283, observed RSS increases96.348/90.863MiB. Single-process costs remain descriptive/unaccepted. Rusage peak is below sampledVmRSS; peak accounting requires a bounded diagnostic.

Next: Archive/commit/push the completed Gaussian evidence, then execute filter_gradient_nonlinear_directions_20260929.md with fresh exact fixtures. Later diagnose OS memory-counter disagreement and matched owner residency/costs per the cost result. Preserve streaming slowdown, uncontended GPU, native-memory, DZ5/geometry and remaining terminal gates; main stays unmerged.

Profiling04750--04755 completed in40.675878 CPU seconds: all six workers pass,
including161 final checks. RNG-only streaming timings are lower; this does not
explain the full-filter slowdown or satisfy its cost gate. The result and35-member
verified archive are committed in54ba5c538.

Gaussian direction qualification04756--04777 is complete: six actual consumers
pass CPU/GPU and161 terminal checks. Complete direction outputs agree exactly;
maximum five-point error9.49e-12. Three failures are preserved: incorrect zero-
design rejection assumption, then two fixed output-schema shape defects.
Four cases were refreshed after preserving diagnostic failure payload ordering.

Matched CPU cost study04729--04748 is complete;04749 passes161 readback/policy
checks. Every shared numerical/status field is exact; sources, inputs, RNG,
threads and affinity match. Streaming geometric warm ratios1.08239(T32) and
1.09678(T128), with conditional95% upper bounds1.11687/1.15608, triggered the completed
profiling. Median observed RSS is about11MiB lower, but compile-associated RSS
still about526MiB. No cost acceptance. Failed schema pilot04728 is preserved.
Archive streaming-paired-04749 has142 verified members,9,968,524 bytes, SHA256
5ab3ee9719113d9af0450e4949942bd5abd0ca7f9f0be53ba2a46785e7c2aae8.

F14's identified implicit-pfor sites closed through04727. Optional batch and
five library/reference sites repaired;28 runner/benchmark sites have enforced
dispositions. P91 captured-tape XLA failures were repaired with shared loop-local
tape ownership. CPU/GPU and163 final readback/policy checks pass;300 guarded
sources/1436 existing exceptions. No HMC/training or old LEDH route executed.

Other gaps: nonlinear score-study EKF/UKF and LEDH/SGQF/mixture consumers
still assemble six directions in Python; next reviewed continuation is
filter_gradient_nonlinear_directions_20260929.md after this Gaussian unit.
Registered score owners/costs, source-measurement applicability, native/compiler
residency, uncontended GPU costs/capacity, DZ5 locator121 strict trajectory
differences/unconverged optimizers,4539 fitted-geometry CPU/GPU differences,
isotropic angles and other F01--F20 dispositions remain. Reuse unchanged
actual-DZ504618--04628/import04629--04630/index04631 evidence. Do not repeat
unrelated dtype/trajectory trials; the locator invariant-copy lead stays open.

GPU0 is remote desktop and GPU1 display; other campaigns holdGPU2/3 contexts.
Do not stop them or use display fallback unless the owner's condition holds.
CPU is explicit reference; trusted GPU growth/placement required. Preserve
shared analytical authorities, explicit invalidity, LEDH streams and canonical
NeuTra IAF. No subagents, training/HMC, live MacroFinance edits, package/system/
cache changes, relaxed tolerances, canonical LEDH rebuild or main merge.
