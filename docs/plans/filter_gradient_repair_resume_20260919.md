# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Pushed base8f6e94633.
Main remains unmerged. Active plan: filter_gradient_streaming_profile_20260929.md.

Through 04755; active workers: none.
Charged/reserved CPU 109959.160346s / GPU 97122.849622s.
Remaining CPU 25.455789h / GPU 25.021431h.
Global caps56 CPU/52 GPU hours include the extra24 CPU hours. Active allocation
8 workers/900 CPU/0 GPU seconds; used/reserved 6 workers,
40.675878 CPU/0.000000 GPU seconds. One numerical worker at a time.

Profiling04750--04755 completed: all six workers passed, including161 final readback/policy checks. Streaming removes one global loop and moves Philox/Box-Muller into the observation loop; three additional reachable copies/six fusions are explanatory only. RNG component warm times are descriptively lower for streaming at both horizons, so raw RNG cost alone does not explain the full-filter slowdown. Full-filter performance remains unaccepted.

Next: Archive and commit the completed profile, then execute the reviewed score-study analytical direction consumer repair. Preserve the separate full-filter compiler-interaction investigation for a bounded causal intervention; do not change RNG, prefetch past validity gates, or waive performance limits. Remaining GPU, source, DZ5 and terminal gates stay open.

Matched CPU cost study04729--04748 is complete;04749 passes161 readback/policy
checks. Every shared numerical/status field is exact; sources, inputs, RNG,
threads and affinity match. Streaming geometric warm ratios1.08239(T32) and
1.09678(T128), with conditional95% upper bounds1.11687/1.15608, trigger current
profiling. Median observed RSS is about11MiB lower, but compile-associated RSS
still about526MiB. No cost acceptance. Failed schema pilot04728 is preserved.
Archive streaming-paired-04749 has142 verified members,9,968,524 bytes, SHA256
5ab3ee9719113d9af0450e4949942bd5abd0ca7f9f0be53ba2a46785e7c2aae8.

F14's identified implicit-pfor sites closed through04727. Optional batch and
five library/reference sites repaired;28 runner/benchmark sites have enforced
dispositions. P91 captured-tape XLA failures were repaired with shared loop-local
tape ownership. CPU/GPU and163 final readback/policy checks pass;300 guarded
sources/1436 existing exceptions. No HMC/training or old LEDH route executed.

Other gaps: score-study evaluate_gaussian still assembles six analytical
directions using Python; execute filter_gradient_score_study_directions_20260929.md.
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
