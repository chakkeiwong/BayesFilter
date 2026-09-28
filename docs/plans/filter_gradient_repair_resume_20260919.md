# Filter and gradient repair resume checkpoint

Branch repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Pushed based8c23fc41.
Main remains unmerged. Active plan: filter_gradient_streaming_paired_cost_20260929.md.

Through 04749; active workers: none.
Charged/reserved CPU 109918.484468s / GPU 97122.849622s.
Remaining CPU 25.467088h / GPU 25.021431h.
Global caps56 CPU/52 GPU hours include the extra24 CPU hours. Active allocation
24 workers/1200 CPU/0 GPU seconds; used/reserved 22 workers,
449.669978 CPU/0.000000 GPU seconds. One numerical worker at a time.

All20 matched CPU cost workers04729--04748 pass exact shared records, RNG diagnostics, source/environment and enclosing-XLA gates. Readback04749 passes161 checks. Geometric warm ratios are1.08239 atT32 and1.09678 atT128; conditional95% upper bounds1.11687/1.15608 trigger profiling. Median observed RSS is about11MiB lower for streaming; compile-associated RSS still about526MiB. Failed schema pilot04728 is preserved and excluded by documented harness disposition, not timing. No runtime/gate/seed changes.

Next: Execute filter_gradient_streaming_profile_20260929.md (saved-HLO comparison and bounded RNG component costs) before accepting streaming performance. Then continue filter_gradient_score_study_directions_20260929.md for actual analytical-score consumer loops. Preserve GPU sharing, source-applicability, DZ5 and other terminal gates.

F14's identified implicit-pfor sites are closed through04727. The optional batch
route and five library/reference sites were repaired; all28 runner/benchmark
sites now have enforced dispositions. P91 captured-tape XLA failures04721/04722
were localized by04723 and repaired with shared loop-local tape ownership;
CPU04724/GPU04725 pass. Final04727 passes163 policy/readback checks over300
guarded files/1436 existing exceptions. Archive pfor-runner-04727 has86 verified
members,2,397,345 bytes, SHA256355017c3a37b4de52f56f287d2fcba21efbf3b9749041e797f6678322be6367e.
No historical LEDH route or HMC/training run executed, no whole-master claim.

Memory/performance remains open: fresh XLA owners retain native/compiler RSS
after Python GC; retained-owner reuse is boundedly qualified. Prior T128/N64
CPU streaming was43.9ms versus36.9ms buffered, which this paired study tests.
Uncontended GPU cost/allocator/capacity awaits unshared non-display hardware.
Other campaigns hold GPU2/3 contexts. GPU0 is remote desktop and GPU1 display;
no display fallback unless the owner's availability condition holds. CPU-only
runs are explicit references. GPU runs require trusted access and verified growth.

Other gaps: registered analytical-score consumer migration/costs, current-source
measurement applicability, DZ5 locator121 strict trajectory differences and
unconverged optimizers,4539 strict fitted-geometry CPU/GPU record differences,
isotropic angles and remaining F01--F20 terminal dispositions. Reuse unchanged
actual-DZ504618--04628/import04629--04630/index04631 evidence. Do not repeat
unrelated dtype or full-trajectory trials; the locator invariant-copy compiler
mechanism lead remains the next bounded causal intervention. See terminal queue.

Preserve analytical shared authorities, explicit invalidity reporting, original
LEDH seed streams, author-profile NeuTra IAF and scientific gates. No subagents,
training/HMC, live MacroFinance edits, package/system/cache changes, tolerance
relaxation, canonical LEDH rebuild, unsupported promotion or main merge.
