# Filter and gradient repair resume checkpoint

Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Base commit c7c0b88c2;
this file records the streaming checkpoint through04706. Main remains unmerged.

Active question: remove O(T*N*d) seeded process-noise storage and qualify
reusable versus fresh-owner memory. Active plan:
`filter_gradient_ledh_streaming_memory_20260929.md`.
Through 04706; active workers: none.
Charged/reserved CPU 109174.884363s / GPU 97012.611230s.
Remaining CPU 25.673643h / GPU 25.052052h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Streaming allocation24 workers/3600 CPU/2400 GPU seconds; used/reserved
19 workers/637.001795 CPU/411.837878 GPU seconds.
One numerical worker at a time. CPU is explicit reference only. GPU requires
trusted non-display availability and verified memory growth. Timing requires
unshared device preflight. Do not stop other campaigns.

Streaming buffer removal is qualified: CPU/GPU9 endpoint checks each, exact buffered records and original RNG scheduling. CPU isolated costs and8 healthy capacity workers pass; original full-horizon random storage is absent throughT128/N64. CPU/GPU numerical regressions04704/04705 pass43 each after preserving the04703 HLO metadata harness failure; final source-bound readback/policy04706 passes164. Fresh-owner native/compiler RSS and possible CPU long-horizon slowdown remain open. Unshared GPU costs/capacity blocked by PID2260909 contexts, no worker launched.

Next: Execute filter_gradient_ledh_pfor_disposition_20260929.md. Keep GPU cost/capacity and controlled performance attribution in the terminal queue.

Preserve prior qualified evidence: seeded public/LM result through04687,
validity boundary04662--04668, actual DZ5 renewal04618--04628/import isolation
04629--04630/evidence index04631. Older current-source readbacks retain their
original source scope; do not silently relax them after runtime changes.

Other gates: F14 unapproved optional pfor, registered analytical-score migration
and costs, current-source timing/memory applicability, DZ5 locator121 strict
trajectory differences and unconverged optimizers,4539 strict fitted-geometry
CPU/GPU record differences/isotropic angles, and all F01--F20 dispositions.
Locator optimized-HLO invariant-copy lead is not a qualified causal repair.
See `filter_gradient_terminal_gap_queue_20260928.md` for the work order.

Keep callback configuration fixed per retained owner; one-shot convenience must
refresh mutable closures. No global callback cache or system/cache mutation.
LEDH SeedSequence/PCG64/Philox streams remain unchanged; geometry's approved
TensorFlow stream migration does not apply to them. No copied numerical kernels.
Preserve strong reset rejection: unusable public value is NaN with validity,
code/index and separate raw diagnostic outputs. No tolerance relaxation.
No live MacroFinance edits, subagents, training, HMC, package/environment change
or canonical LEDH rebuild. Preserve author-profile NeuTra IAF. Unsupported
scientific/default claims remain blocked. No main merge until every master gate.
