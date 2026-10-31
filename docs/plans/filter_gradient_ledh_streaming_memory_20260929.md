# LEDH streaming seeds and owner-memory follow-up

Continue the master after04687. The registered value wrapper and shared LM
precision repair have bounded CPU/GPU numerical qualification; the result is
`filter_gradient_ledh_seeded_public_result_20260929.md`. Memory/capacity and
uncontended GPU costs remain open. This phase stays within the existing global
56 CPU/52 GPU process-hour caps and one numerical-worker limit.

Question: can the shared seeded value owner consume one Philox process draw
per executed time step, preserving seeded realizations and healthy/rejected
records, while removing its O(T*N*d) random-input buffer? Separately determine
the host-memory behavior of repeated fixed-owner reuse versus fresh callback
configurations. CPU04686 shows521.86MiB warm RSS growth and993.70MiB after two
extra one-shot calls despite Python collection. This is observed native/compiler
residency, not proof of an unbounded leak or a justified global-cache purge.

The numerical authority must remain `ledh_canonical_value_program_tf.py`.
Refactor its existing recurrence to accept either supplied random arrays or
seeded state as an explicit fixed factory configuration, with shared flow,
weight, reset and diagnostic logic. Carry Philox state in the time loop; draw
process noise only after the same prediction-validity gate as the old wrapper.
For annealed resampling, initialize PCG64(resample_seed+time) once per executed
time step and advance once per stage. Preserve large seed-word carry, fixed
signatures and all dynamic observation/seed inputs. Do not copy the filter,
change the stream, change callback ownership or adopt a row-mapped target.
Preserve the supplied-array factory as the independent execution comparator.

Before implementation, freeze the current seed123/124 complete records and
capture compiled buffer shapes/cost boundaries. Qualification compares
streaming seeded, current array-composed seeded, supplied independent arrays
and the frozen eager filter with repaired shared LM arithmetic. Preserve
healthy atol=rtol1e-6, invalid diagnostic roles, exact RNG state/uniform gates,
changed seeds/observations, time-dependent callbacks and one trace/HLO/no host
callbacks. Check early termination, including first invalid prediction, to
prove original draw scheduling. The score recurrence remains analytical.

Use a small horizon/particle ladder selected before launch to show random
storage scaling, with explicit memory expectations and invalid-case labels.
Do not use rejected trajectories for numerical speed rankings. HLO/buffer
evidence plus fresh-process allocator/RSS samples must distinguish removal of
the random buffer from compiler/context overhead. Repeat the same retained
owner and separately several fresh owners at fixed signatures, record weak
references and host/device memory before/after release and process exit. If
native memory persists, qualify process-boundary containment and document the
need for explicit owner reuse; do not silently cache mutable closures or call
global cache-clearing/system-limit APIs. Keep one-shot convenience costs visible.

Renew matched GPU costs only when a non-display device is unshared. Preserve
the declined preflight04687 receipt and leave other campaigns alone. No desktop
fallback is permitted unless the owner's stated all-other-devices load/memory
condition holds. GPU remains default; CPU is reference/diagnostic only. Verify
memory growth before initialization, preserve environment/seeds/source hashes,
and write unique run artifacts via the existing campaign runner/allowlist.

Initial allocation: at most16 workers,3600 CPU seconds and2400 GPU seconds,
drawn from25.850588 CPU/25.166452 GPU hours remaining after04687. At most two
localized harness retries, charged inside the allocation. Use300-second
workers initially; a900-second capacity worker requires a concrete reviewed
fixture and expected memory. Stop a unit on source/input drift, seeded/healthy
numerical regression, missing placement/memory diagnostics, incompatible
callbacks, device shortage or exhausted allocation. Localize failures rather
than increasing tolerances or implying scientific rejection.

Skeptical review: moving RNG into the loop can change draw order after invalid
prediction and can alter compiler fusion even with identical mathematical
inputs. Both are explicit gates. Graph buffer sizes alone do not measure actual
allocator peaks; RSS after GC does not identify a leak. A single small owner
does not qualify long-running public consumers. Initial work should establish
the smallest failing resource comparison before a ladder, with the exact
ladder reviewed before launch. This plan does not close F14 optional-pfor
disposition, score public migration, the DZ5 locator/geometry gaps, current
source applicability or F01--F20 terminal review. No canonical LEDH, posterior,
HMC, training, precision/default change or main merge follows from this phase.
Primary-agent review recorded; no independent review asserted.

Activation, 2026-09-29: recovered clean pushed commit c7c0b88c2, with no
campaign worker in trusted host process state. Freeze that commit's registered
array-composed owner and native recurrence before any runtime edit. Its shared
LM/reset dependencies are the repaired current source, not the erroneous prior
eager LM comparator. Keep independent supplied draws and eager/current-reset
comparators. Frozen Git modules are test authorities only.

Predeclared capacity ladder: (T,N,d)=(3,8,2),(3,64,2),(32,64,2),(128,64,2),
float64 with float32 reset, seed123/resample17, dual-trust controls from the
qualified small endpoint. Buffered process arrays are respectively384,3072,
32768,131072 bytes; streaming uses one N*d draw plus three uint64 Philox words.
Per-step diagnostic histories legitimately remain O(T). Other cloud/reset
workspaces remain O(N*d*d) and O(N*N); this is not a whole-filter O(N*d) claim.
The ladder is a mechanism/capacity diagnostic, with three warm evaluations per
point and both buffered and streaming owners; rejected trajectories are labeled
and excluded from speed ratios. Fixture sizes are deliberately bounded and do
not qualify large production capacity. Inspect graph shapes and optimized HLO
separately from timed retained-owner samples, whose compiler export costs must
not contaminate the measurements.

Fresh-process cost arms: frozen buffered XLA, streaming graph reference and
streaming XLA, each at the original small healthy fixture, 15 retained calls
and two extra fresh owners, with memory/weak-reference samples. Reuse earlier
unchanged eager CPU cost evidence; renew eager GPU and matched GPU costs if
unshared. Observe child exit from the runner and device process accounting;
process exit is containment, not proof of allocator leak absence. One numerical
worker at a time, maximum16 total including baseline freeze and regressions.
Add only explicit registered runner groups, preserve the approved runner prefix,
and apply unshared-device preflight to every new GPU timing group.

Activation skeptical review: numerical tolerances, seeds, callback semantics,
reset controls, dtype and TF32 stay fixed. Cost statistics remain descriptive.
Graph-buffer removal is the engineering criterion, numerical regression a veto,
owner-memory growth a repair trigger, and missing diagnostics or exhausted
allocation a stop. Nonlinear/scientific/score admission cannot follow from the
linear fixture. No material baseline or contract flaw found; execute the freeze.

Before timing launch, measurement review found that two arms in one process
would confound native/compiler residency. Split every ladder point into two
fresh-process workers. GPU runs the capacity ladder; CPU is the small matched
reference cost arm. Keep all four predeclared sizes. This increases the local
worker ceiling from16 to24, without changing3600 CPU/2400 GPU seconds, global
caps, scientific target or any numerical gate. Planned22 workers:2 freeze,
2 qualification,6 isolated costs,1 prior eager GPU cost,8 capacity,2 regression,
1 final policy/readback; up to2 localized harness retries remain inside24.
Each worker is sequential. This is measurement isolation under the existing
campaign authorization, not additional compute authorization.

Availability update before GPU costs: the runner declined launch, preserving
cost-preflight-declined-20260928T172518388312Z.json. PID2260909 owns compute
contexts on both non-display GPUs2/3; GPU3 is idle but shared. GPU0 belongs to
gnome-remote-desktop despite display_active=false, and GPU1 has active display.
Do not use either for timing or stop the other campaign. Execute the same
predeclared four-point capacity ladder on CPU as an explicit reference/debug
exception, with one fresh process per arm/point and no GPU/default-capacity
claim. CPU process-time remains under the original3600-second allocation.
Preserve GPU costs/capacity as pending; no unshared preflight or timing gate is
weakened. GPU numerical qualification04691 remains valid for its recorded
shared-device numerical scope. This reference ladder resolves buffer-scaling
and host-memory questions while hardware timing is unavailable.

Regression04703 is preserved with42 passes
and one HLO-text failure in the supplied-array annealed test. Numerical checks
passed. The shown difference is duplicate TensorFlow debug op_name suffixes
(zeros/_0 versus zeros/_1) on repeated compiler export, with one graph trace.
Localized harness repair1/2 archives both full exports and compares every
instruction/constant/operand/shape byte after stripping only HLO metadata={...}
source/debug annotations. Numerical tolerances, signatures, changed-input
checks and no-host-callback gates remain unchanged. The renewed run must prove
that metadata is the only difference; otherwise localize further and stop costs.
No runtime source changes follow from this diagnostic repair.
