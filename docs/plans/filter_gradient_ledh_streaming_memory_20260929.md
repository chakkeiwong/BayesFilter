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
