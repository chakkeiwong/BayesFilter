# Batched locator GPU accounting repair

Run03297 preserves the unmodified pinned d6a568384 GPU failure: TensorFlow
places its int32 resource counters on CPU, and GPU XLA rejects access to them.
The internal reusable candidate has the same declarations. The repair changes
only seven resource counters/indices and matching integer producers to int64.
Floating-point target/chart/search/replay arithmetic, TFP optimizer state,
comparisons, cap values and the original tie sentinel remain unchanged.

The independent GPU comparator uses the pinned original function with the same
accounting-only adaptation, recording its exact edits and source hashes. CPU
comparisons retain the completely unmodified original. Test fixtures use at most
1200 recorded target rows, so both integer representations are exact and stay
below the original tie sentinel. This does not certify unbounded counters.

Evidence contract: renew all eight complete records (batch1/3, quadratic,
nonquadratic, flat and invalid rows) plus the enclosing invalid-to-valid
recurrence on CPU/GPU. Require existing full-record tolerances, exact target
order/counts and decisions, changing starts/scales with one trace and unchanged
HLO, and owner/callback/graph collection. Failures are repair triggers; no
comparator tolerance, scientific method or public default changes are allowed.
Shared-device runs answer correctness only. This unit does not establish public
integration, costs, native memory eviction or actual DZ5 readiness.

Skeptical review: the failed original must not be passed off as GPU evidence.
Its compatibility adaptation is explicit and integer-only; the unmodified CPU
authority cross-checks the change. No CPU fallback or target arithmetic change
is included. There are no numerical workers active while sources are edited.

The existing unit has used12 workers/256.802863 seconds through03297. Expand
its attempt ceiling from20 to32 under the unchanged3600-second unit limit and
32CPU/52GPU-hour cumulative campaign caps. This covers nine renewed workers per
device, one policy worker and one localized retry. Each numerical worker has a
120-second ceiling. Use the stable runner's `batched_center_reuse_cpu/gpu`
matrices, `batched_center_enclosing_cpu/gpu` groups and `policy`; frozen sources
through each cohort. Manifests and complete result records remain in fresh
numbered campaign directories. Stop for invalid evidence, a changed method or
exhausted limits. Main stays unmerged.

All nine GPU checks03362--03370 now pass complete adapted-original records,
exact target order/counts, changed starts/scales, unchanged HLO and collection.
The enclosing invalid-to-valid recurrence also passes. The unit uses30/32workers
and778.283893/3600seconds. Public wiring and actual dense-initializer integration
remain open; the seven-counter GPU reference adaptation stays explicit.
