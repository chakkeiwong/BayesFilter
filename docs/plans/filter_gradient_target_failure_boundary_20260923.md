# Target failure numerical boundary repair

Question: can the diagnostic exception adapter preserve its complete public
semantics while its finite-output checks and fallback array construction run
in a stable XLA segment? The authority is unmodified Git3582b4ac
`bayesfilter/inference/target_failure_policy.py`, loaded only by diagnostic
tests. Current owned call sites are tests and exports; the public callable
accepts arbitrary Python callbacks and catches per-call Python exceptions.

Preserve callback invocation exactly once with its original position object,
allowed and forbidden failure/branch labels, details, exception types, scalar
value and matching score-shape checks, nonfinite-output opt-out, fallback
values/scores, classifications and every report field. No new scientific
fallback, threshold or status classification is introduced. Configuration and
exception/report formatting stay on the host. Put finite checks, fallback
selection and score construction in a stateless XLA program with explicit
value, score, declared-failure and fallback-parameter operands. Keep a bounded
shape-only cache; it must not retain callbacks, positions or policies.

The public API remains explicitly a host diagnostic exception adapter, not a
compiled target endpoint. Tracing a callback that throws based on a runtime
Python value would change its semantics. A compiled consumer must instead
return a tensor failure status and enforce its label permissions inside its
own enclosing graph; it cannot call this Python adapter. Test composition of
the numerical segment under one enclosing XLA target, but do not infer
qualification of an absent production caller from that test.

Acceptance: compare complete original records, scores and classification,
plus exact exception type/message and callback counts for healthy, every
declared error, nonfinite value/score, forbidden labels, disabled catch and
programmer/shape errors. Cover scalar, vector, matrix and empty positions.
Check changed operands and fallback parameters without retracing or HLO change,
healthy and fallback derivatives, enclosing composition, no Python callbacks
in the graph, bounded cache and callback collection. Run existing seven
target-failure/linear-Kalman consumer checks on CPU/GPU and renew policy checks.
CPU tests explicitly hide GPUs; GPU workers verify growth and provenance.

Budget: at most8 fresh workers/1200 charged seconds (120s each, or a300s
localized retry) within the existing32CPU/52GPU-hour caps. Exact commands use
the stable campaign runner's `target_failure_endpoint_cpu/gpu` groups and
`policy`. Each numbered directory preserves manifests, pinned hashes, complete
records and HLO. Stop on changed scientific semantics, invalid evidence or
exhausted limits. Numerical mismatch triggers diagnosis without relaxed
tolerances. Shared GPU timing is not clean performance evidence; this unit
does not close campaign costs, lifetime memory, HMC/default readiness or merge.

Skeptical review: a compiled output checker alone cannot make arbitrary host
callbacks compliant. Explicitly retain that boundary and exercise a callback
whose exception changes between calls. Preserve the original valid-label
behavior even when its allow-list excludes `valid`; introducing a new check
would be a separate API change. Empty score finiteness remains vacuously true.
Only fixed float64 conversion and finite selection are moved, so exact public
output/decision equality is the criterion. No source edits during another
numerical worker or matrix; execute this unit after the locator cohort ends.
