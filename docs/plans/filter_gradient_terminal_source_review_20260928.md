# Terminal source review preparation

Refresh the whole-repository syntax and static-call inventory at the current
repair branch, then reconcile findings F01--F20 with actual consumer evidence
and remaining gaps. The last syntax snapshot and several progress fields in
the accumulated ledger are stale; neither historical counts nor the passing
276-source policy guard can establish whole-repository completion.

This first step is a read-only standard-library AST scan. It imports no
TensorFlow/NumPy and runs no algorithm, so it may proceed while the single
authorized numerical worker performs the actual-consumer lifetime test. Use
`scripts/audit_filter_gradient_policy.py --output <root>/audit.json.gz
--markdown <root>/audit.md` with the existing tf-gpu Python and unique root
`artifacts/filter-gradient-repair-20260917/terminal-source-audit-20260928-r1`.
Allow at most120 CPU process-seconds for this scan from the existing campaign
budget; record its actual elapsed time and count it once. Do not launch a
second numerical worker or change runtime source during the active cohort.

The deliverable is a current inventory and an explicit list of unresolved
consumer/evidence obligations. Inspect candidate numerical operations in
context and follow their reachable public endpoints before classifying them.
Metadata, serialization/reporting traversal, diagnostics and numerical
recurrences must remain distinct. Existing historical/vendor parse errors
must be identified, not counted as repaired or silently omitted. Tests, HLO,
callback and dependency records supply execution evidence; syntax alone does
not. Any newly confirmed violation blocks only the affected terminal finding
and triggers a bounded repair plan. No finding closes from this scan alone.

Skeptical review: matching names is incomplete, and static call resolution
misses dynamic callbacks and factories. Preserve those limitations, compare
the committed guard coverage, and reuse qualified component evidence only
when its actual dependency bytes remain applicable. Do not require re-running
unaffected numerical suites merely because documentation or an unrelated
module changed. No Zhao-Cui faithfulness, canonical LEDH, convergence,
posterior, performance-ranking or whole-program claim is authorized here.

Attempt r1 reached compressed serialization but exceeded120 seconds, leaving
an incomplete13MiB gzip and no Markdown. Preserve both as failed output. The
default gzip compression level9 adds avoidable reporting cost; use level1 with
unchanged JSON content, then retry once in `terminal-source-audit-20260928-r2`
under a300-second limit. Reserve420 seconds total for this read-only unit,
including the first120-second failed attempt, from the same global cap. Record
timeouts in a `finally` path so missing completion cannot erase the charge.
After success, reopen gzip and verify the complete JSON/schema and counts.
This changes only inventory serialization, not numerical code or audit rules.
