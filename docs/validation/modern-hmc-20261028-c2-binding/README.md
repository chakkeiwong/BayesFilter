# Explicit Phase 9 source-binding checks

The deterministic gzip-compressed consumer-scoped-binding.patch.gz is relative to consumer files immediately before this phase and
preserves earlier edits. Installed hashes identify the exact final five files.
JSON manifests/results are gzip-compressed with mtime zero. Failed attempts
and JUnit outcomes are retained. The inventory records SHA-256 hashes. No raw
chain arrays are duplicated.

Direct public tests use explicit binding without changing global pins.
Regression attempts retain earlier test-only process-local injection; results
distinguish these roles. No sampler transitions were executed.
See [plan](../../plans/bayesfilter-modern-hmc-c2-scoped-binding-2026-10-28.md)
and [result](../../plans/bayesfilter-modern-hmc-c2-scoped-binding-result-2026-10-28.md).
Compatibility is limited to the tested CPU target; posterior, GPU, consumer
master dispatch and release remain separate.
