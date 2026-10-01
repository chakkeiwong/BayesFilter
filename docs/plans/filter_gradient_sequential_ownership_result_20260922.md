# Sequential program ownership repair

A complete sequential controller now owns the callback-dependent factory programs
created during construction and tracing. Replacing its root no longer leaves
those callbacks in separate global LRUs. Standalone factory callers retain the
existing bounded cache API, and shape-only caches retain their existing behavior.
The numerical controller and public result contract are unchanged.

The scope uses a ContextVar while factories are built and traced. A controller
holds its own dependency memo; numerical calls do not enter the scope or change
it. Tracing completes inside that scope, including lazy nested factory calls.
There is no sweep of global caches after a call. The existing invocation lock
continues to protect buffered locator resources. Construction-time tracing is
included in the next full public cold-cost comparison.

Runs02944--02955 pass two pure construction/isolation checks and15 CPU checks;
02956--02966 pass the same15 numerical/lifetime checks on GPU2, UUID
541e1e19-2df4-9064-4db9-9d0d2abc3eba. The six actual lifetime probes cover terminal,
factor and batched-locator programs. After root replacement, an explicitly retained
old compiled handle still executes. After releasing that handle, its callbacks,
root, scope and9/11/13 dependency functions are all collected. Releasing the
successor works too, with unchanged standalone cache sizes.

Different-curvature targets produce different precisions, excluding stale callback
reuse. Original3582b4ac full records and target order pass for terminal, factor-one,
factor-two, scalar locator, batched locator and moving-budget cases on both devices.
The public changed-input/one-trace, buffered progress, overflow, no-eager-retry and
frozen-derivative boundary checks also pass. The earlier full23-case public and
legacy consumer qualification remains in the separate public result; final-source
E6 qualification is still required. The new scope source is fully guarded, giving
224 sources/1325 exact exceptions without any new exemption.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain scoped sequential ownership | CPU/GPU lifetime, identity and original records pass | No stale target, callback retention or numerical mismatch in these cases | Broader roots and native allocation lifetime | Complete public costs and ordered-block capture | Python collection does not evict native executables |
| Continue E3–E6 | Qualified public dependency can support ordered blocks | Initializer strict rounding and external telemetry/deadlines remain open | Full endpoint and final-source costs | Implement ordered block execution and actual consumer repairs | No whole-repository or main-merge readiness |

Post-run review: the strongest competing explanation for high RSS is native
compiler/allocator retention after Python collection, already observed in separate
posterior probes. This change repairs confirmed Python ownership; it does not
resolve the known many-signature executable-mapping ceiling. The graph/RSS/cache
measurements and process-lifetime limits must continue to be reported separately.
The posterior and full-geometry factories do not expose the same global callback
LRUs; source inspection alone does not qualify every other consumer's lifetime.

02967 passes all128 policy/provenance/registration checks. The full unit uses24
workers and697.07 seconds within24/3600. Source identities match throughout
`artifacts/filter-gradient-repair-20260917/sequential-ownership-qualification-02967.json`;
the saved analyzer validates every lifetime report. Focused Ruff and whitespace
checks pass. Cumulative charges are59081.88946123973 CPU/58769.39307047443 GPU
seconds under unchanged32/52-hour caps.
