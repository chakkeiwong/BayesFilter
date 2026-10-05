# Remaining SVD composition lifetime and resource decision

Runs05246--05248 pass CPU/GPU lifetime checks and161 terminal readback/policy
checks. Seven direct numerical files are byte-identical toea94aac96, so the
accepted48-worker cost cohort remains applicable to its dense/block/sequential/
quadratic/COD composition. The readback verifies its receipt and archived report
checksums. The old inaccurate XLA arm remains excluded from timing comparisons.
Angle/subspace pair diagnostics are a separate caller, still awaiting costs.

Each lifetime worker evaluates the existing exact-quadratic D3/D5 fixtures256
times with alternating offsets and live precision resources. Exact replay,
independent condition/precision/rank/status checks at1e-10 and one trace pass.
Both shape-only primitives retain zero captured inputs. Twelve replacement
callers use those same two primitives; all outputs pass the independent checks
and all caller programs, resource owners and resource objects collect.

| Device | Late128-call RSS growth D3/D5 (bytes) | Additional RSS for12 replacement callers (MiB) | Final RSS (bytes) |
| --- | --- | --- | --- |
|CPU reference|20480/20480|998.238|2231771136|
|GPU3|28672/32768|723.844|2154061824|

Late growth stays below16MiB; configuration growth stays below2GiB. GPU live
allocator bytes plateau within each dimension at2304/2816; peaks are54272/
110848. After replacements, current/peak bytes are2560/111360, whereas process
reservation is446693376 bytes. The latter includes context, compiler and library
memory. GPU3 used UUID GPU-b8045e28-4433-ec7a-77a5-db0636748322 with verified
growth and unshared observations. The earlier paired costs used GPU2 of the
same hardware class; these lifetime observations are not a cross-device timing
pair. Parent observations verify exit and absence of each worker's GPU context.

Accept the scoped resource tradeoff for the accurate compiled composition.
The unchanged cost study reports current-XLA/current-graph cold ratios2.997/
3.154 on CPU and1.865/2.072 on GPU; added host RSS is344.1/354.7MiB CPU and
20.3/20.6MiB GPU. Warm ratios are0.352/1.184 CPU and0.350/1.509 GPU for D3/D5.
The D5 slowdown is retained, not called an improvement. The accurate SVD and
enclosing XLA execution meet the numerical/default contract; the cheaper old
XLA route fails independent numerical checks and cannot be substituted. Current
graph is an explicit reference. No numerical method or tolerance was changed
to improve timing. This is a measured engineering tradeoff under the master,
not a universal performance bound.

Native/compiler residency survives Python collection. Reuse a fixed owner
where possible and bound replacement work by a numerical worker lifetime;
the demonstrated bound is two dimensions and12 replacements, not arbitrary
configuration capacity. The exact native allocation arena is unidentified.
These observations establish stable measured reuse and practical process-exit
containment, not TensorFlow eviction or leak-freedom at all scales. Actual DZ5
process-lifetime qualification04618--04628 remains separate applicable evidence.

The unit consumes58.179977 CPU and142.276204 GPU process-seconds. Readback is
run05248/remaining-svd-resource-readback.json under the campaign raw root.
The verified remaining-svd-resource-05246-05248-evidence.tar.gz and adjacent
verification JSON preserve all three manifests/logs/results and checkpoint
sources. No worker failed. The resource allocation now uses46/48 workers.

Primary-agent review:12 caller replacements test more than repeated identical
calls, but they share just two numerical shapes; do not infer unlimited shape
coverage. Small GPU live allocations do not explain total native residency.
Receipt/source applicability permits reusing unchanged costs without promoting
their failed arm. Remaining angle/subspace and SQMC costs, affected-use and
current-caller dispositions, and terminal merge gates remain open.
