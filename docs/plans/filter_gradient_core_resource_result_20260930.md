# Core filter matched resource disposition

The ten scoped core numerical owners pass90 qualified GPU cost workers and
terminal readback05400 passes161 checks. All30 original/current-graph/current-XLA
blocks pass the unchanged1e-10 FP64 bounds and exact discrete fields; maximum
absolute discrepancy is1.31902266886641e-11. There are30 current-XLA128-call exact-replay
checks with one trace and no callbacks. Late64-call RSS growth is
0–16384 bytes; live GPU allocator
occupancy is unchanged between64 and128 in every worker. Process exit and
uncontended observations qualify. No numerical implementation changed.

The plan is filter_gradient_core_resource_renewal_20260930.md. Baseline3582b4ac
and the currente7d2432c4 Python closure use the same frozen prepared inputs,
TensorFlow2.19.1 environment, FP64 and physical GPU3. Original TT host routes
are explicit eager references; other originals use XLA. Current graph execution
is a nondefault control. Growth is verified before device initialization.
Commands, hashes, environment, seeds, hardware and timings are in run manifests
05306–05400. Original and current closures are complete, independently loaded.

Times below are medians of three independent processes;20 inner warm calls
per process are not statistical replicates. Intervals are paired log-ratio t
intervals with only three blocks. Setup+cold includes preparation, trace and
first synchronized execution. RSS delta is the median paired warm difference;
allocator peaks are live TensorFlow allocation measures, separate from process
reservation. The saved readback also contains all graph-control values.

| Fixture | Original→XLA warm ms | Paired ratio [95% interval] | Setup+cold s | Extra host RSS MiB | Allocator peak MiB |
|---|---:|---:|---:|---:|---:|
| contract_e | 37.550→39.692 | 1.0412 [0.9633,1.1254] | 12.847→11.177 | -0.062 | 0.076→0.069 |
| tt | 22.182→15.110 | 0.6781 [0.6472,0.7105] | 10.247→14.742 | +193.277 | 0.128→0.141 |
| tt_adapted | 108.944→15.950 | 0.1465 [0.1416,0.1517] | 11.384→16.840 | +178.816 | 8.031→0.148 |
| tt_gaussian | 116.164→16.335 | 0.1408 [0.1394,0.1423] | 11.562→16.845 | +147.305 | 8.030→0.143 |
| tt_adjoint | 738.421→23.321 | 0.0314 [0.0301,0.0327] | 3.685→25.267 | +851.344 | 8.220→0.231 |
| tt_actual | 12.907→20.525 | 1.5923 [1.5312,1.6557] | 16.440→6.592 | -177.098 | 8.133→4.133 |
| tt_scalar | 864.493→96.093 | 0.1111 [0.1106,0.1117] | 4.199→23.184 | +936.234 | 9.241→0.436 |
| apf | 1.018→1.157 | 1.1325 [1.0585,1.2117] | 2.713→2.614 | +2.320 | 0.016→0.014 |
| dns | 0.742→0.749 | 1.0088 [0.9892,1.0289] | 0.942→1.625 | +29.121 | 0.123→0.124 |
| retained_moments | 0.742→0.854 | 1.1483 [1.0139,1.3005] | 1.323→1.944 | +32.711 | 0.007→0.010 |

The TT analytical adjoint and scalar finite-program AD diagnostic trigger the
host-memory and cold-cost investigation. Graph-only controls already require
1823.87MiB and1736.08MiB warm RSS respectively, versus1137.83MiB and1160.23MiB
for the original eager controls. Current XLA requires1988.79MiB and2096.47MiB.
Thus graph construction and execution account for much of the increase; XLA
adds approximately165MiB and360MiB over graph. Stage observations localize the
increase to preparation, tracing and first execution. Pair0 adjoint prepared /
traced / cold RSS is1047.07/1275.16/1988.75MiB; its subsequent late64-call RSS
growth is12KiB with9984 live device bytes. The three blocks and128-call checks
show bounded fixed-shape reuse, not proof of native allocator eviction.

Retain these compiled routes with an explicit small-fixture engineering
tradeoff: about2.0–2.1GiB resident host memory per worker, materially larger cold
cost, much lower repeated-call latency and lower device allocator peaks. Reuse
retained owners and release native residency through process exit when a worker
is retired. The bounded Python factory caches do not promise that eviction
returns all compiler allocations to the OS. No arbitrary-T/N capacity or
universal speed/memory claim follows. The scalar AD measurement cannot establish
an analytical-score claim.

Actual-SV TT triggers a warm-time investigation:12.907→20.525ms, ratio1.5923.
The original graph expands Python date/ALS/reverse loops; the current source
uses shared native recurrences, core checkpoints and exact design reconstruction.
The matched graph records change from10390 outer nodes/no functions to2614
outer nodes/28 functions (function bodies are additional, so outer-node counts
are not whole-graph size). Current graph-only warm83.808ms is slower still.
Current setup+cold improves16.440→6.592s, host RSS decreases177.098MiB and device
peak decreases8.133→4.133MiB. This comparison identifies the native recurrence /
checkpoint-replay execution change; it does not isolate every kernel cost.
Accept the measured7.62ms warm overhead in this T4 fixture to retain bounded
native control, lower setup/memory and policy compliance. No claim of a warm
speedup is made; optimizing that scope remains optional engineering work.

APF and retained-moment point estimates increase13.25% and14.83%; their95%
upper bounds exceed20%. These are qualified descriptive costs, not evidence of
a universal20% ceiling. DNS and ordinary TT add host residency below the256MiB
trigger. No fixture doubles live allocator peak. All fixed-shape lifetime gates
pass; none of these results establishes native memory release without exit.

Failures remain preserved.05306 has142 passes/18 harness failures because an
optional selection attribute was absent from internal Namespaces; the
absent-safe lookup repairs it and05307 passes160.05310 and05347 complete their
numerical/reuse work but fail five-second nvidia-smi query deadlines. Their
timing remains unqualified; same-source retries05311 and05348 pass, without
relaxed gates or observed foreign processes. No failure is erased or substituted
with a different numerical method.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action / nonclaim |
|---|---|---|---|---|
| Close scoped core-resource renewal |90 costs,30 three-arm numerical blocks and161 readback checks pass | Failed workers retained; no numeric/resource reuse veto | Only fixed fixtures and measured owner boundaries | Reuse this evidence; no universal cost or whole-program closure |
| Retain adjoint/scalar compiled owners | Values/scores preserved; warm improvement and stable reuse | Large host/cold increases explicitly accepted | Native residency ownership/large-shape capacity not proved | Retain owners, bounded workers and exit-based lifetime |
| Retain actual-SV native recurrence | Same numerical outputs with lower cold/memory |59.23% warm regression explicitly accepted in T4 scope | Kernel-level attribution and larger shapes unmeasured | No universal speedup claim |
| Keep C2 preparation open | Prepared evaluator costs do not cover enclosing proposal construction | Confirmed Python feedback loops remain | Complete public preparation still unqualified | Execute the bounded C2 repair; no main merge yet |

Primary-agent terminal review checked source/input provenance, old eager versus
XLA scope, inner-call pseudoreplication, monitor failures, graph-control memory
and the distinction between live allocation and reservation. The strongest
alternative explanation is that finite small fixtures hide larger-shape
compiler/capacity costs. The qualified claim is deliberately confined to these
owners/fixtures. A changed active dependency or a demonstrated caller regression
reopens its scope. Adaptive iAPF/KDM remain deferred; canonical LEDH rebuilding,
training, HMC and historical optimizer research remain outside this unit.
