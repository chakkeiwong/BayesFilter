# Fixed SSL-LSTM replay execution result

The public fixed-replay value and analytical score now execute through one
bounded TensorFlow/XLA owner, including parameter unpacking, replay noise,
native time control, complete output tensors and numerical diagnostics.
Theta and observations remain dynamic inputs. Public metadata stays outside
the numerical program. The Phase6 benchmark reports the mode actually executed
and labels explicit non-JIT output as debug/reference. This is the existing
local extension, not source-faithful Zhao--Cui or canonical LEDH admission.

Numerical qualification passes17 tests per backend in04981/04982. Both the
original direct-call Philox stream and the original enclosing-XLA stream are
preserved. Complete records, changed-input reuse, T1, multivariate observations,
original errors, default HLO/native looping and one-trace execution pass.
Analytical derivatives pass independent finite differences and tape diagnostics;
the runtime score remains analytical. The independent dense derivative oracle
now clears its compiled owner before tracing patched callbacks, avoiding reuse
of the candidate as its own reference.

Run04980 preserves a compatibility failure: list-valued seeds accepted by the
pinned original were unhashable in the initial cache. The repair normalizes the
numerical manifest key, retains caller metadata, and validates seeds through
TensorFlow before cache lookup. This prevents a populated integer key from
admitting numerically equal but invalid float seeds. Earlier costs04944--04979
are superseded; their evidence is retained.

The complete corrected-source cohort is04983--05054: three fresh-process
repeats per before/after, public-default/explicit-graph/explicit-XLA, T2/T8 and
CPU/GPU combination. Baseline is c4950a827. Identical frozen theta, observations,
nine particles, seeds, float64 and disabled TF32 are used; explicit XLA retains
its own original stream. GPU3 UUID
GPU-b8045e28-4433-ec7a-77a5-db0636748322 has verified growth and clean sampled
sharing. CPU is explicitly hidden-GPU reference. Runs05055/05056 independently
verify all36 records per backend, provenance, replay and paired numerical
agreement. Maximum absolute error is2.67e-15; explicit-XLA GPU records are exact.

| Device/mode | T | Before warm ms | After warm ms | Paired geometric ratio (95% interval) | After-minus-before RSS MiB |
|---|---:|---:|---:|---|---|
| CPU public default |2|36.760|1.016|0.0271 (0.0240--0.0307)|193.3--194.0|
| CPU public default |8|167.908|1.036|0.00604 (0.00544--0.00670)|193.1--194.1|
| GPU public default |2|58.132|1.866|0.031 (0.028--0.034)|-29.4---26.0|
| GPU public default |8|261.624|2.036|0.007 (0.006--0.010)|-31.4---28.0|
| CPU explicit graph |2|1.143|1.573|1.460 (1.120--1.902)|14.7--15.0|
| CPU explicit graph |8|3.421|4.661|1.335 (1.274--1.399)|-44.1---43.8|
| GPU explicit graph |2|3.962|6.247|1.572 (1.357--1.822)|13.2--24.4|
| GPU explicit graph |8|8.984|12.610|1.396 (1.265--1.541)|-46.9---34.1|
| CPU explicit XLA |2|0.621|0.650|1.000 (0.727--1.374)|9.8--10.4|
| CPU explicit XLA |8|0.694|0.653|1.005 (0.773--1.305)|-126.5---124.7|
| GPU explicit XLA |2|1.394|1.535|1.067 (0.963--1.183)|7.6--8.6|
| GPU explicit XLA |8|1.473|1.777|1.235 (1.090--1.399)|-119.3---117.2|

Warm milliseconds are medians of process medians. Paired intervals use
Student-t on three log ratios (df2), with no multiple-comparison or general
superiority claim. The old public default is eager; that comparison must not
be confused with matched compiled modes. At T8 GPU explicit-XLA cold cost
falls5.950 to1.967 seconds. The original graph grows1169 to5201 nodes from
T2 toT8; candidate graphs stay constant:1311 direct/default,1346 explicit graph,
1166 explicit XLA. Different RNG compatibility contexts account for different
fixed graphs. Compiled-context warm regressions remain measured costs despite
the much faster new ordinary default.

Lifetime05057/05058 runs2000 alternating-input complete public calls, with exact
per-input replay and one trace. Final1000-call sampled RSS increases are0.227
MiB CPU and0.234MiB GPU. GPU allocator current/peak stays5632/34304 bytes.
Twenty distinct seed configurations obey the16-entry cache cap and retained
entries do not retrace. Their extra sampled RSS is880.0MiB CPU/786.3MiB GPU,
within the declared2GiB diagnostic cap. Ordinary cache clear removes entries
but releases no observed native RSS. A bounded Python cache is therefore not
a guarantee of native executable eviction. Reuse a fixed manifest where
applicable; independently scoped campaigns that create many specializations
need bounded process lifetimes. No unlimited-capacity or leak-freedom claim.

| Decision | Primary criterion/veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|
| Qualify execution and numerical repair in tested scope | Full records, independent score diagnostics, seeded/error compatibility and enclosing CPU/GPU XLA pass | Broader fixtures and caller configurations | Preserve evidence and current source guard | Whole-program or scientific admission |
| Record bounded residency | Fixed owner stabilizes;20 configurations stay within declared capacity; cache clear does not release RSS | Native registry/compiler lifetime beyond this finite process | Preserve practical process-lifetime limitation | LRU eviction releases native memory |
| Keep compiled-context cost gate open | GPU T8 XLA ratio1.235; graph modes also slower | Repeated conditional inside native time loop is a concrete candidate overhead | Test unconditional T-1 observation/transition loop plus one final observation | Automatic acceptance based on public-default speedup |

Inference status: no numerical/provenance veto in the corrected cohort;
statistical speed superiority and default readiness are not established.
Reported differences are fixture-specific. A conditional-split intervention
must preserve all equations, RNG contexts, records and errors and pass fresh
cost comparisons before replacing this result. No tolerance changes, training,
HMC, canonical rebuild or optimizer-convergence research are included.

Primary-agent terminal review: pinned original execution is independent and
source-bound; metadata coercion and invalid-seed cache collisions were tested.
The strongest cost alternative is launch/loop overhead in this tiny fixture;
static graph size cannot establish its cause. Failed timings remain visible.
No independent-agent review is claimed. Focused Ruff and whitespace checks pass;
the legacy benchmark's10 and dense-oracle test's3 Ruff findings are unchanged
from HEAD. Final164 readback/policy checks05059 pass. Guard coverage is317
sources/1465 exact exceptions; new exceptions cover schema/reporting only.

The119-worker unit used384.029219 CPU and666.028694 GPU process-seconds,
including the preserved cache failure and earlier superseded cost workers.
Archive ssl-lstm-replay-05059-evidence.tar.gz contains504 reopened/verified
members,22372503 bytes, SHA256
e42d7999b4640e20e9080c1d156349397fa14b59ad19ffa671e7589814887d08.
The adjacent verification JSON binds every archived run/log/result. Files are
under docs/plans/artifacts/filter-gradient-repair-20260917. Current branch/main
status and remaining global budget are in the concise resume checkpoint.
