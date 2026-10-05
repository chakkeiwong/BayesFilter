# Streaming owner resource acceptance

The bounded streaming unit completes through05245. Twelve fresh GPU cost
workers05228--05239 preserve exact complete shared outputs and RNG records;
three lifetime workers05240/05243/05244 and161 terminal checks05245 pass.
This accepts the measured execution tradeoff for the declared streaming value
owner. The prior stricter CPU timing study remains failed, as reported below.
No runtime algorithm, random stream, tolerance or policy allowance changed.

| Horizon | GPU warm before -> after (ms) | After/before ratio, approximate95% interval | Cold before -> after (s) | Mean warm RSS change (MiB) |
| --- | --- | --- | --- | --- |
|32|78.291 ->77.423|0.9890 [0.9113,1.0734]|10.906 ->11.098|-10.55|
|128|307.921 ->305.072|0.9908 [0.9524,1.0307]|11.258 ->11.360|-10.79|

These are three counterbalanced fresh-process pairs per horizon, with30
synchronized warm calls per process. Times are means of arm medians; the
interval uses paired log ratios and Student-t with two degrees of freedom.
Both upper limits are below the predeclared1.10 bound; neither comparison
establishes a speed improvement. The LGSSM fixture is seed13,N64,D2,T32/128,
process123/resample17. GPU3 UUID GPU-b8045e28-4433-ec7a-77a5-db0636748322 had
unshared preflight and in-run observations; all workers verified memory growth.
Timing and primary memory measurements precede comparison compilation or HLO.

The earlier CPU study04729--04748 has ratios1.08239/1.09678 and upper95%
limits1.11687/1.15608, above1.10. That stricter verdict remains failed. The
component investigation did not establish RNG as the slowdown's cause; the
tested state intervention worsened it and was rejected. The master permits a
documented measured tradeoff: retaining the exact stream in the compiled
streaming owner avoids a horizon-sized random preparation buffer and preserves
early-stop draw consumption. The default GPU route has equivalent outputs,
bounded measured cost and stable reuse. Accept the scoped CPU overhead under
that engineering tradeoff; do not relabel the stricter study or claim universal
speed. The precise low-level cause of the CPU timing difference is unproved.

Each current owner makes128 valid synchronized calls. T32 alternates both
predeclared seed/observation inputs. T128's second input is invalid:05241
preserves the failed harness assumption, and05242 proves the frozen buffered
reference and streaming owner reject with exactly matching records, including
NaN locations and statuses. Runs05243/05244 measure the valid first input and
retain the second as a changed-input refusal witness. No seed was substituted.

| Lifetime scope | Late64-call RSS growth (bytes) | Extra RSS for four configurations (MiB) | Validity at N8/16/32/64 |
| --- | --- | --- | --- |
|GPU,T32 (05240)|0|692.523|false,false,true,true|
|GPU,T128 (05243)|16384|698.852|false,false,false,true|
|CPU reference,T128 (05244)|24576|755.215|false,false,false,true|

All configurations use K=N. Invalid configurations measure allocation behavior
only; they do not establish usable scientific capacity. Late growth is below
the16MiB investigation trigger; four additional configurations stay below the
2GiB incremental RSS bound. Every owner has one trace, exact replay and
collectible Python ownership. GPU allocator current/peak bytes plateau at
8192/184561152 (T32) and12800/184565760 (T128). These differ from process GPU
reservation, which includes context/compiler/library memory. Native residency
survives Python collection: final worker RSS is about2.27GB GPU/1.95GB CPU.
Use a bounded worker lifetime when changing configurations; saved parent
observations verify worker exit and absence of its GPU context. No native
eviction or unlimited-configuration guarantee follows.

The exact sources, environment, commands, inputs, hardware, measurements,
process observations and failed05241 are preserved in raw runs05228--05245
under /home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917.
Run05245/streaming-resource-readback.json independently checks the comparisons.
The frozen pre-repair harness and AST identity of its timed functions preserve
the applicability of05228--05241 after the test assumption was corrected.
All numerical dependencies remained unchanged. The verified archive
streaming-resource-05228-05245-evidence.tar.gz and adjacent verification JSON
preserve every raw record and checkpoint sources.

Primary-agent review: accepting this tradeoff does not override the CPU failure,
establish speed superiority, or turn rejected particle configurations into
capacity evidence. Three timing pairs and sampled sharing limit inference.
Remaining-SVD lifetime, angle/subspace guard and SQMC preparation resources,
affected-use dispositions and final integration gates remain open. No general
leak-freedom, canonical LEDH, posterior, HMC or whole-program conclusion follows.
