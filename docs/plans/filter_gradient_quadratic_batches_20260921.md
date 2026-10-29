# Quadratic-center batch evaluation repair

Question: can the Python chunk loop in `refine_batched_quadratic_center.evaluate`
be replaced by one stable-signature XLA program without changing target call
order, duplicated padding, strict incumbent selection, invalid-row stopping,
replay rows, physical-row counts, or any reported candidate batch?

Baseline: exact evaluator source at pushed `bd36a89b`, which still contains the
original Python chunk recurrence. Complete public numerical comparisons also
load original `3582b4ac` and its isolated dependency closure. That original source
remains the numerical authority; the intermediate evaluator is an execution
comparator only. Use fixed binary64 targets and unchanged configured seeds,
limits and tolerances (atol=rtol=1e-10, discrete fields exactly equal).

Mechanism: fixed-capacity input/output tensors, runtime active-row count, and a
single `tf.while_loop` with sequential batched calls. Reject a nonfinite input
chunk before calling the target but after charging its physical/padded rows, as
the original does. Retain selection from valid rows of a partly invalid batch
before stopping. Replay keeps padded values/scores and cannot change the
incumbent. Materialize candidate history only at the host reporting boundary.
The kernel defaults to XLA; explicit graph mode is diagnostic only. No pfor,
Python numerical callback, NumPy, retuning or alternative score is permitted.

Pass criteria: exact callback order/counts and full discrete reports, complete
numeric records at the unchanged tolerance, graph/XLA equality, a single trace
across changed inputs/counts/record flags, and successful enclosing GPU XLA/HLO.
Test partial/full batches, ties, all-invalid and mixed-invalid rows, nonfinite
points before/after a valid batch, replay duplicates and callback shape/dtype
errors. Stateful test callbacks must count actual TF executions with int64
resources, not Python tracing. Compare complete public records for uniform and
paired probes, rejection and successful geometry. Current outer fit-round and
probe-partition loops, non-XLA fitting and raw trust eigensystem remain explicit
debt; this dependency cannot close F18/F19 or establish whole-initializer XLA.

After numerical qualification, compare the exact original evaluator and current
graph/XLA at two capacities in separate fresh CPU/GPU processes, same targets,
inputs/output lifetime and 20 warmed calls. Preserve compile/cold/warm/copy,
host RSS/high-water observations, TF allocator current/peak, trace and graph/HLO
counts. Costs are descriptive; terminal three-process repeats and complete
endpoint costs remain required. Existing >20% warm regression, >2x device peak,
>256 MiB host increase and ongoing allocation-growth investigation triggers apply.

Execution uses only the existing absolute campaign-driver prefix, registered
test groups, GPU2 idle preflight, verified growth and sequential workers. Each
focused group has a 300-second ceiling (900 only if measured group size requires
splitting or that existing bound); each cost worker has a 120-second ceiling.
All attempts consume the unchanged 52 GPU / 32 CPU process-hour caps. Results
go into fresh `run-*` directories under the existing campaign artifact root.
Stop the trial on broken source isolation, missing observations, uncontrolled
resource growth or exhausted budget. A numerical failure rejects this candidate
and triggers localization, without waiving the gate or changing the algorithm.

Skeptical pre-execution review: a final-incumbent-only check would miss padded
replay and invalid-row accounting; full history and callback counters are
mandatory. A graph callback may change rounding even when the chunk algorithm
is identical, so complete original-source records precede acceptance. A Python
counter inside the callback counts traces and is an invalid oracle. Large static
history buffers can increase memory: measure at both extents and keep complete
outer costs open. The kernel is not training, a tuner, HMC, a filter-score
substitute, or a claim of scientific/default readiness. No external source or
pin changes are included. This primary-agent audit finds the bounded dependency
repair appropriate; independent terminal review remains pending.

Numerical qualification 01840/01842 passes all 25 focused CPU/GPU cases;
01841/01843 pass all 51 public CPU/GPU cases; 01844/01845 pass all 38 paired
pilot CPU/GPU cases. Stateful test counters now count TensorFlow executions
with int64 resources. 01846 passes all 67 policy/controller checks. The guard
adds the complete new numerical module and only the public evaluator scope;
one exact exception assembles completed history and does no target selection.

Cost-harness review caught an output-lifetime mismatch: the exact extracted
original evaluator was bound to a diagnostic namespace that retained the last
details/incumbent after the returned tensors were released. The real original
closure was invocation-local. Runs 01847--01850 are preserved as preliminary
harness evidence and are excluded from accepted comparisons. The matrix paused
between workers; removing those two temporary namespace bindings changes no
arithmetic. Restart both matched cost matrices under the same limits. No
runtime source or numerical gate changed for this correction.

Runs 01851--01862 complete the corrected six-arm CPU/GPU cost matrices. Every
returned field matches the source-pinned baseline at the unchanged tolerance;
the baseline evaluator AST is exactly identical in `3582b4ac` and `bd36a89b`.
The comparison is `quadratic-batch-cost-comparison-01862.json` in the campaign
artifact root, produced by `/tmp/analyze_quadratic_batch_costs_20260921.py`.
No warm-time, device-peak or 256-MiB host regression trigger fires. CPU XLA
adds about 161 MiB of observed host footprint; GPU adds 7.3/0.2 MiB. These
figures use stage/call RSS and reported VmHWM, not exact maximum resident memory.
The raw report caveat referring to driver memory samples was inaccurate for
these test workers; the analysis records that correction and future reports
now state it explicitly.

Small warm host changes remain (CPU XLA 60/52 KiB across 20 calls, GPU XLA
20/16 KiB). GPU live allocations are constant, but short traces do not establish
a plateau. Next run four fresh diagnostic processes at D5/capacity128:
original/current evaluator on CPU/GPU, 20 warm calls followed by four blocks of
100 calls, discarding outputs each call and retaining only block-end memory.
Compare these sparse observations with the earlier per-call observation growth;
record final optional garbage collection separately. This diagnoses recorder/
allocator warm-up versus continued retention and is not a timing comparison.
Use the same callback/arguments, one trace and <=120 seconds per worker. A
continued material increase triggers further attribution; no cleanup is added
to runtime and no leak-freedom claim follows from a finite plateau.

| Device / capacity | Original warm ms | Graph warm ms | XLA warm ms | XLA minus original observed host footprint | Original / XLA device peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / 32 | 21.545 | 1.697 | 0.711 | 168,792,064 | not applicable |
| CPU / 128 | 85.719 | 4.828 | 1.026 | 168,669,184 | not applicable |
| GPU2 / 32 | 34.262 | 8.142 | 1.261 | 7,626,752 | 17,408 / 12,800 |
| GPU2 / 128 | 133.473 | 24.931 | 2.328 | 176,128 | 61,184 / 26,880 |

Warm figures are medians of 20 calls within each single process. The baseline
oracle includes stacking its completed history into the same reported tensors;
the candidate returns that history natively. Both materialize all fields.
These costs exclude the outer fit rounds and public list/report assembly, so
they cannot establish complete initializer speed. Candidate graph node count is
211 at both extents with one trace. GPU XLA live allocations remain constant
across the measured warm calls (9,216 / 23,040 bytes).

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain the batch evaluator repair on the validation branch | Original records and CPU/GPU consumer suites pass | No numerical failure in tested scope; no stated ratio trigger | Small host growth and full outer execution | Complete sparse growth diagnosis, then outer fitter/controller migration | Complete execution repair, scientific admission or merge readiness |
| Preserve all prior D5 numerical blockers | Complete original record equality required | 66 CPU / 53 GPU changed-input fields still fail | Coupled optimizer rounding | Continue discrimination at unchanged thresholds | That passing this independent helper waives the lifecycle failure |

Post-run primary-agent review: these fixtures exercise both probe modes and
failure reporting but are low dimensional. The strongest alternative to a
general speed benefit is avoided host dispatch dominating these small targets.
Actual model callbacks and full fit rounds can change the cost balance; their
required future comparisons remain explicit. Statistical speed ranking,
default readiness and broad memory stability are unsupported by these single
process observations.

01871--01874 retain identical records across 421 calls in each process. GPU
allocator current is constant after output release (original 9,728 bytes;
XLA 8,960 bytes). Original host RSS settles by 100--200 additional calls; XLA
GPU settles in the last 100-call interval. CPU XLA still grows by 84 KiB across
400 calls, and final garbage collection does not reduce it. This is not proof
of a leak or of stability. Run two XLA-only longer diagnostics, CPU and GPU2,
10 blocks of 1,000 calls after the same 20-call warmup, keeping only block-end
observations and identical outputs. The measured ~1--2 ms calls support a
120-second worker bound. All original/candidate evidence remains preserved;
no cleanup or numerical change is installed. Continued growth will trigger
ownership/allocation attribution, not a claimed plateau.

01875/01876 complete 10,021 identical calls in each CPU/GPU process. CPU host
RSS rises 233,472 bytes after the short warmup, then is unchanged from the
7,000 through 10,000 additional-call observations. GPU host RSS rises 163,840
bytes and is unchanged from 2,000 through 10,000 additional calls. GPU current
remains 8,960 bytes after output release and peak is 26,880 bytes; both factories
trace once. Final diagnostic garbage collection is outside ordinary execution.
These observations support bounded warm-up in the tested fixed shape, with no
continuing growth at the final sampled intervals. They do not establish cache
eviction, leak freedom across shapes/callbacks, or full-controller memory costs.

Final source review confirms batch evaluation is numerical TensorFlow/XLA;
the one allowlisted Python loop only transports completed report rows. Callback
shape/dtype errors still propagate. Two old tests' Python trace counters were
replaced with TensorFlow execution counters without changing their assertions.
The public outer fit-round, design partition and forced non-XLA fitter remain
tracked migration debt. Source-authority disposition is documented separately
in the eigenpair checkpoint, with all original-source gates retained.
