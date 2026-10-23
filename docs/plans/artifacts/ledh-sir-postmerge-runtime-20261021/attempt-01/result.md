# Why the frozen SIR replay became slower

21 October 2026. The merge introduces a concrete extra computation in the plain
and conditional endpoints: the compiled filter now retains the flow-step QR
factorization and determinant derivative because the new validity checks depend
on them. Before the merge, XLA eliminated this work from those two endpoints.
The physical mixture already required it. Alternating unchanged before/after
endpoints reproduces this selective slowdown. This is a verified change in
compiled work, with timing evidence consistent with that mechanism; the exact
fraction of total overhead attributable to QR alone was not isolated.

## Checked consumer and compiler dependencies

The diagnostic reconstructs the replay's actual consumer, including moment
loss, raw moments, analytical derivative and every returned validity diagnostic:

`make_moment_endpoint` → `build_backward_guided_ledh_kernel` →
`_value_and_analytical_score_impl` → `_flow_substeps_with_tangent` →
`_flow_substep_body_impl`.

In the merged snapshot, `bayesfilter/highdim/ledh_canonical_score_tf.py:1432`
factorizes `I + eps*A` with QR. Lines1444–1448 make the validity flags depend on
the factorization, finite accumulated log determinant and finite analytical
log-determinant derivative. The filter consumes those flags at line682 and
propagates them into subsequent states and the returned validity status.
The moment consumer reduces those flags into its actual `valid` output
(`bayesfilter/score_study/moment_training_tf.py:92`). They are live computation.

The plain Gaussian density uses the pushed covariance of the accumulated affine
map (`ledh_marginal_weights_tf.py:101`); it does not consume the accumulated
per-step log determinant. The conditional beta=1 proposal returns only its
conditional Gaussian density (`ledh_physical_proposal_tf.py:126`). The beta=.9
mixture also evaluates the LEDH density, which consumes the log determinant
(lines129–142). This explains the differing compiler behavior.

| Endpoint | Flow QR in optimized HLO before | Flow QR after |
| --- | --- | --- |
| Plain beta=0 | Absent | Present |
| Conditional beta=1 | Absent | Present |
| Physical mixture beta=.9 | Present | Present |

The complete optimized HLO files and extracted QR lines are preserved in
`before/` and `after/`; `analysis.json` records their checksums. Metadata-line
counts are not operation counts. The executed loop has5 observation steps,
24 flow substeps and64 particles: 7,680 individual18-by-18 QR factorizations
per filter evaluation, implemented in batches. Their tangents and validity
reductions also remain live. These are not extra Sinkhorn iterations.

Two scope issues explain why a performance-oriented merge did not guarantee a
speedup here. Commit65a379b2c adds native analytical-score batching in
`covariance_proposal_score_batch_tf.py` and related shared/sharded implementations.
The guided consumer above calls the scalar canonical score implementation,
not that batch factory. Also, `ledh_small_matrix_tf.py:13` specializes only
static matrix orders1 and2; this SIR scope's main matrices have orders9 and18.
These changes can benefit their intended scopes without offsetting the extra
work in this consumer. No claim about their performance in other scopes is made.

## Alternating timing diagnostic

Use the reviewed plan `docs/plans/ledh-sir-postmerge-runtime-20261021.md`.
Two persistent isolated workers loaded the original frozen source snapshots.
After compilation, eight paired blocks alternated which version ran first and
rotated arm order. Only one worker received an execution request at a time.
Every returned tensor was materialized before stopping its timer. Both zero
direction (the original moment workload) and coordinate direction0 (a nonzero
analytical-score control) used the same single-trace compiled functions.

| Endpoint | Zero direction before / after, seconds | Observed ratio | Nonzero direction before / after, seconds | Observed ratio |
| --- | ---: | ---: | ---: | ---: |
| Plain | .238865 / .435226 | 1.822 | .199005 / .477087 | 2.397 |
| Conditional | .214721 / .403183 | 1.878 | .204617 / .497219 | 2.430 |
| Physical mixture | .430423 / .404406 | .940 | .423734 / .498255 | 1.176 |

These are descriptive means of eight calls per cell on one exposed dataset.
The physical-mixture differences do not establish an improvement or regression.
Machine load still varies; for example, individual zero-direction paired ratios
range widely. The table does not identify a general SIR slowdown factor.
It does show the predicted selective pattern even when the two versions are
interleaved. The original sequential2.249/2.360 ratios remain measurements of
that campaign, not universal causal estimates.

Python conversion after tensor synchronization averaged.000125 seconds and
never exceeded.000519 seconds. It cannot explain the roughly.2-second mean
increase in the zero-direction plain/conditional calls. Compilation is excluded
from the table. Every endpoint traced once; no retracing occurred.

## Integrity, budget, and reproducibility

All96 warm measured calls and12 initialization/control calls passed same-source
tie-outs. Compared loss, moments, value and score reproduce the archived replay
exactly (maximum absolute difference0 for each available field). No invalid
result or source/input mismatch occurred. Both source snapshots remained
unchanged. This does not repair the earlier cross-source plain numerical parity
failure; the diagnostic reproduces each source's own result.

The exact supervisor command was:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/diagnose_sir_postmerge_runtime.py --repository /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922 --output /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/ledh-sir-postmerge-runtime-20261021/attempt-01
```

Each worker's `manifest.json` records its exact command, source commit/index,
TensorFlow version, custom-op hash, controls, input checksum and memory policy.
TensorFlow2.20.0-dev0+selfbuilt ran on the same RTX4080SUPER with escalated GPU
access, FP64, TF32off, XLAon, memory growth verified before initialization,
`CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, and the existing
WSL exception `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. The custom-op directory
was the worktree's `bayesfilter/ops`; bytecode writing was disabled.
Dataset263043001, particle seed283030000 and reset seed263040002 remain frozen.
OT remains(1,5), N64, T5, chunk64 and24 flow substeps. No training or tuning ran.

The before worker lived249.052378seconds; the after worker163.932426seconds.
Their combined412.984804 GPU-process seconds conservatively include resident
idle time while the other worker ran. The total replay-plus-diagnostic charge
is5142.898765 of19800seconds;14657.101235 remain. Both workers exited normally.
Allocator peaks were8,828,160 and8,832,256bytes, respectively. Full timings,
outputs, cold costs and terminal trace counts are in `result.json`, `rows.jsonl`
and the worker logs. `analysis.json` is reproducible with the CPU-only stdlib
command `CUDA_VISIBLE_DEVICES=-1 python scripts/analyze_sir_postmerge_runtime.py
<this-attempt>`; no framework or NumPy computation is involved in reporting.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Explanation confirmed at compiled-work level | QR absent before and live after only in predicted endpoints; alternating timing pattern agrees | No invalid output, source mismatch, tie-out failure or retracing | Precise QR share of overhead and wider-scope speed ratio | Optimize the shared flow determinant/validity calculation while preserving all failure checks; separately evaluate whether guided batching should be integrated | No numerical repair, new default, score improvement or general speed ranking |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No diagnostic validity veto fired; earlier cross-source numerical non-equivalence remains |
| Statistically supported ranking | None claimed from this one-dataset timing diagnostic |
| Descriptive-only differences | All timing means and ratios in the table |
| Default-readiness | Not evaluated; no runtime implementation changed |
| Next evidence needed | A shared optimization with unchanged validity semantics, healthy/invalid-input regression checks, and a paired representative timing study |

Terminal red-team review: live QR is established by optimized consumer HLO,
not merely finding a QR call in source. The strongest alternative explanation
for part of the measured slowdown is changing load or other added validity
work. A controlled optimization that removes redundant factorization while
preserving all checks and leaves runtime unchanged would weaken QR as the
dominant time explanation, but would not undo the verified extra compiled
work. The weakest evidence is the exact speed ratio from one dataset. Removing
validity checks to recover old timings is not an acceptable repair. Neither
Sinkhorn tuning nor guide retraining answers this runtime problem.
