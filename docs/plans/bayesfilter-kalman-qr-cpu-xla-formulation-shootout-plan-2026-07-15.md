# Kalman QR CPU/XLA Formulation Shootout Plan

Date: 2026-07-15

Risk tier: `TIER_2_MATERIAL_RESEARCH_ENGINEERING`

Status: `AUTHORIZED_AFTER_SKEPTICAL_AUDIT`

## Research Question

Does the surprising CPU result come from the Kalman benchmark or work contract,
from the current native `[B,...]` tensor formulation, or from a limitation of
TensorFlow/XLA CPU lowering that persists across reasonable single-process XLA
formulations?

This experiment does not modify the accepted `r3` throughput harness or its
frozen source fingerprint. It creates a separate diagnostic harness and treats
the prior 16-worker result as an external reference.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can a single CPU process using XLA evaluate the same 16 canonical proposals substantially faster or with materially better core utilization than the current native `B=16` formulation? |
| Exact baseline | `native_batch`: `build_batch_native_analytic_fn(..., batch_size=16, jit_compile=True)` on CPUs `16..31`, NUMA node 0, intra-op 16, inter-op 1. |
| Candidate mechanisms | Strict `tf.vectorized_map` pfor over the scalar analytical score; fallback-enabled `tf.vectorized_map`; `tf.map_fn` with `parallel_iterations=1` and `16`; a statically unrolled 16-row scalar analytical graph. |
| External references | One process making 16 sequential calls to one compiled XLA `B=1` function, plus the already accepted 16-process XLA `B=1` result. Neither is eligible to win the single-process formulation comparison. |
| Expected failure modes | Mapping transformations lower to the same inefficient batched primitives; map loops remain serial under XLA; static unrolling causes graph/compile explosion; or one alternative exposes row-level concurrency that the native batch formulation does not. |
| Nomination criterion | On `(D,P,T,B)=(10,50,120,16)`, a parity-valid single-process candidate must compile within 300 seconds and have descriptive median warm makespan at most `0.80` times native batch in the first fresh-process block. |
| Confirmation criterion | A nominee advances to eight fresh-process paired blocks against native batch. Its paired bootstrap 95% interval for candidate/native warm makespan must be entirely below `0.90`, with no parity, source, placement, contamination, compile, cleanup, or memory veto. |
| Promotion veto | Wrong/missing proposal rows; value or score parity failure at `rtol=atol=2e-4`; non-finite output; silent `vectorized_map` fallback in the strict arm; source drift; invalid affinity/NUMA placement; admitted CPU contamination; or aggregate RSS above 16 GiB. |
| Continuation veto | Tiny XLA smoke fails for the native baseline; timer/synchronization or process CPU accounting is invalid; the fixed CPU topology cannot be verified; or the source fingerprint changes during a phase. A candidate-only compile/runtime failure rejects that candidate and does not stop other arms. |
| Explanatory diagnostics | First executable call, two warmups, five synchronized measured calls, process CPU seconds, average cores used, target-CPU contamination, RSS, GraphDef nodes/bytes, optimized HLO bytes/hash/op census, and raw per-round timings. |
| Nonclaims | No GPU conclusion; no universal XLA, TensorFlow, CPU, or batching conclusion; no default-policy change; no claim that `map_fn` or `vectorized_map` is production-ready; no HMC, posterior, or scientific-validity claim. |

## Formulation Semantics

All arms receive the same explicit rows `0..15` from `_make_parameter_cloud`.
The scalar mapping body uses `_model_tensors` and the scalar analytical QR
score. The native baseline uses `_batched_model_tensors` and
`tf_qr_sqrt_kalman_score_batched_static` through the existing builder.

- `native_batch`: one native `[B,...]` graph.
- `vectorized_strict`: `tf.vectorized_map(..., fallback_to_while_loop=False)`.
- `vectorized_fallback`: `tf.vectorized_map(..., fallback_to_while_loop=True)`;
  this is diagnostic and cannot establish successful pfor vectorization.
- `map_sequential`: `tf.map_fn(..., parallel_iterations=1)`.
- `map_parallel_16`: `tf.map_fn(..., parallel_iterations=16)`.
- `static_unrolled`: Python constructs 16 scalar graph branches at trace time;
  no Python work occurs inside a warm execution.
- `sequential_b1_calls`: one compiled `B=1` XLA executable invoked 16 times by
  the worker; explanatory only because it is not one compiled `B=16` program.

`tf.vectorized_map` and `tf.map_fn` are comparator mechanisms in this plan.
Their historical exclusion from the production batch-native contract is not a
performance veto and must not predetermine this result.

## Workload Ladder

### Phase 0: Harness And Tiny XLA Smoke

- Add
  `docs/benchmarks/run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py`.
- Add `tests/test_kalman_qr_cpu_xla_formulation_shootout.py`.
- Unit-test formulation identities, worker commands, parity, summaries,
  nomination logic, source drift, and fail-closed error records.
- Run every formulation at `(D,P,T,B)=(2,3,4,4)` on CPUs `16..19` with XLA,
  exact output parity, timing synchronization, and HLO capture.

Stop only if the native baseline or harness validity fails. Candidate-only
failures are recorded and the remaining formulations continue.

### Phase 1: Small-Cell Formulation Canary

Run one fresh process per formulation at `(10,50,120,16)` on CPUs `16..31`.
Use two untimed warm calls and five measured calls. Preserve compile time,
warm timing, CPU utilization, RSS, parity, and HLO/GraphDef diagnostics.

Nominate every eligible single-process candidate meeting the `0.80` timing
screen. If none qualifies, write `NO_SINGLE_PROCESS_FORMULATION_REPAIR_NOMINATED`
and stop without rerunning the large throughput comparison.

### Phase 2: Paired Confirmation

For the fastest eligible nominee only, run eight fresh-process paired blocks
against `native_batch` on `(10,50,120,16)`, with balanced order and seed
`20260715`. Use the same CPUs, source identity, warmup, parity, and resource
contract. Report paired raw ratios, geometric mean, bootstrap interval, and
sign test. Do not treat five within-process calls as independent samples.

### Phase 3: Transfer Check

Only after Phase 2 passes, run one fresh-process diagnostic block per arm on
`(30,50,120,16)`. Do not run `(30,150,120,16)` until this intermediate cell
passes the 300-second compile and 180-second measured-call bounds. Transfer
timings are descriptive and cannot replace Phase 2 confirmation.

## Environment And Resource Contract

- CPU only: `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.
- `TF_FORCE_GPU_ALLOW_GROWTH=true` remains set as repository policy.
- XLA JIT enabled for every timed formulation.
- Fixed physical CPUs `16..31` on NUMA node 0; no SMT siblings `144..159`.
- `OMP_NUM_THREADS=16`, `TF_NUM_INTRAOP_THREADS=16`,
  `TF_NUM_INTEROP_THREADS=1` for single-process `B=16` arms.
- Tiny smoke uses the first four CPUs and four intra-op threads.
- Prelaunch target-CPU maximum occupancy below 10% and one-minute load at most
  16. Measured unattributed target CPU time above
  `max(0.25 seconds, 2% of allocated core-seconds)` invalidates the block.
- First-call timeout 300 seconds; measured call timeout 180 seconds; child wall
  timeout 900 seconds; aggregate RSS stop at 16 GiB.
- Raw HLO text uses ignored `.txt` sidecars. Structured JSON and the result note
  preserve the evidence needed for interpretation.

## Evidence Contract

| Evidence item | Required artifact |
| --- | --- |
| Exact work | Explicit row IDs and full value/score outputs per arm. |
| Numerical equivalence | Rowwise residuals versus `native_batch`, tolerances, finite/shape status. |
| Timing | First call, two warmups, five measured calls, synchronization method, and per-call wall time. |
| CPU behavior | Per-call process CPU seconds, average cores used, and target-CPU contamination accounting. |
| Compiler structure | GraphDef node/byte counts and optimized-HLO byte count, hash, and operation/custom-call census; raw HLO sidecar when available. |
| Reproducibility | Git commit, source manifest/fingerprint, Python/TensorFlow versions, CPU topology, environment variables, exact command, and output paths. |
| Result | `docs/plans/bayesfilter-kalman-qr-cpu-xla-formulation-shootout-result-2026-07-15.md`. |

## Skeptical Pre-Execution Audit

Status: `PASS_AFTER_SCOPE_REPAIR`.

- Wrong baseline: avoided. The current native `B=16` XLA function is the exact
  baseline; multiprocessing is not mislabeled as non-XLA.
- Unfair work: avoided prospectively. Every arm receives explicit rows `0..15`
  and must return rowwise-parity-equivalent values and scores.
- Proxy promotion: avoided. HLO size, utilization, and one-block timings are
  explanatory or nomination evidence; only paired fresh-process confirmation
  can support a formulation repair.
- Hidden fallback: repaired. Strict and fallback-enabled `vectorized_map` arms
  are separate, and fallback-enabled success cannot be called pfor success.
- Stale design assumption: repaired. Historical prohibitions on mapping APIs
  are recognized as production-contract choices, not evidence that the
  alternatives are slow.
- Statistical weakness: bounded. Within-process repeats are summarized as one
  block. Any nominated repair receives eight fresh-process pairs.
- Environment mismatch: avoided. CPU affinity, NUMA, XLA, threads, fixture,
  timing boundary, and source identity are fixed across formulations.
- Commands answer the question: the harness records parity, compile behavior,
  HLO structure, wall time, CPU time, and contamination. A speed-only artifact
  without these fields is inadmissible.
- Resource risk: bounded by the tiny smoke, compile/call/wall timeouts, 16 GiB
  RSS cap, candidate-local failure handling, and no large-cell execution before
  a confirmed small-cell repair.

Execution is authorized because the plan now distinguishes correctness of the
existing implementation from optimality of its XLA formulation and includes
the missing mapping counterfactuals without weakening numerical or resource
gates.
