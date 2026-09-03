# Phase 8 C1 graph-cost repair subplan

Date: 2026-08-30  
Parent plan:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`  
Prior result:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-c1-result-2026-08-30.md`  
Status: `ATTEMPT_02_PREFIX_PASS_FULL_BANK_TIMEOUT_GRAPH_BLOCKER`

## Why this subplan exists

The previous Phase 8 C1 cost pilot exhausted its own 5,400-second allocation
before completing a receipt. That stop remains valid and is not reopened by
this document. The independent GPU boundary probe now passes, so the next
smallest discriminating question is whether the q=20 XLA graph is expensive
because the validation diagnostic uses one large static batch rather than a
sequence of ordinary batched calls.

The latest user instruction to continue is treated as authorization for this
bounded infrastructure/graph diagnostic. It does **not** authorize a new
claim-bearing training campaign, reassignment of the reserved Phase 9 budget,
or a change to the scientific target and promotion criteria.

## Research intent ledger

| Field | Definition |
|---|---|
| Question | Can a 256-row held-out Gaussianization diagnostic be evaluated as a finite collection of static batches of size 8 or 32, avoiding the unresolved single `B=256` XLA graph? |
| Baseline | The analytic CPU fixture is the direct-vs-chunked parity authority. The known q=20 direct `B=256` attempt and the new q=20 direct `B=32` attempt are preserved as timeouts and are not rerun merely for comparison. |
| Mechanism | Chunk only the outer validation bank. Each target call still receives a static rank-2 tensor with more than one row; row values, scores, statuses, and residuals are concatenated without replacement or filtering. |
| Expected failure | A chunk still fails to compile, a chunk returns invalid/nonfinite rows, aggregate statistics differ from the direct small-bank calculation, or total cost remains outside the bounded diagnostic cap. |
| Pass criterion | Analytic direct-vs-chunked parity passes within the declared float64 tolerance; a q=20 256-row bank completes using one declared chunk size (the retry uses 8); all rows are accounted for; GPU/XLA and memory-growth receipts pass; no forbidden row-mapping construct is introduced. |
| Promotion role | Feasibility/repair trigger only. A pass permits drafting a new C1 cost receipt and reviewed continuation plan; it does not select a transport or establish whitening, mode coverage, posterior correctness, HMC readiness, or scaling. |
| Vetoes | GPU or memory-policy failure, target/bridge signature drift, nonfinite or invalid rows, singleton/scalar target calls, pfor/`tf.map_fn`/`tf.vectorized_map`, incomplete row accounting, output collision, or timeout at the diagnostic cap. |
| Nonclaims | No statement about candidate quality, Gaussianization, mode discovery, convergence, or statistical ranking follows from this subplan. |

## Scope and budget

This is a fresh output root and a separate repair budget. The bounded command
cap is **600 seconds**, chosen as a short diagnostic envelope: it is longer
than the observed first `B=8` target compilation (about 55 seconds) but far
shorter than the closed C1 campaign. The number is a feasibility cap, not a
performance target and not an authorization to spend the reserved Phase 9
allocation. One launch is allowed for the first implementation; a localized
invocation failure may receive one same-scope retry after a focused repair,
with the total cap still 600 seconds.

Output root:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-graph-repair/
```

Every attempted output directory is fresh. Existing timeout and partial
checkpoint artifacts under the 2026-08-29 root are read-only historical
evidence and are not overwritten or used as warm starts.

## Mathematical and implementation contract

Let the held-out latent bank be

\[
 Z=(z_1,\ldots,z_{256})\in\mathbb R^{256\times4}.
\]

For a fixed chart and beta, the existing diagnostic defines a per-row map

\[
 D_C(Z_{a:b})
 = \bigl(r_j,u_j,\ell_j,s_j,\text{status}_j\bigr)_{j=a}^{b-1}
\]

by one batch-native call on the static slice `Z[a:b]`, where the chunk size
`C=b-a` is 8 or 32. The repaired aggregate is the concatenation

\[
 D_{C}^{\mathrm{all}}(Z)
 = D_C(Z_{0:C})\,\Vert\,D_C(Z_{C:2C})\,\Vert\,\cdots .
\]

For a 256-row bank, both candidate sizes divide 256 exactly. Since the target
value, score, transport map, and Jacobian are applied independently to each
row and no statistic is used to alter a later row, concatenation followed by
the same reductions (mean, RMS, median, and q90) is the same finite-bank
diagnostic functional, up to floating-point reduction order. The target law is
not changed: chunking changes only graph batching and evaluation order.

The implementation must preserve these conditions:

- every target invocation has a statically known rank-2 shape `[C,4]` with
  `C > 1`;
- no Python loop iterates over individual samples and no scalar target call is
  introduced;
- no row is dropped, replaced, resampled, or conditionally used to influence
  another row;
- the bridge and chart are the same frozen objects and signatures as the
  existing diagnostic;
- aggregation is outside the XLA target kernel and records the exact chunk
  size and row count; and
- the helper is diagnostic-only until a separate review proves that its use in
  a claim-bearing route preserves the route's batching and evidence contract.

## Skeptical pre-run audit

The audit was completed before execution:

| Risk checked | Finding and control |
|---|---|
| Wrong baseline | The direct large-bank timeout is not treated as a numerical baseline. A direct 32-row call is the parity authority because it completes under the existing route. |
| Proxy promoted as science | Completion time and chunk parity are feasibility diagnostics only; no whitening or candidate gate is changed. |
| Hidden scalar fallback | Static source scan and runtime shape receipt must show `[C,4]`, `C in {8,32}`, for every call. |
| Changed target measure | Target/bridge signatures, beta, latent bank, and chart state are frozen and hashed; only partitioning changes. |
| Missing stop condition | The 600-second total cap, one launch, fresh output root, and explicit failure classes stop the repair. |
| Memory illusion | GPU growth is configured before TensorFlow import and the manifest records current/peak allocator bytes; `nvidia-smi` reservation is not used as live tensor memory. |
| Unfair comparison | The same latent rows and chart are used for direct/chunked parity; reduction-order tolerance is declared rather than silently ignored. |
| Stale context | C1 remains closed. A successful repair must lead to a new reviewed subplan before any C2--C5 launch. |

The first implementation of this design had a material ordering flaw: it
compiled a q=20 direct `B=32` parity call before the chunked path, which timed
out and answered neither parity nor chunk feasibility. That attempt is
preserved as evidence-only. The repair moves algebraic parity to the analytic
CPU fixture and compares the q=20 prefix returned by the chunked path with the
same prefix in the full chunked bank. No material flaw remains in the repaired
bounded design; its result can only be a graph-feasibility receipt.

## Execution sequence

1. Add a small repository helper with an explicit `chunk_size` and static shape
   validation. Keep the current one-shot function unchanged as the parity
   authority.
2. Add CPU-hidden unit tests using an analytic affine chart and a fake
   batch-native bridge. Test exact row accounting, rejection of chunk sizes
   1/non-divisors, finite aggregation, and direct-vs-chunked parity.
3. Run the focused tests and a static forbidden-token scan.
4. Run one trusted GPU diagnostic on GPU 0 through
   `scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh` or a dedicated
   bounded diagnostic entry point. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before
   TensorFlow import, use XLA, and write a fresh JSON manifest under the output
   root.
5. If the 256-row chunked diagnostic completes, record per-chunk target calls,
   compile/steady timings, allocator bytes, statuses, and aggregate parity.
   Do not reuse its output as training or confirmation data.
6. At the inter-phase repair review, classify the result. A pass permits a new
   C1 cost-receipt subplan; a timeout or numerical failure preserves C1's
   continuation veto and triggers graph-level investigation. Neither result
   permits C2 automatically.

## Required artifact fields

The JSON manifest must contain:

```text
schema = bayesfilter.ssl_lstm_q20.c1_graph_repair.v1
status = PASS_C1_CHUNKED_DIAGNOSTIC or a named failure status
target_signature and bridge_signature
chart/component identity and beta
latent_row_count = 256
chunk_size and chunk_count
per_chunk_static_shape and per_chunk_status counts
all_rows_accounted = true
direct_32_vs_chunked_32 parity metrics and tolerance
logical_gpus, gpu_launch_mode, gpu_trust_basis
memory_policy and TF_FORCE_GPU_ALLOW_GROWTH
XLA/TF32 settings, command, environment, seed, git commit, wall time
nonclaims
```

The final note must include a decision table and an inference-status table.
It must state explicitly whether the result invalidated the harness,
implementation, target, or only the current graph batching choice.

## Stop and continuation rules

- Stop immediately on a hard veto or when 600 seconds of this repair budget is
  consumed.
- A pass does not reopen the closed C1 allocation. It only justifies drafting a
  new cost receipt with a new budget allocation and output root.
- A localized import or artifact-writing error may be repaired and retried once
  under the same contract. Do not change target, beta, chart, chunk semantics,
  or scientific promotion criteria during that retry.
- Any change to target code, derivative backend, validation semantics, or
  campaign budget requires a new reviewed subplan.

## Execution update (2026-08-30)

The first implementation was repaired after its q=20 direct-`B=32` parity
call consumed the cap before chunking. The repaired `B=8`-only attempt passed
the 32-row prefix in `192.6942829299951` seconds but timed out in the full
256-row stage at 600 seconds. The terminal result is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-result-2026-08-30.md`.
This subplan is closed; a compiled batch-body loop, target-kernel
optimization, or justified multi-device evaluator must be specified in a new
reviewed subplan before another q=20 launch.
