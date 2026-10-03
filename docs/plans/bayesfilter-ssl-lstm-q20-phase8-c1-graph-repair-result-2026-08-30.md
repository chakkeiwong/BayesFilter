# Phase 8 C1 graph-repair result

Date: 2026-08-30  
Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-subplan-2026-08-30.md`  
Status: `C1_REPAIR_CLOSED_CHUNK8_PREFIX_PASS_FULL_BANK_TIMEOUT`

## Question and evidence contract

The question was whether the q=20 held-out validation diagnostic could avoid
the unresolved single large static graph by evaluating 256 rows as fixed,
non-singleton `B=8` TensorFlow/XLA chunks. The diagnostic was allowed to test
graph feasibility only. A pass required all 256 rows, finite statuses, exact
prefix replay parity, verified GPU memory growth, and a complete manifest.
No training, sampling, candidate selection, or posterior claim was authorized.

## Commands and artifacts

The first implementation (`attempt-01-chunk8`) compiled a q=20 direct `B=32`
parity graph before entering the chunked path and hit its 600-second cap. Its
timeout record is preserved at:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-graph-repair/attempt-01-chunk8/timeout.json
```

The repaired command used only static `B=8` calls:

```text
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
BAYESFILTER_GPU_LAUNCH_MODE=c1_graph_repair_direct \
timeout --signal=TERM --kill-after=30s 600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_phase8_c1_graph_repair_2026_08_30.py \
--output-dir docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-graph-repair/attempt-02-chunk8 \
--chunk-size 8
```

The second attempt reached GPU 0 with XLA and on-demand allocation. It
completed the four-chunk 32-row prefix in `192.6942829299951` seconds. All 32
rows were finite, valid, and had reference-chart density/score residuals at
float64 roundoff. It then entered the 256-row full-bank stage and exited with
code `124` at the 600-second cap before writing a manifest. The preserved
records are:

```text
.../attempt-02-chunk8/stage-chunked-prefix-done.json
.../attempt-02-chunk8/stage-full-bank-start.json
.../attempt-02-chunk8/timeout.json
```

The measured prefix implies roughly 1,540 seconds for 32 sequential `B=8`
chunks under the same process, although that extrapolation is descriptive and
not a performance claim.

## Decision table

| Decision | Primary criterion | Veto status | Interpretation | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this graph-repair attempt | Complete 256-row receipt within 600 s | Timeout veto | Chunking avoids the `B=256` graph but repeated q=20 target calls remain too costly | Design a new graph-level optimization or multi-device/target-evaluation plan with a fresh budget and output root | No claim that chunking is mathematically wrong or that the transport failed |
| Retain the helper | Analytic direct/chunked tests and static batch contract | Pass | Every call remains a static non-singleton batch; no row mapping or pfor was introduced | Keep as diagnostic utility pending a broader cost plan | No production/default promotion |
| Gateway readiness | GPU probe and process reach GPU 0 | Pass | Service/device boundary works independently of the q=20 graph cost | Keep the repository launcher and narrow service rule | No scientific or memory-capacity claim |

## Inference-status table

| Evidence class | Result |
|---|---|
| Hard veto screen | Focused tests passed; GPU memory growth and XLA initialization passed; full-bank receipt was vetoed by timeout. |
| Statistically supported ranking | None. No candidate comparison ran. |
| Descriptive-only differences | The 32-row prefix was finite and roundoff-accurate; repeated `B=8` work did not fit 600 s. |
| Default readiness | Not established for the transport ensemble, whitening, posterior, or NeuTra/HMC. The repository GPU execution default is independently verified. |
| Next evidence needed | A reviewed graph-level repair (for example, one compiled batch body with an explicitly batch-native chunk loop, a target-kernel optimization, or a justified multi-GPU evaluator), followed by a complete cost receipt. |

## Classification and stop

The result invalidates only the available 600-second feasibility evidence. It
does not invalidate the target mathematics, bridge, transport implementation,
or research direction. The first attempt exposed a diagnostic ordering flaw;
the repaired attempt removed that flaw and still found a real per-call graph
cost limit. C2--C5 remain closed under the existing Phase 8 authority.

Continuing requires a materially new graph implementation or execution design
and a fresh reviewed budget. The service approval problem is not a blocker:
the independent probe and both repair launches reached GPU 0 without a
per-run idle-GPU or Luna gate.
