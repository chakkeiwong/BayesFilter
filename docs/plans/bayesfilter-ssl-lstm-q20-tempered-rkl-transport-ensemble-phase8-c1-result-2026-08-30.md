# Phase 8 C1 result: q=20 GPU cost feasibility

Date: 2026-08-30  
Plan:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md`  
Status: `C1_CLOSED_BUDGET_EXHAUSTED_LARGE_BATCH_GRAPH_LIMIT`

## Question and evidence contract

The question was whether the frozen q=20 tempered reverse-KL ensemble can be
run as a batch-native TensorFlow/TFP GPU workload within the Phase 8 cost
envelope. The comparator was the same q=20 target and bridge at the two frozen
batch hypotheses (`B=8`, `B=32`), with two independent charts, XLA enabled,
and on-demand GPU allocation. A pass required a complete two-batch cost
manifest, finite target/training values, immutable checkpoint replay, learned-
map reliability, and peak allocation below 4 GiB. A timeout was a feasibility
failure for the attempt, not evidence against the transport mathematics or a
candidate-quality result.

## Commands and environment

All three GPU attempts used the repository-owned launcher:

```text
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

The launcher selected GPU 0, exported `TF_FORCE_GPU_ALLOW_GROWTH=true` before
TensorFlow import, and did not invoke `codex-gpu-probe`, an idle-GPU probe, or a
per-run Luna reviewer. TensorFlow initialized one NVIDIA RTX 4080 SUPER
(30,103 MiB reported), with XLA and TF32 enabled. The outer service permission
was an execution boundary only; `external_approval_is_runner_gate=false` in
the runner contract. CPU-hidden compatibility and the focused 13-test
transport suite passed before these attempts.

## Attempts

| Attempt | Mode | Cap (s) | Last completed operation | Exit | Interpretation |
|---|---|---:|---|---:|---|
| `attempt-02-default-gpu` | cost pilot, validation 256 | 1,800 | B=8 chart-0 beta-0 checkpoint/replay | 124 | Large q=20 graph/compile cost; no candidate result |
| `attempt-03-target-localization` | target localization | 900 | B=8 beta-0 and beta-0.5 target calls; B=256 beta-0 started | 124 | B=8 target calls finite; B=256 static graph did not finish |
| `attempt-04-cost-smallbank` | cost pilot, validation 8 | 2,700 | B=8 both charts; B=32 chart-0 beta-0.5 checkpoint | 124 | Reduced bank progressed, but complete two-batch receipt did not finish |

The timeout records are:

- `phase8-calibration/attempt-02-default-gpu/timeout.json`;
- `phase8-calibration/attempt-03-target-localization/timeout.json`; and
- `phase8-calibration/attempt-04-cost-smallbank/timeout.json`.

The final attempt consumed the remaining C1 allocation. The partial checkpoint
hashes are retained in its timeout record and are not promoted to a run
manifest or candidate artifact.

## Decision table

| Decision | Primary criterion | Hard veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C1 as infeasible under this budget | Complete two-batch cost/reliability receipt | Timeout veto; no numerical invalidity observed | q=20 static graph and post-training reliability compile cost | Obtain a new budget or optimize the target graph, then create a new reviewed subplan and output root | No whitening, mode-discovery, posterior, HMC, superiority, or scaling claim |
| Keep repository GPU default | Launcher reaches GPU 0 with pre-import memory growth and no idle probe | Pass for execution-boundary implementation | Outer service may still deny access in another session | Configure one narrow persistent service rule for the launcher | No claim about service-wide GPU availability |

## Inference-status table

| Inference status | Result |
|---|---|
| Hard veto screen | C0 fixtures and GPU mechanics passed. C1 was vetoed only by its declared wall-time cap; no nonfinite target, memory-growth failure, checkpoint mismatch, or pfor violation was observed. |
| Statistically supported ranking | None; no candidate comparison or confirmation stream ran. |
| Descriptive-only differences | B=8 target calls were finite; partial chart checkpoints were written. These observations are feasibility diagnostics only. |
| Default readiness | Not established for the transport ensemble or NeuTra/HMC. Repository GPU execution default is implemented, but scientific default readiness is separate. |
| Next evidence needed | A new bounded graph-optimization/cost plan, followed by a complete cost receipt and then C2--C5; untouched Phase 9 streams remain preserved. |

## Classification and stop

The result invalidates the available C1 feasibility evidence, not the harness,
target mathematics, data, or reverse-KL ensemble direction. The repeated
pattern is a q=20 TensorFlow/XLA graph-cost limitation: B=8 target calls finish
and the small-bank arm trains multiple charts, while the full validation and
remaining B=32 receipt exceed the authorized C1 wall cap. C2--C5 are therefore
not launched. Continuing requires a materially new target-graph optimization
or compute allocation, which is outside this subplan and needs a new reviewed
experiment contract.
